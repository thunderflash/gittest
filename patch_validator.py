#!/usr/bin/env python3
"""
Stage S4 — Patch Validation

对 Gold Patch 进行多层验证，确保训练数据的高质量：

Gate 1 — Diff Parse: 统一 diff 可解析为 FILE/OLD_STRING/NEW_STRING 编辑块
Gate 2 — File Existence: 修改的文件在 base_commit 上存在
Gate 3 — Apply: patch 可用 Python 字符串替换应用（模拟 git apply）
Gate 4 — Non-empty: patch 至少有一个非空编辑块
Gate 5 — Syntax: 修改后的文件可通过 Python AST 解析（无语法错误）
Gate 6 — Relevance: 修改的文件与 src_files / test_files 相关

输出：
  <output_dir>/validation/<instance_id>.json       验证结果
  <output_dir>/validation/summary.json             汇总统计（Gate 通过率）

验证结果决定标签是否进入正式训练：
- 所有 Gate 通过 -> Strong 标签，可用于训练
- 部分 Gate 失败 -> 标记为 verified=False，降低 confidence
- 关键 Gate 失败 (Parse/Apply/Non-empty) -> 标签废弃
"""

import json
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import paths_config as pc  # noqa: E402
except ImportError:  # pragma: no cover
    pc = None


# ---------------------------------------------------------------------------
# Diff 解析（统一 diff -> 编辑块）
# ---------------------------------------------------------------------------

def parse_gold_diff(gold_diff: str) -> list[dict[str, str]]:
    """将统一 diff 解析为编辑块列表。

    每个编辑块: {filepath, old_string, new_string}
    """
    edits: list[dict[str, str]] = []
    if not gold_diff or "@@" not in gold_diff:
        return edits

    lines = gold_diff.splitlines(keepends=True)
    i = 0
    current_file = None

    while i < len(lines):
        line = lines[i]

        if line.startswith("+++ b/"):
            current_file = line[6:].rstrip("\n")
            i += 1
            continue
        if line.startswith("--- a/"):
            i += 1
            continue

        if line.startswith("@@"):
            old_lines: list[str] = []
            new_lines: list[str] = []
            i += 1
            while i < len(lines):
                l = lines[i]
                if l.startswith("@@") or l.startswith("--- a/") or l.startswith("+++ b/"):
                    break
                if l.startswith("-") and not l.startswith("---"):
                    old_lines.append(l[1:])
                elif l.startswith("+") and not l.startswith("+++"):
                    new_lines.append(l[1:])
                elif l.startswith(" ") or l == "\\":
                    if l.startswith(" "):
                        old_lines.append(l[1:])
                        new_lines.append(l[1:])
                i += 1

            if current_file and old_lines and new_lines:
                edits.append({
                    "filepath": current_file,
                    "old_string": "".join(old_lines),
                    "new_string": "".join(new_lines),
                })
            continue

        i += 1

    return edits


# ---------------------------------------------------------------------------
# Patch 应用（Python 字符串替换，模拟 git apply）
# ---------------------------------------------------------------------------

def apply_edit_to_text(text: str, old_string: str, new_string: str) -> tuple[bool, str]:
    """对单个文本应用单个编辑块。

    返回 (成功, 结果消息或修改后文本)。
    """
    if old_string not in text:
        # 尝试归一化匹配（忽略尾随空白）
        norm_old = "\n".join(l.rstrip() for l in old_string.splitlines())
        norm_lines = [l.rstrip() for l in text.splitlines()]
        found = False
        for i in range(len(norm_lines) - len(norm_old.splitlines()) + 1):
            window = "\n".join(norm_lines[i:i + len(norm_old.splitlines())])
            if window == norm_old:
                # 找到归一化匹配，用实际文本替换
                actual_old = "\n".join(text.splitlines()[i:i + len(norm_old.splitlines())])
                text = text.replace(actual_old, new_string, 1)
                found = True
                break
        if not found:
            return False, f"old_string not found (best-effort normalization failed)"
        return True, text

    text = text.replace(old_string, new_string, 1)
    return True, text


def git_apply_patch(work_dir: Path, gold_diff: str) -> tuple[bool, str]:
    """F3: 使用 git apply 在临时副本上真实应用 patch（首选路径）。

    返回 (成功, 消息)。git apply 在非 repo 目录也可用（按相对路径应用）。
    """
    # 统一 LF 行尾，否则 diff 解析可能失败
    diff_text = gold_diff.replace("\r\n", "\n")
    patch_file = work_dir / "_gold.patch"
    try:
        with open(patch_file, "w", encoding="utf-8", newline="\n") as f:
            f.write(diff_text)
    except Exception as e:
        return False, f"write patch file failed: {e}"

    common = dict(cwd=str(work_dir), capture_output=True, text=True, timeout=120)
    try:
        chk = subprocess.run(["git", "apply", "--check", "_gold.patch"], **common)
        if chk.returncode != 0:
            return False, f"git apply --check failed: {(chk.stderr or chk.stdout)[:500]}"
        app = subprocess.run(["git", "apply", "_gold.patch"], **common)
        if app.returncode != 0:
            return False, f"git apply failed: {(app.stderr or app.stdout)[:500]}"
        return True, "git_apply ok"
    except FileNotFoundError:
        return False, "git not available"
    except Exception as e:
        return False, f"git apply error: {e}"


def apply_patch_to_repo(
    repo_dir: Path,
    edits: list[dict[str, str]],
    gold_diff: str = "",
) -> tuple[bool, str, list[dict[str, Any]], Path | None, str]:
    """将 patch 应用到 repo 的临时工作副本。

    F3: 首选 git apply（真实 patch 语义：hunk 上下文/行号/新文件/删除），
    不可用或失败时回退到字符串替换，并标记 apply_method 供审计。
    注意：fallback 结果不能视为正式 patch 验证（多 hunk/重名等语义不受保证）。

    返回 (全部成功, 消息, 逐文件详情, 临时工作目录或 None, apply_method)。
    """
    file_details: list[dict[str, Any]] = []
    work_dir = Path(tempfile.mkdtemp(prefix="patch_validate_"))

    # 先复制整个 repo 到工作目录
    try:
        shutil.copytree(repo_dir, work_dir, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    except Exception as e:
        return False, f"copy repo failed: {e}", file_details, None, "copy_failed"

    # 首选：git apply 真实应用
    if gold_diff.strip():
        git_ok, git_msg = git_apply_patch(work_dir, gold_diff)
        if git_ok:
            for edit in edits:
                filepath = edit["filepath"]
                p = work_dir / filepath
                detail = {
                    "filepath": filepath,
                    "exists": p.exists(),
                    "applied": p.exists(),
                    "error": None,
                    "modified_path": str(p) if p.exists() else None,
                    "modified_content": None,
                }
                file_details.append(detail)
            return True, "applied via git apply", file_details, work_dir, "git_apply"
        print(f"[VALIDATE] git apply failed ({git_msg[:120]}); falling back to string replace")

    # 回退：字符串替换（仅作为降级路径，结果标记 fallback）
    for edit in edits:
        filepath = edit["filepath"]
        old_s = edit["old_string"]
        new_s = edit["new_string"]
        p = work_dir / filepath

        detail = {
            "filepath": filepath,
            "exists": p.exists(),
            "applied": False,
            "error": None,
            "modified_path": None,
            "modified_content": None,
            "is_new_file": not old_s,
        }

        if not p.exists():
            # 新文件：old_string 为空时直接创建
            if not old_s:
                p.parent.mkdir(parents=True, exist_ok=True)
                try:
                    p.write_text(new_s, encoding="utf-8")
                    detail["applied"] = True
                    detail["exists"] = True
                    detail["modified_path"] = str(p)
                    detail["modified_content"] = new_s
                except Exception as e:
                    detail["error"] = f"create new file failed: {e}"
                file_details.append(detail)
                continue
            # 尝试模糊查找同名文件
            matches = list(work_dir.rglob(Path(filepath).name))
            matches = [m for m in matches if m.is_file()]
            if len(matches) == 1:
                p = matches[0]
                detail["filepath"] = str(p.relative_to(work_dir))
                detail["exists"] = True
            elif len(matches) > 1:
                detail["error"] = f"ambiguous path: {filepath}"
                file_details.append(detail)
                continue
            else:
                detail["error"] = f"file not found: {filepath}"
                file_details.append(detail)
                continue

        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            detail["error"] = f"read error: {e}"
            file_details.append(detail)
            continue

        ok, result = apply_edit_to_text(text, old_s, new_s)
        if ok:
            detail["applied"] = True
            detail["modified_path"] = str(p)
            detail["modified_content"] = result
            try:
                p.write_text(result, encoding="utf-8")
            except Exception as e:
                detail["error"] = f"write error: {e}"
                detail["applied"] = False
        else:
            detail["error"] = result

        file_details.append(detail)

    all_ok = all(d["applied"] for d in file_details if d["exists"])
    method = "string_replace_fallback"
    return all_ok, "applied" if all_ok else "partial_failure", file_details, work_dir, method


# ---------------------------------------------------------------------------
# 语法验证
# ---------------------------------------------------------------------------

def check_python_syntax(filepath: Path) -> tuple[bool, str | None]:
    """检查 Python 文件语法是否正确。"""
    try:
        source = filepath.read_text(encoding="utf-8", errors="replace")
        compile(source, str(filepath), "exec")
        return True, None
    except SyntaxError as e:
        return False, f"SyntaxError: {e.msg} at line {e.lineno}"
    except Exception as e:
        return False, f"Error: {e}"


# ---------------------------------------------------------------------------
# 主验证流程
# ---------------------------------------------------------------------------

def validate_patch(
    instance_id: str,
    gold_diff: str,
    repo_dir: Path,
    src_files: list[str],
    test_files: list[str],
) -> dict[str, Any]:
    """对单个 patch 进行完整验证。"""
    result = {
        "instance_id": instance_id,
        "gold_diff_length": len(gold_diff) if gold_diff else 0,
        "gates": {},
        "edits_count": 0,
        "file_details": [],
        "verified": False,
        "confidence": 0.0,
        "fail_reasons": [],
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }

    if not gold_diff.strip():
        result["fail_reasons"].append("empty_gold_diff")
        result["gates"]["parse"] = {"passed": False, "reason": "empty gold_diff"}
        return result

    # Gate 1 — Diff Parse
    edits = parse_gold_diff(gold_diff)
    result["edits_count"] = len(edits)
    parse_ok = len(edits) > 0
    result["gates"]["parse"] = {
        "passed": parse_ok,
        "edits_count": len(edits),
    }
    if not parse_ok:
        result["fail_reasons"].append("parse_failed")
        return result

    # Gate 2 & 3 — File Existence + Apply (写入临时工作副本)
    apply_ok, apply_msg, file_details, work_dir, apply_method = \
        apply_patch_to_repo(repo_dir, edits, gold_diff=gold_diff)
    result["file_details"] = file_details
    result["apply_method"] = apply_method

    files_exist = all(d["exists"] for d in file_details)
    files_applied = all(d["applied"] for d in file_details if d["exists"])
    result["gates"]["file_existence"] = {
        "passed": files_exist,
        "files_checked": len(file_details),
        "files_found": sum(1 for d in file_details if d["exists"]),
    }
    result["gates"]["apply"] = {
        "passed": files_applied,
        "message": apply_msg,
        "method": apply_method,
    }

    if not files_exist:
        result["fail_reasons"].append("file_not_found")
    if not files_applied:
        result["fail_reasons"].append("apply_failed")

    # Gate 4 — Non-empty
    non_empty = any(e["old_string"] or e["new_string"] for e in edits)
    result["gates"]["non_empty"] = {"passed": non_empty}
    if not non_empty:
        result["fail_reasons"].append("empty_edits")

    # Gate 5 — Syntax on MODIFIED files (from temp work dir, not originals)
    syntax_results = {}
    syntax_passed = True
    if work_dir is not None:
        for detail in file_details:
            if detail["exists"] and detail["applied"] and detail.get("modified_path"):
                p = Path(detail["modified_path"])
                ok, err = check_python_syntax(p)
                syntax_results[detail["filepath"]] = {"passed": ok, "error": err}
                if not ok:
                    syntax_passed = False
        # 清理临时目录
        try:
            shutil.rmtree(work_dir, ignore_errors=True)
        except Exception:
            pass
    result["gates"]["syntax_modified"] = {
        "passed": syntax_passed,
        "results": syntax_results,
    }

    # Gate 6 — Relevance（F3: 分类记录，仅作为质量指标，不作为 correctness）
    modified_files = [d["filepath"] for d in file_details if detail_applied(d)]
    src_set = set(src_files or [])
    test_set = set(test_files or [])
    categories: dict[str, list[str]] = {"source": [], "test": [], "other": []}
    for f in modified_files:
        norm = f.replace("\\", "/")
        if norm in src_set:
            categories["source"].append(f)
        elif norm in test_set or "/tests/" in norm or norm.startswith("tests/") \
                or Path(norm).name.startswith("test_") or Path(norm).name.endswith("_test.py"):
            categories["test"].append(f)
        else:
            categories["other"].append(f)
    relevant = any(f in src_set for f in modified_files)
    result["gates"]["relevance"] = {
        "passed": relevant,
        "quality_metric_only": True,   # F3: 不参与 correctness 判定
        "categories": categories,
        "modified_files": modified_files,
        "src_files": src_files,
        "test_files": test_files,
    }

    # 综合判定：critical = parse + file_existence + apply + non_empty + syntax_modified
    # relevance 为质量指标（F3），git_apply 为首选方法（fallback 需人工复核）
    critical_passed = (
        parse_ok and files_exist and files_applied and non_empty and syntax_passed
    )
    if apply_method == "string_replace_fallback":
        # fallback 语义不受保证：不判 verified=True，交由人工/云端测试确认
        critical_passed = False
        result["fail_reasons"].append("apply_method_fallback_requires_review")
    result["verified"] = critical_passed

    # 置信度计算
    gate_scores = {
        "parse": 1.0 if parse_ok else 0.0,
        "file_existence": 1.0 if files_exist else 0.0,
        "apply": 1.0 if files_applied else 0.0,
        "non_empty": 1.0 if non_empty else 0.0,
        "syntax_modified": 1.0 if syntax_passed else 0.0,
        "relevance": 1.0 if relevant else 0.0,
    }
    result["confidence"] = round(sum(gate_scores.values()) / len(gate_scores), 2)

    if not critical_passed:
        result["verified"] = False

    return result


def detail_applied(d: dict) -> bool:
    return d.get("exists", False) and d.get("applied", False)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def validate_patches_for_tasks(
    task_ids: list[str],
    output_dir: Path,
) -> list[dict[str, Any]]:
    """为多个任务验证 patch。"""
    recon_path = pc.RECON_PATH if pc is not None else None
    recon = {}
    if recon_path is not None and recon_path.exists():
        recon = json.loads(recon_path.read_text(encoding="utf-8"))

    results = []
    for tid in task_ids:
        task = recon.get(tid, {})
        if not task:
            print(f"[SKIP] {tid}: not found in reconstructed.json")
            continue

        gold_diff = task.get("gold_diff", "")
        src_files = task.get("src_files", [])
        test_files = task.get("test_files", [])

        repo_dir = (pc.REPOS_DIR / tid) if pc is not None else Path(tid)
        if not repo_dir.exists():
            print(f"[WARN] {tid}: repo directory not found, skipping validation")
            continue

        print(f"[VALIDATE] {tid}")
        res = validate_patch(tid, gold_diff, repo_dir, src_files, test_files)
        results.append(res)

        status = "PASS" if res["verified"] else "FAIL"
        print(f"  [{status}] confidence={res['confidence']} edits={res['edits_count']} "
              f"fail_reasons={res['fail_reasons']}")

    # 汇总
    summary_path = output_dir / "validation_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    total = len(results)
    passed = sum(1 for r in results if r["verified"])
    gate_stats = {}
    for r in results:
        for gate, g in r["gates"].items():
            if gate not in gate_stats:
                gate_stats[gate] = {"passed": 0, "total": 0}
            gate_stats[gate]["total"] += 1
            if isinstance(g, dict) and g.get("passed"):
                gate_stats[gate]["passed"] += 1

    summary = {
        "total_tasks": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": round(passed / max(total, 1), 4),
        "gate_stats": gate_stats,
        "results": results,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nValidation summary: {passed}/{total} passed ({summary['pass_rate']*100:.1f}%)")
    print(f"Summary saved to {summary_path}")
    return results


def main():
    ap = __import__("argparse").ArgumentParser(description="Stage S4: Patch Validation")
    ap.add_argument("--tasks", nargs="+", required=True,
                    help="要验证的 task id 列表")
    ap.add_argument("--output-dir", default=str(pc.PROCESSED_DIR) if pc is not None else "./data/processed",
                    help="处理后数据输出目录")
    args = ap.parse_args()

    validate_patches_for_tasks(
        task_ids=args.tasks,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
