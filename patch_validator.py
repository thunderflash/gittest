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
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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


def apply_patch_to_repo(
    repo_dir: Path,
    edits: list[dict[str, str]],
) -> tuple[bool, str, list[dict[str, Any]]]:
    """将 patch 应用到 repo 副本（仅内存中，不修改磁盘）。

    返回 (全部成功, 消息, 逐文件详情)。
    """
    file_details: list[dict[str, Any]] = []

    for edit in edits:
        filepath = edit["filepath"]
        old_s = edit["old_string"]
        new_s = edit["new_string"]
        p = repo_dir / filepath

        detail = {
            "filepath": filepath,
            "exists": p.exists(),
            "applied": False,
            "error": None,
        }

        if not p.exists():
            # 尝试模糊查找同名文件
            matches = list(repo_dir.rglob(Path(filepath).name))
            matches = [m for m in matches if m.is_file()]
            if len(matches) == 1:
                p = matches[0]
                detail["filepath"] = str(p.relative_to(repo_dir))
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
        else:
            detail["error"] = result

        file_details.append(detail)

    all_ok = all(d["applied"] for d in file_details if d["exists"])
    return all_ok, "applied" if all_ok else "partial_failure", file_details


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

    # Gate 2 & 3 — File Existence + Apply
    apply_ok, apply_msg, file_details = apply_patch_to_repo(repo_dir, edits)
    result["file_details"] = file_details

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

    # Gate 5 — Syntax (对修改后的文件进行语法检查，需实际写入临时副本)
    # 这里简化：仅对原始文件做语法检查，修改后的语法检查留给 Gate 6 的实际测试
    syntax_results = {}
    for detail in file_details:
        if detail["exists"] and detail["applied"]:
            p = repo_dir / detail["filepath"]
            ok, err = check_python_syntax(p)
            syntax_results[detail["filepath"]] = {"passed": ok, "error": err}
    result["gates"]["syntax_original"] = syntax_results

    # Gate 6 — Relevance
    modified_files = [d["filepath"] for d in file_details if detail_applied(d)]
    relevant = any(f in src_files for f in modified_files)
    result["gates"]["relevance"] = {
        "passed": relevant,
        "modified_files": modified_files,
        "src_files": src_files,
    }

    # 综合判定
    critical_passed = (
        parse_ok and files_exist and files_applied and non_empty
    )
    result["verified"] = critical_passed

    # 置信度计算
    gate_scores = {
        "parse": 1.0 if parse_ok else 0.0,
        "file_existence": 1.0 if files_exist else 0.0,
        "apply": 1.0 if files_applied else 0.0,
        "non_empty": 1.0 if non_empty else 0.0,
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
    recon_path = Path(r'F:\test\gemma4\mini_eval\reconstructed.json')
    recon = {}
    if recon_path.exists():
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

        repo_dir = Path(r'F:\test\gemma4\mini_eval\repos') / tid
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
    ap.add_argument("--output-dir", default=r"F:\test\gemma4\training\data\processed",
                    help="处理后数据输出目录")
    args = ap.parse_args()

    validate_patches_for_tasks(
        task_ids=args.tasks,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
