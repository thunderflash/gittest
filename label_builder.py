#!/usr/bin/env python3
"""
Stage S3 — 6-Task Label Construction with Label Provenance

为每个 Issue 构建 6 种训练任务标签，每个标签都带完整的来源追踪（Label Provenance）：

1. Issue Understanding      - 从 Issue 提取症状、预期行为、约束
2. Bug Localization         - 定位缺陷文件/符号（Weak 标签，不默认入训练）
3. Test Reasoning           - 理解测试意图、边界、失败机制
4. Root Cause & Repair Plan - 根因分析 + 修复方案
5. Patch Generation         - 生成正确 Patch（Strong 标签，来自 Gold Patch）
6. Test Result Interpretation - 基于测试执行日志的错误诊断

Label Provenance 字段：
- label_source: gold_patch / test_execution / human_review / model_proposal / derived
- label_strength: strong / medium / weak
- evidence: 支持标签的原始证据
- verified: 是否经过自动或人工验证
- confidence: 标签置信度

防泄漏：所有标签构造仅使用 base_commit 状态的代码和测试，不包含修复后代码。

输出：
  <output_dir>/labels/<instance_id>/<task_type>.jsonl
  <output_dir>/labels/summary.json
"""

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# 复用 context_builder 的符号提取工具
sys.path.insert(0, r'F:\test\gemma4\training\src')
from context_builder import (  # noqa: E402
    extract_symbols_from_text,
    extract_imports_from_test_diff,
    extract_test_funcs_from_test_diff,
    build_issue_from_test_diff,
    posix_path,
)

# ---------------------------------------------------------------------------
# Label Provenance 构造器
# ---------------------------------------------------------------------------

def make_provenance(
    label_source: str,
    label_strength: str,
    evidence: str | None = None,
    verified: bool = False,
    confidence: float = 1.0,
) -> dict[str, Any]:
    return {
        "label_source": label_source,
        "label_strength": label_strength,
        "evidence": evidence or "",
        "verified": verified,
        "confidence": confidence,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Task 1 — Issue Understanding
# ---------------------------------------------------------------------------

def build_issue_understanding(
    instance_id: str,
    issue: str,
    test_diff: str,
) -> dict[str, Any] | None:
    """从 Issue 和 test_diff 提取问题理解。"""
    if not issue.strip():
        return None

    imports = extract_imports_from_test_diff(test_diff)
    test_funcs = extract_test_funcs_from_test_diff(test_diff)

    # 从 issue 中提取症状关键词
    symptom_keywords = extract_symbols_from_text(issue)

    output = {
        "symptom": issue[:500],
        "expected_behavior": "",
        "actual_behavior": "",
        "constraints": [],
        "unknowns": [],
        "relevant_imports": imports[:10],
        "relevant_test_functions": test_funcs[:10],
    }

    # 从 test_diff 中提取预期行为（断言）
    for m in re.finditer(r'assert\s+(.+)', test_diff):
        output["expected_behavior"] += m.group(1).strip() + "; "

    provenance = make_provenance(
        label_source="derived",
        label_strength="medium",
        evidence=f"test_diff imports={imports[:5]}, test_funcs={test_funcs[:5]}",
        verified=False,
        confidence=0.6,
    )

    return {
        "sample_id": f"{instance_id}_issue_understanding",
        "instance_id": instance_id,
        "task_type": "issue_understanding",
        "difficulty": "single_file",
        "issue": issue,
        "code_context": "",
        "test_context": test_diff[:2000],
        "evidence": {
            "source_files": [],
            "tests": [],
            "verified": False,
        },
        "target": json.dumps(output, ensure_ascii=False, indent=2),
        "provenance": provenance,
    }


# ---------------------------------------------------------------------------
# Task 2 — Bug Localization (Weak，不默认入训练)
# ---------------------------------------------------------------------------

def build_bug_localization(
    instance_id: str,
    issue: str,
    src_files: list[str],
    test_diff: str,
    gold_diff: str,
) -> dict[str, Any] | None:
    """构建缺陷定位标签（Weak，基于 Gold Patch 修改文件）。

    注意：Patch 修改文件不等于缺陷唯一位置，仅作为弱标签。
    """
    if not src_files:
        return None

    # 从 gold_diff 中提取修改的文件
    modified_files = []
    for m in re.finditer(r'^\+\+\+\s+b/(.+)', gold_diff, re.MULTILINE):
        modified_files.append(m.group(1).strip())

    output = {
        "suspected_files": modified_files[:10],
        "suspected_symbols": [],
        "evidence": [
            f"Gold patch modifies: {', '.join(modified_files[:5])}"
        ],
        "confidence": "medium",
        "note": "Weak label: patch files may include refactoring, not just defect location.",
    }

    # 从 issue 中提取可能相关的符号作为 suspected_symbols
    symbols = extract_symbols_from_text(issue + " " + test_diff)
    output["suspected_symbols"] = symbols[:10]

    provenance = make_provenance(
        label_source="derived",
        label_strength="weak",
        evidence=f"gold_diff modified files: {modified_files[:5]}",
        verified=False,
        confidence=0.4,
    )

    return {
        "sample_id": f"{instance_id}_bug_localization",
        "instance_id": instance_id,
        "task_type": "bug_localization",
        "difficulty": "multi_file" if len(modified_files) > 1 else "single_file",
        "issue": issue,
        "code_context": "",
        "test_context": test_diff[:2000],
        "evidence": {
            "source_files": src_files,
            "tests": [],
            "verified": False,
        },
        "target": json.dumps(output, ensure_ascii=False, indent=2),
        "provenance": provenance,
    }


# ---------------------------------------------------------------------------
# Task 3 — Test Reasoning
# ---------------------------------------------------------------------------

def build_test_reasoning(
    instance_id: str,
    issue: str,
    test_diff: str,
    test_files: list[str],
) -> dict[str, Any] | None:
    """构建测试推理标签（Medium，需要测试证据支持）。"""
    if not test_diff.strip():
        return None

    test_funcs = extract_test_funcs_from_test_diff(test_diff)

    # 提取断言
    assertions = []
    for m in re.finditer(r'assert\s+(.+)', test_diff):
        assertions.append(m.group(1).strip())

    # 提取 pytest.raises / pytest.warns
    raises_patterns = []
    for m in re.finditer(r'pytest\.(raises|warns)\s*\(\s*([^,\)]+)', test_diff):
        raises_patterns.append(f"{m.group(1)}({m.group(2).strip()})")

    output = {
        "existing_tests_relevant": test_files[:10],
        "missing_coverage": [],
        # P0-4: 不再编造 proposed_tests，仅从 test_diff 静态提取已有断言
        "assertions": assertions[:10],
        "raises_patterns": raises_patterns[:10],
        "note": "Derived statically from test_diff; not verified by execution.",
    }

    provenance = make_provenance(
        label_source="test_diff_derived",   # P0-4: 不是 test_execution
        label_strength="weak",              # P0-4: 弱标签，静态推导
        evidence=f"test_diff has {len(test_funcs)} test functions, {len(assertions)} assertions",
        verified=False,
        confidence=0.4,
    )

    return {
        "sample_id": f"{instance_id}_test_reasoning",
        "instance_id": instance_id,
        "task_type": "test_reasoning",
        "difficulty": "single_file",
        "issue": issue,
        "code_context": "",
        "test_context": test_diff,
        "evidence": {
            "source_files": [],
            "tests": test_files,
            "verified": False,
        },
        "target": json.dumps(output, ensure_ascii=False, indent=2),
        "provenance": provenance,
    }


# ---------------------------------------------------------------------------
# Task 4 — Root Cause & Repair Plan
# ---------------------------------------------------------------------------

def build_root_cause_repair_plan(
    instance_id: str,
    issue: str,
    src_files: list[str],
    test_files: list[str],
    gold_diff: str,
    test_diff: str,
) -> dict[str, Any] | None:
    """构建根因与修复方案标签（Medium，需要证据门禁验证）。

    不能仅凭 Gold Patch 反推根因，需要证据支持。
    """
    if not gold_diff.strip():
        return None

    # 从 gold_diff 中提取修改的文件和大致逻辑
    modified_files = []
    for m in re.finditer(r'^\+\+\+\s+b/(.+)', gold_diff, re.MULTILINE):
        modified_files.append(m.group(1).strip())

    # 提取添加的行（+ 开头，非 +++）
    added_lines = []
    removed_lines = []
    for line in gold_diff.splitlines():
        if line.startswith('+') and not line.startswith('+++'):
            added_lines.append(line[1:])
        elif line.startswith('-') and not line.startswith('---'):
            removed_lines.append(line[1:])

    output = {
        "root_cause": "Derived from gold patch analysis - pseudo-supervision, see evidence",
        "evidence": [
            f"Modified files: {', '.join(modified_files[:5])}",
            f"Added {len(added_lines)} lines, removed {len(removed_lines)} lines",
        ],
        "repair_strategy": "Apply gold patch modifications",
        "target_components": modified_files[:5],
        "implementation_steps": [
            "Identify defect location",
            "Implement fix per gold patch",
            "Verify against target tests",
        ],
        "risks": ["Patch may include refactoring not related to defect"],
        "alternatives_rejected": [],
        "exclude_from_training": True,   # P0-3: 伪监督，不默认入正式训练
    }

    provenance = make_provenance(
        label_source="derived",
        label_strength="weak",             # P0-3: 从 gold_diff 推导，弱标签
        evidence=f"gold_diff modified {len(modified_files)} files, {len(added_lines)} added lines",
        verified=False,
        confidence=0.3,
    )

    return {
        "sample_id": f"{instance_id}_root_cause_repair_plan",
        "instance_id": instance_id,
        "task_type": "root_cause_repair_plan",
        "difficulty": "multi_file" if len(modified_files) > 1 else "single_file",
        "issue": issue,
        "code_context": "",
        "test_context": test_diff[:2000],
        "evidence": {
            "source_files": src_files,
            "tests": test_files,
            "verified": False,
        },
        "target": json.dumps(output, ensure_ascii=False, indent=2),
        "provenance": provenance,
    }


# ---------------------------------------------------------------------------
# Task 5 — Patch Generation (Strong)
# ---------------------------------------------------------------------------

def build_patch_generation(
    instance_id: str,
    issue: str,
    gold_diff: str,
    src_files: list[str],
) -> dict[str, Any] | None:
    """构建 Patch Generation 标签（Strong，来自 Gold Patch）。

    需要验证：
    1. Diff 可解析
    2. 可在 base_commit 上 apply
    3. 文件存在
    4. Patch 非空
    """
    if not gold_diff.strip():
        return None

    # 验证 patch 可解析
    edits = []
    current_file = None
    for line in gold_diff.splitlines(keepends=True):
        if line.startswith("+++ b/"):
            current_file = line[6:].rstrip("\n")
        elif line.startswith("@@"):
            # 简单解析：收集 +/- 行
            pass

    # 这里只做基础验证：gold_diff 非空且包含 diff 标记
    if "@@" not in gold_diff:
        return None

    output = gold_diff

    provenance = make_provenance(
        label_source="gold_patch",
        label_strength="strong",
        evidence=f"gold_diff from base_commit, {len(gold_diff)} chars",
        verified=False,  # 需要后续 Gate 验证 apply 和测试
        confidence=0.9,
    )

    return {
        "sample_id": f"{instance_id}_patch_generation",
        "instance_id": instance_id,
        "task_type": "patch_generation",
        "difficulty": "multi_file" if len(src_files) > 1 else "single_file",
        "issue": issue,
        "code_context": "",
        "test_context": "",
        "evidence": {
            "source_files": src_files,
            "tests": [],
            "verified": False,
        },
        "target": output,
        "provenance": provenance,
    }


# ---------------------------------------------------------------------------
# Task 6 — Test Result Interpretation (Strong，需要运行日志)
# ---------------------------------------------------------------------------

def build_test_result_interpretation(
    instance_id: str,
    issue: str,
    patch: str | None,
    test_evidence: dict | None,
) -> dict[str, Any] | None:
    """构建测试结果解释标签（Strong，必须有运行日志）。

    如果没有实际测试执行日志，返回 None。
    """
    if not test_evidence:
        return None

    stdout = test_evidence.get("stdout", "")
    stderr = test_evidence.get("stderr", "")
    exit_code = test_evidence.get("exit_code", -1)

    # 判断失败类别
    failure_category = "unknown"
    if "SyntaxError" in stderr:
        failure_category = "syntax_error"
    elif "AttributeError" in stderr:
        failure_category = "attribute_error"
    elif "FAILED" in stdout or "failed" in stdout.lower():
        failure_category = "test_failure"
    elif exit_code == 0:
        failure_category = "pass"

    output = {
        "failure_category": failure_category,
        "failure_location": "",
        "patch_related": bool(patch),
        "diagnosis": "",
        "next_action": "",
        "stdout_tail": stdout[-500:] if stdout else "",
        "stderr_tail": stderr[-500:] if stderr else "",
        "exit_code": exit_code,
    }

    # 提取失败位置
    for m in re.finditer(r'File "([^"]+)", line (\d+)', stderr):
        output["failure_location"] = f"{m.group(1)}:{m.group(2)}"
        break

    provenance = make_provenance(
        label_source="test_execution",
        label_strength="strong",
        evidence=f"exit_code={exit_code}, category={failure_category}",
        verified=True,
        confidence=1.0,
    )

    return {
        "sample_id": f"{instance_id}_test_result_interpretation",
        "instance_id": instance_id,
        "task_type": "test_result_interpretation",
        "difficulty": "single_file",
        "issue": issue,
        "code_context": patch or "",
        "test_context": json.dumps(test_evidence, ensure_ascii=False),
        "evidence": {
            "source_files": [],
            "tests": [],
            "verified": True,
        },
        "target": json.dumps(output, ensure_ascii=False, indent=2),
        "provenance": provenance,
    }


# ---------------------------------------------------------------------------
# 主流程：为单个 Issue 构建所有任务标签
# ---------------------------------------------------------------------------

def build_labels_for_issue(
    instance_id: str,
    task: dict,
    output_dir: Path,
) -> dict[str, Any]:
    """为单个 Issue 构建 6 种任务标签。"""
    issue = task.get("problem_statement", "") or build_issue_from_test_diff(
        task.get("test_diff", ""), instance_id
    )
    test_diff = task.get("test_diff", "")
    gold_diff = task.get("gold_diff", "")
    src_files = task.get("src_files", [])
    test_files = task.get("test_files", [])

    results = {
        "instance_id": instance_id,
        "tasks": {},
    }

    # Task 1: Issue Understanding
    t1 = build_issue_understanding(instance_id, issue, test_diff)
    if t1:
        results["tasks"]["issue_understanding"] = t1

    # Task 2: Bug Localization (Weak)
    t2 = build_bug_localization(instance_id, issue, src_files, test_diff, gold_diff)
    if t2:
        results["tasks"]["bug_localization"] = t2

    # Task 3: Test Reasoning
    t3 = build_test_reasoning(instance_id, issue, test_diff, test_files)
    if t3:
        results["tasks"]["test_reasoning"] = t3

    # Task 4: Root Cause & Repair Plan
    t4 = build_root_cause_repair_plan(instance_id, issue, src_files, test_files, gold_diff, test_diff)
    if t4:
        results["tasks"]["root_cause_repair_plan"] = t4

    # Task 5: Patch Generation (Strong)
    t5 = build_patch_generation(instance_id, issue, gold_diff, src_files)
    if t5:
        results["tasks"]["patch_generation"] = t5

    # Task 6: Test Result Interpretation (需要测试证据，暂时跳过)
    # 需要后续从 patch_correctness_exp 或实际执行中获取
    results["tasks"]["test_result_interpretation"] = None

    # 保存
    issue_dir = output_dir / instance_id
    issue_dir.mkdir(parents=True, exist_ok=True)

    for task_type, record in results["tasks"].items():
        if record is None:
            continue
        task_file = issue_dir / f"{task_type}.jsonl"
        with open(task_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 保存 provenance 汇总
    prov_file = issue_dir / "provenance.json"
    prov_data = {
        "instance_id": instance_id,
        "task_types": list(results["tasks"].keys()),
        "label_sources": {
            k: (v["provenance"]["label_source"] if v else "skipped")
            for k, v in results["tasks"].items()
        },
        "label_strengths": {
            k: (v["provenance"]["label_strength"] if v else "skipped")
            for k, v in results["tasks"].items()
        },
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    prov_file.write_text(json.dumps(prov_data, ensure_ascii=False, indent=2), encoding="utf-8")

    return results


def build_labels_for_tasks(
    task_ids: list[str],
    output_dir: Path,
) -> list[dict[str, Any]]:
    """为多个任务构建标签。"""
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
        print(f"[LABEL] {tid}")
        res = build_labels_for_issue(tid, task, output_dir)
        results.append(res)

    # 汇总
    summary_path = output_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps({"results": results, "count": len(results)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nLabel builder summary: {len(results)} issues labeled")
    print(f"Summary saved to {summary_path}")
    return results


def main():
    ap = __import__("argparse").ArgumentParser(description="Stage S3: 6-Task Label Construction")
    ap.add_argument("--tasks", nargs="+", required=True,
                    help="要构建标签的 task id 列表")
    ap.add_argument("--output-dir", default=r"F:\test\gemma4\training\data\processed",
                    help="处理后数据输出目录")
    args = ap.parse_args()

    build_labels_for_tasks(
        task_ids=args.tasks,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
