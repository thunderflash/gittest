#!/usr/bin/env python3
"""
Stage S3 — 6-Task Label Construction with Label Provenance

F6 修复后的设计：
- 只允许两类来源进入标签：problem_statement（Issue 文本）与 gold_diff（监督答案）。
- gold test_diff 只能作为分析材料（gold_test_patch_analysis），产物一律 weak /
  training_eligible=False；test_diff 从未被真实执行，不得声称 test_execution。
- 不存在可信证据的任务（Issue Understanding / Root Cause）默认排除在正式训练外，
  由未来的模型提议 + 证据校验 + 人工审计流程产生 Strong 标签后才会放行。
- 每条记录携带 training_eligible 与 eligibility_note，最终资格由 S5 Gate 决定。

任务与默认资格：
1. Issue Understanding       - weak/derived,      training_eligible=False（Issue 复制不是理解）
2. Bug Localization          - weak/gold_patch,   training_eligible=False（仅候选文件证据）
3. Test Reasoning            - weak/gold_test_patch_analysis, training_eligible=False
4. Root Cause & Repair Plan  - weak/derived,      training_eligible=False（模板伪监督）
5. Patch Generation          - strong/gold_patch, training_eligible=True（最终需 S4 验证）
6. Test Result Interpretation - 仅在有真实执行日志时生成（strong/test_execution）

输出：
  <output_dir>/<instance_id>/<task_type>.jsonl
  <output_dir>/<instance_id>/provenance.json
  <output_dir>/summary.json
"""

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import paths_config as pc  # noqa: E402
except ImportError:  # pragma: no cover
    pc = None

from context_builder import extract_symbols_from_text, posix_path  # noqa: E402


# ---------------------------------------------------------------------------
# Label Provenance 构造器
# ---------------------------------------------------------------------------

def make_provenance(
    label_source: str,
    label_strength: str,
    evidence: str | None = None,
    verified: bool = False,
    confidence: float = 1.0,
    training_eligible: bool = False,
    eligibility_note: str = "",
) -> dict[str, Any]:
    return {
        "label_source": label_source,
        "label_strength": label_strength,
        "evidence": evidence or "",
        "verified": verified,
        "confidence": confidence,
        "training_eligible": training_eligible,
        "eligibility_note": eligibility_note,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Task 1 — Issue Understanding（F6: 仅从 problem_statement 构建，禁止 test_diff）
# ---------------------------------------------------------------------------

def build_issue_understanding(
    instance_id: str,
    issue: str,
) -> dict[str, Any] | None:
    """从 Issue 文本构建问题理解标签。

    F6: 不再从 test_diff 抽取 imports/assert 充当 expected_behavior —
    那会把 "Issue 理解" 变成 "Issue 复制 + 测试 diff 解析"。
    在没有人工/模型审计前，target 只包含可从 Issue 直接得到的结构化信息，
    其余字段留空并标记为 proposed；标签默认不进入正式训练。
    """
    if not issue.strip():
        return None

    symptom_keywords = extract_symbols_from_text(issue)

    output = {
        "symptom": issue[:500],
        "expected_behavior": "",       # 需人工/模型审计后填充，禁止从 test_diff 提取
        "actual_behavior": "",
        "constraints": [],
        "unknowns": [],
        "symptom_keywords": symptom_keywords[:15],
        "note": "Proposed label derived from issue text only. "
                "expected_behavior MUST come from human/model audit, not gold test diff.",
    }

    provenance = make_provenance(
        label_source="derived",
        label_strength="weak",
        evidence=f"issue text only ({len(issue)} chars)",
        verified=False,
        confidence=0.3,
        training_eligible=False,
        eligibility_note="issue copying is not issue understanding; needs audited expected_behavior",
    )

    return {
        "sample_id": f"{instance_id}_issue_understanding",
        "instance_id": instance_id,
        "task_type": "issue_understanding",
        "issue": issue,
        "target": json.dumps(output, ensure_ascii=False, indent=2),
        "provenance": provenance,
        "training_eligible": False,
    }


# ---------------------------------------------------------------------------
# Task 2 — Bug Localization（F6: 区分 gold_changed_files 与 suspected_bug_files）
# ---------------------------------------------------------------------------

def build_bug_localization(
    instance_id: str,
    issue: str,
    src_files: list[str],
    gold_diff: str,
) -> dict[str, Any] | None:
    """构建缺陷定位候选标签（Weak）。

    F6: gold patch 修改文件 ≠ 缺陷唯一位置。输出区分：
    - gold_changed_files: gold patch 实际修改的文件（事实）
    - suspected_bug_files: 结合 Issue 符号匹配的候选文件（Hit@K 候选）
    每个候选附 evidence。标签默认不进入正式训练。
    """
    gold_changed_files: list[str] = []
    if gold_diff:
        for m in re.finditer(r'^\+\+\+\s+b/(.+)', gold_diff, re.MULTILINE):
            gold_changed_files.append(m.group(1).strip())

    # 候选排序：gold_changed_files 优先（作为 top-1 候选），其余来自 Issue 符号匹配
    issue_symbols = extract_symbols_from_text(issue)
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rank, f in enumerate(gold_changed_files[:10]):
        candidates.append({
            "file": f,
            "rank": rank,
            "evidence": "modified by gold patch (weak evidence for defect location)",
        })
        seen.add(f)
    extra = [f for f in (src_files or []) if f not in seen]
    for f in extra[:10]:
        candidates.append({
            "file": f,
            "rank": len(candidates),
            "evidence": "listed in src_files metadata",
        })
        seen.add(f)

    if not candidates:
        return None

    output = {
        "gold_changed_files": gold_changed_files[:10],
        "suspected_bug_files": [c["file"] for c in candidates[:10]],
        "candidates": candidates,
        "suspected_symbols": issue_symbols[:10],
        "note": "gold_changed_files (what the fix touched) is a different concept from "
                "bug_location (where the defect is). Weak label; do not treat as ground truth.",
    }

    provenance = make_provenance(
        label_source="gold_patch",
        label_strength="weak",
        evidence=f"gold patch modifies {len(gold_changed_files)} file(s)",
        verified=False,
        confidence=0.4,
        training_eligible=False,
        eligibility_note="localization needs Hit@K evaluation and human review",
    )

    return {
        "sample_id": f"{instance_id}_bug_localization",
        "instance_id": instance_id,
        "task_type": "bug_localization",
        "issue": issue,
        "target": json.dumps(output, ensure_ascii=False, indent=2),
        "provenance": provenance,
        "training_eligible": False,
    }


# ---------------------------------------------------------------------------
# Task 3 — Test Reasoning（F6: gold_test_patch_analysis，无执行证据）
# ---------------------------------------------------------------------------

def build_test_reasoning(
    instance_id: str,
    issue: str,
    test_diff: str,
    test_files: list[str],
) -> dict[str, Any] | None:
    """构建测试推理标签（Weak，静态分析 gold test patch，未执行任何测试）。

    F6: label_source 更正为 gold_test_patch_analysis（此前误标 test_execution）。
    test_diff 的存在 ≠ 测试执行；assert 的存在 ≠ 测试推理。
    """
    if not test_diff.strip():
        return None

    test_funcs = [m.group(1) for m in
                  re.finditer(r'^\+\s*def\s+(test_[a-zA-Z0-9_]*)\s*\(', test_diff, re.MULTILINE)]
    assertions = [m.group(1).strip() for m in re.finditer(r'assert\s+(.+)', test_diff)]
    raises_patterns = [f"{m.group(1)}({m.group(2).strip()})" for m in
                       re.finditer(r'pytest\.(raises|warns)\s*\(\s*([^,\)]+)', test_diff)]

    output = {
        "existing_tests_relevant": test_files[:10],
        "new_test_functions": test_funcs[:10],
        "assertions": assertions[:10],
        "raises_patterns": raises_patterns[:10],
        "note": "Statically derived from gold test patch. No test was executed. "
                "Real test reasoning requires execution evidence or human review.",
    }

    provenance = make_provenance(
        label_source="gold_test_patch_analysis",
        label_strength="weak",
        evidence=f"test patch has {len(test_funcs)} new test funcs, {len(assertions)} assertions",
        verified=False,
        confidence=0.3,
        training_eligible=False,
        eligibility_note="no execution evidence; label_source was previously mislabeled as test_execution",
    )

    return {
        "sample_id": f"{instance_id}_test_reasoning",
        "instance_id": instance_id,
        "task_type": "test_reasoning",
        "issue": issue,
        "target": json.dumps(output, ensure_ascii=False, indent=2),
        "provenance": provenance,
        "training_eligible": False,
    }


# ---------------------------------------------------------------------------
# Task 4 — Root Cause & Repair Plan（F6: 默认排除出正式训练）
# ---------------------------------------------------------------------------

def build_root_cause_repair_plan(
    instance_id: str,
    issue: str,
    gold_diff: str,
) -> dict[str, Any] | None:
    """构建根因与修复方案标签（Weak，模板伪监督，默认排除）。

    F6: 当前没有能力从 gold_diff 推断真实 root cause — 任何固定模板句子
    （"Apply gold patch modifications"）都不是根因分析。此任务从正式自动构建
    中剔除，未来由模型基于 base context 提议诊断候选，再经代码证据、
    Test Evidence、Patch 行为一致性验证 + 人工审计后才会成为 Strong 标签。
    """
    if not gold_diff.strip():
        return None

    modified_files = [m.group(1).strip() for m in
                      re.finditer(r'^\+\+\+\s+b/(.+)', gold_diff, re.MULTILINE)]
    added = sum(1 for l in gold_diff.splitlines() if l.startswith('+') and not l.startswith('+++'))
    removed = sum(1 for l in gold_diff.splitlines() if l.startswith('-') and not l.startswith('---'))

    output = {
        "root_cause": "NOT_DERIVED — template placeholder, excluded from training",
        "evidence": [
            f"gold patch modifies {len(modified_files)} file(s); +{added}/-{removed} lines",
        ],
        "repair_strategy": "NOT_DERIVED — needs model-proposed diagnosis with evidence",
        "target_components": modified_files[:5],
        "exclude_from_training": True,
        "note": "Patch statistics are not root cause analysis. "
                "Future: model-proposed diagnosis validated against code/test evidence.",
    }

    provenance = make_provenance(
        label_source="derived",
        label_strength="weak",
        evidence=f"gold_diff modified {len(modified_files)} files, +{added}/-{removed} lines",
        verified=False,
        confidence=0.2,
        training_eligible=False,
        eligibility_note="template pseudo-supervision; would teach shortcut behavior",
    )

    return {
        "sample_id": f"{instance_id}_root_cause_repair_plan",
        "instance_id": instance_id,
        "task_type": "root_cause_repair_plan",
        "issue": issue,
        "target": json.dumps(output, ensure_ascii=False, indent=2),
        "provenance": provenance,
        "training_eligible": False,
    }


# ---------------------------------------------------------------------------
# Task 5 — Patch Generation（Strong；最终资格由 S4/S5 验证决定）
# ---------------------------------------------------------------------------

def build_patch_generation(
    instance_id: str,
    issue: str,
    gold_diff: str,
    src_files: list[str],
) -> dict[str, Any] | None:
    """构建 Patch Generation 标签（Strong，来自 Gold Patch）。

    资格条件（S4 验证后由 Gate 最终决定）：
    gold_diff 可解析、可在 base_commit 上 git apply、修改后文件语法通过。
    """
    if not gold_diff.strip() or "@@" not in gold_diff:
        return None

    provenance = make_provenance(
        label_source="gold_patch",
        label_strength="strong",
        evidence=f"gold_diff from base_commit, {len(gold_diff)} chars",
        verified=False,   # S4 git apply 验证后由 Gate 置 True
        confidence=0.9,
        training_eligible=True,
        eligibility_note="gold patch; final eligibility requires S4 validation pass",
    )

    return {
        "sample_id": f"{instance_id}_patch_generation",
        "instance_id": instance_id,
        "task_type": "patch_generation",
        "issue": issue,
        "target": gold_diff,
        "provenance": provenance,
        "training_eligible": True,
    }


# ---------------------------------------------------------------------------
# Task 6 — Test Result Interpretation（Strong，必须来自真实执行日志）
# ---------------------------------------------------------------------------

def build_test_result_interpretation(
    instance_id: str,
    issue: str,
    patch: str | None,
    test_evidence: dict | None,
) -> dict[str, Any] | None:
    """构建测试结果解释标签（Strong，必须有真实测试执行日志）。

    没有执行日志 → 返回 None（不生成）。
    """
    if not test_evidence:
        return None

    stdout = test_evidence.get("stdout", "")
    stderr = test_evidence.get("stderr", "")
    exit_code = test_evidence.get("exit_code", -1)

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

    for m in re.finditer(r'File "([^"]+)", line (\d+)', stderr):
        output["failure_location"] = f"{m.group(1)}:{m.group(2)}"
        break

    provenance = make_provenance(
        label_source="test_execution",
        label_strength="strong",
        evidence=f"exit_code={exit_code}, category={failure_category}",
        verified=True,
        confidence=1.0,
        training_eligible=True,
        eligibility_note="derived from real execution log",
    )

    return {
        "sample_id": f"{instance_id}_test_result_interpretation",
        "instance_id": instance_id,
        "task_type": "test_result_interpretation",
        "issue": issue,
        "target": json.dumps(output, ensure_ascii=False, indent=2),
        "provenance": provenance,
        "training_eligible": True,
    }


# ---------------------------------------------------------------------------
# 主流程：为单个 Issue 构建所有任务标签
# ---------------------------------------------------------------------------

def build_labels_for_issue(
    instance_id: str,
    task: dict,
    output_dir: Path,
) -> dict[str, Any]:
    """为单个 Issue 构建 6 种任务标签（F6 语义）。"""
    problem = (task.get("problem_statement", "") or "").strip()
    gold_diff = task.get("gold_diff", "") or ""
    test_diff = task.get("test_diff", "") or ""
    src_files = task.get("src_files", []) or []
    test_files = task.get("test_files", []) or []

    # F6: problem_statement 缺失 → 不构造 issue（禁止 test_diff 造 issue）
    issue = problem if problem else f"[MISSING_ISSUE_TEXT] {instance_id}"
    issue_source = "problem_statement" if problem else "missing"

    results: dict[str, Any] = {
        "instance_id": instance_id,
        "issue_source": issue_source,
        "tasks": {},
    }

    # Task 1: Issue Understanding（仅 problem_statement 存在时才有意义）
    if problem:
        t1 = build_issue_understanding(instance_id, issue)
        if t1:
            results["tasks"]["issue_understanding"] = t1

    # Task 2: Bug Localization（weak）
    t2 = build_bug_localization(instance_id, issue, src_files, gold_diff)
    if t2:
        results["tasks"]["bug_localization"] = t2

    # Task 3: Test Reasoning（weak，静态分析）
    t3 = build_test_reasoning(instance_id, issue, test_diff, test_files)
    if t3:
        results["tasks"]["test_reasoning"] = t3

    # Task 4: Root Cause（weak，默认排除）
    t4 = build_root_cause_repair_plan(instance_id, issue, gold_diff)
    if t4:
        results["tasks"]["root_cause_repair_plan"] = t4

    # Task 5: Patch Generation（strong）
    t5 = build_patch_generation(instance_id, issue, gold_diff, src_files)
    if t5:
        results["tasks"]["patch_generation"] = t5

    # Task 6: Test Result Interpretation（需要真实执行日志；此处无）
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

    # provenance 汇总（含 training_eligible map，供 S5 Gate 使用）
    prov_data = {
        "instance_id": instance_id,
        "issue_source": issue_source,
        "task_types": [k for k, v in results["tasks"].items() if v is not None],
        "label_sources": {
            k: (v["provenance"]["label_source"] if v else "skipped")
            for k, v in results["tasks"].items()
        },
        "label_strengths": {
            k: (v["provenance"]["label_strength"] if v else "skipped")
            for k, v in results["tasks"].items()
        },
        "training_eligible": {
            k: (v["training_eligible"] if v else False)
            for k, v in results["tasks"].items()
        },
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "note": "F6 semantics: only patch_generation (and real-execution task 6) "
                "can be training eligible; all derived/analyzed labels are weak.",
    }
    prov_file = issue_dir / "provenance.json"
    prov_file.write_text(json.dumps(prov_data, ensure_ascii=False, indent=2), encoding="utf-8")

    return results


def build_labels_for_tasks(
    task_ids: list[str],
    output_dir: Path,
) -> list[dict[str, Any]]:
    """为多个任务构建标签。"""
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
        print(f"[LABEL] {tid}")
        res = build_labels_for_issue(tid, task, output_dir)
        results.append(res)

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
    ap.add_argument("--output-dir", default=str(pc.PROCESSED_DIR) if pc is not None else "./data/processed",
                    help="处理后数据输出目录")
    args = ap.parse_args()

    build_labels_for_tasks(
        task_ids=args.tasks,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
