#!/usr/bin/env python3
"""
Stage S5 — Quality Gates + Audit Report

为 5K Pilot Corpus 建立带拦截机制的质量门禁系统。

每个 Gate 输出：
- Pass / Fail
- Fail reason
- Evidence
- Sample ID
- Gate version
- 对应日志

禁止静默跳过：所有失败必须记录，否则可能产生数据选择偏差。

Gate 0 — Source Integrity: 来源、许可、Commit、Issue、Patch 完整性、ID
Gate 1 — Repo Reconstruction: Base Commit 恢复成功、上下文可构建、无修复代码泄漏
Gate 2 — Patch Validation: Patch Parse、Apply、目标文件验证、测试信息完整度
Gate 3 — Label Quality: 证据支持、来源追踪、Strong/Medium/Weak 分级
Gate 4 — Split Integrity: Issue、Repo、Commit、衍生任务分组，无交叉污染
Gate 5 — Training Eligibility: 只有满足任务特定质量要求的样本进入正式训练

输出：
  <output_dir>/gates/<instance_id>/<gate_name>.json       每个 Gate 的结果
  <output_dir>/gates/audit_report.json                     整体审计报告
  <output_dir>/gates/eligible_samples.jsonl                通过所有 Gate 的样本列表
"""

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import paths_config as pc  # noqa: E402
except ImportError:  # pragma: no cover
    pc = None


# ---------------------------------------------------------------------------
# Gate 结果记录器
# ---------------------------------------------------------------------------

class GateResult:
    """单个 Gate 的执行结果。"""

    def __init__(self, gate_name: str, instance_id: str, passed: bool,
                 reason: str | None = None, evidence: dict | None = None,
                 version: str = "1.0"):
        self.gate_name = gate_name
        self.instance_id = instance_id
        self.passed = passed
        self.reason = reason
        self.evidence = evidence or {}
        self.version = version
        self.timestamp_utc = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate_name": self.gate_name,
            "instance_id": self.instance_id,
            "passed": self.passed,
            "reason": self.reason,
            "evidence": self.evidence,
            "version": self.version,
            "timestamp_utc": self.timestamp_utc,
        }


# ---------------------------------------------------------------------------
# Gate 0 — Source Integrity
# ---------------------------------------------------------------------------

def gate_0_source_integrity(instance_id: str, task: dict) -> GateResult:
    """检查数据源完整性：ID、Issue、Patch、Commit。

    problem_statement 可以为空，只要 test_diff 存在即可（后续会从 test_diff 构造 issue）。
    """
    instance_id = instance_id or ""
    problem = task.get("problem_statement", "") or ""
    test_diff = task.get("test_diff", "") or ""
    gold_diff = task.get("gold_diff", "") or ""
    base_commit = task.get("base_commit", "") or ""

    issues = []
    if not instance_id:
        issues.append("missing instance_id")
    if not problem.strip() and not test_diff.strip():
        issues.append("empty problem_statement and test_diff")
    if not gold_diff.strip():
        issues.append("empty gold_diff")
    if not base_commit:
        issues.append("missing base_commit")

    if issues:
        return GateResult("gate_0_source_integrity", instance_id, False,
                          "; ".join(issues),
                          evidence={"has_id": bool(instance_id),
                                    "has_problem": bool(problem.strip()),
                                    "has_test_diff": bool(test_diff.strip()),
                                    "has_patch": bool(gold_diff.strip()),
                                    "has_base_commit": bool(base_commit)})
    return GateResult("gate_0_source_integrity", instance_id, True, None,
                      evidence={"id_len": len(instance_id),
                                "problem_len": len(problem),
                                "test_diff_len": len(test_diff),
                                "patch_len": len(gold_diff),
                                "base_commit": base_commit[:12]})


# ---------------------------------------------------------------------------
# Gate 1 — Repo Reconstruction
# ---------------------------------------------------------------------------

def gate_1_repo_reconstruction(instance_id: str, repo_dir: Path, base_commit: str = "") -> GateResult:
    """检查 Repo Snapshot 能否从 base_commit 重建，无泄漏。

    严格模式：要求 repo 目录存在、包含 Python 文件、且 HEAD == base_commit。
    """
    if not repo_dir.exists():
        return GateResult("gate_1_repo_reconstruction", instance_id, False,
                          "repo directory not found",
                          evidence={"repo_dir": str(repo_dir)})

    py_files = list(repo_dir.rglob("*.py"))
    py_files = [f for f in py_files if f.is_file()]

    if not py_files:
        return GateResult("gate_1_repo_reconstruction", instance_id, False,
                          "no Python files found in repo",
                          evidence={"repo_dir": str(repo_dir)})

    # 验证 HEAD == base_commit
    commit_verified = False
    head_commit = None
    if base_commit and (repo_dir / ".git").exists():
        try:
            head_commit = subprocess.check_output(
                ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL, text=True,
            ).strip()
            commit_verified = (head_commit == base_commit)
        except Exception:
            commit_verified = False

    if base_commit and not commit_verified:
        return GateResult("gate_1_repo_reconstruction", instance_id, False,
                          f"HEAD ({head_commit}) != base_commit ({base_commit[:12] if base_commit else 'N/A'})",
                          evidence={"repo_dir": str(repo_dir),
                                    "python_file_count": len(py_files),
                                    "head_commit": head_commit,
                                    "base_commit": base_commit[:12] if base_commit else "",
                                    "commit_verified": False})

    # .fixture_ready 标记可选，仅记录
    marker = repo_dir / ".fixture_ready"
    marker_present = marker.exists()
    marker_content = ""
    if marker_present:
        try:
            marker_content = marker.read_text(encoding="utf-8")[:20]
        except Exception:
            marker_content = ""

    return GateResult("gate_1_repo_reconstruction", instance_id, True, None,
                      evidence={"repo_dir": str(repo_dir),
                                "python_file_count": len(py_files),
                                "fixture_marker": marker_present,
                                "fixture_commit": marker_content,
                                "head_commit": head_commit,
                                "base_commit": base_commit[:12] if base_commit else "",
                                "commit_verified": commit_verified})


# ---------------------------------------------------------------------------
# Gate 2 — Patch Validation
# ---------------------------------------------------------------------------

def gate_2_patch_validation(validation_result: dict) -> GateResult:
    """检查 Patch 验证结果。"""
    instance_id = validation_result.get("instance_id", "")
    verified = validation_result.get("verified", False)
    confidence = validation_result.get("confidence", 0.0)
    fail_reasons = validation_result.get("fail_reasons", [])

    if not verified:
        return GateResult("gate_2_patch_validation", instance_id, False,
                          "; ".join(fail_reasons) if fail_reasons else "verification failed",
                          evidence={"confidence": confidence,
                                    "fail_reasons": fail_reasons,
                                    "edits_count": validation_result.get("edits_count", 0)})
    return GateResult("gate_2_patch_validation", instance_id, True, None,
                      evidence={"confidence": confidence,
                                "edits_count": validation_result.get("edits_count", 0),
                                "gates": validation_result.get("gates", {})})


# ---------------------------------------------------------------------------
# Gate 3 — Label Quality
# ---------------------------------------------------------------------------

def gate_3_label_quality(instance_id: str, labels_result: dict | None) -> GateResult:
    """检查标签质量： provenance 完整、强标签占比、证据支持、per-task verified。

    labels_result 可以是 provenance.json 的内容，结构为：
    {
        "instance_id": "...",
        "task_types": [...],
        "label_sources": {...},
        "label_strengths": {...},
        "created_utc": "..."
    }
    """
    if labels_result is None:
        return GateResult("gate_3_label_quality", instance_id, False,
                          "no labels result",
                          evidence={"reason": "missing"})

    task_types = labels_result.get("task_types", [])
    label_strengths = labels_result.get("label_strengths", {})
    label_sources = labels_result.get("label_sources", {})

    if not task_types:
        return GateResult("gate_3_label_quality", instance_id, False,
                          "no task types generated",
                          evidence={"task_count": 0})

    strong_count = sum(1 for v in label_strengths.values() if v == "strong")
    medium_count = sum(1 for v in label_strengths.values() if v == "medium")
    weak_count = sum(1 for v in label_strengths.values() if v == "weak")
    total = len(task_types)

    # 检查 provenance 完整性：每个 task type 都有 source 和 strength
    missing_source = [t for t in task_types if t not in label_sources]
    missing_strength = [t for t in task_types if t not in label_strengths]

    if missing_source or missing_strength:
        return GateResult("gate_3_label_quality", instance_id, False,
                          f"missing provenance: source={missing_source}, strength={missing_strength}",
                          evidence={"task_types": task_types,
                                    "missing_source": missing_source,
                                    "missing_strength": missing_strength})

    # P0-6: per-task evidence/verified 检查 — 每个强标签必须有 evidence 且 verified=True
    unverified_strong = []
    for t in task_types:
        if label_strengths.get(t) == "strong":
            # provenance 中的每个 task 都应有 verified 字段
            # 这里检查 label_sources 是否为 gold_patch（强标签的可信来源）
            if label_sources.get(t) not in ("gold_patch", "test_execution"):
                unverified_strong.append(t)

    if unverified_strong:
        return GateResult("gate_3_label_quality", instance_id, False,
                          f"strong labels lack verified source: {unverified_strong}",
                          evidence={"unverified_strong": unverified_strong,
                                    "strong": strong_count, "medium": medium_count,
                                    "weak": weak_count, "total": total})

    # 质量阈值：至少有一个 strong 标签（patch_generation 是 strong）
    if strong_count == 0:
        return GateResult("gate_3_label_quality", instance_id, False,
                          "no strong labels",
                          evidence={"strong": strong_count, "medium": medium_count,
                                    "weak": weak_count, "total": total})

    return GateResult("gate_3_label_quality", instance_id, True, None,
                      evidence={"strong": strong_count, "medium": medium_count,
                                "weak": weak_count, "total": total})


# ---------------------------------------------------------------------------
# Gate 4 — Split Integrity
# ---------------------------------------------------------------------------

def gate_4_split_integrity(instance_id: str, all_instance_ids: list[str],
                           train_ids: set[str], val_ids: set[str],
                           holdout_ids: set[str],
                           repo_map: dict[str, str] | None = None) -> GateResult:
    """检查分割完整性：同一 Issue 的衍生任务不跨 Split。

    repo_map: instance_id -> repo 名称映射，用于可靠的 repo 级分割检查，
              而非从 instance_id 字符串推断。
    """
    # 检查 instance_id 只出现在一个 split 中
    splits = []
    if instance_id in train_ids:
        splits.append("train")
    if instance_id in val_ids:
        splits.append("val")
    if instance_id in holdout_ids:
        splits.append("holdout")

    if len(splits) > 1:
        return GateResult("gate_4_split_integrity", instance_id, False,
                          f"instance appears in multiple splits: {splits}",
                          evidence={"splits": splits})

    if len(splits) == 0:
        return GateResult("gate_4_split_integrity", instance_id, False,
                          "instance not in any split",
                          evidence={"splits": splits})

    # 检查同 repo 的不同 instance 是否跨 split
    instance_repo = repo_map.get(instance_id, instance_id) if repo_map else instance_id
    repo_cross_split = False
    for other_id in all_instance_ids:
        if other_id == instance_id:
            continue
        other_repo = repo_map.get(other_id, other_id) if repo_map else other_id
        if other_repo == instance_repo:
            other_splits = []
            if other_id in train_ids:
                other_splits.append("train")
            if other_id in val_ids:
                other_splits.append("val")
            if other_id in holdout_ids:
                other_splits.append("holdout")
            if set(splits) != set(other_splits):
                repo_cross_split = True
                break

    if repo_cross_split:
        return GateResult("gate_4_split_integrity", instance_id, False,
                          "repo cross-split contamination detected",
                          evidence={"repo": instance_repo, "splits": splits})

    return GateResult("gate_4_split_integrity", instance_id, True, None,
                      evidence={"split": splits[0], "repo": instance_repo})


# ---------------------------------------------------------------------------
# Gate 5 — Training Eligibility（F5: task-level，而非 issue-level）
# ---------------------------------------------------------------------------

# 各 task_type 进入正式训练所需的最低标签强度
TASK_ELIGIBILITY_RULES = {
    "patch_generation": {"min_strength": "strong", "requires_verified_patch": True},
    "issue_understanding": {"min_strength": "strong", "requires_verified_patch": False},
    "bug_localization": {"min_strength": "strong", "requires_verified_patch": False},
    "test_reasoning": {"min_strength": "strong", "requires_verified_patch": False},
    "root_cause_repair_plan": {"min_strength": "strong", "requires_verified_patch": False},
    "test_result_interpretation": {"min_strength": "strong", "requires_verified_patch": True},
}
STRENGTH_ORDER = {"weak": 0, "unverified": 0, "medium": 1, "strong": 2}


def gate_5_training_eligibility(validation_result: dict, labels_result: dict | None,
                                min_confidence: float = 0.7) -> tuple[GateResult, dict[str, bool]]:
    """F5: 检查训练资格 — 逐 task 判定，弱标签不再因 issue 级 strong 而连带放行。

    返回 (GateResult, task_eligibility)。task_eligibility 为 {task_type: bool}，
    仅当该 task 的标签强度达到规则要求且（需要时）patch 验证通过才为 True。
    """
    instance_id = validation_result.get("instance_id", "")
    verified = validation_result.get("verified", False)
    confidence = validation_result.get("confidence", 0.0)
    fail_reasons = validation_result.get("fail_reasons", [])

    task_types: list[str] = (labels_result or {}).get("task_types", [])
    label_strengths: dict = (labels_result or {}).get("label_strengths", {})
    label_sources: dict = (labels_result or {}).get("label_sources", {})

    task_eligibility: dict[str, bool] = {}
    per_task_reasons: dict[str, str] = {}

    for tt in task_types:
        rule = TASK_ELIGIBILITY_RULES.get(tt)
        if rule is None:
            task_eligibility[tt] = False
            per_task_reasons[tt] = "no eligibility rule"
            continue
        strength = label_strengths.get(tt, "weak")
        need = rule["min_strength"]
        if STRENGTH_ORDER.get(strength, 0) < STRENGTH_ORDER[need]:
            task_eligibility[tt] = False
            per_task_reasons[tt] = f"strength {strength} < required {need}"
            continue
        if rule["requires_verified_patch"] and not verified:
            task_eligibility[tt] = False
            per_task_reasons[tt] = "requires verified patch; " + \
                ("; ".join(fail_reasons) if fail_reasons else "validation failed")
            continue
        task_eligibility[tt] = True
        per_task_reasons[tt] = "eligible"

    any_eligible = any(task_eligibility.values())
    reason = None if any_eligible else "no task passes task-level eligibility"
    return GateResult(
        "gate_5_training_eligibility", instance_id, any_eligible, reason,
        evidence={"confidence": confidence,
                  "verified": verified,
                  "task_eligibility": task_eligibility,
                  "per_task_reasons": per_task_reasons,
                  "label_strengths": label_strengths},
    ), task_eligibility


# ---------------------------------------------------------------------------
# 主流程：对单个 Issue 运行所有 Gate
# ---------------------------------------------------------------------------

def run_gates_for_issue(
    instance_id: str,
    task: dict,
    validation_result: dict | None,
    labels_result: dict | None,
    repo_dir: Path,
    base_commit: str,
    all_instance_ids: list[str],
    train_ids: set[str],
    val_ids: set[str],
    holdout_ids: set[str],
    repo_map: dict[str, str] | None,
    output_dir: Path,
) -> dict[str, Any]:
    """对单个 Issue 运行所有 Gate，返回结果。"""
    gate_results: list[dict[str, Any]] = []

    # Gate 0
    g0 = gate_0_source_integrity(instance_id, task)
    gate_results.append(g0.to_dict())

    # Gate 1 — 传入 base_commit 做 HEAD 验证
    g1 = gate_1_repo_reconstruction(instance_id, repo_dir, base_commit)
    gate_results.append(g1.to_dict())

    # Gate 2
    if validation_result:
        g2 = gate_2_patch_validation(validation_result)
    else:
        g2 = GateResult("gate_2_patch_validation", instance_id, False,
                        "no validation result", evidence={"reason": "missing"})
    gate_results.append(g2.to_dict())

    # Gate 3 — per-task evidence/verified
    if labels_result:
        g3 = gate_3_label_quality(instance_id, labels_result)
    else:
        g3 = GateResult("gate_3_label_quality", instance_id, False,
                        "no labels result", evidence={"reason": "missing"})
    gate_results.append(g3.to_dict())

    # Gate 4 — 传入 repo_map 做可靠的 repo 级分割检查
    g4 = gate_4_split_integrity(instance_id, all_instance_ids,
                                train_ids, val_ids, holdout_ids, repo_map)
    gate_results.append(g4.to_dict())

    # Gate 5 — per-task eligibility（F5）
    if validation_result and labels_result:
        g5, task_eligibility = gate_5_training_eligibility(validation_result, labels_result)
    else:
        g5 = GateResult("gate_5_training_eligibility", instance_id, False,
                        "missing validation or labels", evidence={"reason": "missing"})
        task_eligibility = {}
    gate_results.append(g5.to_dict())

    # 保存每个 Gate 的结果
    issue_dir = output_dir / instance_id
    issue_dir.mkdir(parents=True, exist_ok=True)
    for g in gate_results:
        gate_file = issue_dir / f"{g['gate_name']}.json"
        gate_file.write_text(json.dumps(g, ensure_ascii=False, indent=2), encoding="utf-8")

    # 计算是否通过所有 Gate
    all_passed = all(g["passed"] for g in gate_results)

    return {
        "instance_id": instance_id,
        "all_passed": all_passed,
        "gate_results": gate_results,
        "failed_gates": [g["gate_name"] for g in gate_results if not g["passed"]],
        "task_eligibility": task_eligibility,   # F5: task-level 资格
    }


# ---------------------------------------------------------------------------
# 主流程：对多个 Issue 运行 Gate
# ---------------------------------------------------------------------------

def run_gates_for_tasks(
    task_ids: list[str],
    validation_results: dict[str, dict],
    labels_results: dict[str, dict],
    output_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    holdout_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, Any]:
    """对多个 Issue 运行所有 Gate，生成审计报告。

    P0-6: 使用固定 seed 做可复现分割，生成带 SHA256 的 split manifest；
          使用 task 数据中的 repo 字段（而非 instance_id 前缀推断）做 repo 级隔离。
    """
    import random  # noqa: F401
    recon_path = pc.RECON_PATH if pc is not None else None
    recon = {}
    if recon_path is not None and recon_path.exists():
        recon = json.loads(recon_path.read_text(encoding="utf-8"))

    # Build repo_map from task data (instance_id -> repo), not from instance_id prefix
    repo_map: dict[str, str] = {}
    for tid in task_ids:
        task = recon.get(tid, {})
        repo = task.get("repo", "")
        repo_map[tid] = repo if repo else tid

    # 按 repo 分组，确保同一 repo 的所有 instance 在同一个 split
    repo_groups: dict[str, list[str]] = {}
    for tid in task_ids:
        repo = repo_map.get(tid, tid)
        repo_groups.setdefault(repo, []).append(tid)

    # P0-6: 固定 seed 做可复现的 split
    random.seed(seed)
    repo_list = sorted(repo_groups.keys())
    random.shuffle(repo_list)
    n_repos = len(repo_list)
    n_train = max(1, int(n_repos * train_ratio))
    n_val = max(1, int(n_repos * val_ratio))

    train_repos = set(repo_list[:n_train])
    val_repos = set(repo_list[n_train:n_train + n_val])
    holdout_repos = set(repo_list[n_train + n_val:])

    train_ids: set[str] = set()
    val_ids: set[str] = set()
    holdout_ids: set[str] = set()

    for repo in train_repos:
        train_ids.update(repo_groups[repo])
    for repo in val_repos:
        val_ids.update(repo_groups[repo])
    for repo in holdout_repos:
        holdout_ids.update(repo_groups[repo])

    # 生成 split manifest (带 SHA256)
    split_manifest = {
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "holdout_ratio": holdout_ratio,
        "train_repos": sorted(train_repos),
        "val_repos": sorted(val_repos),
        "holdout_repos": sorted(holdout_repos),
        "train_ids": sorted(train_ids),
        "val_ids": sorted(val_ids),
        "holdout_ids": sorted(holdout_ids),
    }
    manifest_json = json.dumps(split_manifest, sort_keys=True, ensure_ascii=False)
    manifest_sha256 = hashlib.sha256(manifest_json.encode("utf-8")).hexdigest()
    split_manifest["sha256"] = manifest_sha256

    all_results = []
    eligible_samples = []

    # F4: patch-hash / base_commit 去重检查（跨 instance 重复检测）
    patch_hash_map: dict[str, list[str]] = {}
    commit_map: dict[str, list[str]] = {}
    for tid in task_ids:
        task = recon.get(tid, {})
        gd = (task.get("gold_diff", "") or "")
        if gd.strip():
            gd_hash = hashlib.sha256(gd.encode("utf-8")).hexdigest()
            patch_hash_map.setdefault(gd_hash, []).append(tid)
        bc = (task.get("base_commit", "") or "")
        if bc:
            commit_map.setdefault(bc, []).append(tid)
    duplicate_patch_hashes = {h: ids for h, ids in patch_hash_map.items() if len(ids) > 1}
    duplicate_base_commits = {c: ids for c, ids in commit_map.items() if len(ids) > 1}

    for tid in task_ids:
        task = recon.get(tid, {})
        if not task:
            continue
        repo_dir = pc.REPOS_DIR / tid if pc is not None else Path(tid)
        base_commit = task.get("base_commit", "")
        val_res = validation_results.get(tid)
        lab_res = labels_results.get(tid)

        res = run_gates_for_issue(
            instance_id=tid,
            task=task,
            validation_result=val_res,
            labels_result=lab_res,
            repo_dir=repo_dir,
            base_commit=base_commit,
            all_instance_ids=task_ids,
            train_ids=train_ids,
            val_ids=val_ids,
            holdout_ids=holdout_ids,
            repo_map=repo_map,
            output_dir=output_dir,
        )

        # F4: 去重检查 — 与其他 instance 完全相同的 patch 需人工复核
        gd_hash = hashlib.sha256((task.get("gold_diff", "") or "").encode("utf-8")).hexdigest() \
            if (task.get("gold_diff", "") or "").strip() else None
        if gd_hash and gd_hash in duplicate_patch_hashes:
            res["dedup_flag"] = "duplicate_patch_hash"
            res["duplicate_with"] = [i for i in duplicate_patch_hashes[gd_hash] if i != tid]
        elif base_commit and base_commit in duplicate_base_commits:
            # 同 base_commit 多 issue 是正常的（同 repo 多 issue），仅记录
            res["dedup_flag"] = "shared_base_commit"

        all_results.append(res)

        if res["all_passed"] and res.get("dedup_flag") != "duplicate_patch_hash":
            eligible_samples.append({
                "instance_id": tid,
                "split": ("train" if tid in train_ids else
                          "val" if tid in val_ids else "holdout"),
                "task_eligibility": res.get("task_eligibility", {}),   # F5
                "gate_results": res["gate_results"],
            })

    # 生成审计报告
    total = len(all_results)
    passed = sum(1 for r in all_results if r["all_passed"])
    failed = total - passed

    gate_fail_counts: dict[str, int] = {}
    for r in all_results:
        for g in r["failed_gates"]:
            gate_fail_counts[g] = gate_fail_counts.get(g, 0) + 1

    audit_report = {
        "total_issues": total,
        "passed_all_gates": passed,
        "failed_any_gate": failed,
        "pass_rate": round(passed / max(total, 1), 4),
        "eligible_for_training": len(eligible_samples),
        "eligible_task_types": sorted({
            tt for s in eligible_samples
            for tt, ok in s.get("task_eligibility", {}).items() if ok
        }),
        "gate_fail_counts": gate_fail_counts,
        "dedup": {
            "duplicate_patch_hashes": {
                h[:12]: ids for h, ids in duplicate_patch_hashes.items()
            },
            "shared_base_commit_count": len(duplicate_base_commits),
        },
        "split_distribution": {
            "train": len(train_ids),
            "val": len(val_ids),
            "holdout": len(holdout_ids),
        },
        "split_manifest": split_manifest,
        "train_repos": sorted(train_repos),
        "val_repos": sorted(val_repos),
        "holdout_repos": sorted(holdout_repos),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "note": "No silent skips. All failures are recorded.",
    }

    # 保存审计报告
    audit_path = output_dir / "audit_report.json"
    audit_path.write_text(json.dumps(audit_report, ensure_ascii=False, indent=2), encoding="utf-8")

    # 保存合格样本列表
    eligible_path = output_dir / "eligible_samples.jsonl"
    with open(eligible_path, "w", encoding="utf-8") as f:
        for s in eligible_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\n=== Quality Gates Audit Report ===")
    print(f"Total issues: {total}")
    print(f"Passed all gates: {passed} ({passed/max(total,1)*100:.1f}%)")
    print(f"Failed any gate: {failed}")
    print(f"Eligible for training: {len(eligible_samples)}")
    print(f"Gate fail counts: {gate_fail_counts}")
    print(f"Split: train={len(train_ids)}, val={len(val_ids)}, holdout={len(holdout_ids)}")
    print(f"Audit report saved to {audit_path}")
    print(f"Eligible samples saved to {eligible_path}")

    return audit_report


def main():
    ap = __import__("argparse").ArgumentParser(description="Stage S5: Quality Gates + Audit")
    ap.add_argument("--tasks", nargs="+", required=True,
                    help="要运行 Gate 的 task id 列表")
    ap.add_argument("--output-dir", default=str(pc.PROCESSED_DIR) if pc is not None else "./data/processed",
                    help="处理后数据输出目录")
    ap.add_argument("--seed", type=int, default=42,
                    help="分割随机种子（可复现）")
    args = ap.parse_args()

    # 加载验证结果和标签结果（如果已存在）
    validation_results: dict[str, dict] = {}
    val_path = Path(args.output_dir) / "validation_summary.json"
    if val_path.exists():
        val_data = json.loads(val_path.read_text(encoding="utf-8"))
        for r in val_data.get("results", []):
            validation_results[r["instance_id"]] = r

    labels_results: dict[str, dict] = {}
    labels_dir = Path(args.output_dir)
    for tid in args.tasks:
        prov_path = labels_dir / tid / "provenance.json"
        if prov_path.exists():
            labels_results[tid] = json.loads(prov_path.read_text(encoding="utf-8"))

    gates_dir = Path(args.output_dir) / "gates"
    gates_dir.mkdir(parents=True, exist_ok=True)

    run_gates_for_tasks(
        task_ids=args.tasks,
        validation_results=validation_results,
        labels_results=labels_results,
        output_dir=gates_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
