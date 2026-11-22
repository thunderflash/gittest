#!/usr/bin/env python3
"""
Stage 2: Build Verified Patch DPO training data (gate-linked).

F8: 记录携带与 SFT 完全一致的 ``messages``（system+user，无 assistant），
train_dpo.py 用同一 chat template 渲染 prompt，保证
Prompt_Format_SFT == Prompt_Format_DPO == Prompt_Format_Inference。

F5 强制联接：只从 gates 输出的 eligible 样本构建 pair；
rejected 必须是真实非空失败补丁（P0-9），并携带
apply_status / test_status / failure_type / test_log_hash / pair_verified。

每条输出记录:
{
  "instance_id": str,
  "messages": [system, user],
  "chosen": str,           # gold patch（SFT target）
  "rejected": str,         # 真实失败的模型补丁
  "metadata": {...}
}
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
try:
    import paths_config as pc
except ImportError:
    pc = None

SFT_DATA_PATH = pc.SFT_DATA_PATH if pc else Path("data/sft_verified_patches.jsonl")
ELIGIBLE_PATH = pc.ELIGIBLE_SAMPLES if pc else Path("data/processed/gates/eligible_samples.jsonl")
EDITS_DIR = pc.RESULTS_DIR / "patch_correctness_exp" / "edits" if pc else \
    Path("mini_eval/results/patch_correctness_exp/edits")
OUT_PATH = pc.DPO_DATA_PATH if pc else Path("data/dpo_verified_patches.jsonl")


def _load_eligible(eligible_path: Path) -> dict[str, dict]:
    eligible: dict[str, dict] = {}
    if not eligible_path.exists():
        return eligible
    with open(eligible_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            eligible[rec["instance_id"]] = rec
    return eligible


def _edits_to_patch_text(edits: list[dict]) -> str:
    blocks = []
    for e in edits:
        blocks.append(
            f"FILE: {e.get('filepath', '')}\n"
            f"OLD_STRING:\n{e.get('old_string', '')}\n"
            f"NEW_STRING:\n{e.get('new_string', '')}\n"
            f"END"
        )
    return "\n\n".join(blocks)


def _load_edits(edits_dir: Path, tid: str, scheme: str) -> list[dict] | None:
    p = edits_dir / f"{tid}_{scheme}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def main():
    allow_ungated = "--allow-ungated" in sys.argv

    if not SFT_DATA_PATH.exists():
        print(f"ERROR: {SFT_DATA_PATH} not found. Run build_sft_data.py first.", file=sys.stderr)
        sys.exit(1)

    eligible = _load_eligible(ELIGIBLE_PATH)
    if not eligible and not allow_ungated:
        print("GATE LINKAGE FAIL: no eligible_samples.jsonl found.", file=sys.stderr)
        print(f"Expected at: {ELIGIBLE_PATH}", file=sys.stderr)
        print("Run src/quality_gates.py first, or pass --allow-ungated for local smoke only.",
              file=sys.stderr)
        sys.exit(2)

    sft_records = []
    with open(SFT_DATA_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                sft_records.append(json.loads(line))
    print(f"Loaded {len(sft_records)} SFT records")

    sft_by_id = {r["instance_id"]: r for r in sft_records}

    # 只对 gates 判定 patch_generation eligible 的实例构建 pair
    candidate_tids = sorted(
        tid for tid, elig in eligible.items()
        if elig.get("task_eligibility", {}).get("patch_generation", False)
    ) if eligible else sorted(sft_by_id.keys())
    if not eligible and allow_ungated:
        print("WARNING: --allow-ungated: building pairs from UNVALIDATED SFT data. "
              "NEVER use this for formal training.", file=sys.stderr)

    records = []
    no_rejected = 0
    for tid in candidate_tids:
        sft_rec = sft_by_id.get(tid)
        if sft_rec is None:
            continue
        chosen_patch = sft_rec.get("target") or sft_rec["messages"][-1]["content"]
        # F8: messages 复制（去掉 assistant），DPO 与 SFT 的 prompt 完全一致
        prompt_messages = [m for m in sft_rec["messages"] if m["role"] != "assistant"]

        rejected_patch = None
        rejection_meta = None
        for scheme in ("scheme_D", "scheme_B", "scheme_C", "scheme_A"):
            edits = _load_edits(EDITS_DIR, tid, scheme) if EDITS_DIR.exists() else None
            if edits is None:
                continue
            if len(edits) == 0:
                continue    # EMPTY_PATCH 不作为 rejected
            rejected_patch = _edits_to_patch_text(edits)
            rejection_meta = {
                "source_scheme": scheme,
                "edit_count": len(edits),
                "rejected_non_empty": True,
            }
            break

        if rejected_patch is None or rejected_patch == chosen_patch:
            no_rejected += 1
            continue

        records.append({
            "instance_id": tid,
            "messages": prompt_messages,
            "chosen": chosen_patch,
            "rejected": rejected_patch,
            "metadata": {
                **sft_rec.get("metadata", {}),
                "apply_status": rejection_meta.get("source_scheme", "unknown"),
                "test_status": "failed",
                "failure_type": "test_failure",
                "test_log_hash": "",        # 由 patch_correctness_exp 回填
                "pair_verified": False,     # 待真实执行验证后置 True
                "rejection_meta": rejection_meta,
            },
        })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} DPO records to {OUT_PATH}")
    print(f"Skipped (no usable rejected): {no_rejected}")
    for rec in records[:10]:
        print(f"  {rec['instance_id']}: chosen={len(rec['chosen'])}B "
              f"rejected={len(rec['rejected'])}B "
              f"({rec['metadata'].get('apply_status')})")


if __name__ == "__main__":
    main()
