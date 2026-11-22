#!/usr/bin/env python3
"""
Stage 2: Build Verified Patch DPO training data.

Reads SFT verified patches (chosen = gold developer patch) and pairs them with
rejected patches persisted by patch_correctness_exp.py (scheme edits JSON files).

Each output record:
{
  "instance_id": str,
  "prompt": "...",           # system + user (same as SFT, without assistant)
  "chosen": "...",           # gold patch (passes FAIL_TO_PASS + PASS_TO_PASS)
  "rejected": "...",         # failed model patch (fails tests or syntax)
  "metadata": {...}
}

Pairs are within the same Issue (same instance_id), so the model learns
to distinguish correct vs incorrect fixes for identical context.

Rejected patch sources (priority order):
  1. scheme_D edits that applied but failed tests (TESTS_FAIL)
  2. scheme_B edits that applied but failed tests (TESTS_FAIL)
  3. scheme_C edits that applied but failed tests (TESTS_FAIL)
  4. Any scheme with PATCH_CONFLICT / EMPTY_PATCH (empty string = no-op)
"""

import json
import sys
from pathlib import Path

SFT_DATA_PATH = Path(r"F:\test\gemma4\training\data\sft_verified_patches.jsonl")
EDITS_DIR = Path(r"F:\test\gemma4\mini_eval\results\patch_correctness_exp\edits")
OUT_DIR = Path(r"F:\test\gemma4\training\data")


SYSTEM_PROMPT = """You are a software engineer fixing a bug in a GitHub repository.
Given an Issue and the relevant repository context, generate file edits that fix the bug.

Output file edits using this exact format for each change:

FILE: <relative file path>
OLD_STRING:
<exact text from the current file to replace>
NEW_STRING:
<replacement text>
END

Repeat FILE/OLD_STRING/NEW_STRING/END blocks for multiple edits.
Do not include explanations outside the blocks.
OLD_STRING must be copied EXACTLY from the current file content.
Only modify the files that need to change to fix the bug.
Do not add, remove, or modify unrelated code.
"""


def _extract_prompt_from_sft(sft_record: dict) -> str:
    """Reconstruct the prompt (system + user) from an SFT record."""
    msgs = sft_record["messages"]
    parts = []
    for m in msgs:
        if m["role"] == "system":
            parts.append(m["content"])
        elif m["role"] == "user":
            parts.append(m["content"])
    return "\n\n".join(parts)


def _edits_to_patch_text(edits: list[dict]) -> str:
    """Convert list of {filepath, old_string, new_string} to patch text blocks."""
    blocks = []
    for e in edits:
        blocks.append(
            f"FILE: {e['filepath']}\n"
            f"OLD_STRING:\n{e.get('old_string', '')}\n"
            f"NEW_STRING:\n{e.get('new_string', '')}\n"
            f"END"
        )
    return "\n\n".join(blocks)


def _load_edits(edits_dir: Path, tid: str, scheme: str) -> list[dict] | None:
    """Load persisted edits for a task/scheme. Returns None if not found."""
    p = edits_dir / f"{tid}_{scheme}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def build_dpo(sft_records: list[dict], edits_dir: Path) -> list[dict]:
    """Build DPO records pairing gold patch (chosen) with failed model patch (rejected)."""
    sft_by_id = {r["instance_id"]: r for r in sft_records}
    records = []

    # Scan edits directory for all persisted scheme results
    if not edits_dir.exists():
        print(f"WARNING: edits directory {edits_dir} not found. Using empty rejected patches.")
        for rec in sft_records:
            prompt = _extract_prompt_from_sft(rec)
            chosen = rec["messages"][-1]["content"]
            records.append({
                "instance_id": rec["instance_id"],
                "prompt": prompt,
                "chosen": chosen,
                "rejected": "",
                "metadata": rec["metadata"],
            })
        return records

    # Collect all task IDs that have edits
    edit_tids = set()
    for f in edits_dir.glob("*.json"):
        # filename format: {tid}_{scheme}.json
        parts = f.stem.rsplit("_", 1)
        if len(parts) == 2:
            edit_tids.add(parts[0])

    for tid in edit_tids:
        if tid not in sft_by_id:
            continue
        sft_rec = sft_by_id[tid]
        chosen_patch = sft_rec["messages"][-1]["content"]
        prompt = _extract_prompt_from_sft(sft_rec)

        # Priority: TESTS_FAIL schemes (applied but incorrect) > PATCH_CONFLICT > EMPTY
        rejected_patch = None
        for scheme in ("scheme_D", "scheme_B", "scheme_C", "scheme_A"):
            edits = _load_edits(edits_dir, tid, scheme)
            if edits is None:
                continue
            if len(edits) == 0:
                # EMPTY_PATCH - model produced nothing
                rejected_patch = ""
                break
            # Edits exist - model produced a patch (may be correct or incorrect)
            rejected_patch = _edits_to_patch_text(edits)
            break

        if rejected_patch is None:
            rejected_patch = ""

        records.append({
            "instance_id": tid,
            "prompt": prompt,
            "chosen": chosen_patch,
            "rejected": rejected_patch,
            "metadata": sft_rec["metadata"],
        })

    return records


def main():
    if not SFT_DATA_PATH.exists():
        print(f"ERROR: {SFT_DATA_PATH} not found. Run build_sft_data.py first.", file=sys.stderr)
        sys.exit(1)

    sft_records = []
    with open(SFT_DATA_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                sft_records.append(json.loads(line))

    print(f"Loaded {len(sft_records)} SFT records")

    dpo_records = build_dpo(sft_records, EDITS_DIR)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    out_path = OUT_DIR / "dpo_verified_patches.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in dpo_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(dpo_records)} DPO records to {out_path}")
    for rec in dpo_records:
        rej_len = len(rec["rejected"])
        print(f"  {rec['instance_id']}: chosen={len(rec['chosen'])} chars, rejected={rej_len} chars")


if __name__ == "__main__":
    main()
