#!/usr/bin/env python3
"""
Stage 1: Build Verified Patch SFT training data (gate-linked).

F5 强制联接：默认必须存在 quality gates 输出（eligible_samples.jsonl），且
instance 的 task_eligibility["patch_generation"] 为 True 才会生成训练样本。
没有 gates 输出时直接阻断（--allow-ungated 仅用于本地 smoke，会打印显著警告）。

P0-7 保持：prompt 只含 problem_statement + base_commit，不泄漏
test_diff / merge_sha / gold-patch 衍生文件清单。

每条输出记录（统一契约）:
{
  "instance_id": str,
  "task_type": "patch_generation",
  "messages": [system, user, assistant],
  "target": <assistant patch text>,
  "label_provenance": {...},
  "metadata": {...}
}
"""

import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
try:
    import paths_config as pc
except ImportError:
    pc = None

RECON_PATH = pc.RECON_PATH if pc else Path("mini_eval/reconstructed.json")
ELIGIBLE_PATH = pc.ELIGIBLE_SAMPLES if pc else Path("data/processed/gates/eligible_samples.jsonl")
VALIDATION_SUMMARY = pc.VALIDATION_SUMMARY if pc else Path("data/processed/validation_summary.json")
OUT_PATH = pc.SFT_DATA_PATH if pc else Path("data/sft_verified_patches.jsonl")

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


def _split_gold_diff_into_edits(gold_diff: str) -> list[dict]:
    """Parse a unified diff into {filepath, old_string, new_string} edits."""
    edits = []
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
            old_lines, new_lines = [], []
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
            if current_file:
                edits.append({"filepath": current_file,
                              "old_string": "".join(old_lines),
                              "new_string": "".join(new_lines)})
            continue
        i += 1
    return edits


def _format_edits_as_patch(edits: list[dict]) -> str:
    blocks = []
    for e in edits:
        blocks.append(
            f"FILE: {e['filepath']}\n"
            f"OLD_STRING:\n{e['old_string']}\n"
            f"NEW_STRING:\n{e['new_string']}\n"
            f"END"
        )
    return "\n\n".join(blocks)


def _load_eligible(eligible_path: Path) -> dict[str, dict]:
    """读取 gates 输出 → {instance_id: {split, task_eligibility}}。"""
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


def build_record(tid: str, task: dict, validation: dict | None) -> dict | None:
    gold_diff = task.get("gold_diff") or ""
    if not gold_diff.strip():
        return None
    edits = _split_gold_diff_into_edits(gold_diff)
    if not edits:
        return None

    patch_text = _format_edits_as_patch(edits)
    problem = (task.get("problem_statement", "") or "").strip()

    user_lines = []
    if problem:
        user_lines.append(f"Issue: {problem}")
    else:
        user_lines.append(f"Fix the bug in task {tid}.")
    user_lines.append("")
    user_lines.append("Repository Context:")
    user_lines.append(f"  Base commit: {task.get('base_commit', 'unknown')}")
    user_lines.append("")
    user_lines.append(
        "Task: Generate file edits (FILE/OLD_STRING/NEW_STRING/END blocks) that fix the bug. "
        "Use only the issue description and repository context at base_commit."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(user_lines)},
        {"role": "assistant", "content": patch_text},
    ]

    return {
        "instance_id": tid,
        "task_type": "patch_generation",
        "messages": messages,
        "target": patch_text,
        "label_provenance": {
            "label_source": "gold_patch",
            "label_strength": "strong",
            "verified": bool(validation and validation.get("verified")),
            "apply_method": (validation or {}).get("apply_method", "unknown"),
            "context_hash": (validation or {}).get("context_hash", ""),
        },
        "metadata": {
            "base_commit": task.get("base_commit"),
            "repo": task.get("repo", ""),
            "gold_diff": gold_diff,
            "num_edits": len(edits),
            "validation": validation or {},
        },
    }


def _load_validations(vs_path: Path) -> dict[str, dict]:
    """读取 S4 验证结果 → {instance_id: validation_result}。"""
    out: dict[str, dict] = {}
    if not vs_path.exists():
        return out
    try:
        data = json.loads(vs_path.read_text(encoding="utf-8"))
    except Exception:
        return out
    for r in data.get("results", []):
        out[r.get("instance_id", "")] = r
    return out


def main():
    allow_ungated = "--allow-ungated" in sys.argv

    if not RECON_PATH.exists():
        print(f"ERROR: {RECON_PATH} not found", file=sys.stderr)
        sys.exit(1)
    recon = json.loads(RECON_PATH.read_text(encoding="utf-8"))

    eligible = _load_eligible(ELIGIBLE_PATH)
    validations = _load_validations(VALIDATION_SUMMARY)
    if not eligible and not allow_ungated:
        print("GATE LINKAGE FAIL: no eligible_samples.jsonl found.", file=sys.stderr)
        print(f"Expected at: {ELIGIBLE_PATH}", file=sys.stderr)
        print("Run src/quality_gates.py first, or pass --allow-ungated for local smoke only.",
              file=sys.stderr)
        sys.exit(2)
    if not eligible and allow_ungated:
        print("WARNING: --allow-ungated: emitting UNVALIDATED data. "
              "NEVER use this for formal training.", file=sys.stderr)

    records = []
    skipped_ungated = 0
    skipped_task = 0
    for tid, task in recon.items():
        if eligible:
            elig = eligible.get(tid)
            if elig is None:
                skipped_ungated += 1
                continue
            if not elig.get("task_eligibility", {}).get("patch_generation", False):
                skipped_task += 1
                continue
        validation = validations.get(tid)
        rec = build_record(tid, task, validation)
        if rec is None:
            continue
        records.append(rec)

    random.seed(42)
    random.shuffle(records)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} SFT records to {OUT_PATH}")
    print(f"Skipped: {skipped_ungated} not-in-gates, {skipped_task} task-not-eligible")

    # 数据 SHA 记录，供训练侧校验
    if records:
        sha = hashlib.sha256(OUT_PATH.read_bytes()).hexdigest()
        sidecar = OUT_PATH.with_suffix(".sha256")
        sidecar.write_text(sha, encoding="utf-8")
        print(f"Data sha256: {sha[:16]}... -> {sidecar}")


if __name__ == "__main__":
    main()
