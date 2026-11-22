#!/usr/bin/env python3
"""
Verify that a LoRA adapter actually changes model output and produces
valid patch-format responses for bug-fixing tasks.

P1 verification dimensions:
- Logits diff: adapter is active (weights affect output).
- Target module: LoRA weights are loaded and non-zero on expected modules.
- Task quality: model output matches FILE/OLD_STRING/NEW_STRING/END format.

Usage (cloud GPU):
  python verify_adapter.py \
    --model google/gemma-4-31B-it-qat-w4a16-ct \
    --adapter training/adapters/patch_sft_v1 \
    --hf-token $HF_TOKEN \
    --threshold 1e-4

Exit 0 if all checks pass, exit 1 if any check fails.
"""

import argparse
import glob
import hashlib
import json
import os
import re
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig

MODEL_ID = "google/gemma-4-31B-it-qat-w4a16-ct"

TASK_PROMPTS = [
    "Fix the bug in fastapi where ORJSONResponse should emit a deprecation warning when used.",
    "Fix the null pointer dereference in the request handler when the header is missing.",
    "Fix the off-by-one error in the pagination offset calculation.",
]

PATCH_BLOCK_RE = re.compile(
    r"FILE:\s*\S+\s*\nOLD_STRING:\s*\n.*?\nNEW_STRING:\s*\n.*?\nEND",
    re.DOTALL,
)


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def v0_adapter_files(adapter_dir: str) -> tuple[bool, str]:
    """V0: 文件完整性 — safetensors 存在、非零、有 config；记录哈希。"""
    weights = glob.glob(os.path.join(adapter_dir, "adapter_model.safetensors"))
    if not weights or os.path.getsize(weights[0]) == 0:
        return False, "adapter_model.safetensors missing or empty (.filepart?)"
    cfg = os.path.join(adapter_dir, "adapter_config.json")
    if not os.path.exists(cfg):
        return False, "adapter_config.json missing"
    sha = sha256_file(weights[0])
    return True, f"weights {os.path.getsize(weights[0])}B sha256={sha[:16]}..."


def v1_base_model_match(adapter_dir: str, model_id: str) -> tuple[bool, str]:
    """V1: 训练元数据中的 base_model 必须与当前 --model 一致。"""
    meta_path = os.path.join(adapter_dir, "training_meta.json")
    if not os.path.exists(meta_path):
        return True, "no training_meta.json (skip; recommend recording base_model)"
    try:
        meta = json.load(open(meta_path, encoding="utf-8"))
    except Exception as e:
        return False, f"training_meta.json unreadable: {e}"
    trained_on = meta.get("base_model", "")
    if trained_on and os.path.basename(trained_on) != os.path.basename(model_id) \
            and trained_on != model_id:
        return False, f"adapter trained on '{trained_on}' but verifying against '{model_id}'"
    stage = meta.get("stage", "?")
    return True, f"base_model match ({trained_on or 'unknown'}), stage={stage}"


def check_lora_weights(model) -> tuple[bool, str]:
    """检查 LoRA 权重是否已加载且非零（target module 验证）。"""
    lora_params = []
    for name, param in model.named_parameters():
        if "lora" in name.lower() and param.requires_grad:
            lora_params.append((name, param.data.abs().max().item()))

    if not lora_params:
        return False, "No trainable LoRA parameters found"

    nonzero = [(n, v) for n, v in lora_params if v > 0]
    if not nonzero:
        return False, "All LoRA weights are zero (adapter not loaded)"

    top = sorted(nonzero, key=lambda x: -x[1])[:5]
    detail = "; ".join(f"{n}={v:.6f}" for n, v in top)
    return True, f"{len(nonzero)} nonzero LoRA params, top: {detail}"


def check_task_output(model, tok, threshold: float) -> tuple[bool, str]:
    """任务质量验证：模型对 bug-fixing prompt 的输出是否符合 patch 格式。"""
    base_pass = 0
    adapter_pass = 0

    # 使用 adapter 模型生成
    for prompt in TASK_PROMPTS:
        messages = [
            {"role": "system", "content": "You are a software engineer fixing a bug. Output FILE/OLD_STRING/NEW_STRING/END blocks."},
            {"role": "user", "content": prompt},
        ]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=200, do_sample=False, pad_token_id=tok.pad_token_id)
        gen = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        if PATCH_BLOCK_RE.search(gen):
            adapter_pass += 1

    task_ok = adapter_pass >= max(1, len(TASK_PROMPTS) // 2)
    return task_ok, f"Adapter produced valid patch blocks: {adapter_pass}/{len(TASK_PROMPTS)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--adapter", required=True, help="Path to trained LoRA adapter")
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))
    ap.add_argument("--threshold", type=float, default=1e-4,
                    help="Min logits max-diff to consider adapter active")
    args = ap.parse_args()

    # ---- V0: 文件完整性 ----
    v0_ok, v0_msg = v0_adapter_files(args.adapter)
    print(f"[V0] files: {v0_msg}", flush=True)

    # ---- V1: base model 元数据匹配 ----
    v1_ok, v1_msg = v1_base_model_match(args.adapter, args.model)
    print(f"[V1] metadata: {v1_msg}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, token=args.hf_token or None)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kw = dict(token=args.hf_token or None, torch_dtype=torch.bfloat16, device_map="auto")
    base_model = AutoModelForCausalLM.from_pretrained(args.model, **kw)
    try:
        base_model.config.use_cache = False
    except Exception:
        pass

    test_input = "Fix this bug: ORJSONResponse should emit a deprecation warning."
    inputs = tok(test_input, return_tensors="pt").to(base_model.device)

    with torch.no_grad():
        base_logits = base_model(**inputs).logits

    # ---- V2a: logits diff ----
    adapter_model = PeftModel.from_pretrained(base_model, args.adapter, is_trainable=False)
    adapter_model.eval()
    with torch.no_grad():
        adapter_logits = adapter_model(**inputs).logits

    diff = (base_logits - adapter_logits).abs().max().item()
    logits_ok = diff > args.threshold
    print(f"[V2] logits max diff: {diff:.6f} (threshold {args.threshold})", flush=True)

    # ---- V2b: LoRA weights ----
    lora_ok, lora_detail = check_lora_weights(adapter_model)
    print(f"[V2] lora weights: {lora_detail}", flush=True)

    # ---- V2c: task quality ----
    task_ok, task_detail = check_task_output(adapter_model, tok, args.threshold)
    print(f"[V2] task quality: {task_detail}", flush=True)

    all_ok = v0_ok and v1_ok and logits_ok and lora_ok and task_ok
    if all_ok:
        print("VERIFY_PASSED: files intact, base model matches, adapter active, "
              "LoRA weights loaded, task output valid.", flush=True)
        sys.exit(0)
    else:
        print("VERIFY_FAILED:", flush=True)
        if not v0_ok:
            print(f"  [V0] {v0_msg}", flush=True)
        if not v1_ok:
            print(f"  [V1] {v1_msg}", flush=True)
        if not logits_ok:
            print("  [V2] logits diff below threshold (adapter may be inactive)", flush=True)
        if not lora_ok:
            print(f"  [V2] {lora_detail}", flush=True)
        if not task_ok:
            print(f"  [V2] {task_detail}", flush=True)
        print("NOTE: passing all V0-V2 checks is necessary but NOT sufficient — "
              "frozen repair eval (V3) is required before release.", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()