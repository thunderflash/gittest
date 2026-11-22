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

    # 1. Logits diff check
    adapter_model = PeftModel.from_pretrained(base_model, args.adapter, is_trainable=False)
    adapter_model.eval()
    with torch.no_grad():
        adapter_logits = adapter_model(**inputs).logits

    diff = (base_logits - adapter_logits).abs().max().item()
    logits_ok = diff > args.threshold
    print(f"Base vs Adapter logits max diff: {diff:.6f} (threshold {args.threshold})", flush=True)

    # 2. Target module / LoRA weights check
    lora_ok, lora_detail = check_lora_weights(adapter_model)
    print(f"LoRA weights: {lora_detail}", flush=True)

    # 3. Task quality check
    task_ok, task_detail = check_task_output(adapter_model, tok, args.threshold)
    print(f"Task quality: {task_detail}", flush=True)

    all_ok = logits_ok and lora_ok and task_ok
    if all_ok:
        print("VERIFY_PASSED: adapter active, LoRA weights loaded, task output valid.", flush=True)
        sys.exit(0)
    else:
        print("VERIFY_FAILED:", flush=True)
        if not logits_ok:
            print("  - Logits diff below threshold (adapter may be inactive)", flush=True)
        if not lora_ok:
            print(f"  - {lora_detail}", flush=True)
        if not task_ok:
            print(f"  - {task_detail}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()