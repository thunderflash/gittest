#!/usr/bin/env python3
"""
Verify that a LoRA adapter actually changes model output (logits diff).

Usage (cloud GPU):
  python verify_adapter.py \
    --model google/gemma-4-31B-it-qat-w4a16-ct \
    --adapter training/adapters/patch_sft_v1 \
    --hf-token $HF_TOKEN

Exit 0 if adapter is active (logits diff > threshold), exit 1 if not.
"""

import argparse
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_ID = "google/gemma-4-31B-it-qat-w4a16-ct"


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

    adapter_model = PeftModel.from_pretrained(base_model, args.adapter, is_trainable=False)
    adapter_model.eval()
    with torch.no_grad():
        adapter_logits = adapter_model(**inputs).logits

    diff = (base_logits - adapter_logits).abs().max().item()
    print(f"Base vs Adapter logits max diff: {diff:.6f}", flush=True)

    if diff > args.threshold:
        print("ADAPTER_ACTIVE: LoRA weights are affecting model output.", flush=True)
        sys.exit(0)
    else:
        print("ADAPTER_INACTIVE: LoRA weights are NOT affecting model output.", flush=True)
        print("  Possible causes: adapter weights not loaded, base model mismatch,", flush=True)
        print("  or adapter file is empty/corrupt.", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
