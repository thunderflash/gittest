#!/usr/bin/env python3
"""
Stage 1: Verified Patch SFT training for Gemma 4 31B.

Trains a LoRA adapter on true Issue→Patch pairs. Only the assistant's patch
content is supervised (system/user tokens are masked with -100).

Designed to run on cloud GPU (Kaggle, etc.) with the gated Gemma 4 31B QAT model.

Usage:
  python train_sft.py \
    --data training/data/sft_verified_patches.jsonl \
    --out training/adapters/patch_sft_v1 \
    --model google/gemma-4-31B-it-qat-w4a16-ct \
    --hf-token $HF_TOKEN \
    --r 16 --alpha 32 --dropout 0.05 --lr 5e-5 \
    --epochs 3 --max_len 8192 --grad_steps 8 --batch 1
"""

import argparse
import json
import os
import random
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL_ID = "google/gemma-4-31B-it-qat-w4a16-ct"


class SFTDataset(Dataset):
    """Each item: list of messages with loss mask on assistant tokens only."""

    def __init__(self, data, tokenizer, max_len=8192):
        self.data = data
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        msgs = self.data[idx]["messages"]
        tokens = []
        labels = []
        for m in msgs:
            text = self.tok.apply_chat_template(
                [m], tokenize=False, add_generation_prompt=False
            )
            t = self.tok.encode(text, add_special_tokens=False)
            if m["role"] == "assistant":
                tokens.extend(t)
                labels.extend(t)
            else:
                tokens.extend(t)
                labels.extend([-100] * len(t))
        tokens = tokens[: self.max_len]
        labels = labels[: self.max_len]
        input_ids = torch.tensor(tokens, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        attn = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attn}


def collate(batch, pad_id):
    maxlen = max(b["input_ids"].numel() for b in batch)
    inp = torch.full((len(batch), maxlen), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), maxlen), -100, dtype=torch.long)
    attn = torch.zeros((len(batch), maxlen), dtype=torch.long)
    for i, b in enumerate(batch):
        n = b["input_ids"].numel()
        inp[i, :n] = b["input_ids"]
        lab[i, :n] = b["labels"]
        attn[i, :n] = 1
    return {"input_ids": inp, "labels": lab, "attention_mask": attn}


def hash_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="SFT data jsonl: [{messages:[...]}]")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""),
                    help="HF token for gated model")
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max_len", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grad_steps", type=int, default=8)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--dequantize", action="store_true",
                    help="load ct base and dequantize to bf16 for training")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    data = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    if not data:
        print("NO TRAINING DATA -> BLOCKED. refusing to emit empty adapter.", flush=True)
        sys.exit(2)

    if not os.path.isdir(args.model) and not args.hf_token:
        print("BLOCKED_NO_BASE_WEIGHTS: model is not a local folder and no HF_TOKEN provided.",
              flush=True)
        sys.exit(2)

    print(f"Loading {len(data)} training samples", flush=True)
    print(f"Tokenizer: {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, token=args.hf_token or None)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"Loading base model: {args.model}", flush=True)

    def _load(use_quant=False):
        kw = dict(token=args.hf_token or None, torch_dtype=torch.bfloat16, device_map="auto")
        if use_quant:
            try:
                from compressed_tensors.quantization import CompressedTensorsConfig
                kw["quantization_config"] = CompressedTensorsConfig()
                return AutoModelForCausalLM.from_pretrained(args.model, **kw)
            except Exception:
                return None
        return AutoModelForCausalLM.from_pretrained(args.model, **kw)

    model = None
    if args.dequantize:
        try:
            model = _load(use_quant=True)
            if model is not None:
                model = model.dequantize()
                print("dequantized ct base -> bf16", flush=True)
        except Exception as e:
            print(f"dequantize failed ({repr(e)[:150]}); plain load", flush=True)
            model = None
    if model is None:
        model = _load(use_quant=False)

    try:
        model.config.use_cache = False
    except Exception:
        pass

    model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=args.r,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        # Gemma 4 target modules: match q_proj and v_proj in language_model
        target_modules=[".*language_model.*q_proj", ".*language_model.*v_proj"],
    )
    model = get_peft_model(model, lora)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    ds = SFTDataset(data, tok, args.max_len)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch, shuffle=True,
        collate_fn=lambda b: collate(b, tok.pad_token_id)
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = max(1, len(loader) * args.epochs // args.grad_steps)
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=0,
                                            num_training_steps=total_steps)

    os.makedirs(args.out, exist_ok=True)
    loss_hist = []
    model.train()
    step = 0
    for ep in range(args.epochs):
        for bi, b in enumerate(loader):
            out = model(
                input_ids=b["input_ids"].to(model.device),
                attention_mask=b["attention_mask"].to(model.device),
                labels=b["labels"].to(model.device),
            )
            loss = out.loss / args.grad_steps
            loss.backward()
            if (bi + 1) % args.grad_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1
                loss_hist.append(float(loss.item() * args.grad_steps))
                print(f"epoch={ep+1} step={step} loss={loss_hist[-1]:.4f}", flush=True)

    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)

    data_sha = hash_file(args.data)
    meta = {
        "base_model": args.model,
        "data_file": args.data,
        "data_sha256": data_sha,
        "seed": args.seed,
        "r": args.r, "alpha": args.alpha, "dropout": args.dropout,
        "lr": args.lr, "epochs": args.epochs, "max_len": args.max_len,
        "n_samples": len(data),
        "final_steps": step,
        "loss_curve": loss_hist,
        "stage": "sft_verified_patch",
    }
    json.dump(meta, open(os.path.join(args.out, "training_meta.json"), "w",
                         encoding="utf-8"), indent=2)

    import glob
    adapt = glob.glob(os.path.join(args.out, "adapter_model.safetensors"))
    if not adapt:
        print("WARNING: no adapter_model.safetensors produced -> GATE FAIL", flush=True)
        sys.exit(3)
    print(f"Saved adapter + meta to {args.out}", flush=True)
    print(f"Adapter bytes: {os.path.getsize(adapt[0])}", flush=True)


if __name__ == "__main__":
    main()
