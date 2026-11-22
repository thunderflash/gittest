#!/usr/bin/env python3
"""
Stage 2: Verified Patch DPO training for Gemma 4 31B.

Fine-tunes the Stage 1 SFT adapter using Direct Preference Optimization.
chosen = verified gold patch (passes tests), rejected = failed model patch.

Usage:
  python train_dpo.py \
    --data training/data/dpo_verified_patches.jsonl \
    --sft-adapter training/adapters/patch_sft_v1 \
    --out training/adapters/patch_sft_dpo_v1 \
    --model google/gemma-4-31B-it-qat-w4a16-ct \
    --hf-token $HF_TOKEN \
    --r 16 --alpha 32 --dropout 0.05 --lr 1e-5 \
    --epochs 2 --max_len 8192 --grad_steps 16 --batch 1 \
    --beta 0.1
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
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

MODEL_ID = "google/gemma-4-31B-it-qat-w4a16-ct"


class DPODataset(Dataset):
    """Each item: prompt + chosen + rejected patch text."""

    def __init__(self, data, tokenizer, max_len=8192):
        self.data = data
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item["prompt"]
        chosen = item["chosen"]
        rejected = item["rejected"]

        # Tokenize full sequence: prompt + completion
        def encode_full(prompt_text, completion_text):
            full_text = prompt_text + "\n\n" + completion_text
            # Tokenize prompt part separately to know where completion starts
            prompt_ids = self.tok.encode(prompt_text, add_special_tokens=False)
            full_ids = self.tok.encode(full_text, add_special_tokens=False)
            return prompt_ids, full_ids

        p_chosen, f_chosen = encode_full(prompt, chosen)
        p_rej, f_rej = encode_full(prompt, rejected)

        return {
            "prompt_chosen": p_chosen,
            "full_chosen": f_chosen,
            "prompt_rejected": p_rej,
            "full_rejected": f_rej,
            "instance_id": item.get("instance_id", ""),
        }


def collate_dpo(batch, pad_id):
    def pad_seq(seqs, pad_val):
        maxlen = max(len(s) for s in seqs)
        out = torch.full((len(seqs), maxlen), pad_val, dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        return out

    return {
        "prompt_chosen": pad_seq([b["prompt_chosen"] for b in batch], pad_id),
        "full_chosen": pad_seq([b["full_chosen"] for b in batch], pad_id),
        "prompt_rejected": pad_seq([b["prompt_rejected"] for b in batch], pad_id),
        "full_rejected": pad_seq([b["full_rejected"] for b in batch], pad_id),
        "instance_id": [b["instance_id"] for b in batch],
    }


def get_batch_logits(model, input_ids, attention_mask):
    """Run model forward and return logits."""
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    return out.logits


def sequence_log_prob(logits, input_ids, prompt_len):
    """
    Compute log probability of the completion tokens (after prompt_len).
    logits: [B, T, V], input_ids: [B, T], prompt_len: int
    Returns: [B] log prob per sequence
    """
    # Shift: predict token at position t from logits at position t-1
    shift_logits = logits[:, :-1, :]  # [B, T-1, V]
    shift_ids = input_ids[:, 1:]      # [B, T-1]
    # Only completion tokens (after prompt)
    # completion starts at prompt_len in input_ids, which is position prompt_len-1 in shift_ids
    comp_start = max(prompt_len - 1, 0)
    comp_logits = shift_logits[:, comp_start:, :]   # [B, T_comp, V]
    comp_ids = shift_ids[:, comp_start:]            # [B, T_comp]

    log_probs = F.log_softmax(comp_logits, dim=-1)  # [B, T_comp, V]
    gathered = log_probs.gather(-1, comp_ids.unsqueeze(-1)).squeeze(-1)  # [B, T_comp]
    # Mask padding tokens
    pad_mask = (comp_ids != -100)
    # If padding, log_prob contribution is 0
    gathered = gathered * pad_mask
    return gathered.sum(dim=-1)  # [B]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="DPO data jsonl")
    ap.add_argument("--sft-adapter", required=True, help="Path to Stage 1 SFT adapter")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grad-steps", type=int, default=16)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--beta", type=float, default=0.1, help="DPO temperature")
    ap.add_argument("--dequantize", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    data = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    if not data:
        print("NO TRAINING DATA -> BLOCKED.", flush=True)
        sys.exit(2)

    print(f"Loaded {len(data)} DPO pairs", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, token=args.hf_token or None)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def _load_base(use_quant=False):
        kw = dict(token=args.hf_token or None, torch_dtype=torch.bfloat16, device_map="auto")
        if use_quant:
            try:
                from compressed_tensors.quantization import CompressedTensorsConfig
                kw["quantization_config"] = CompressedTensorsConfig()
                return AutoModelForCausalLM.from_pretrained(args.model, **kw)
            except Exception:
                return None
        return AutoModelForCausalLM.from_pretrained(args.model, **kw)

    # Policy model: base + SFT adapter (trainable)
    base_model = _load_base(args.dequantize)
    if base_model is None:
        base_model = _load_base(False)
    try:
        base_model.config.use_cache = False
    except Exception:
        pass
    base_model = prepare_model_for_kbit_training(base_model)
    lora = LoraConfig(
        r=args.r, lora_alpha=args.alpha, lora_dropout=args.dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=[".*language_model.*q_proj", ".*language_model.*v_proj"],
    )
    policy = get_peft_model(base_model, lora)
    # Load SFT adapter weights into the policy model
    policy = PeftModel.from_pretrained(base_model, args.sft_adapter, is_trainable=True)
    policy.gradient_checkpointing_enable()
    policy.enable_input_require_grads()
    policy.print_trainable_parameters()

    # Reference model: base + SFT adapter (frozen)
    ref_base = _load_base(args.dequantize)
    if ref_base is None:
        ref_base = _load_base(False)
    try:
        ref_base.config.use_cache = False
    except Exception:
        pass
    ref_base = prepare_model_for_kbit_training(ref_base)
    ref_model = PeftModel.from_pretrained(ref_base, args.sft_adapter, is_trainable=False)
    ref_model.eval()

    ds = DPODataset(data, tok, args.max_len)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch, shuffle=True,
        collate_fn=lambda b: collate_dpo(b, tok.pad_token_id)
    )
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr)
    total_steps = max(1, len(loader) * args.epochs // args.grad_steps)
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=0,
                                            num_training_steps=total_steps)

    os.makedirs(args.out, exist_ok=True)
    loss_hist = []
    policy.train()
    step = 0

    for ep in range(args.epochs):
        for bi, b in enumerate(loader):
            # Chosen forward
            chosen_ids = b["full_chosen"].to(policy.device)
            chosen_mask = torch.ones_like(chosen_ids)
            chosen_prompt_lens = [len(p) for p in b["prompt_chosen"]]

            with torch.no_grad():
                ref_chosen_logits = get_batch_logits(ref_model, chosen_ids, chosen_mask)
            policy_chosen_logits = get_batch_logits(policy, chosen_ids, chosen_mask)

            # Rejected forward
            rej_ids = b["full_rejected"].to(policy.device)
            rej_mask = torch.ones_like(rej_ids)
            rej_prompt_lens = [len(p) for p in b["prompt_rejected"]]

            with torch.no_grad():
                ref_rej_logits = get_batch_logits(ref_model, rej_ids, rej_mask)
            policy_rej_logits = get_batch_logits(policy, rej_ids, rej_mask)

            # Compute log probs
            chosen_log_pi = []
            chosen_log_ref = []
            for i in range(chosen_ids.size(0)):
                pl = chosen_prompt_lens[i]
                chosen_log_pi.append(sequence_log_prob(policy_chosen_logits[i:i+1],
                                                        chosen_ids[i:i+1], pl)[0])
                chosen_log_ref.append(sequence_log_prob(ref_chosen_logits[i:i+1],
                                                         chosen_ids[i:i+1], pl)[0])

            rej_log_pi = []
            rej_log_ref = []
            for i in range(rej_ids.size(0)):
                pl = rej_prompt_lens[i]
                rej_log_pi.append(sequence_log_prob(policy_rej_logits[i:i+1],
                                                     rej_ids[i:i+1], pl)[0])
                rej_log_ref.append(sequence_log_prob(ref_rej_logits[i:i+1],
                                                      rej_ids[i:i+1], pl)[0])

            chosen_log_pi = torch.stack(chosen_log_pi)
            chosen_log_ref = torch.stack(chosen_log_ref)
            rej_log_pi = torch.stack(rej_log_pi)
            rej_log_ref = torch.stack(rej_log_ref)

            # DPO loss
            # L = -log sigmoid(beta * (log_pi_chosen - log_ref_chosen - log_pi_rej + log_ref_rej))
            logits_diff = (chosen_log_pi - chosen_log_ref) - (rej_log_pi - rej_log_ref)
            loss = -F.logsigmoid(args.beta * logits_diff).mean()

            loss = loss / args.grad_steps
            loss.backward()

            if (bi + 1) % args.grad_steps == 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1
                loss_hist.append(float(loss.item() * args.grad_steps))
                print(f"epoch={ep+1} step={step} dpo_loss={loss_hist[-1]:.4f}", flush=True)

    policy.save_pretrained(args.out)
    tok.save_pretrained(args.out)

    data_sha = hash_file(args.data)
    meta = {
        "base_model": args.model,
        "sft_adapter": args.sft_adapter,
        "data_file": args.data,
        "data_sha256": data_sha,
        "seed": args.seed,
        "r": args.r, "alpha": args.alpha, "dropout": args.dropout,
        "lr": args.lr, "epochs": args.epochs, "max_len": args.max_len,
        "beta": args.beta,
        "n_samples": len(data),
        "final_steps": step,
        "loss_curve": loss_hist,
        "stage": "dpo_verified_patch",
    }
    json.dump(meta, open(os.path.join(args.out, "training_meta.json"), "w",
                         encoding="utf-8"), indent=2)

    import glob
    adapt = glob.glob(os.path.join(args.out, "adapter_model.safetensors"))
    if not adapt:
        print("WARNING: no adapter_model.safetensors produced -> GATE FAIL", flush=True)
        sys.exit(3)
    print(f"Saved DPO adapter + meta to {args.out}", flush=True)
    print(f"Adapter bytes: {os.path.getsize(adapt[0])}", flush=True)


if __name__ == "__main__":
    main()
