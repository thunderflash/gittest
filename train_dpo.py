#!/usr/bin/env python3
"""
Stage 2: Verified Patch DPO for Gemma 4 31B.

F8 fixes (audit):
- Prompt format unified with SFT/inference: prompt is rendered through the same
  chat template (system+user messages + generation prompt). Completion tokens
  are the chosen/rejected patch text appended after the generation prompt.
- Padding/attention: attention masks built from real sequence lengths; completion
  log-prob masks derived from prompt boundary + attention, never from pad id alone.
- max_len enforced: prompt left-truncated (tail preserved), completion kept whole;
  samples whose completion alone exceeds budget are dropped.
- Prefix alignment: prompt token boundary verified as true prefix; on mismatch the
  actual common-prefix length is used and the event is counted.
- Gradient accumulation: tail windows stepped with correct scaling (shared logic
  with SFT).
- --preflight: loads both models, runs one batch forward/backward/step, verifies
  reference stays frozen, saves/reloads adapter — before any real training.

Usage:
  python train_dpo.py \
    --data training/data/dpo_verified_patches.jsonl \
    --sft-adapter training/adapters/patch_sft_v1 \
    --out training/adapters/patch_sft_dpo_v1 \
    --model google/gemma-4-31B-it-qat-w4a16-ct \
    --hf-token $HF_TOKEN \
    --r 16 --alpha 32 --dropout 0.05 --lr 1e-5 \
    --epochs 2 --max_len 8192 --grad_steps 16 --batch 1 \
    --beta 0.1 --val_ratio 0.1 --early_stopping_patience 2 \
    --preflight --dequantize
"""

import argparse
import glob
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

MODEL_ID = "google/gemma-4-31B-it-qat-w4a16-ct"
IGNORE_INDEX = -100


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


class DPODataset(Dataset):
    """prompt（chat template 渲染）+ chosen/rejected completion。

    输入记录要求: {"messages": [system, user], "chosen": str, "rejected": str}
    （build_dpo_data.py 从 SFT 记录复制 messages，保证与 SFT 完全一致）。
    """

    def __init__(self, data, tokenizer, max_len=8192):
        self.tok = tokenizer
        self.max_len = max_len
        self.prefix_mismatch_count = 0
        self.items = []
        skipped = 0
        for item in data:
            messages = item.get("messages")
            chosen = item.get("chosen", "")
            rejected = item.get("rejected", "")
            if not messages or not chosen or not rejected or chosen == rejected:
                skipped += 1
                continue
            try:
                prompt_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                skipped += 1
                continue
            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

            def enc_full(completion: str) -> list[int]:
                return tokenizer.encode(prompt_text + "\n\n" + completion,
                                        add_special_tokens=False)

            full_chosen = enc_full(chosen)
            full_rejected = enc_full(rejected)

            # max_len: completion 保持完整；prompt 与两个 full 序列同量左截
            comp_budget = max(64, max_len - 2)
            if len(full_chosen) - len(prompt_ids) > comp_budget or \
               len(full_rejected) - len(prompt_ids) > comp_budget:
                skipped += 1
                continue
            overflow = max(len(full_chosen), len(full_rejected)) - max_len
            if overflow > 0:
                if overflow >= len(prompt_ids):
                    skipped += 1
                    continue
                prompt_ids = prompt_ids[overflow:]
                full_chosen = full_chosen[overflow:]
                full_rejected = full_rejected[overflow:]

            # 前缀校验与对齐（F8）
            def align(full: list[int]) -> int:
                nonlocal prompt_ids
                n = min(len(prompt_ids), len(full))
                if full[:n] == prompt_ids[:n]:
                    return n
                self.prefix_mismatch_count += 1
                # 回退到实际公共前缀
                common = 0
                for a, b in zip(prompt_ids, full):
                    if a != b:
                        break
                    common += 1
                return max(common, 1)

            p_len_c = align(full_chosen)
            p_len_r = align(full_rejected)

            self.items.append({
                "full_chosen": full_chosen[:max_len],
                "p_len_chosen": min(p_len_c, max_len),
                "full_rejected": full_rejected[:max_len],
                "p_len_rejected": min(p_len_r, max_len),
                "instance_id": item.get("instance_id", ""),
            })
        self.skipped = skipped

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        return {
            "full_chosen": it["full_chosen"],
            "p_len_chosen": it["p_len_chosen"],
            "full_rejected": it["full_rejected"],
            "p_len_rejected": it["p_len_rejected"],
            "instance_id": it["instance_id"],
        }


def collate_dpo(batch, pad_id):
    def pad_seq(seqs):
        maxlen = max(len(s) for s in seqs)
        out = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
        attn = torch.zeros((len(seqs), maxlen), dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, :len(s)] = torch.tensor(s, dtype=torch.long)
            attn[i, :len(s)] = 1                     # F8: 真实长度 attention mask
        return out, attn

    fc, fa = pad_seq([b["full_chosen"] for b in batch])
    fr, ra = pad_seq([b["full_rejected"] for b in batch])
    return {
        "full_chosen": fc, "chosen_attn": fa,
        "full_rejected": fr, "rejected_attn": ra,
        "p_len_chosen": [b["p_len_chosen"] for b in batch],
        "p_len_rejected": [b["p_len_rejected"] for b in batch],
        "instance_id": [b["instance_id"] for b in batch],
    }


def sequence_log_prob(logits, input_ids, attn_mask, prompt_len) -> torch.Tensor:
    """Completion token log-prob。

    F8: mask = attention mask ∧ (position >= prompt_len-1 after shift)；
    padding 永不参与（attention=0 且位于 completion 之外的 token 被排除）。
    """
    shift_logits = logits[:, :-1, :]
    shift_ids = input_ids[:, 1:]
    shift_attn = attn_mask[:, 1:].to(logits.dtype)
    comp_start = max(prompt_len - 1, 0)
    comp_logits = shift_logits[:, comp_start:, :]
    comp_ids = shift_ids[:, comp_start:]
    comp_attn = shift_attn[:, comp_start:]

    log_probs = F.log_softmax(comp_logits, dim=-1)
    gathered = log_probs.gather(-1, comp_ids.unsqueeze(-1)).squeeze(-1)
    return (gathered * comp_attn).sum(dim=-1)


def dpo_loss_for_batch(policy, ref_model, b, device, beta):
    chosen_ids = b["full_chosen"].to(device)
    chosen_attn = b["chosen_attn"].to(device)
    rej_ids = b["full_rejected"].to(device)
    rej_attn = b["rejected_attn"].to(device)

    with torch.no_grad():
        ref_c = ref_model(input_ids=chosen_ids, attention_mask=chosen_attn).logits
        ref_r = ref_model(input_ids=rej_ids, attention_mask=rej_attn).logits
    pol_c = policy(input_ids=chosen_ids, attention_mask=chosen_attn).logits
    pol_r = policy(input_ids=rej_ids, attention_mask=rej_attn).logits

    c_pi, c_ref = [], []
    for i in range(chosen_ids.size(0)):
        pl = b["p_len_chosen"][i]
        c_pi.append(sequence_log_prob(pol_c[i:i+1], chosen_ids[i:i+1],
                                      chosen_attn[i:i+1], pl)[0])
        c_ref.append(sequence_log_prob(ref_c[i:i+1], chosen_ids[i:i+1],
                                       chosen_attn[i:i+1], pl)[0])
    r_pi, r_ref = [], []
    for i in range(rej_ids.size(0)):
        pl = b["p_len_rejected"][i]
        r_pi.append(sequence_log_prob(pol_r[i:i+1], rej_ids[i:i+1],
                                      rej_attn[i:i+1], pl)[0])
        r_ref.append(sequence_log_prob(ref_r[i:i+1], rej_ids[i:i+1],
                                       rej_attn[i:i+1], pl)[0])

    logits_diff = (torch.stack(c_pi) - torch.stack(c_ref)) - \
                  (torch.stack(r_pi) - torch.stack(r_ref))
    return -F.logsigmoid(beta * logits_diff).mean()


def run_preflight(policy, ref_model, loader, device, tok, out_dir, opt):
    """F8: DPO preflight — 一个 batch 的完整训练路径 + 冻结校验 + 保存/重载。"""
    print("[PREFLIGHT] running single-batch forward/backward/step...", flush=True)
    ref_frozen_before = all(not p.requires_grad for p in ref_model.parameters())
    b = next(iter(loader))
    loss = dpo_loss_for_batch(policy, ref_model, b, device, beta=0.1)
    (loss / 1).backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    opt.step()
    opt.zero_grad()
    ref_frozen_after = all(not p.requires_grad for p in ref_model.parameters())
    pre_dir = Path(out_dir) / "_preflight_adapter"
    policy.save_pretrained(str(pre_dir))
    reloaded = PeftModel.from_pretrained(
        policy.get_base_model(), str(pre_dir), is_trainable=False)
    print(f"[PREFLIGHT] loss={float(loss):.4f} ref_frozen={ref_frozen_before}->{ref_frozen_after} "
          f"adapter_saved_and_reloaded={pre_dir.exists()}", flush=True)
    ok = ref_frozen_before and ref_frozen_after and pre_dir.exists()
    print("[PREFLIGHT] PASS" if ok else "[PREFLIGHT] FAIL", flush=True)
    return ok


def evaluate_dpo(policy, ref_model, loader, device, beta):
    policy.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for b in loader:
            total += float(dpo_loss_for_batch(policy, ref_model, b, device, beta))
            n += 1
    policy.train()
    return total / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--sft-adapter", required=True)
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
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--early_stopping_patience", type=int, default=2)
    ap.add_argument("--target_modules", type=str, default=None)
    ap.add_argument("--preflight", action="store_true",
                    help="只跑单 batch preflight（双 31B 显存验证），不训练")
    ap.add_argument("--dequantize", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    raw = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    # F8: rejected 质量校验 — 空或与 chosen 相同的 pair 不进入
    data = [r for r in raw
            if r.get("rejected") and r["rejected"] != r.get("chosen", "")
            and r.get("messages")]
    print(f"Loaded {len(raw)} DPO records, {len(data)} valid pairs", flush=True)
    if not data:
        print("NO VALID DPO PAIRS -> BLOCKED.", flush=True)
        sys.exit(2)

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

    base_model = _load_base(args.dequantize) or _load_base(False)
    try:
        base_model.config.use_cache = False
    except Exception:
        pass
    base_model = prepare_model_for_kbit_training(base_model)

    target_mods = ([m.strip() for m in args.target_modules.split(",") if m.strip()]
                   if args.target_modules
                   else [".*language_model.*q_proj", ".*language_model.*v_proj"])

    lora = LoraConfig(r=args.r, lora_alpha=args.alpha, lora_dropout=args.dropout,
                      bias="none", task_type="CAUSAL_LM", target_modules=target_mods)
    policy = get_peft_model(base_model, lora)
    policy = PeftModel.from_pretrained(base_model, args.sft_adapter, is_trainable=True)
    matched = sorted({n for n, _ in policy.named_parameters() if "lora_" in n.lower()})
    print(f"[LORA AUDIT] matched lora tensors: {len(matched)}", flush=True)
    if not matched:
        print("LORA AUDIT FAIL: zero matched modules -> BLOCKED.", flush=True)
        sys.exit(5)
    policy.gradient_checkpointing_enable()
    policy.enable_input_require_grads()
    policy.print_trainable_parameters()

    ref_base = _load_base(args.dequantize) or _load_base(False)
    try:
        ref_base.config.use_cache = False
    except Exception:
        pass
    ref_base = prepare_model_for_kbit_training(ref_base)
    ref_model = PeftModel.from_pretrained(ref_base, args.sft_adapter, is_trainable=False)
    ref_model.eval()

    ds = DPODataset(data, tok, args.max_len)
    print(f"DPODataset items={len(ds)} skipped={ds.skipped} "
          f"prefix_mismatches={ds.prefix_mismatch_count}", flush=True)
    if len(ds) == 0:
        print("NO DATASET ITEMS -> BLOCKED.", flush=True)
        sys.exit(2)

    n_val = int(len(ds) * args.val_ratio) if args.val_ratio > 0 else 0
    train_items = ds.items[n_val:]
    val_items = ds.items[:n_val]

    def _subset(items):
        sub = DPODataset.__new__(DPODataset)
        sub.tok, sub.max_len = tok, args.max_len
        sub.items = items
        sub.prefix_mismatch_count = 0
        sub.skipped = 0
        return sub

    loader_train = DataLoader(_subset(train_items), batch_size=args.batch, shuffle=True,
                              collate_fn=lambda b: collate_dpo(b, tok.pad_token_id),
                              drop_last=False)
    loader_val = (DataLoader(_subset(val_items), batch_size=args.batch, shuffle=False,
                             collate_fn=lambda b: collate_dpo(b, tok.pad_token_id),
                             drop_last=False) if val_items else None)

    device = policy.device
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr)

    os.makedirs(args.out, exist_ok=True)

    if args.preflight:
        ok = run_preflight(policy, ref_model, loader_train, device, tok, args.out, opt)
        sys.exit(0 if ok else 6)

    total_steps = max(1, (len(loader_train) + args.grad_steps - 1)
                      // args.grad_steps * args.epochs)
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=0,
                                            num_training_steps=total_steps)
    best_val, best_step, patience = float("inf"), 0, 0
    loss_hist, val_hist = [], []
    policy.train()
    step = 0

    for ep in range(args.epochs):
        micro, pending = 0, False
        for bi, b in enumerate(loader_train):
            loss = dpo_loss_for_batch(policy, ref_model, b, device, args.beta)
            (loss / args.grad_steps).backward()
            micro += 1
            pending = True
            if micro == args.grad_steps:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1
                loss_hist.append(float(loss.item()))
                print(f"epoch={ep+1} step={step} dpo_loss={loss_hist[-1]:.4f}", flush=True)
                micro, pending = 0, False

        if pending and micro > 0:   # F8: 尾窗补偿
            scale = args.grad_steps / micro
            for p in policy.parameters():
                if p.grad is not None:
                    p.grad.mul_(scale)
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            pending = False
            print(f"epoch={ep+1} step={step} (tail window scaled x{scale:.2f})", flush=True)

        if loader_val is not None:
            val = evaluate_dpo(policy, ref_model, loader_val, device, args.beta)
            val_hist.append(val)
            print(f"epoch={ep+1} val_dpo_loss={val:.4f}", flush=True)
            if val < best_val:
                best_val, best_step, patience = val, step, 0
                policy.save_pretrained(str(Path(args.out) / "best"))
                tok.save_pretrained(str(Path(args.out) / "best"))
            else:
                patience += 1
                if args.early_stopping_patience > 0 and patience >= args.early_stopping_patience:
                    print(f"Early stopping at epoch={ep+1}", flush=True)
                    break

    policy.save_pretrained(args.out)
    tok.save_pretrained(args.out)

    meta = {
        "base_model": args.model,
        "sft_adapter": args.sft_adapter,
        "data_file": args.data,
        "data_sha256": sha256_file(Path(args.data)),
        "seed": args.seed,
        "r": args.r, "alpha": args.alpha, "dropout": args.dropout,
        "lr": args.lr, "epochs": args.epochs, "max_len": args.max_len,
        "beta": args.beta,
        "n_samples": len(data), "n_train": len(train_items), "n_val": len(val_items),
        "final_steps": step,
        "loss_curve": loss_hist, "val_loss_curve": val_hist,
        "best_val_loss": best_val, "best_step": best_step,
        "prefix_mismatches": ds.prefix_mismatch_count,
        "target_modules": target_mods,
        "stage": "dpo_verified_patch_f8",
    }
    with open(os.path.join(args.out, "training_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    adapt = glob.glob(os.path.join(args.out, "adapter_model.safetensors"))
    if not adapt:
        print("WARNING: no adapter_model.safetensors produced -> GATE FAIL", flush=True)
        sys.exit(3)
    print(f"Saved DPO adapter + meta to {args.out}", flush=True)
    print(f"Adapter bytes: {os.path.getsize(adapt[0])}", flush=True)


if __name__ == "__main__":
    main()
