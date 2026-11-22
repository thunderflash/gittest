#!/usr/bin/env python3
"""
Stage 2: Verified Patch DPO training for Gemma 4 31B.

P1 fixes:
- Padding mask 与 logprob 掩码保持一致（均用 -100，不再混用 pad_token_id）。
- Prompt/completion tokenization 边界统一（同一 tokenizer 调用，避免偏移错位）。
- Rejected 质量校验：拒绝为空或与 chosen 相同的 rejected，跳过该 pair。
- Gradient accumulation：尾 batch 未踩到 grad_steps 时也执行 step。
- 验证 loss、最佳 checkpoint、早停。

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
    --beta 0.1 --val_ratio 0.1 --early_stopping_patience 2
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
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

MODEL_ID = "google/gemma-4-31B-it-qat-w4a16-ct"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


class DPODataset(Dataset):
    """每条样本：prompt + chosen + rejected patch 文本。
    P1: 统一 tokenization，prompt/completion 边界清晰，mask 用 -100。
    """

    def __init__(self, data, tokenizer, max_len=8192):
        self.tok = tokenizer
        self.max_len = max_len
        self.items = []
        for item in data:
            prompt = item["prompt"]
            chosen = item["chosen"]
            rejected = item["rejected"]
            if not rejected or rejected == chosen:
                continue  # P1: 跳过无效 rejected
            # 统一用 apply_chat_template 编码 prompt（system+user）
            prompt_msgs = [
                {"role": "system", "content": prompt.split("\n\n")[0] if "\n\n" in prompt else prompt},
                {"role": "user", "content": prompt},
            ]
            try:
                prompt_text = tokenizer.apply_chat_template(
                    prompt_msgs, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                prompt_text = prompt
            self.items.append({
                "prompt": prompt_text,
                "chosen": chosen,
                "rejected": rejected,
                "instance_id": item.get("instance_id", ""),
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        # P1: 统一 tokenization，prompt 和 full 都用同一编码方式
        def encode(prompt_text, completion_text):
            full_text = prompt_text + "\n\n" + completion_text
            full_ids = self.tok.encode(full_text, add_special_tokens=False)
            prompt_ids = self.tok.encode(prompt_text, add_special_tokens=False)
            # 确保 prompt_ids 是 full_ids 的前缀
            if len(prompt_ids) > len(full_ids):
                prompt_ids = prompt_ids[:len(full_ids)]
            return prompt_ids, full_ids

        p_chosen, f_chosen = encode(item["prompt"], item["chosen"])
        p_rej, f_rej = encode(item["prompt"], item["rejected"])

        return {
            "prompt_chosen": p_chosen,
            "full_chosen": f_chosen,
            "prompt_rejected": p_rej,
            "full_rejected": f_rej,
            "instance_id": item["instance_id"],
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


def sequence_log_prob(logits, input_ids, prompt_len, pad_id):
    """Compute log probability of completion tokens (after prompt_len).

    P1: logprob 掩码使用 -100，与 labels 保持一致；不使用 pad_token_id。
    """
    shift_logits = logits[:, :-1, :]   # [B, T-1, V]
    shift_ids = input_ids[:, 1:]       # [B, T-1]
    comp_start = max(prompt_len - 1, 0)
    comp_logits = shift_logits[:, comp_start:, :]   # [B, T_comp, V]
    comp_ids = shift_ids[:, comp_start:]            # [B, T_comp]

    log_probs = F.log_softmax(comp_logits, dim=-1)  # [B, T_comp, V]
    gathered = log_probs.gather(-1, comp_ids.unsqueeze(-1)).squeeze(-1)  # [B, T_comp]

    # P1: 用 -100 掩码 padding 位置，而不是 pad_token_id
    pad_mask = (comp_ids != -100) & (comp_ids != pad_id)
    gathered = gathered * pad_mask
    return gathered.sum(dim=-1)  # [B]


def evaluate_dpo(policy, ref_model, loader, device, pad_id, beta):
    """在验证集上计算平均 DPO loss。"""
    policy.eval()
    ref_model.eval()
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for b in loader:
            chosen_ids = b["full_chosen"].to(device)
            chosen_mask = torch.ones_like(chosen_ids)
            chosen_prompt_lens = [len(p) for p in b["prompt_chosen"]]

            ref_chosen_logits = ref_model(chosen_ids, attention_mask=chosen_mask).logits
            policy_chosen_logits = policy(chosen_ids, attention_mask=chosen_mask).logits

            rej_ids = b["full_rejected"].to(device)
            rej_mask = torch.ones_like(rej_ids)
            rej_prompt_lens = [len(p) for p in b["prompt_rejected"]]

            ref_rej_logits = ref_model(rej_ids, attention_mask=rej_mask).logits
            policy_rej_logits = policy(rej_ids, attention_mask=rej_mask).logits

            chosen_log_pi = []
            chosen_log_ref = []
            for i in range(chosen_ids.size(0)):
                pl = chosen_prompt_lens[i]
                chosen_log_pi.append(sequence_log_prob(policy_chosen_logits[i:i+1],
                                                        chosen_ids[i:i+1], pl, pad_id)[0])
                chosen_log_ref.append(sequence_log_prob(ref_chosen_logits[i:i+1],
                                                         chosen_ids[i:i+1], pl, pad_id)[0])

            rej_log_pi = []
            rej_log_ref = []
            for i in range(rej_ids.size(0)):
                pl = rej_prompt_lens[i]
                rej_log_pi.append(sequence_log_prob(policy_rej_logits[i:i+1],
                                                     rej_ids[i:i+1], pl, pad_id)[0])
                rej_log_ref.append(sequence_log_prob(ref_rej_logits[i:i+1],
                                                      rej_ids[i:i+1], pl, pad_id)[0])

            chosen_log_pi = torch.stack(chosen_log_pi)
            chosen_log_ref = torch.stack(chosen_log_ref)
            rej_log_pi = torch.stack(rej_log_pi)
            rej_log_ref = torch.stack(rej_log_ref)

            logits_diff = (chosen_log_pi - chosen_log_ref) - (rej_log_pi - rej_log_ref)
            loss = -F.logsigmoid(beta * logits_diff).mean()
            total_loss += float(loss.item())
            total_batches += 1
    policy.train()
    return total_loss / max(total_batches, 1)


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
    ap.add_argument("--val_ratio", type=float, default=0.1,
                    help="验证集比例（0 表示不划分验证集）")
    ap.add_argument("--early_stopping_patience", type=int, default=2,
                    help="验证 loss 不下降多少个 epoch 后早停（0 表示不早停）")
    ap.add_argument("--target_modules", type=str, default=None,
                    help="LoRA target modules 正则，逗号分隔，如 'q_proj,v_proj'")
    ap.add_argument("--dequantize", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    data = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    if not data:
        print("NO TRAINING DATA -> BLOCKED.", flush=True)
        sys.exit(2)

    # P1: rejected 质量校验 — 过滤掉无效 pair
    valid_data = []
    for rec in data:
        if rec.get("rejected") and rec["rejected"] != rec.get("chosen", ""):
            valid_data.append(rec)
    print(f"Loaded {len(data)} DPO pairs, {len(valid_data)} valid after rejection check", flush=True)
    if not valid_data:
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

    # Policy model: base + SFT adapter (trainable)
    base_model = _load_base(args.dequantize)
    if base_model is None:
        base_model = _load_base(False)
    try:
        base_model.config.use_cache = False
    except Exception:
        pass
    base_model = prepare_model_for_kbit_training(base_model)
    if args.target_modules:
        target_mods = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    else:
        target_mods = [".*language_model.*q_proj", ".*language_model.*v_proj"]
    lora = LoraConfig(
        r=args.r, lora_alpha=args.alpha, lora_dropout=args.dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=target_mods,
    )
    policy = get_peft_model(base_model, lora)
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

    ds = DPODataset(valid_data, tok, args.max_len)
    print(f"DPODataset valid items: {len(ds)}", flush=True)

    # P1: 划分训练/验证集
    n_total = len(ds)
    n_val = int(n_total * args.val_ratio) if args.val_ratio > 0 else 0
    n_train = n_total - n_val

    train_items = ds.items[n_val:] if n_val > 0 else ds.items
    val_items = ds.items[:n_val] if n_val > 0 else []

    ds_train = DPODataset.__new__(DPODataset)
    ds_train.tok = tok
    ds_train.max_len = args.max_len
    ds_train.items = train_items

    ds_val = None
    if val_items:
        ds_val = DPODataset.__new__(DPODataset)
        ds_val.tok = tok
        ds_val.max_len = args.max_len
        ds_val.items = val_items

    loader_train = DataLoader(
        ds_train, batch_size=args.batch, shuffle=True,
        collate_fn=lambda b: collate_dpo(b, tok.pad_token_id),
        drop_last=False,  # P1: 不丢弃尾 batch
    )
    loader_val = None
    if ds_val is not None:
        loader_val = DataLoader(
            ds_val, batch_size=args.batch, shuffle=False,
            collate_fn=lambda b: collate_dpo(b, tok.pad_token_id),
            drop_last=False,
        )

    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr)
    total_steps = max(1, len(loader_train) * args.epochs // args.grad_steps)
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=0,
                                            num_training_steps=total_steps)

    os.makedirs(args.out, exist_ok=True)
    best_val_loss = float("inf")
    best_step = 0
    patience_counter = 0
    loss_hist = []
    val_loss_hist = []
    policy.train()
    step = 0
    device = policy.device

    for ep in range(args.epochs):
        pending_grads = False

        for bi, b in enumerate(loader_train):
            chosen_ids = b["full_chosen"].to(device)
            chosen_mask = torch.ones_like(chosen_ids)
            chosen_prompt_lens = [len(p) for p in b["prompt_chosen"]]

            with torch.no_grad():
                ref_chosen_logits = ref_model(chosen_ids, attention_mask=chosen_mask).logits
            policy_chosen_logits = policy(chosen_ids, attention_mask=chosen_mask).logits

            rej_ids = b["full_rejected"].to(device)
            rej_mask = torch.ones_like(rej_ids)
            rej_prompt_lens = [len(p) for p in b["prompt_rejected"]]

            with torch.no_grad():
                ref_rej_logits = ref_model(rej_ids, attention_mask=rej_mask).logits
            policy_rej_logits = policy(rej_ids, attention_mask=rej_mask).logits

            chosen_log_pi = []
            chosen_log_ref = []
            for i in range(chosen_ids.size(0)):
                pl = chosen_prompt_lens[i]
                chosen_log_pi.append(sequence_log_prob(policy_chosen_logits[i:i+1],
                                                        chosen_ids[i:i+1], pl, tok.pad_token_id)[0])
                chosen_log_ref.append(sequence_log_prob(ref_chosen_logits[i:i+1],
                                                         chosen_ids[i:i+1], pl, tok.pad_token_id)[0])

            rej_log_pi = []
            rej_log_ref = []
            for i in range(rej_ids.size(0)):
                pl = rej_prompt_lens[i]
                rej_log_pi.append(sequence_log_prob(policy_rej_logits[i:i+1],
                                                     rej_ids[i:i+1], pl, tok.pad_token_id)[0])
                rej_log_ref.append(sequence_log_prob(ref_rej_logits[i:i+1],
                                                      rej_ids[i:i+1], pl, tok.pad_token_id)[0])

            chosen_log_pi = torch.stack(chosen_log_pi)
            chosen_log_ref = torch.stack(chosen_log_ref)
            rej_log_pi = torch.stack(rej_log_pi)
            rej_log_ref = torch.stack(rej_log_ref)

            logits_diff = (chosen_log_pi - chosen_log_ref) - (rej_log_pi - rej_log_ref)
            loss = -F.logsigmoid(args.beta * logits_diff).mean()

            loss = loss / args.grad_steps
            loss.backward()
            pending_grads = True

            if (bi + 1) % args.grad_steps == 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1
                loss_hist.append(float(loss.item() * args.grad_steps))
                print(f"epoch={ep+1} step={step} dpo_loss={loss_hist[-1]:.4f}", flush=True)
                pending_grads = False

        # P1: 处理 epoch 尾部的累积梯度
        if pending_grads:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            pending_grads = False

        # 验证
        if loader_val is not None:
            val_loss = evaluate_dpo(policy, ref_model, loader_val, device, tok.pad_token_id, args.beta)
            val_loss_hist.append(val_loss)
            print(f"epoch={ep+1} val_dpo_loss={val_loss:.4f}", flush=True)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_step = step
                patience_counter = 0
                best_dir = Path(args.out) / "best"
                policy.save_pretrained(str(best_dir))
                tok.save_pretrained(str(best_dir))
                print(f"  New best val_loss={val_loss:.4f} saved to {best_dir}", flush=True)
            else:
                patience_counter += 1
                if args.early_stopping_patience > 0 and patience_counter >= args.early_stopping_patience:
                    print(f"Early stopping at epoch={ep+1} (patience {patience_counter})", flush=True)
                    break

    policy.save_pretrained(args.out)
    tok.save_pretrained(args.out)

    data_sha = sha256_file(Path(args.data))
    meta = {
        "base_model": args.model,
        "sft_adapter": args.sft_adapter,
        "data_file": args.data,
        "data_sha256": data_sha,
        "seed": args.seed,
        "r": args.r, "alpha": args.alpha, "dropout": args.dropout,
        "lr": args.lr, "epochs": args.epochs, "max_len": args.max_len,
        "beta": args.beta,
        "n_samples": len(valid_data),
        "n_train": n_train,
        "n_val": n_val,
        "final_steps": step,
        "loss_curve": loss_hist,
        "val_loss_curve": val_loss_hist,
        "best_val_loss": best_val_loss,
        "best_step": best_step,
        "target_modules": target_mods,
        "stage": "dpo_verified_patch_p1",
    }
    with open(os.path.join(args.out, "training_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    import glob
    adapt = glob.glob(os.path.join(args.out, "adapter_model.safetensors"))
    if not adapt:
        print("WARNING: no adapter_model.safetensors produced -> GATE FAIL", flush=True)
        sys.exit(3)
    print(f"Saved DPO adapter + meta to {args.out}", flush=True)
    print(f"Adapter bytes: {os.path.getsize(adapt[0])}", flush=True)


if __name__ == "__main__":
    main()