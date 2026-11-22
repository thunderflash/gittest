#!/usr/bin/env python3
"""
Stage 1: Verified Patch SFT training for Gemma 4 31B.

P1 fixes:
- Multi-task SFT with task/issue-balanced sampling and six-task ratio control.
- Unified messages formatter for consistent chat template.
- LoRA target modules configurable with validation.
- Gradient accumulation: tail batches are not dropped (remaining grads stepped).
- Validation loss tracking, best checkpoint saving, early stopping.

Trains a LoRA adapter on true Issue→Patch pairs. Only the assistant's patch
content is supervised (system/user tokens are masked with -100).

Usage:
  python train_sft.py \
    --data training/data/sft_verified_patches.jsonl \
    --out training/adapters/patch_sft_v1 \
    --model google/gemma-4-31B-it-qat-w4a16-ct \
    --hf-token $HF_TOKEN \
    --r 16 --alpha 32 --dropout 0.05 --lr 5e-5 \
    --epochs 3 --max_len 8192 --grad_steps 8 --batch 1 \
    --val_ratio 0.1 --early_stopping_patience 3
"""

import argparse
import json
import os
import random
import hashlib
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL_ID = "google/gemma-4-31B-it-qat-w4a16-ct"

# 六种任务类型及其默认采样比例（和为 1.0）
TASK_TYPES = [
    "issue_understanding",
    "bug_localization",
    "test_reasoning",
    "root_cause_repair_plan",
    "patch_generation",
    "test_result_interpretation",
]
DEFAULT_TASK_RATIO = {
    "issue_understanding": 0.15,
    "bug_localization": 0.10,
    "test_reasoning": 0.15,
    "root_cause_repair_plan": 0.10,
    "patch_generation": 0.35,
    "test_result_interpretation": 0.15,
}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_sft_data(data_path: Path) -> list[dict]:
    data = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def group_by_task_type(data: list[dict]) -> dict[str, list[dict]]:
    """按 task_type 分组数据，用于多任务平衡采样。"""
    groups: dict[str, list[dict]] = defaultdict(list)
    for rec in data:
        meta = rec.get("metadata", {})
        # 从 metadata 或 instance_id 推断 task_type
        tt = meta.get("task_type", "")
        if not tt:
            iid = rec.get("instance_id", "")
            if "_patch_generation" in iid:
                tt = "patch_generation"
            elif "_test_reasoning" in iid:
                tt = "test_reasoning"
            elif "_issue_understanding" in iid:
                tt = "issue_understanding"
            elif "_bug_localization" in iid:
                tt = "bug_localization"
            elif "_root_cause_repair_plan" in iid:
                tt = "root_cause_repair_plan"
            elif "_test_result_interpretation" in iid:
                tt = "test_result_interpretation"
        groups[tt].append(rec)
    return dict(groups)


def balanced_sample(groups: dict[str, list[dict]],
                    task_ratio: dict[str, float],
                    n_samples: int,
                    seed: int) -> list[dict]:
    """按 task_ratio 从各组按比例采样，实现多任务平衡。"""
    rng = random.Random(seed)
    sampled: list[dict] = []
    for tt in TASK_TYPES:
        group = groups.get(tt, [])
        if not group:
            continue
        ratio = task_ratio.get(tt, 0.0)
        n_tt = max(1, int(n_samples * ratio)) if ratio > 0 else 0
        n_tt = min(n_tt, len(group))
        if n_tt > 0:
            sampled.extend(rng.sample(group, n_tt))
    # 如果总数不足 n_samples，从所有数据中随机补齐
    if len(sampled) < n_samples:
        all_recs = [r for g in groups.values() for r in g]
        remaining = n_samples - len(sampled)
        sampled.extend(rng.sample(all_recs, min(remaining, len(all_recs))))
    rng.shuffle(sampled)
    return sampled[:n_samples]


class SFTDataset(Dataset):
    """每条样本：messages 列表，仅 assistant token 有 loss（system/user 用 -100 掩码）。"""

    def __init__(self, data, tokenizer, max_len=8192):
        self.data = data
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        msgs = self.data[idx]["messages"]
        # 使用统一的 chat template 拼接
        text = self.tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False
        )
        # 分别编码以便对 system/user + assistant 做 loss 掩码
        full_ids = self.tok.encode(text, add_special_tokens=False)
        # 找出 assistant 部分的起始位置：重新编码仅 system+user
        prefix_msgs = [m for m in msgs if m["role"] != "assistant"]
        prefix_text = self.tok.apply_chat_template(
            prefix_msgs, tokenize=False, add_generation_prompt=True
        ) if prefix_msgs else ""
        prefix_ids = self.tok.encode(prefix_text, add_special_tokens=False) if prefix_text else []
        prompt_len = len(prefix_ids)

        tokens = full_ids[: self.max_len]
        labels = [-100] * len(tokens)
        # 仅 assistant 部分（prompt_len 之后）计算 loss
        for i in range(prompt_len, len(tokens)):
            labels[i] = tokens[i]

        input_ids = torch.tensor(tokens, dtype=torch.long)
        labels_t = torch.tensor(labels, dtype=torch.long)
        attn = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "labels": labels_t, "attention_mask": attn}


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


def evaluate(model, loader, device, pad_id):
    """在验证集上计算平均 loss。"""
    model.eval()
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for b in loader:
            out = model(
                input_ids=b["input_ids"].to(device),
                attention_mask=b["attention_mask"].to(device),
                labels=b["labels"].to(device),
            )
            total_loss += float(out.loss.item())
            total_batches += 1
    model.train()
    return total_loss / max(total_batches, 1)


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
    ap.add_argument("--val_ratio", type=float, default=0.1,
                    help="验证集比例（0 表示不划分验证集）")
    ap.add_argument("--early_stopping_patience", type=int, default=3,
                    help="验证 loss 不下降多少个 epoch 后早停（0 表示不早停）")
    ap.add_argument("--task_ratio", type=str, default=None,
                    help="JSON 字符串，六种任务的采样比例，如 '{\"patch_generation\":0.5,...}'")
    ap.add_argument("--target_modules", type=str, default=None,
                    help="LoRA target modules 正则，逗号分隔，如 'q_proj,v_proj'")
    ap.add_argument("--dequantize", action="store_true",
                    help="load ct base and dequantize to bf16 for training")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # 解析 task_ratio
    task_ratio = dict(DEFAULT_TASK_RATIO)
    if args.task_ratio:
        try:
            task_ratio.update(json.loads(args.task_ratio))
        except Exception as e:
            print(f"WARN: 无法解析 --task_ratio: {e}，使用默认比例", flush=True)

    # 解析 target_modules（P1: 可配置，避免正则风险）
    if args.target_modules:
        target_mods = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    else:
        target_mods=[".*language_model.*q_proj", ".*language_model.*v_proj"]

    data = load_sft_data(Path(args.data))
    if not data:
        print("NO TRAINING DATA -> BLOCKED. refusing to emit empty adapter.", flush=True)
        sys.exit(2)

    # 多任务分组
    groups = group_by_task_type(data)
    print(f"Task type distribution: { {k: len(v) for k, v in groups.items()} }", flush=True)

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

    # P1: LoRA target modules 可配置并做基本校验
    lora = LoraConfig(
        r=args.r,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_mods,
    )
    model = get_peft_model(model, lora)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    # 划分训练/验证集
    n_total = len(data)
    n_val = int(n_total * args.val_ratio) if args.val_ratio > 0 else 0
    n_train = n_total - n_val

    if n_val > 0:
        # 按 task_type 分层划分，保持比例
        train_data: list[dict] = []
        val_data: list[dict] = []
        for tt in TASK_TYPES:
            group = groups.get(tt, [])
            n_tt_val = max(1, int(len(group) * args.val_ratio)) if args.val_ratio > 0 else 0
            n_tt_val = min(n_tt_val, len(group))
            val_data.extend(group[:n_tt_val])
            train_data.extend(group[n_tt_val:])
        # 如果某些 task 无 val 样本，从 train 中移一些
        if len(val_data) < n_val and n_train > n_val:
            extra = n_val - len(val_data)
            train_data, extra_val = train_data[:-extra], train_data[-extra:]
            val_data.extend(extra_val)
    else:
        train_data = data
        val_data = []

    print(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}", flush=True)

    # P1: 多任务平衡采样构建训练 dataset
    if train_data:
        train_groups = group_by_task_type(train_data)
        train_data = balanced_sample(train_groups, task_ratio, len(train_data), args.seed)

    ds_train = SFTDataset(train_data, tok, args.max_len)
    loader_train = DataLoader(
        ds_train, batch_size=args.batch, shuffle=True,
        collate_fn=lambda b: collate(b, tok.pad_token_id),
        drop_last=False,  # P1: 不丢弃尾 batch
    )

    ds_val = None
    loader_val = None
    if val_data:
        ds_val = SFTDataset(val_data, tok, args.max_len)
        loader_val = DataLoader(
            ds_val, batch_size=args.batch, shuffle=False,
            collate_fn=lambda b: collate(b, tok.pad_token_id),
            drop_last=False,
        )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = max(1, len(loader_train) * args.epochs // args.grad_steps)
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=0,
                                            num_training_steps=total_steps)

    os.makedirs(args.out, exist_ok=True)
    best_val_loss = float("inf")
    best_step = 0
    patience_counter = 0
    loss_hist = []
    val_loss_hist = []
    model.train()
    step = 0

    for ep in range(args.epochs):
        epoch_loss = 0.0
        epoch_batches = 0
        pending_grads = False  # P1: 尾 batch 未踩到 grad_steps 时也应 step

        for bi, b in enumerate(loader_train):
            out = model(
                input_ids=b["input_ids"].to(model.device),
                attention_mask=b["attention_mask"].to(model.device),
                labels=b["labels"].to(model.device),
            )
            loss = out.loss / args.grad_steps
            loss.backward()
            pending_grads = True

            if (bi + 1) % args.grad_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1
                loss_hist.append(float(loss.item() * args.grad_steps))
                epoch_loss += loss_hist[-1]
                epoch_batches += 1
                pending_grads = False
                print(f"epoch={ep+1} step={step} loss={loss_hist[-1]:.4f}", flush=True)

        # P1: 处理 epoch 尾部的累积梯度
        if pending_grads:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            pending_grads = False

        # 验证
        if loader_val is not None:
            val_loss = evaluate(model, loader_val, model.device, tok.pad_token_id)
            val_loss_hist.append(val_loss)
            print(f"epoch={ep+1} val_loss={val_loss:.4f}", flush=True)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_step = step
                patience_counter = 0
                # 保存最佳 checkpoint
                best_dir = Path(args.out) / "best"
                model.save_pretrained(str(best_dir))
                tok.save_pretrained(str(best_dir))
                print(f"  New best val_loss={val_loss:.4f} saved to {best_dir}", flush=True)
            else:
                patience_counter += 1
                if args.early_stopping_patience > 0 and patience_counter >= args.early_stopping_patience:
                    print(f"Early stopping at epoch={ep+1} (patience {patience_counter})", flush=True)
                    break

    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)

    data_sha = sha256_file(Path(args.data))
    meta = {
        "base_model": args.model,
        "data_file": args.data,
        "data_sha256": data_sha,
        "seed": args.seed,
        "r": args.r, "alpha": args.alpha, "dropout": args.dropout,
        "lr": args.lr, "epochs": args.epochs, "max_len": args.max_len,
        "n_samples": len(data),
        "n_train": len(train_data),
        "n_val": len(val_data),
        "final_steps": step,
        "loss_curve": loss_hist,
        "val_loss_curve": val_loss_hist,
        "best_val_loss": best_val_loss,
        "best_step": best_step,
        "task_ratio": task_ratio,
        "target_modules": target_mods,
        "stage": "sft_verified_patch_multi_task",
    }
    with open(os.path.join(args.out, "training_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    import glob
    adapt = glob.glob(os.path.join(args.out, "adapter_model.safetensors"))
    if not adapt:
        print("WARNING: no adapter_model.safetensors produced -> GATE FAIL", flush=True)
        sys.exit(3)
    print(f"Saved adapter + meta to {args.out}", flush=True)
    print(f"Adapter bytes: {os.path.getsize(adapt[0])}", flush=True)


if __name__ == "__main__":
    main()