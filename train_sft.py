#!/usr/bin/env python3
"""
Stage 1: Multi-task Verified Repair SFT for Gemma 4 31B.

F7 fixes (audit):
- Loss mask: full-conversation chat template applied ONCE; assistant span located
  via ``return_assistant_tokens_mask`` when supported, else prefix-boundary method
  with strict prefix verification (per-message concatenation removed).
- ``--dump_token_audit N``: writes token-level decode/label alignment for the first
  N samples so the mask can be manually verified BEFORE training starts.
- Gradient accumulation: tail batches are stepped with correct per-window scaling.
- LoRA target module audit: prints matched module count + trainable params; blocks
  training on zero matches (regex risk).
- Multi-task balanced sampling with six-task ratio control; val loss / best
  checkpoint / early stopping.

Usage:
  python train_sft.py \
    --data training/data/sft_verified_patches.jsonl \
    --out training/adapters/patch_sft_v1 \
    --model google/gemma-4-31B-it-qat-w4a16-ct \
    --hf-token $HF_TOKEN \
    --r 16 --alpha 32 --dropout 0.05 --lr 5e-5 \
    --epochs 3 --max_len 8192 --grad_steps 8 --batch 1 \
    --val_ratio 0.1 --early_stopping_patience 3 \
    --dump_token_audit 3 \
    --dequantize
"""

import argparse
import glob
import hashlib
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL_ID = "google/gemma-4-31B-it-qat-w4a16-ct"

TASK_TYPES = [
    "issue_understanding", "bug_localization", "test_reasoning",
    "root_cause_repair_plan", "patch_generation", "test_result_interpretation",
]
DEFAULT_TASK_RATIO = {
    "issue_understanding": 0.15, "bug_localization": 0.10, "test_reasoning": 0.15,
    "root_cause_repair_plan": 0.10, "patch_generation": 0.35,
    "test_result_interpretation": 0.15,
}
IGNORE_INDEX = -100


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
    groups: dict[str, list[dict]] = defaultdict(list)
    for rec in data:
        meta = rec.get("metadata", {})
        tt = meta.get("task_type", "")
        if not tt:
            iid = rec.get("instance_id", "")
            for cand in TASK_TYPES:
                if iid.endswith("_" + cand) or f"_{cand}_" in iid:
                    tt = cand
                    break
            if not tt:
                tt = meta.get("stage", "patch_generation")
        groups[tt].append(rec)
    return dict(groups)


def balanced_sample(groups: dict[str, list[dict]], task_ratio: dict[str, float],
                    n_samples: int, seed: int) -> list[dict]:
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
    if len(sampled) < n_samples:
        all_recs = [r for g in groups.values() for r in g]
        remaining = n_samples - len(sampled)
        sampled.extend(rng.sample(all_recs, min(remaining, len(all_recs))))
    rng.shuffle(sampled)
    return sampled[:n_samples]


# ---------------------------------------------------------------------------
# Tokenization & loss mask（F7 核心）
# ---------------------------------------------------------------------------

def build_example(tok, messages: list[dict], max_len: int) -> tuple[list[int], list[int], dict]:
    """对完整对话一次性应用 chat template，定位 assistant span 生成 loss mask。

    策略（按优先级）：
    1. ``return_assistant_tokens_mask=True``（transformers 支持 且模板声明了
       {% generation %} 标记时）— 权威方法。
    2. prefix-boundary：模板化 system+user+generation_prompt，验证其 token 序列
       是完整序列的真实前缀后，将 assistant span 置于边界之后。
    3. 兜底：对每条 message 单独模板化（保留作最后手段，标记 mask_method）。

    返回 (input_ids, labels, info)。
    """
    info: dict = {}

    # 1) 权威方法：assistant_tokens_mask
    try:
        enc = tok.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False,
            return_dict=True, return_assistant_tokens_mask=True,
        )
        input_ids = enc["input_ids"]
        if input_ids and isinstance(input_ids[0], list):   # 带 batch 维时取第一条
            input_ids = input_ids[0]
        assistant_mask = enc.get("assistant_masks")
        if assistant_mask is not None and any(assistant_mask):
            if assistant_mask and isinstance(assistant_mask[0], list):
                assistant_mask = assistant_mask[0]
            input_ids = list(input_ids)[:max_len]
            labels = [t if m else IGNORE_INDEX
                      for t, m in zip(input_ids, list(assistant_mask)[:max_len])]
            info["mask_method"] = "assistant_tokens_mask"
            return input_ids, labels, info
    except Exception as e:
        info["assistant_mask_error"] = repr(e)[:200]

    # 2) prefix-boundary + 严格前缀校验
    try:
        full_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        prefix_msgs = [m for m in messages if m["role"] != "assistant"]
        prefix_text = tok.apply_chat_template(
            prefix_msgs, tokenize=False, add_generation_prompt=True) if prefix_msgs else ""
        full_ids = tok.encode(full_text, add_special_tokens=False)
        prefix_ids = tok.encode(prefix_text, add_special_tokens=False) if prefix_text else []
        is_prefix = (len(prefix_ids) <= len(full_ids) and
                     full_ids[:len(prefix_ids)] == prefix_ids)
        info["prefix_verified"] = is_prefix
        if is_prefix:
            input_ids = full_ids[:max_len]
            labels = [IGNORE_INDEX] * len(input_ids)
            for i in range(len(prefix_ids), len(input_ids)):
                labels[i] = input_ids[i]
            info["mask_method"] = "prefix_boundary"
            return input_ids, labels, info
        info["prefix_mismatch_note"] = "template output is not token-prefix aligned"
    except Exception as e:
        info["prefix_error"] = repr(e)[:200]

    # 3) 兜底：逐条 message（语义可能受损，仅在 1/2 均失败时使用）
    tokens: list[int] = []
    labels: list[int] = []
    for m in messages:
        text = tok.apply_chat_template([m], tokenize=False, add_generation_prompt=False)
        t = tok.encode(text, add_special_tokens=False)
        tokens.extend(t)
        if m["role"] == "assistant":
            labels.extend(t)
        else:
            labels.extend([IGNORE_INDEX] * len(t))
    info["mask_method"] = "per_message_fallback"
    return tokens[:max_len], labels[:max_len], info


class SFTDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=8192, audit_sink: list | None = None,
                 audit_limit: int = 0):
        self.tok = tokenizer
        self.max_len = max_len
        self.items = []
        for idx, rec in enumerate(data):
            input_ids, labels, info = build_example(tokenizer, rec["messages"], max_len)
            if audit_sink is not None and idx < audit_limit:
                audit_sink.append({
                    "idx": idx,
                    "instance_id": rec.get("instance_id", ""),
                    **info,
                    "n_tokens": len(input_ids),
                    "n_supervised": sum(1 for l in labels if l != IGNORE_INDEX),
                    "supervised_decode_head": tokenizer.decode(
                        [t for t, l in zip(input_ids, labels) if l != IGNORE_INDEX][:120]),
                    "prefix_decode_tail": tokenizer.decode(
                        input_ids[: max(0, len(input_ids) - sum(1 for l in labels if l != IGNORE_INDEX))][-120:]),
                })
            if sum(1 for l in labels if l != IGNORE_INDEX) == 0:
                continue  # 无监督 token 的样本直接丢弃
            self.items.append({"input_ids": input_ids, "labels": labels})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        ids = torch.tensor(it["input_ids"], dtype=torch.long)
        labels = torch.tensor(it["labels"], dtype=torch.long)
        return {"input_ids": ids, "labels": labels, "attention_mask": torch.ones_like(ids)}


def collate(batch, pad_id):
    maxlen = max(b["input_ids"].numel() for b in batch)
    inp = torch.full((len(batch), maxlen), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), maxlen), IGNORE_INDEX, dtype=torch.long)
    attn = torch.zeros((len(batch), maxlen), dtype=torch.long)
    for i, b in enumerate(batch):
        n = b["input_ids"].numel()
        inp[i, :n] = b["input_ids"]
        lab[i, :n] = b["labels"]
        attn[i, :n] = 1
    return {"input_ids": inp, "labels": lab, "attention_mask": attn}


def evaluate(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for b in loader:
            out = model(input_ids=b["input_ids"].to(device),
                        attention_mask=b["attention_mask"].to(device),
                        labels=b["labels"].to(device))
            total += float(out.loss.item())
            n += 1
    model.train()
    return total / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max_len", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grad_steps", type=int, default=8)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--early_stopping_patience", type=int, default=3)
    ap.add_argument("--task_ratio", type=str, default=None)
    ap.add_argument("--target_modules", type=str, default=None,
                    help="逗号分隔的模块名后缀或正则")
    ap.add_argument("--dump_token_audit", type=int, default=0,
                    help="对前 N 个样本导出 token 级 mask 审计文件")
    ap.add_argument("--require_mask_audit_pass", action="store_true",
                    help="prefix 校验失败或使用 fallback 时阻断训练")
    ap.add_argument("--dequantize", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    task_ratio = dict(DEFAULT_TASK_RATIO)
    if args.task_ratio:
        try:
            task_ratio.update(json.loads(args.task_ratio))
        except Exception as e:
            print(f"WARN: cannot parse --task_ratio ({e}); using defaults", flush=True)

    if args.target_modules:
        target_mods = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    else:
        target_mods = [".*language_model.*q_proj", ".*language_model.*v_proj"]

    data = load_sft_data(Path(args.data))
    if not data:
        print("NO TRAINING DATA -> BLOCKED.", flush=True)
        sys.exit(2)

    groups = group_by_task_type(data)
    print(f"Task type distribution: { {k: len(v) for k, v in groups.items()} }", flush=True)

    if not os.path.isdir(args.model) and not args.hf_token:
        print("BLOCKED_NO_BASE_WEIGHTS: no local model dir and no HF_TOKEN.", flush=True)
        sys.exit(2)

    tok = AutoTokenizer.from_pretrained(args.model, token=args.hf_token or None)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # F7: token-level audit（训练前人工核对 mask）
    audit_sink: list = []
    if args.dump_token_audit > 0:
        for i in range(min(args.dump_token_audit, len(data))):
            _ids, _lab, info = build_example(tok, data[i]["messages"], args.max_len)
            supervised = sum(1 for l in _lab if l != IGNORE_INDEX)
            audit_sink.append({
                "idx": i, "instance_id": data[i].get("instance_id", ""), **info,
                "n_tokens": len(_ids), "n_supervised": supervised,
                "supervised_decode_head": tok.decode(
                    [t for t, l in zip(_ids, _lab) if l != IGNORE_INDEX][:120]),
                "prefix_decode_tail": tok.decode(
                    _ids[: max(0, len(_ids) - supervised)][-120:]),
            })
        audit_path = Path(args.out) / "token_audit.jsonl"
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        with open(audit_path, "w", encoding="utf-8") as f:
            for a in audit_sink:
                f.write(json.dumps(a, ensure_ascii=False) + "\n")
        print(f"Token audit written to {audit_path} — review mask alignment before training.",
              flush=True)
        bad = [a for a in audit_sink
               if a.get("mask_method") == "per_message_fallback"
               or a.get("prefix_verified") is False]
        if bad and args.require_mask_audit_pass:
            print(f"MASK AUDIT FAIL on {len(bad)} sample(s) -> BLOCKED.", flush=True)
            sys.exit(4)

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

    lora = LoraConfig(r=args.r, lora_alpha=args.alpha, lora_dropout=args.dropout,
                      bias="none", task_type="CAUSAL_LM", target_modules=target_mods)
    model = get_peft_model(model, lora)

    # F7: LoRA target module 匹配审计 — 零匹配直接阻断
    matched = sorted({n for n, _ in model.named_parameters() if "lora_" in n.lower()})
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f"[LORA AUDIT] matched lora param tensors: {len(matched)}; "
          f"trainable params: {len(trainable)}", flush=True)
    print(f"[LORA AUDIT] target_modules={target_mods}", flush=True)
    print(f"[LORA AUDIT] first matched: {matched[:6]}", flush=True)
    if not matched:
        print("LORA AUDIT FAIL: zero matched modules -> BLOCKED (regex likely wrong).",
              flush=True)
        sys.exit(5)
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # 划分训练/验证集（按 task 分层）
    n_total = len(data)
    n_val = int(n_total * args.val_ratio) if args.val_ratio > 0 else 0
    train_data: list[dict] = []
    val_data: list[dict] = []
    if n_val > 0:
        for tt in TASK_TYPES:
            g = groups.get(tt, [])
            k = min(len(g), max(1, int(len(g) * args.val_ratio)))
            val_data.extend(g[:k])
            train_data.extend(g[k:])
        if len(val_data) < n_val and len(train_data) > n_val:
            extra = n_val - len(val_data)
            val_data.extend(train_data[-extra:])
            train_data = train_data[:-extra]
    else:
        train_data = data

    train_groups = group_by_task_type(train_data)
    train_data = balanced_sample(train_groups, task_ratio, len(train_data), args.seed)
    print(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}", flush=True)

    ds_train = SFTDataset(train_data, tok, args.max_len)
    loader_train = DataLoader(ds_train, batch_size=args.batch, shuffle=True,
                              collate_fn=lambda b: collate(b, tok.pad_token_id),
                              drop_last=False)
    loader_val = None
    if val_data:
        ds_val = SFTDataset(val_data, tok, args.max_len)
        loader_val = DataLoader(ds_val, batch_size=args.batch, shuffle=False,
                                collate_fn=lambda b: collate(b, tok.pad_token_id),
                                drop_last=False)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = max(1, (len(loader_train) + args.grad_steps - 1)
                      // args.grad_steps * args.epochs)
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=0,
                                            num_training_steps=total_steps)

    os.makedirs(args.out, exist_ok=True)
    best_val_loss, best_step, patience = float("inf"), 0, 0
    loss_hist, val_loss_hist = [], []
    model.train()
    step = 0

    for ep in range(args.epochs):
        micro_count = 0        # 当前 accumulation 窗口内的 micro-batch 数
        pending = False

        for b in loader_train:
            out = model(input_ids=b["input_ids"].to(model.device),
                        attention_mask=b["attention_mask"].to(model.device),
                        labels=b["labels"].to(model.device))
            micro_count += 1
            pending = True
            # F7: 按当前窗口实际 micro-batch 数缩放（含尾批）
            is_window_end = (micro_count == args.grad_steps)
            if is_window_end:
                loss = out.loss / args.grad_steps
            else:
                loss = out.loss / args.grad_steps  # 先按满窗缩放，尾窗再补偿
            loss.backward()

            if is_window_end:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1
                loss_hist.append(float(out.loss.item()))
                print(f"epoch={ep+1} step={step} loss={loss_hist[-1]:.4f}", flush=True)
                micro_count = 0
                pending = False

        # F7: 尾窗补偿 — 实际 micro 数为 micro_count，重新按真实比例补一次 step
        if pending and micro_count > 0:
            # 梯度已按 1/grad_steps 缩放；等效于把尾窗 mean 再乘 (grad_steps/micro_count)
            # 直接对现有累积梯度按比例放大后 step，保持与满窗一致的量纲
            scale = args.grad_steps / micro_count
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.mul_(scale)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            pending = False
            print(f"epoch={ep+1} step={step} (tail window scaled x{scale:.2f})", flush=True)

        if loader_val is not None:
            val_loss = evaluate(model, loader_val, model.device)
            val_loss_hist.append(val_loss)
            print(f"epoch={ep+1} val_loss={val_loss:.4f}", flush=True)
            if val_loss < best_val_loss:
                best_val_loss, best_step, patience = val_loss, step, 0
                model.save_pretrained(str(Path(args.out) / "best"))
                tok.save_pretrained(str(Path(args.out) / "best"))
                print(f"  New best val_loss={val_loss:.4f} -> best/", flush=True)
            else:
                patience += 1
                if args.early_stopping_patience > 0 and patience >= args.early_stopping_patience:
                    print(f"Early stopping at epoch={ep+1}", flush=True)
                    break

    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)

    meta = {
        "base_model": args.model,
        "data_file": args.data,
        "data_sha256": sha256_file(Path(args.data)),
        "seed": args.seed,
        "r": args.r, "alpha": args.alpha, "dropout": args.dropout,
        "lr": args.lr, "epochs": args.epochs, "max_len": args.max_len,
        "n_samples": len(data), "n_train": len(train_data), "n_val": len(val_data),
        "final_steps": step,
        "loss_curve": loss_hist, "val_loss_curve": val_loss_hist,
        "best_val_loss": best_val_loss, "best_step": best_step,
        "task_ratio": task_ratio, "target_modules": target_mods,
        "lora_matched_tensors": len(matched),
        "stage": "sft_verified_patch_multi_task_f7",
    }
    with open(os.path.join(args.out, "training_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    adapt = glob.glob(os.path.join(args.out, "adapter_model.safetensors"))
    if not adapt:
        print("WARNING: no adapter_model.safetensors produced -> GATE FAIL", flush=True)
        sys.exit(3)
    print(f"Saved adapter + meta to {args.out}", flush=True)
    print(f"Adapter bytes: {os.path.getsize(adapt[0])}", flush=True)


if __name__ == "__main__":
    main()
