# Gemma 4 31B Patch Capability Enhancement — Training Pipeline

Multi-stage training pipeline to improve Gemma 4 31B's code patch generation capability.
All training scripts run on cloud GPU with the gated `gemma-4-31b-it-qat-w4a16-ct` QAT model.
Data preparation (S1–S5) runs locally (no GPU needed).

## Directory Layout

```
training/
  data/
    raw/              # S1: 原始数据集导出 (swe_fixer.jsonl, swe_smith_py.jsonl)
    snapshots/        # S1: Repo 克隆快照 (base_commit 状态)
    processed/        # S2–S5: 处理后数据 (contexts, labels, validation, gates)
    sft_verified_patches.jsonl   # Stage 1: gold Issue→Patch pairs
    dpo_verified_patches.jsonl   # Stage 2: chosen/rejected patch pairs
  adapters/
    patch_sft_v1/                 # Stage 1 output (SFT adapter, best/ 子目录)
    patch_sft_dpo_v1/             # Stage 2 output (DPO adapter, best/ 子目录)
  src/
    acquire_datasets.py   # S1: 数据集采集、仓库快照、完整性审计
    context_builder.py    # S2: Repo 快照 + 上下文构建（防泄漏）
    label_builder.py      # S3: 六任务标签构建（带 Label Provenance）
    patch_validator.py    # S4: Patch 多 Gate 验证（git apply、语法、相关性）
    quality_gates.py      # S5: 质量门禁 + 审计报告（split manifest, SHA256）
  build_sft_data.py       # 构建 SFT 训练数据（P0-7: 无 test_diff 泄漏）
  build_dpo_data.py       # 构建 DPO 训练数据（P0-9: 无空 rejected）
  train_sft.py            # Stage 1: 多任务平衡 SFT（P1: ratio、验证、早停）
  train_dpo.py            # Stage 2: DPO（P1: mask、边界、质量校验、早停）
  verify_adapter.py       # 适配器验证（logits + LoRA 权重 + 任务质量）
  README.md
```

## Prerequisites (Cloud)

- GPU with ≥ 48GB VRAM (single GPU, 31B QAT model ~ 20GB bf16 + LoRA overhead)
- `HF_TOKEN` with access to `google/gemma-4-31B-it-qat-w4a16-ct`
- Python packages: `torch`, `transformers`, `peft`, `compressed-tensors` (for QAT dequant)
- `datasets`, `huggingface_hub` (for S1 data acquisition)

## P0 Fix Summary (data leakage & correctness)

| Fix | File | Description |
|-----|------|-------------|
| P0-1 | `src/context_builder.py` | Removed `test_diff` from symbol extraction and `context_hash` |
| P0-2 | `src/context_builder.py` | Added git `HEAD == base_commit` verification instead of hardcoded `True` |
| P0-3 | `src/label_builder.py` | `root_cause_repair_plan` changed to `weak`/`derived`, `exclude_from_training=True` |
| P0-4 | `src/label_builder.py` | `test_reasoning` changed to `test_diff_derived`/`weak`, no fabricated `proposed_tests` |
| P0-5 | `src/patch_validator.py` | Apply patch to temp work copy, compile on MODIFIED files, syntax+relevance in critical gates |
| P0-6 | `src/quality_gates.py` | Gate 1: HEAD==base_commit; Gate 3: per-task evidence/verified; Gate 4: fixed seed+SHA256 split manifest; Gate 5: per-task eligibility |
| P0-7 | `build_sft_data.py` | Removed `test_diff`, `merge_sha`, gold-patch-derived `src_files` from prompt |
| P0-8 | `src/acquire_datasets.py` | Dataset ID: `internlm/SWE-Fixer-Train-110K`; repo snapshot; tightened schema |
| P0-9 | `build_dpo_data.py` | No empty `rejected`; required metadata: `apply_status`, `test_status`, `failure_type`, `test_log_hash`, `pair_verified` |

## Stage 0: Local Data Pipeline (S1–S5, no GPU)

```bash
# S1: 采集数据集并建仓库快照
python training/src/acquire_datasets.py --sources swe_fixer --max-rows 200 --dry-run
python training/src/acquire_datasets.py --sources swe_fixer

# S2: 构建上下文（需 reconstructed.json 中的任务）
python training/src/context_builder.py --tasks <task_id_1> <task_id_2> ...

# S3: 构建六任务标签
python training/src/label_builder.py --tasks <task_id_1> <task_id_2> ...

# S4: Patch 验证
python training/src/patch_validator.py --tasks <task_id_1> <task_id_2> ...

# S5: 质量门禁 + 审计
python training/src/quality_gates.py --tasks <task_id_1> <task_id_2> ... --seed 42

# 构建 SFT 训练数据（P0-7 修复后）
python training/build_sft_data.py
# → training/data/sft_verified_patches.jsonl

# 构建 DPO 训练数据（P0-9 修复后）
python training/build_dpo_data.py
# → training/data/dpo_verified_patches.jsonl
```

Upload `training/data/` to your cloud training environment.

## Stage 1: Verified Patch SFT (Multi-Task)

Train on true Issue→Patch pairs with multi-task balanced sampling and six-task ratio control.

```bash
python training/train_sft.py \
  --data training/data/sft_verified_patches.jsonl \
  --out training/adapters/patch_sft_v1 \
  --model google/gemma-4-31B-it-qat-w4a16-ct \
  --hf-token $HF_TOKEN \
  --r 16 --alpha 32 --dropout 0.05 --lr 5e-5 \
  --epochs 3 --max_len 8192 --grad_steps 8 --batch 1 \
  --val_ratio 0.1 --early_stopping_patience 3 \
  --dequantize
```

P1 options:
- `--task_ratio '{"patch_generation":0.5,...}'` — 六种任务采样比例
- `--target_modules q_proj,v_proj` — LoRA target modules
- `--val_ratio` / `--early_stopping_patience` — 验证与早停

Expected output: `training/adapters/patch_sft_v1/adapter_model.safetensors` + `training_meta.json`
Best checkpoint saved to `training/adapters/patch_sft_v1/best/`.

## Stage 2: Verified Patch DPO

Fine-tune SFT adapter with chosen/rejected pairs. Only valid (non-empty, non-identical) rejected patches are used.

```bash
python training/train_dpo.py \
  --data training/data/dpo_verified_patches.jsonl \
  --sft-adapter training/adapters/patch_sft_v1 \
  --out training/adapters/patch_sft_dpo_v1 \
  --model google/gemma-4-31B-it-qat-w4a16-ct \
  --hf-token $HF_TOKEN \
  --r 16 --alpha 32 --dropout 0.05 --lr 1e-5 \
  --epochs 2 --max_len 8192 --grad_steps 16 --batch 1 \
  --beta 0.1 --val_ratio 0.1 --early_stopping_patience 2 \
  --dequantize
```

Expected output: `training/adapters/patch_sft_dpo_v1/adapter_model.safetensors`

## Stage 3: Verify Adapter

**Critical:** Confirm the adapter is active, LoRA weights loaded, and task output is valid patch format.

```bash
python training/verify_adapter.py \
  --model google/gemma-4-31B-it-qat-w4a16-ct \
  --adapter training/adapters/patch_sft_dpo_v1 \
  --hf-token $HF_TOKEN \
  --threshold 1e-4
```

Exit 0 = all checks pass. Exit 1 = any check failed (do not use adapter).

## Data Quality Checklist

Before training, verify:

- [ ] S1: Dataset source is `internlm/SWE-Fixer-Train-110K` (not `princeton-nlp/`)
- [ ] S2: Context built from `base_commit` only, `test_diff` excluded from context_hash
- [ ] S2: Repo HEAD == base_commit verified (not hardcoded)
- [ ] S3: `root_cause_repair_plan` marked `exclude_from_training`, `test_reasoning` is `weak`
- [ ] S4: Patch validated via temp work copy, syntax checked on MODIFIED files
- [ ] S5: Split manifest with SHA256, fixed seed, repo-level isolation
- [ ] SFT data: no `test_diff` or `merge_sha` in prompt
- [ ] DPO data: no empty `rejected`, all pairs have `pair_verified` metadata
- [ ] No training data in evaluation set (split by repository)
- [ ] `adapter_model.safetensors` exists and is > 0 bytes after training

## Troubleshooting

**Adapter not loading**: Check `adapter_model.safetensors` exists (not `.filepart`).

**Base model mismatch**: Training and inference must use the same `gemma-4-31b-it-qat-w4a16-ct` QAT model.

**OOM**: Reduce `--batch` to 1, increase `--grad_steps`, reduce `--max_len` to 4096.

**DPO no valid pairs**: Ensure `build_dpo_data.py` found non-empty rejected patches from scheme edits.