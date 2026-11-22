# Gemma 4 31B Patch Capability Enhancement — Training Pipeline

Three-stage training to improve Gemma 4 31B's code patch generation capability.
All scripts run on cloud GPU (Kaggle, etc.) with the gated `gemma-4-31b-it-qat-w4a16-ct` model.

## Directory Layout

```
training/
  data/
    sft_verified_patches.jsonl    # Stage 1: gold Issue→Patch pairs
    dpo_verified_patches.jsonl    # Stage 2: chosen/rejected patch pairs
  adapters/
    patch_sft_v1/                 # Stage 1 output (SFT adapter)
    patch_sft_dpo_v1/             # Stage 2 output (DPO adapter)
  build_sft_data.py               # Build SFT data from reconstructed.json
  build_dpo_data.py               # Build DPO data (chosen=gold, rejected=failed)
  train_sft.py                    # Stage 1: Verified Patch SFT
  train_dpo.py                    # Stage 2: Verified Patch DPO
  verify_adapter.py               # Verify adapter actually changes logits
```

## Prerequisites (Cloud)

- GPU with ≥ 48GB VRAM (single GPU, 31B QAT model ~ 20GB bf16 + LoRA overhead)
- `HF_TOKEN` with access to `google/gemma-4-31B-it-qat-w4a16-ct`
- Python packages: `torch`, `transformers`, `peft`, `compressed-tensors` (for QAT dequant)

## Stage 0: Build Training Data

Run locally first (no GPU needed):

```bash
# Step 1: Build SFT data from reconstructed.json
python training/build_sft_data.py
# → training/data/sft_verified_patches.jsonl

# Step 2: Build DPO data (pairs gold patch with failed patches)
python training/build_dpo_data.py
# → training/data/dpo_verified_patches.jsonl
```

Upload `training/data/` to your cloud training environment.

## Stage 1: Verified Patch SFT

Train on true Issue→Patch pairs. Only assistant tokens supervised.

```bash
python training/train_sft.py \
  --data training/data/sft_verified_patches.jsonl \
  --out training/adapters/patch_sft_v1 \
  --model google/gemma-4-31B-it-qat-w4a16-ct \
  --hf-token $HF_TOKEN \
  --r 16 --alpha 32 --dropout 0.05 --lr 5e-5 \
  --epochs 3 --max_len 8192 --grad-steps 8 --batch 1 \
  --dequantize
```

Expected output: `training/adapters/patch_sft_v1/adapter_model.safetensors` + `training_meta.json`

## Stage 2: Verified Patch DPO

Fine-tune SFT adapter with chosen/rejected pairs. Good patch ≻ bad patch.

```bash
python training/train_dpo.py \
  --data training/data/dpo_verified_patches.jsonl \
  --sft-adapter training/adapters/patch_sft_v1 \
  --out training/adapters/patch_sft_dpo_v1 \
  --model google/gemma-4-31B-it-qat-w4a16-ct \
  --hf-token $HF_TOKEN \
  --r 16 --alpha 32 --dropout 0.05 --lr 1e-5 \
  --epochs 2 --max_len 8192 --grad-steps 16 --batch 1 \
  --beta 0.1 --dequantize
```

Expected output: `training/adapters/patch_sft_dpo_v1/adapter_model.safetensors`

## Stage 3: Verify Adapter Actually Works

**Critical:** Before using the adapter, confirm it actually changes model output (the previous LoRA had missing weights).

```bash
python training/verify_adapter.py \
  --model google/gemma-4-31B-it-qat-w4a16-ct \
  --adapter training/adapters/patch_sft_dpo_v1 \
  --hf-token $HF_TOKEN \
  --threshold 1e-4
```

Exit 0 = adapter active. Exit 1 = adapter NOT affecting output (do not use).

## Hyperparameter Guidance

| Parameter | Stage 1 (SFT) | Stage 2 (DPO) | Rationale |
|-----------|--------------|---------------|-----------|
| r (rank) | 16 | 16 | Match previous config; 64 also viable |
| alpha | 32 | 32 | 2×r |
| dropout | 0.05 | 0.05 | Regularization |
| lr | 5e-5 | 1e-5 | DPO uses lower LR |
| epochs | 3 | 2 | SFT needs more passes |
| max_len | 8192 | 8192 | Code context can be long |
| grad_steps | 8 | 16 | DPO needs larger effective batch |
| beta | — | 0.1 | DPO temperature |

## Data Quality Checklist

Before training, verify:

- [ ] SFT data has ≥ 1000 records (5,000–10,000 target)
- [ ] Each SFT record has valid gold_diff that applies on base_commit
- [ ] DPO chosen = verified patch (passes FAIL_TO_PASS + PASS_TO_PASS)
- [ ] DPO rejected = failed patch (same Issue, different patch)
- [ ] No training data appears in evaluation set (split by repository)
- [ ] adapter_model.safetensors exists and is > 0 bytes after training

## Troubleshooting

**Adapter not loading**: Check `adapter_model.safetensors` exists (not `.filepart`). The previous LoRA failed because only `.filepart` was present.

**Base model mismatch**: Training and inference must use the same `gemma-4-31b-it-qat-w4a16-ct` QAT model. Adapter trained on non-QAT base will not work on QAT inference.

**Logits diff near zero**: Adapter weights are not influencing output. Verify `--hf-token` is set, model loaded correctly, and adapter file is complete.

**OOM**: Reduce `--batch` to 1, increase `--grad-steps`, reduce `--max_len` to 4096.