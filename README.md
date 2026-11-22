# Gemma 4 31B Repair Training Pipeline

多阶段修复能力训练管线。数据准备（S1–S5 + 构建）本地 CPU 运行；训练（SFT/DPO）在云端 GPU 上以
`gemma-4-31b-it-qat-w4a16-ct` QAT 模型运行。

**当前状态：GO — Code Repair + CPU Tests + 200-Issue Audit；NO-GO — Full 5K Pilot Training**
（须先通过 200 条真实 Issue 的端到端审计，再扩至 5K。）

## 审计修复索引（F1–F9）

| Fix | 文件 | 内容 | 验收 |
|-----|------|------|------|
| F0 | `src/paths_config.py` | 可移植路径（`GEMMA4_ROOT` 环境变量覆盖） | Kaggle/Linux 无硬编码 Windows 路径 |
| F1 | `src/context_builder.py` | `build_context` 接口移除 `test_diff`；不再从 test_diff 构造 issue；`audit_context_leak` 自动泄漏检测；真实 tokenizer 预算（可选） | provenance 中 `leak_check.audit.leak_found=False` |
| F2 | `src/acquire_datasets.py` | 数据源 `internlm/SWE-Fixer-Train-110K`；S1 只标 `unverified`；`--streaming` | S4 之后才赋 strong |
| F3 | `src/patch_validator.py` | 首选 `git apply`（真实 patch 语义）；字符串替换仅为 fallback 且不判 verified；relevance 降为质量指标（分类记录） | `apply_method=="git_apply"` |
| F4 | `src/quality_gates.py` | HEAD==base_commit 验证；真实 repo 字段分组；固定 seed 随机 split + SHA256 manifest；patch-hash 去重 | audit_report.dedup |
| F5 | `src/quality_gates.py` + 构建入口 | **Task-level eligibility**（弱标签不因 issue 级 strong 连带放行）；SFT/DPO 构建默认必须消费 gates 输出，否则阻断 | eligible_samples 带 `task_eligibility` |
| F6 | `src/label_builder.py` | Issue Understanding 禁用 test_diff；Test Reasoning 改名 `gold_test_patch_analysis`；Root Cause 默认排除；全量 `training_eligible` 字段 | provenance.training_eligible |
| F7 | `train_sft.py` | 整对话一次性 chat template + assistant mask（权威 `assistant_tokens_mask` → 前缀校验 → fallback）；`--dump_token_audit`；尾批补偿缩放；LoRA 匹配审计（零匹配阻断） | token_audit.jsonl 人工核对 |
| F8 | `train_dpo.py` | prompt 与 SFT/inference 同一 chat template；真实长度 attention mask；max_len 同量左截；前缀对齐校验；`--preflight` 双 31B 显存/冻结校验；尾批补偿 | prefix_mismatches 计数 + preflight PASS |
| F9 | `verify_adapter.py` | V0 文件完整性/哈希 → V1 base_model 元数据匹配 → V2 logits/权重/任务格式 → （V3 冻结评测另行） | exit 0 = V0–V2 全过 |

## 目录结构

```
training/
  src/
    paths_config.py       # F0: 可移植路径（GEMMA4_ROOT 可覆盖）
    acquire_datasets.py   # S1: 采集 + 仓库快照 + streaming
    context_builder.py    # S2: 上下文构建 + 自动泄漏审计
    label_builder.py      # S3: 6 任务标签（F6 语义）
    patch_validator.py    # S4: git apply 验证
    quality_gates.py      # S5: 门禁 + task-level eligibility + split manifest
  build_sft_data.py       # gate 联接的 SFT 数据构建
  build_dpo_data.py       # gate 联接的 DPO 数据构建（messages 统一格式）
  train_sft.py            # 多任务 SFT（F7）
  train_dpo.py            # DPO（F8）
  verify_adapter.py       # V0–V2 适配器验证（F9）
  data/
    raw/ snapshots/ processed/
    sft_verified_patches.jsonl(.sha256)
    dpo_verified_patches.jsonl
  adapters/
```

## 数据管线（本地，CPU）

```bash
# S1 采集（smoke 用 --max-rows 200 --streaming）
python training/src/acquire_datasets.py --sources swe_fixer --max-rows 200 --streaming

# S2 上下文（--use-tokenizer 启用真实 token 计数）
python training/src/context_builder.py --tasks <ids...> [--use-tokenizer]

# S3 标签
python training/src/label_builder.py --tasks <ids...>

# S4 patch 验证（git apply）
python training/src/patch_validator.py --tasks <ids...>

# S5 门禁（固定 seed、SHA256 split manifest、task-level eligibility）
python training/src/quality_gates.py --tasks <ids...> --seed 42

# 构建（强制联接 gates；无 eligible_samples.jsonl 会退出码 2）
python training/build_sft_data.py
python training/build_dpo_data.py
```

本地 smoke 可加 `--allow-ungated`，但正式训练数据必须走 gates。

## Stage 1: SFT（云端）

```bash
python training/train_sft.py \
  --data training/data/sft_verified_patches.jsonl \
  --out training/adapters/patch_sft_v1 \
  --model google/gemma-4-31B-it-qat-w4a16-ct \
  --hf-token $HF_TOKEN \
  --r 16 --alpha 32 --dropout 0.05 --lr 5e-5 \
  --epochs 3 --max_len 8192 --grad_steps 8 --batch 1 \
  --val_ratio 0.1 --early_stopping_patience 3 \
  --dump_token_audit 3 --require_mask_audit_pass \
  --dequantize
```

启动前必做：检查 `token_audit.jsonl`（mask 是否只覆盖 assistant 回答、特殊 token 结构完整）。
LoRA 零匹配会直接阻断（exit 5）。

## Stage 2: DPO（云端）

```bash
# 先 preflight（双 31B 实例显存 / reference 冻结 / 保存重载验证）
python training/train_dpo.py --data ... --sft-adapter ... --out ... --preflight --dequantize

# preflight 通过后再训练
python training/train_dpo.py --data ... --sft-adapter ... --out ... \
  --epochs 2 --grad_steps 16 --batch 1 --beta 0.1 \
  --val_ratio 0.1 --early_stopping_patience 2 --dequantize
```

## Stage 3: 适配器验证

```bash
python training/verify_adapter.py --model google/gemma-4-31B-it-qat-w4a16-ct \
  --adapter training/adapters/patch_sft_dpo_v1 --hf-token $HF_TOKEN
```

V0–V2 通过是必要条件；发布前仍需 V3 冻结 Repair Eval 证明能力提升。

## 200-Issue Pilot 审计清单

- [ ] S1: 云端 smoke 下载验证 Dataset ID/Revision/Schema，抽 20 条核对字段映射
- [ ] S2: 每条 context 的 `leak_check.audit.leak_found == false`，`commit_verified == true`
- [ ] S3: 仅 `patch_generation`（与真实执行的任务 6）可 `training_eligible=true`
- [ ] S4: `apply_method == "git_apply"`；fallback 记录需人工复核
- [ ] S5: audit_report 中 split_manifest.sha256 存在；dedup 无异常重复
- [ ] 构建: SFT/DPO 记录数 == gates eligible 数；训练数据 sha256 与 sidecar 一致
- [ ] SFT 启动前: token_audit.jsonl 人工核对通过
- [ ] DPO 启动前: preflight PASS
