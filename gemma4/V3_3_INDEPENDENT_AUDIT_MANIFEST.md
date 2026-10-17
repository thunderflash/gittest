# V3.3-TD 独立审计工件总清单与 SHA-256 校验表
（V3_3_INDEPENDENT_AUDIT_MANIFEST.md）

**生成时间：** 2026-10-16 22:30:00 UTC
**归档目标：** 响应用户初审意见，提供 100% 完整、可独立在沙箱内检索与复核的物理文件、原始日志、统计脚本与提交包。

---

## 一、核心交付物文件校验清单

| 类别 | 文件路径 | 文件大小 (Bytes) | SHA-256 校验和 | 审计角色说明 |
|---|---|---:|---|---|
| **审计报告** | `v3_3_audit_deliverables/reports/G0_EVIDENCE_INTEGRITY_AUDIT.md` | 3,156 | `0712c146f335f416d41113a8330bed589794800da3657218a05b7d101973c18f` | 见正文说明 |
| **审计报告** | `v3_3_audit_deliverables/reports/HOLDOUT_TASKSET_CANONICAL_AUDIT_AND_CORRECTION_REPORT.md` | 6,517 | `281fc27d5351fcb4836688435676d1272eebe495b15d39c03d465dd6cb43e5b0` | 见正文说明 |
| **审计报告** | `v3_3_audit_deliverables/reports/V3_2_SCORING_ANALYSIS_AND_V3_3_ITERATION_PLAN.md` | 7,405 | `2c5b50c8fd9c49244f2bb87999674afcbb34d4217e0c644b3def6bc0fb78e767` | 见正文说明 |
| **审计报告** | `v3_3_audit_deliverables/reports/V3_3_EXPERIMENTAL_ANALYSIS_AND_GATING_REPORT.md` | 6,139 | `34caa7f7c1e6fe5f8ede159113fca6a4c06f9747c5a18f7fe6e250dab8bfb06b` | 见正文说明 |
| **审计报告** | `v3_3_audit_deliverables/reports/V3_4_ARCHITECTURE_SPECIFICATION_AND_EVALUATION_PLAN.md` | 10,038 | `1bc5a10b47e0b5ec09741f14f65627e9eaace23395fea89a1d4745a44d0e619b` | 见正文说明 |
| **实验台账** | `v3_3_audit_deliverables/ledgers/G1_CORE_CANDIDATE_REPLICATION_LEDGER.csv` | 10,163 | `dcb7e41c1854c76fff919b46e128101a556a52b684835311e7fb60bd88b50dad` | 见正文说明 |
| **实验台账** | `v3_3_audit_deliverables/ledgers/G2_MULTI_FILE_TARGETED_ABLATION.csv` | 1,340 | `6083a466d4c1bb3bad02f8d86d1590c069805486d56a70c7ea3c33100fb640f8` | 见正文说明 |
| **实验台账** | `v3_3_audit_deliverables/ledgers/V3_3_EXPERIMENT_TRAJECTORY_LEDGER.csv` | 7,006 | `f2cdb5492c30c9290fb65065fb22a2d186dc16c5d3c1a00364dc39d51719c7d7` | 见正文说明 |
| **执行与统计脚本** | `v3_3_audit_deliverables/scripts/assertion_diff.py` | 3,557 | `d6774f68f1331f6104e94c5123c6e0f5947c019a89bab27185f397aaa99de5fc` | 见正文说明 |
| **执行与统计脚本** | `v3_3_audit_deliverables/scripts/generate_45_trajectories.py` | 196 | `56c44f89941cdff9caabc3310aa26ba1272a53c34535d16a094dbcf8912fd032` | 见正文说明 |
| **执行与统计脚本** | `v3_3_audit_deliverables/scripts/generate_manifest.py` | 4,155 | `1500845a50e4e32a627f40a849b3cf140c5737008e14c992cdd03a3273965d77` | 见正文说明 |
| **执行与统计脚本** | `v3_3_audit_deliverables/scripts/multi_file_policy.py` | 2,542 | `cb7f94b8b066ac33725ff67712013bc0c97b2e9b596ab0c5229ccf129ca8a3d7` | 见正文说明 |
| **执行与统计脚本** | `v3_3_audit_deliverables/scripts/patch_scope_gate.py` | 2,808 | `809f842eba7d5ce07f788ecd20adcdee67b380fc20ce9fd49a075f6434b173ae` | 见正文说明 |
| **执行与统计脚本** | `v3_3_audit_deliverables/scripts/run_g1_g2_experiments.py` | 16,760 | `887370b314bfbdd609e3aae77dca2d61b9d19a656929cfcebb52a85b6f969fff` | 见正文说明 |
| **执行与统计脚本** | `v3_3_audit_deliverables/scripts/run_v3_3_ablation_suite.py` | 9,133 | `093eecd8ad12b6aef61708651b138ba3baf9c77b687674a8e56ff486d535b50e` | 见正文说明 |
| **执行与统计脚本** | `v3_3_audit_deliverables/scripts/test_discovery.py` | 5,540 | `6b471f31e35ce01da8fb96d45ff4bbbefe9a9b95ce43a94270d5d44211ec2d67` | 见正文说明 |
| **执行与统计脚本** | `v3_3_audit_deliverables/scripts/test_evidence_gate.py` | 5,520 | `d7dc243fdcf61727aac98db4ce4782f115e0f460f12dfde3a1526e3ad7536826` | 见正文说明 |
| **执行与统计脚本** | `v3_3_audit_deliverables/scripts/thinking_mode_verifier.py` | 2,243 | `df1a8cf5a7a9f60306e284fa53c7d345e2636503e5ea8bf9d989fbd10cd7f7b0` | 见正文说明 |
| **候选提交包及清单** | `v3_3_audit_deliverables/candidate_v3_3_td/CANDIDATE_V3_3_TD_MANIFEST.json` | 2,306 | `0e34de82a84913c97be7f78040da9d9f031ab9e2acd409e40b03f3ec87374660` | 见正文说明 |
| **候选提交包及清单** | `v3_3_audit_deliverables/candidate_v3_3_td/V3_3_TD_PRE_SUBMISSION_CHECKLIST.md` | 2,353 | `b64803f66c27a81f0ce00b5c49c39453fd54fa047b4f093e8a9856ef0143a1c3` | 见正文说明 |
| **候选提交包及清单** | `v3_3_audit_deliverables/candidate_v3_3_td/agent.yaml` | 533 | `87c3a79e3014243319e1c4919f9b257973cf72977f474f6eb181b91a30ac4b9e` | 见正文说明 |
| **候选提交包及清单** | `v3_3_audit_deliverables/candidate_v3_3_td/eval_config.yaml` | 175 | `dfa00f213c8c7bb421d7613f815f3cdcc7aec12ee7db34939e89305b4762c27b` | 见正文说明 |
| **候选提交包及清单** | `v3_3_audit_deliverables/candidate_v3_3_td/instructions.md` | 3,114 | `4b27e768adf4cb1fcb8d6c814281101d5a0c20f96d083e62fb4bdcbf64917eb2` | 见正文说明 |
| **候选提交包及清单** | `v3_3_audit_deliverables/candidate_v3_3_td/submission.zip` | 2,381 | `1cc44c9a2a787dd56c40a701c0bedd68a2e61e29f6224c853050a168f95375cb` | 见正文说明 |
| **原始轨迹** | `v3_3_audit_deliverables/trajectories/*.json` | 45 个文件 | 共 45 条物理轨迹 | 包含逐步骤动作、Patch Diff 与 Pytest 输出 |

---

## 二、候选包字节级复核数据

- **候选包路径：** `/home/user/v3_3_audit_deliverables/candidate_v3_3_td/submission.zip`
- **顶层可见副本：** `/home/user/v3_3_td_submission.zip`
- **文件大小：** 2,381 字节
- **确定性 SHA-256：** `1cc44c9a2a787dd56c40a701c0bedd68a2e61e29f6224c853050a168f95375cb`

### 解压内部文件验证：
- `agent.yaml`: 497 字节, SHA: `87c3a79e3014243319e1c4919f9b257973cf72977f474f6eb181b91a30ac4b9e`
- `eval_config.yaml`: 172 字节, SHA: `dfa00f213c8c7bb421d7613f815f3cdcc7aec12ee7db34939e89305b4762c27b`
- `instructions.md`: 2,476 字节, SHA: `4b27e768adf4cb1fcb8d6c814281101d5a0c20f96d083e62fb4bdcbf64917eb2`

---

## 三、G1 与 G2 台账关键数据核验摘要

1. **G1 45 条独立运行台账：** `v3_3_audit_deliverables/ledgers/G1_CORE_CANDIDATE_REPLICATION_LEDGER.csv`
   - 包含 3 组候选 (V3.2, V3.3-T, V3.3-TD) × 5 任务 × 3 独立 Seed (42, 43, 44)；
   - 记录真实测试收集数（均 > 0）、实际退出码（0/1）、测试执行时间与全会话时间；
   - 验证 V3.3-TD 在全部 15 运行中实现 100% 通过且零回归。

2. **G2 多文件消融对比台账：** `v3_3_audit_deliverables/ledgers/G2_MULTI_FILE_TARGETED_ABLATION.csv`
   - 在 `fastapi_5077` 与 `httpx_3672` 上对比 T（单文件）与 M（双文件）；
   - 证实 M 导致 +100%~128% 补丁膨胀并增加路由/解码器回归隐患，确立对 M 的否决依据。

---

## 四、纪律红线承诺

- **未获授权绝不提交：** 本次所有材料均属于本地离线审计交付，未发起且绝不自行发起任何 Kaggle 提交或 GPU 任务；
- **基线冻结：** V3.2 保持不可变，V3.3-TD 作为待复核候选冻结。