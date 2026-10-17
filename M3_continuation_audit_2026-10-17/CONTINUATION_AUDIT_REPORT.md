# M3 继续工程 — 独立离线复核报告（2026-10-17）

**依据：** 用户指令“解压 Google Drive 工程文件包，读取项目交接文档，继续工程任务”。
**输入：** `workspace-01a12aa8-43c3-7136-b737-a522ea4a8935.zip`（99,040,357 B，Google Drive 文件 `1BJ_LocnehTi5z7Ymk4hwWGBXbaG_DI6p`），内含 `project/` 工程树（489 个文件）。
**交接状态认定：** 按 `project/M3/HANDOVER_TASK_INCOMPLETE_2026-10-11.md`，任务**未完成**，按顺序停在 M3，不进入 M4。本报告不改变该状态。

## 0. 结论摘要

- 交接书规定的停止条件**全部遵守**：本轮只做离线、只读的材料复核与复现性验证；**无 Kaggle API 调用、无认证、无提交/重试/改名、无 GPU 启动、无模型/Agent 运行、无工程代码执行**（仅重跑交付清单自身规定的只读审计脚本以验证可复现性）；未修改任何历史证据；未索要 token/cookie/凭据；未要求恢复已删除 notebook。
- 按交付清单 §4 与 `GateC3_audit/VERIFY_README.md` 给出的独立审计路径完成 **28 项检查：26 PASS、0 FAIL、2 FINDING（新发现的材料完整性观察项，见 §4）**。
- 冻结状态**原样保持**：`real FastAPI CPU calibration = NOT_CONFIRMED`、`Gate C = BLOCKED`、`Run #5 GPU = HOLD`、`new_submission_authorized = FALSE`、`M3→M4 = NOT_ELIGIBLE`、`rich_2725 = 2 valid paired runs, gaps [+1, +1]`。

## 1. 本轮做了什么（全部离线只读）

1. 从 Google Drive 下载工程包并按 ZIP 内 GBK 文件名正确解压（489 文件，编码已修复）。按 `交付清单_20261011.md` §4 的独立审计路径逐项执行：两个主材料包 + M2/M3 各材料包完整性校验、工程树内全部 SHA256SUMS 清单复核、交接书 §4 关键交付物哈希表核对、C-3 裁定脚本复现、授权互锁动态验证、A5 双 slug 档案抽查、冲突登记与结论边界核对、返回证据包内部一致性核对。机器可读结果：`verification_matrix.json`（由 `continuation_audit_verify.py` 生成，可复现）。

## 2. 验证矩阵（26 项 PASS）

| ID | 检查 | 结果 |
|---|---|---|
| B01 | `GateC_审计材料包.zip` 完整性 | **30/30 OK**（31 成员；SHA256SUMS 自身不自哈希，与清单口径一致） |
| B02 | `Phase2D2_Requests6589_材料包.zip` 完整性 | **30/30 OK**（31 成员） |
| B03 | `M2_离线审计材料包.zip` | 11/11 OK |
| B04 | `M2_离线审计材料包_rev2.zip` | 12/12 OK |
| B05 | `M3_离线准备包.zip` | 6/6 OK |
| T01–T05 | 工程树内清单（M3、USER_RUN_KIT、线 B OUTPUT、M2、V1） | 全 OK |
| T07–T14 | M3 各子清单（CPU 校准待证、外链补充、本地外链清单、返回证据审阅、交接书、门槛检查点、线 B 格式复核、线 B INPUT） | 全 OK |
| H01 | 交接书 §4 关键交付物哈希表 | **12/12 匹配** |
| R01 | C-3 裁定复现（重跑 `gatec3_ref_integrity_and_closeout.py`） | exit 0；stdout 逐字匹配；输出 SHA-256 `d3b35a91…459`，与包内 `GateC3_closeout.json` **逐字节一致** |
| R02 | 授权互锁（`authenticated_readonly_verify_DRAFT_NOT_EXECUTED.py`，无授权短语） | **exit=3、stdout 为空、stderr 提示 INTERLOCK、零网络请求** |
| A01 | A5 档案哈希 + 5 成员抽查 + 双 slug 语义 | 档案 `af4febda…` 匹配；5/5 成员哈希与字节数精确匹配；请求侧 `auditoverlay`、服务器返回侧 `audit-overlay`、review 同时记录两者且 `accepted_pushes=1` |
| C01 | 冲突登记 C1R-01..05 | 5 项均在（裁定 CONTRADICTED、artifact_fact=CONFIRMED、kaggle_actual_state=UNRESOLVED） |
| C02 | 结论边界 | Gate C-3 全部 5 候选 `remote_fact=UNCONFIRMED`（A2/A4/A5/A6/Run5_GPU） |
| E01 | 返回证据包 `fastapi_cpu_evidence.zip` | CRC PASS；58 成员无重复；解包树 58/58 哈希闭合 |
| E02 | 根 `uploads/fastapi_cpu_evidence.json` | 与 `M3/fastapi_cpu_evidence.zip` **逐字节一致**（`e57692a1…4c0`） |

## 3. 与交接书状态的一致性核对

- M3-A 的 S1–S6 判定与 `M3/CPU_CALIBRATION_REPORT_PENDING_EXTERNAL_EVIDENCE_v1.md`、`M3/FASTAPI_CPU_RETURNED_EVIDENCE_REVIEW_v1.md`、`M3/M3_CONTINUATION_GATE_CHECKPOINT_v1.md` 完全一致：S1/S2/S4 仍缺 `push_result.json`、`remote_status.json`、`download_summary.json`、`kernel-metadata.json` 与提交目录字节哈希绑定；S5 PASS、S6 PASS_WITH_CAVEAT。**外层证据链缺口未变，用户已声明无更多材料。**
- M3-B 线 B 报告（`OFFLINE_CONCLUSION_COMPLETE_OFFICIAL_BASELINE_STILL_INCOMPLETE`）与其 INPUT/OUTPUT 哈希清单全部复核通过；窄 provider 锁（`pytest-httpbin==2.0.0`、`httpbin==0.10.4`、`brotlicffi==1.2.0.2`）口径未被越权扩展；官方 baseline 仍 exit 124、`resolved=false`、无 `SESSION_FINISH`，最后活动测试 `test_proxy_error`，根因 `UNKNOWN_WITH_PROXY_TEST_LEAD`。
- D1 批准记录、授权申请草案（`DRAFT_NOT_SENT`）、M3A 准备报告与 manifest 草案（`DRAFT_NOT_SUBMITTABLE_NOT_FROZEN`）哈希均与交接书 §4 登记一致。

## 4. 新发现（FINDING，不改写历史报告）

### F01 — `gemma4_project/gate1_official.ipynb` 与其登记哈希不符（Phase 2B 层材料）

- 磁盘文件：43,034 B，SHA-256 `8a012409a274a8683ab3ab8f4ecda243f908a0d3b12b5205543ff03deb9612ad`。
- `gemma4_project/SHA256SUMS.txt` 登记：`3f24600fa59e5705116e66a2d0046ffa9ca7c91cd70d9e58c7fd16e04c1ce037`。
- 登记哈希与 `phase2b_gate1_bundle.zip` 内副本**逐字节一致**（2 cells，标题 “Phase 2B — Official Gate 1 Only”）。
- 磁盘副本实为 “L4×4 ENV_CHECK v2” 诊断 notebook（4 cells；前 3 个 cell 与 `L4_ENV_CHECK_v2.ipynb` 相同，第 4 个为 `RUN_MODEL_AND_HARNESS = False` 守卫 cell），与登记的 Gate-1 notebook 是**两个不同文件共用了同一文件名**。
- 影响评估：该文件属于 Phase 2B L4 诊断层，**不在 Gate C 链的证据锚点上**（Gate C-1→C-3 链锚定 phase2d 五候选档案，本轮 B01/R01/A01/C01/C02 全部通过）；`project/README.md` 引用的路径为 `notebooks/gate1_official.ipynb`（本快照中不存在该目录）。判定为**材料完整性观察项**，非 Gate C 结论变更依据；是否以 bundle 内副本为准由用户/审计方决定，本轮不替换、不改动任何文件。

### F02 — `gemma4_project/expert_review_bundle/` 快照不完整

- 该目录 `SHA256SUMS.txt` 列 41 项，快照中仅 4 项存在（3 项哈希 OK + 清单自身）；38 项（`evidence/*`、`source/remote_verify/*`）缺失。
- 性质：交付快照范围缺口（与交付清单 §3 已记录的《Phase2C 项目交接文档_新Agent.md》缺失同类），**不是哈希漂移**；存在的文件全部哈希匹配。

## 5. 冻结状态（重申，未改变）

```text
real FastAPI CPU calibration = NOT_CONFIRMED
Gate C                        = BLOCKED
Run #5 GPU                    = HOLD
new_submission_authorized     = FALSE
M3 -> M4 transition           = NOT_ELIGIBLE
rich_2725                     = 2 valid paired runs, gaps [+1, +1]
A5 分类（C-3 校准）           = REF_SET_DIVERGENT_DOCUMENTED_SINGLE_PUSH
全部远端事实                   = UNCONFIRMED
M2 Rev2 / M3-A / M3-B 用户独立复核 = 均未确认（不得由执行方记录代替）
```

## 6. 剩余阻塞与解锁条件（不变）

- M3-A 外层链仍缺：精确 ref/version 的 Kaggle 最终终态记录、提交目录/下载输出的哈希绑定、结构化 push 回执（kernelId/error）。只有拿到此类**非敏感外层证据**并完成审阅后，才可追加更新 S1–S6 判定；用户目前表示无更多材料。
- 解锁前不得：重投、换名、轮询 API、启动 GPU、把本地日志/CLI 文本/历史 CPU 结果升级为校准确认、把诊断 kernel 的 `COMPLETE` 写成 baseline 通过。

## 7. 复现方式

```bash
python3 continuation_audit_verify.py \
  --project <解压后的 project/ 路径> --out ./verification_matrix.json
```

脚本零网络、零认证；其内部仅以子进程重跑交付包自带的两个只读审计脚本（C-3 复现与互锁验证），不执行任何工程代码。产出：`verification_matrix.json`（本报告 §2 的机器可读形式）与临时复现文件（不进入本目录持久产物）。

## 8. 本轮没有做的事

没有访问任何 Kaggle URL、没有轮询 API、没有提交/重试/改名、没有启动 GPU、没有运行模型或 Agent、没有执行工程代码（Worker/pytest/notebook）、没有修改或删除任何历史证据、没有把任何本地自洽表述升级为远端事实、没有索要凭据或要求恢复已删除 notebook。F01/F02 仅作记录，未做任何替换或修补。
