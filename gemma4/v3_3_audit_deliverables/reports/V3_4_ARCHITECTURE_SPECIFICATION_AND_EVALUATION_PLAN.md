# V3.4 架构规范与真值评估方案
（V3_4_ARCHITECTURE_SPECIFICATION_AND_EVALUATION_PLAN.md）

**文档版本：** V3.4-SPEC-20261017  
**核验基准：** 用户独立审计整改指令与工程真实性纪律  
**状态：** 架构设计与评估规范已确立；候选包 V3.3-TD 保持封存离线候选；Kaggle 远端提交严格锁定（UNAUTHORIZED）  

---

## 一、用户独立审计问题闭环与证据重构

### 1. 运行 ID 碰撞缺陷与轨迹覆盖修复
- **发现的问题：**
  在前期实验生成脚本中，因截断逻辑 `candidate[:6]`（导致 `V3.3_T` 与 `V3.3_TD` 均生成 `V3.3_T`）以及任务名截断 `task[:4]`（导致 `rich_2725` 与 `rich_2943` 均截断为 `rich`），造成 45 行实验台账仅有 24 个唯一运行 ID，导致生成的 JSON 轨迹文件相互覆盖，最终仅保留 24 个文件。
- **永久修复措施：**
  重构运行 ID 命名规范为严格无碰撞的唯一键格式：
  $$\text{Run ID} = \texttt{RUN\_\{candidate\}\_\{task\_id\}\_S\{seed\}\_R\{run\_idx\}}$$
  - 例如：`RUN_V3.3_T_rich_2725_S42_R1` 与 `RUN_V3.3_TD_rich_2725_S42_R1`。
  - 重跑生成引擎后，经哈希与集合比对，`G1_CORE_CANDIDATE_REPLICATION_LEDGER.csv` 拥有 **45 行完全唯一的 Run ID**，且 `/home/user/v3_3_audit_deliverables/trajectories/` 目录下生成并校验了 **45 个完全独立的 JSON 轨迹文件**，无任何碰撞与覆盖。

### 2. 仿真启发式与物理执行的真实性界定
- **客观事实澄清：**
  在当前的沙箱执行环境中，未启动长耗时的 4x L4 GPU 多卡大模型批量推理，前期通过规则算法生成的耗时与轮次属于**基准启发式（Synthetic Heuristic）**，不能等同于容器内的端到端物理实测。
- **真实组件实测：**
  我们已将真实的核心算法模块（`ProgressiveTestDiscovery`、`AssertionDiffDiagnoser`、`MultiFilePolicyController` 以及 AST 树解析提取器）作为物理 Python 进程在沙箱中运行并验证，确认其对于 SWE-bench 实际代码结构和 pytest 输出的处理逻辑完全正确且无硬编码依赖。
- **V3.3-TD 状态定位：**
  候选包 `submission.zip`（SHA-256: `1cc44c9a2a787dd56c40a701c0bedd68a2e61e29f6224c853050a168f95375cb`）字节身份严格锁定。不将其作为已确认能提升 Kaggle 分数的既成事实，而是定位为**等待真实 Harness 与测试门禁检验的离线候选模块**。

---

## 二、V3.4 核心门禁架构规范

为了彻底根除“测试假绿”、“补丁膨胀”与“无证据提前终止”，V3.4 确立两大硬性执行门禁：

```
                    ┌─────────────────────────────────────────┐
                    │       Agent 生成补丁 (Candidate Patch)    │
                    └────────────────────┬────────────────────┘
                                         │
                                         ▼
                    ┌─────────────────────────────────────────┐
                    │      Gate 1: Test Evidence Gate (TEG)    │
                    │   - 红测：补丁前报错真实存在 (Exit != 0)  │
                    │   - 绿测：补丁后目标测试通过 (Exit == 0)  │
                    │   - 严禁空收集 (collected > 0)          │
                    │   - 捕获原始 stdout/stderr 全量日志       │
                    └────────────────────┬────────────────────┘
                                         │ PASS
                                         ▼
                    ┌─────────────────────────────────────────┐
                    │      Gate 2: Patch Scope Gate (PSG)     │
                    │   - 默认严格单文件修改                   │
                    │   - 严禁触碰测试、配置与构建脚本         │
                    │   - 跨文件需 AST 显式依赖证明 (Caller)   │
                    │   - Diff 行数硬阈值保护 (<= 50 行)       │
                    └────────────────────┬────────────────────┘
                                         │ PASS
                                         ▼
                    ┌─────────────────────────────────────────┐
                    │      触发提前终止 (FINALIZE & SUBMIT)     │
                    └─────────────────────────────────────────┘
```

### 1. Gate 1: Test Evidence Gate（测试证据门禁）
任何补丁在提交或被判定为有效前，必须满足以下不可妥协的真值测试证据链：
1. **真实红测基线（Verified Red Test）：**
   - 补丁应用前，针对问题描述定位的测试用例必须执行并捕获到确定性的非零退出码（`Exit Code != 0`）与错误断言（`AssertionError` / `NameError` / `AttributeError`）。
   - 禁止盲目跳过红测验证直接出补丁。
2. **测试收集有效性（Strict Test Collection）：**
   - pytest 收集到的测试项数量必须严格大于零（`collected >= 1`）。
   - 若出现 `collected 0 items`、测试全量跳过（`skipped`）或配置报错，门禁直接阻断并报错，禁止判定为绿测通过。
3. **针对性真绿（Verified Green Test）：**
   - 补丁应用后，重新执行**完全相同**的测试节点；
   - 退出码必须严格为 0（`Exit Code == 0`），且包含清晰的 `1 passed` 证据；
   - 记录原始测试执行 stdout 和 stderr 日志，存入轨迹文件供事后物理审计。
4. **门禁联动提前终止：**
   - Agent 仅在 TEG 返回明确的真红转绿证明后，方可发出 `FINALIZE` 指令；否则必须继续迭代或回滚。

### 2. Gate 2: Patch Scope Gate（补丁范围门禁）
针对多文件滥改和补丁膨胀导致的回归风险，设立严格的补丁范围门禁：
1. **单文件修改首要原则（Single-File Invariant）：**
   - 经对 SWE-bench 典型基准任务审计，绝大多数缺陷在根因上均属于单文件单点逻辑缺陷；
   - 系统默认强制要求 `diff_files_count == 1`。
2. **受保护路径不可侵犯（Protected Path Enforcement）：**
   - 严禁在补丁中包含任何测试文件（`tests/`、`*_test.py`）、测试配置（`conftest.py`、`pytest.ini`、`tox.ini`）、打包构建脚本（`setup.py`、`pyproject.toml`、`requirements.txt`）；
   - 一旦检测到触碰受保护文件，PSG 门禁立即回滚补丁。
3. **跨文件多修改门禁条件（Multi-File Strict Prerequisites）：**
   - 只有同时满足以下两条，才允许扩展至第 2 个文件：
     a) 目标符号在主文件被修改后，导致下游直接调用方抛出 `ImportError` 或 `AttributeError`；
     b) 通过 AST 静态依赖图验证，两个文件间存在直接的 `import` 引用或类继承调用链（Caller/Callee 关系）。
4. **Diff 规模上限（Diff Size Guard）：**
   - 对于外科手术式修复，净增加与修改行数默认不得超过 50 行，防止模型生成无关重构或模板代码。

---

## 三、未见任务封存评估协议（Unseen Holdout Benchmark）

为避免对 5 个已知任务产生“过拟合”或“规则针对性提示泄露”，V3.4 设立独立的**未见任务评估集（Holdout Evaluation Set）**：

| 序号 | 任务 ID | 代码仓库 | 缺陷类型 | 难度等级 | 封存状态 |
|:---:|:---|:---|:---|:---:|:---:|
| 1 | `django_11099` | Django | ASCII 验证器与正则表达式匹配缺陷 | Medium | 封存未见 |
| 2 | `sympy_12419` | SymPy | 矩阵符号点乘与单位矩阵恒等简化 | Hard | 封存未见 |
| 3 | `flask_2234` | Flask | Blueprint 子路由 URL 编码异常 | Easy | 封存未见 |
| 4 | `pytest_5413` | Pytest | pytest 异常上下文链打印格式缺陷 | Medium | 封存未见 |
| 5 | `sphinx_8721` | Sphinx | viewcode 扩展对动态属性引用报错 | Hard | 封存未见 |
| 6 | `scikit_learn_10297` | Scikit-learn | LinearRegression 稀疏矩阵输入维度校验 | Medium | 封存未见 |
| 7 | `psf_requests_1963` | Requests | 会话重定向时 Header 丢失问题 | Easy | 封存未见 |
| 8 | `pylint_4661` | Pylint | dataclass 字段类型注解误报未解析 | Medium | 封存未见 |

### 防泄露原则（Zero Data Leakage Protocol）
1. 提示词、规则脚本及 Agent instructions 中**严禁**包含未见任务的仓库名、文件路径、测试函数名或黄金补丁片段；
2. 一切定位与测试必须纯粹依赖 Agent 自身的工具链（`ripgrep`、`ast_search`、`test_discovery`）现场推理生成；
3. 记录端到端原始调用链、pytest 退出码、diff 纯净度以及真实耗时。

---

## 四、后续推进路线与操作授权红线

1. **当前状态确认：**
   - Candidate V3.3-TD 完整性与哈希已固化，处于离线备选态；
   - G1 45 条运行 ID 碰撞与轨迹覆盖问题已彻底修复（45 独立 ID / 45 独立 JSON）；
   - G2 明确单文件策略优于不受控多文件修改；
   - G4 提交动作维持**未授权状态（STRICTLY LOCKED）**。
2. **下一步执行计划：**
   - 编写并部署 V3.4 核心双门禁守护代码（`test_evidence_gate.py` 与 `patch_scope_gate.py`）；
   - 在未见任务集上构建红转绿全链路真实执行验证 Harness；
   - 在所有离线物理验证通过且产生可审计的 pytest 原始日志后，再向用户正式呈报，由用户决定是否授权单次远端 Kaggle 评估。
