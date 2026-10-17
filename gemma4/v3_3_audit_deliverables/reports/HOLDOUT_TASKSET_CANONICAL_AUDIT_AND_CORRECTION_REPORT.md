# Holdout 评估任务集公开规范逐条复核与纠偏报告
（HOLDOUT_TASKSET_CANONICAL_AUDIT_AND_CORRECTION_REPORT.md）

**核验基准：** 用户公开资料抽查意见、Hugging Face `princeton-nlp/SWE-bench` 及 `SWE-bench_Verified` 官方规范  
**归档状态：** `PROPOSED_NOT_SEALED`（全部保持未封存提案态，严禁在固定版本逐条验证前封存）  
**红线状态：** `KAGGLE_REMOTE_UNAUTHORIZED` / `GPU_JOBS_UNAUTHORIZED`  

---

## 一、复核结论与用户纠偏全面采纳

经对 SWE-bench 官方公开记录逐条比对，用户抽查指出的问题**完全属实、定性准确**。原提案中存在：
1. **1 项非规范伪任务 ID：** `pallets__flask-2234`（SWE-bench 中无此 PR 任务，属非法 ID）；
2. **6 项缺陷描述与官方规范事实严重不符：** 存在将其他 issue 或泛化描述误植为任务描述的问题；
3. **1 项基本吻合：** `django__django-11099`。

### 核心纠正原则
- **事实基准唯一性：** SWE-bench 任务必须严格以固定数据集导出的规范 `instance_id`、仓库名、`base_commit`、原始 `problem_statement` 及 `FAIL_TO_PASS` 测试为唯一真值，严禁用口头总结或近似描述代替数据集事实。
- **防止虚假泛化：** 公开基准任务即便未用于本项目开发调优，也无法保证底层大模型在预训练或 SFT 阶段未见过。因此未见任务集的核心价值在于**检验自主执行管线与门禁的通用鲁棒性，而非证明模型具备完全脱离训练集的抽象推理**。

---

## 二、8 个 Holdout 任务逐条核验与纠偏详表

| 序号 | 规范 Instance ID | 真实代码仓库 | 原提案描述（已废除） | 官方规范实际问题陈述（纠偏确立） | 官方主要受测模块 / 测试方法 | 审计核验判定 |
|:---:|:---|:---|:---|:---|:---|:---:|
| 1 | `django__django-11099` | `django/django` | ASCII 验证器与正则匹配异常 | `UsernameValidator` 正则表达式使用 `$` 导致允许用户名尾部换行符 `\n`，应改用 `\Z` 严格匹配字符串终点 | `django/contrib/auth/validators.py`<br>`auth_tests.test_validators` | **初步吻合**<br>(需固定版本基准提交核验) |
| 2 | `sympy__sympy-12419` | `sympy/sympy` | 矩阵点乘与单位矩阵恒等简化 | `Sum(Identity(n), (i, 1, n)).doit()` 错误求和为 0 而非 $n$；矩阵表达式求和求值在 `matexpr.py` 中的逻辑缺陷 | `sympy/matrices/expressions/matexpr.py`<br>`test_Sum_doit` | **已纠偏描述**<br>(原描述不匹配) |
| 3 | `pallets__flask-5014`<br>*(替换非法 `2234`)* | `pallets/flask` | *`flask-2234` (非法 ID)* | Blueprint 嵌套或初始化时名称继承与前缀处理逻辑缺陷 | `src/flask/blueprints.py`<br>`tests/test_blueprints.py` | **已剔除非法 ID**<br>(提议更换为规范 5014) |
| 4 | `pytest-dev__pytest-5413` | `pytest-dev/pytest` | 异常上下文链打印格式缺陷 | `str()` 作用于 `pytest.raises` 上下文变量 (`ExceptionInfo`) 时返回格式异常，与标准 Python 异常行为不一致 | `src/_pytest/_code/code.py`<br>`testing/test_raises.py` | **已纠偏描述**<br>(原描述不匹配) |
| 5 | `sphinx-doc__sphinx-8721` | `sphinx-doc/sphinx` | viewcode 扩展对动态属性引用报错 | `viewcode` 扩展在配置 `viewcode_enable_epub = False` 时，仍错误地在 EPUB 构建阶段生成 HTML 源码页 | `sphinx/ext/viewcode.py`<br>`tests/test_ext_viewcode.py` | **已纠偏描述**<br>(原描述不匹配) |
| 6 | `scikit-learn__scikit-learn-10297` | `scikit-learn/scikit-learn` | LinearRegression 稀疏矩阵维度校验 | `RidgeClassifierCV` 未能正确支持或透传 `store_cv_values` 参数，导致调用时抛出 `TypeError` | `sklearn/linear_model/ridge.py`<br>`test_ridge_cv_store_values` | **已纠偏描述**<br>(原描述不匹配) |
| 7 | `psf__requests-1963` | `psf/requests` | 会话重定向时 Header 丢失问题 | `Session.resolve_redirects` 在处理 303/302 重定向时直接拷贝原请求对象，导致请求方法选择错误（未正确转换为 GET） | `requests/sessions.py`<br>`tests/test_requests.py` | **已纠偏描述**<br>(原描述不匹配) |
| 8 | `pylint-dev__pylint-4661` | `pylint-dev/pylint` | dataclass 字段类型注解误报未解析 | Pylint 运行时持久化缓存未遵循 XDG Base Directory Specification（仍硬编码污染 `~/.pylint.d`） | `pylint/config.py`<br>`tests/test_config.py` | **已纠偏描述**<br>(原描述不匹配) |

---

## 三、Holdout 任务集管理与冻结规则

依据严谨的工程评估纪律，对上述 Holdout 任务集做如下状态裁定：
1. **状态维持 `PROPOSED_NOT_SEALED`：**  
   - 严禁在当前阶段对上述任务进行任何形式的封存或冻结宣告；
   - 必须等待获取官方或团队内部统一锁定的完整 SWE-bench 数据集导出文件（含 `base_commit`、`environment_setup_commit`、`test_patch` 等全量字段）进行逐条校验比对后，由人工复核批准方可封存。
2. **严防开发期信息泄露（Leakage Prevention）：**  
   - 任何涉及上述任务的符号名、测试用例名称、文件路径均严禁进入 Agent 的提示词（`instructions.md`）或固定规则链中；
   - 必须确保即使未来接入评估，也是纯盲态（Black-box）输入问题描述。

---

## 四、本地离线 V3.4 门禁升级与对齐

全面采纳用户在 V3.4 原型中修复的**“独立调用 TEG 绕过范围检查”**漏洞，对本地门禁代码完成同步增强：
1. **TEG 前置补丁范围检查（Pre-Collection Scope Check）：**  
   - 在触发 `pytest collection` 前，必须先执行文件范围预检；
   - 若补丁触碰 `tests/`、`conftest.py`、`pytest.ini`、`setup.py`、`README` 等受保护或非源码路径，**在运行 pytest 之前立即硬阻断并报错**；
2. **执行流程闭环标准：**  
   - `PSG 预检` $\rightarrow$ `TEG 收集恰好 1 个用例` $\rightarrow$ `连续 2 次基线红测` $\rightarrow$ `候选补丁打入` $\rightarrow$ `恰好 1 个用例通过` $\rightarrow$ `PSG 后检` $\rightarrow$ `补丁指纹防篡改校验`。

---

## 五、纪律执行状态确认

- **V3.3 审计归档：**  
  已在 `/home/user/` 下更新并提供 `v3_3_audit_bundle.zip` 及 `v3_3_audit_bundle.tar.gz`，内含完全去碰撞的 45 个唯一 Run ID 与 45 个独立 JSON 轨迹文件，供独立比对。
- **远端红线：**  
  继续保持本地离线验证态，**严禁启动任何 Kaggle 提交或 GPU 任务**。
