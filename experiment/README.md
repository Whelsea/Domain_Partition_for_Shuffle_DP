# Experiment Framework — User-Level Shuffle-DP

本文件是 `experiment/` 子项目的总文档，目标是让你可以：
- 快速复现实验
- 理解每个模块作用
- 明确数据集处理细节（包含用了哪些原始列）
- 知道当前实现的最新语义（尤其是 FE1 与 random_tau）

## Directory Structure

```text
experiment/
├── README.md                          # ← 本文件
├── run_sweep.py                       # 批量实验入口（生成数据 + 运行实验）
├── experiment.pdf                     # 预生成实验结果（参考）
│
├── static/                            # 协议实现与单次实验入口
│   ├── run_experiment.py              # 主 CLI：加载数据 -> 跑协议 -> 输出指标
│   ├── two_round_protocol.py          # Two-round protocol
│   ├── one_round_protocol.py          # One-round protocol
│   ├── baseline_clip_M.py             # Baseline 1: m_tau = M
│   ├── baseline_random_tau.py         # Baseline 2: m_tau 从 {1,2,4,...,M} 随机选
│   ├── GKMPS.py                       # GKMPS 计数协议（Round 1）
│   ├── query_protocols.py             # Query protocol 工厂（FE1 / GKMPS）
│   └── results/                       # Static 实验结果输出目录
│
├── Base_Protocol/                     # 第三方/底层协议实现
│   ├── ShuffleFE-main/                # FE1 频率估计
│   │   ├── FE1.py
│   │   └── ...
│   └── RM2-main/                      # 备用，当前未接入主实验入口
│
├── dataset/
│   ├── simulated_data/
│   │   ├── generate_data.py           # 生成模拟数据 CLI
│   │   ├── dataset.py                 # UserLevelDataset 定义 + 读写
│   │   └── data/                      # 模拟数据输出目录
│   └── real_data/
│       ├── README.md                  # 真实数据详细文档
│       ├── aol/
│       │   ├── process_aol.py
│       │   ├── user-ct-test-collection-01.txt
│       │   └── data/
│       ├── CK_pay/
│       │   ├── process_ck_pay.py
│       │   ├── Employee_Payroll.csv
│       │   └── data/
│       ├── ml-32m/
│       │   ├── process_movielens.py
│       │   ├── ratings.csv
│       │   └── data/
│       └── Netf/
│           ├── process_netflix.py
│           ├── training_set/
│           └── data/
└── dynamic/                           # 动态场景占位（当前未接入完整流程）
```

---

## Quick Start

### A. 你要的最小可复现示例（M=128, m_max=16, FE1, 4协议）

在仓库根目录执行：

```bash
# 1) 生成数据集
python experiment/dataset/simulated_data/generate_data.py \
  --n 500 \
  --M 128 \
  --m_max 16 \
  --U 100 \
  --contrib_dist zipf \
  --value_dist uniform \
  --seed 42 \
  --output experiment/dataset/simulated_data/data/n500_U100_zipf_mmax16_seed42.csv

# 2) 跑四个协议
cd experiment/static
python run_experiment.py \
  --dataset ../dataset/simulated_data/data/n500_U100_zipf_mmax16_seed42.csv \
  --base_protocol FE1 \
  --protocols two_round one_round baseline_clip_M baseline_random_tau \
  --times 10 \
  --trim 0.2 \
  --random_select_times 5 \
  --seed 42 \
  --fe_workers 1 \
  --output results/m128_mmax16_fe1_compare_no_mu_search.json
```

结果文件：

```text
experiment/static/results/M128_mmax16_n500_U100_eps1_timeYYYYMMDD_HHMMSS.json
```

### B. 一般单次实验（推荐）

```bash
cd experiment/static
python run_experiment.py \
  --dataset <path/to/dataset.csv> \
  --quick_fe1 \
  --output <path/to/result.csv>
```

`--quick_fe1` 默认展开为：
- `--base_protocol FE1`
- `--protocols two_round one_round baseline_clip_M baseline_random_tau`
- `--epsilon 1.0`
- `--times 50`
- `--trim 0.2`

### C. 真实数据流程

```bash
# 示例：Netflix
cd experiment/dataset/real_data/Netf
python process_netflix.py --n 5000 --M 64 --U 1000000000

cd experiment/static
python run_experiment.py \
  --dataset ../dataset/real_data/Netf/data/netflix_n5000_M64_U1000000000.csv \
  --quick_fe1 \
  --output results/results_netflix.csv
```

### D. 批量 Sweep

```bash
cd experiment
python run_sweep.py --dry-run   # 只预览
python run_sweep.py             # 实际执行
python run_sweep.py --force     # 强制重跑
```

`run_sweep.py` 会：
- 先按 `SWEEP_CONFIG["datasets"]` 生成/复用数据集
- 再按 `dataset × protocol × base_protocol × epsilon` 跑全部组合
- 每个 setting 生成一个结果 CSV，并在末尾汇总

可编辑配置位置：`experiment/run_sweep.py` 顶部 `SWEEP_CONFIG`。
典型配置片段：

```python
SWEEP_CONFIG = {
    "datasets": [
        {"type": "simulated", "n": [1000, 5000], "M": [2**20], "m_max": [1024], "U": [100000], "contrib_dist": ["zipf"], "seed": 42},
        {"type": "aol", "n": [5000], "M": [32, 64], "U": [100000, 1000000]},
    ],
    "protocols": ["two_round", "one_round", "baseline_clip_M", "baseline_random_tau"],
    "base_protocol": ["FE1"],
    "epsilon": [0.5, 1.0, 2.0, 4.0],
    "delta": None,
    "beta": 0.1,
    "times": 50,
    "trim": 0.2,
    "seed": 42,
}
```

---

## Current Semantics (最新实现语义)

### FE1 与 `mu_search`

- 默认 **不使用** `mu_search`。
- 默认使用 FE1 理论 `mu`：
  - `mu = 32 * log(2/delta) / epsilon^2`
  - `b = max(2, epsilon^2 * n / log(2/delta)^c)`
- 如果你要显式启用 `mu_search`：加 `--fe_use_mu_search`。

### Faithful 执行 vs. Simulator

- 默认就是 **faithful** 消息级执行。
- 如果你要启用协议中的 `Simulator` 快速路径：加 `--simulate`。
- `--simulate` 会启用：
  - two-round / one-round / baseline 的 `Simulator(...)` 快捷路径
  - FE1 在未开启 `mu_search` 时的 analytical simulator
- `--no_simulate` 只是兼容开关；当前默认已经是不使用 simulator。
- 默认 faithful 模式下，FE1 会走 `LocalRandomizer + Analyzer` 路径。

### One-round faithful 语义

- faithful `one_round` 仍然会为所有候选阈值 `2^j` 真实生成 query 消息。
- 实现上不再显式执行 shuffle，因为 analyzer 只依赖消息 multiset，不依赖顺序。
- analyzer 端只处理最终选中的 `j* = log2(m_tau)` 这一路 query 通道。

### `baseline_random_tau` 的 `random_select_times`

当前语义：
- `random_select_times=K` 仅用于稳定统计（多次随机 `m_tau` 取平均）。
- **不按 K 切分隐私预算**（每次子运行都用 `eps/m_tau`, `delta/m_tau`）。
- 返回的 `nmessages` 是 K 次子运行的**平均**通信量（不是求和）。

---

## CLI 参数参考（run_experiment.py）

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--dataset` | str | 必填 | 数据集 CSV 路径（从头部读取 n/M/U） |
| `--protocols` | str[] | 全部4个 | `two_round`, `one_round`, `baseline_clip_M`, `baseline_random_tau` |
| `--base_protocol` | str | None | `FE1`（频率估计）或 `GKMPS`（求和） |
| `--epsilon` | float[] | `[1.0]` | 可多值 sweep |
| `--delta` | float | `1/n^2` | 隐私参数 `delta` |
| `--beta` | float | `0.1` | 失败概率 |
| `--gamma` | float | `0.3` | GKMPS 参数 |
| `--times` | int | `50` | 独立重复次数 |
| `--trim` | float | `0.2` | 截断均值比例 |
| `--seed` | int | `42` | 随机种子 |
| `--output` | str | None | 输出路径；使用其目录和扩展名，文件名自动标准化 |
| `--no_output_timestamp` | flag | False | 文件名中的时间部分写成 `timeNA` |
| `--random_select_times` | int | `10` | `baseline_random_tau` 内部随机次数（仅稳定统计） |
| `--simulate` | flag | False | 启用 `Simulator` 快速路径；默认是 faithful 执行 |
| `--no_simulate` | flag | False | 兼容开关；当前默认已经不启用 simulator |
| `--fe_c` | float | `1.0` | FE1 参数 c |
| `--fe_workers` | int | auto | FE1 analyzer worker 数 |
| `--fe_use_mu_search` | flag | False | 启用 FE1 `mu_search` |
| `--quick_fe1` | flag | False | FE1 常用预设 |

---

## 协议与接口

### 协议统一返回

四个协议统一返回三元组：

```text
(m_tau, query_result, nmessages)
```

### 协议对比（高层）

| 协议 | m_tau 来源 | 是否有 Round1 计数 | Query 部分 |
|---|---|---|---|
| `two_round` | GKMPS 私有估计 | 是 | 用估计的 `m_tau` 跑一次 |
| `one_round` | 一轮内联合估计 | 合并在一轮 | 为各候选阈值都构造 query，再选 `j*` |
| `baseline_clip_M` | 固定 `M` | 否 | 直接用 `m_tau=M` |
| `baseline_random_tau` | 随机阈值 | 否 | 随机多次取平均（预算不按 K 切分） |

### Query protocol 工厂

| Key | 工厂类 | 实现文件 | 用途 |
|---|---|---|---|
| `FE1` | `FE1Protocol` | `Base_Protocol/ShuffleFE-main/FE1.py` | 频率估计 |
| `GKMPS` | `GKMPSSumProtocol` | `static/GKMPS.py` | 求和估计 |

### 预算公式（实现口径）

| 协议 | per-record 预算 | 备注 |
|---|---|---|
| `two_round` | `eps/(2*m_tau)`, `delta/(2*m_tau)` | Round1 与 Round2 各占一半预算 |
| `one_round` | `eps/(2*2^j*(logM+1))`, `delta/(2*2^j*(logM+1))` | 每个候选子域一套 query 通道 |
| `baseline_clip_M` | `eps/M`, `delta/M` | 固定 `m_tau=M` |
| `baseline_random_tau` | `eps/m_tau`, `delta/m_tau` | `K` 仅用于统计稳定，不切分预算 |

---

## Algorithm Summary

### Two-round

1. Round1 在几何子域 `I_j=[2^(j-1)+1,2^j]` 上做计数，阈值筛出 `m_tau`。
2. Round2 将每个用户裁剪/填充到 `m_tau`，再调用 base protocol（FE1/GKMPS）。
3. 输出 `(m_tau, query_result, nmessages)`。

### One-round

1. 用户同时为所有候选 `2^j` 生成计数+query 消息。
2. faithful 实现中，消息会完整生成，但不再显式 shuffle。
3. 分析端先确定 `m_tau`，再只聚合 `j*=log2(m_tau)` 对应 query 通道。
4. 输出 `(m_tau, query_result, nmessages)`。

### Baseline 们

1. `baseline_clip_M`：直接令 `m_tau=M`。
2. `baseline_random_tau`：随机取 `m_tau`，重复 `K` 次求均值（预算不按 `K` 切分）。
3. 都输出 `(m_tau, query_result, nmessages)`。

---

## 数据格式

所有数据集（模拟 + 真实）统一输出为 `UserLevelDataset` CSV：

```text
# n=1000,M=64,U=10000000
# source=simulated,contrib_dist=zipf,value_dist=uniform,seed=42
12,8,19
7
...
```

规则：
- 第1行必须有 `n`, `M`, `U`。
- 第2行可选 metadata。
- 每个数据行对应一个用户，逗号分隔整数记录。
- 取值范围 `[0, U]`。

---

## 数据集特性与原始列使用

### 1) Simulated (`dataset/simulated_data/`)

- 生成入口：`generate_data.py`
- 关键参数：`n, M, m_max, U, contrib_dist, value_dist, seed`
- 常用语义：
  - `M`：协议公开上界
  - `m_max`：生成数据中的真实贡献上限（可小于 `M`）
  - `zipf` 贡献分布默认使用折中的 shifted Zipf：`Pr[m_i=x] ∝ (x+1.36)^{-6}`，用于在名义大支撑上保持较小的经验最大贡献
  - 如果要回到旧的标准 Zipf，可显式传 `--zipf_shift 0 --contrib_alpha 1.5`

支持的贡献分布（`contrib_dist`）：
- `uniform_fixed`, `uniform_random`, `zipf`, `geometric`, `gaussian`, `one_heavy`, `mixed`

支持的取值分布（`value_dist`）：
- `uniform`, `zipf`, `gaussian`, `constant`

### 2) AOL (`dataset/real_data/aol/`)

原始文件：`user-ct-test-collection-01.txt`

原始列（tab 分隔）：
- `AnonID`, `Query`, `QueryTime`, `ItemRank`, `ClickURL`

代码实际使用列：
- 用户ID：`AnonID`（按用户分组）
- 记录提取：优先 `ClickURL`，其次 `Query`（当 Query 看起来像域名）
- 其余列不参与编码

编码：域名前三字符 -> 24-bit 整数 -> `% (U+1)`。

### 3) MovieLens 32M (`dataset/real_data/ml-32m/`)

原始文件：`ratings.csv`
原始表头：`userId,movieId,rating,timestamp`

原始列：
- `userId`, `movieId`, `rating`, `timestamp`

代码实际使用列：
- 用户ID：`userId`
- 内容编码：`movieId`, `rating`
- `timestamp` 当前不参与编码

模式：
- `movie_rating`（默认）：`(movieId-1)*10 + half_star_index`
- `movie_id`：`movieId-1`

### 4) Netflix (`dataset/real_data/Netf/`)

原始结构：`training_set/mv_*.txt`（每个电影一个文件）
- 文件首行：`MovieID:`
- 后续行：`CustomerID,Rating,Date`

代码实际使用列：
- 用户ID：`CustomerID`
- 内容编码：
  - `movie_rating_date`（默认）：用 `MovieID + Rating + Date`
  - `movie_rating`：用 `MovieID + Rating`

---

## 调用链（run_experiment.py）

1. 解析 CLI 参数
2. `UserLevelDataset.load_csv()` 读取 `n/M/U` 与用户记录
3. 构造协议实例（四选一）
4. 如果有 `base_protocol`，通过工厂按运行时预算构造 FE1/GKMPS 实例
5. 重复 `times` 次收集指标
6. 计算 trimmed mean 指标（不输出 full error）
7. 输出到终端并可写 CSV/JSON

---

## 输出字段说明

输出每个协议一条记录，常用字段：

- `m_tau_mean`, `m_tau_median`, `m_tau_ge_mmax_ratio`
- `error_trimmed_mean`, `relative_error_trimmed_mean`
- `linf_error_trimmed_mean`
- `error_p50_trimmed_mean`, `error_p90_trimmed_mean`, `error_p95_trimmed_mean`, `error_p99_trimmed_mean`
- `msg_per_user_trimmed_mean`
- FE1 运行参数（若 base protocol = FE1）：
  - `fe1_utility_parameter`
  - `fe1_modular_size`
  - `fe1_big_prime`
  - `fe1_mu`
  - `fe1_sample_prob`
  - `fe1_collision_probability`
- `elapsed_sec`

其中 `relative_error_trimmed_mean = linf_error_trimmed_mean / U`。

CSV 列定义见 `static/run_experiment.py` 的 `CSV_COLUMNS`。

---

## 依赖

- Python 3.10+
- `numpy`
- `numba`（FE1 加速路径需要）

---

## 维护说明

每次改动以下任一项，都要同步更新本 README：
- 目录结构
- CLI 参数
- 默认行为（例如 FE1 `mu_search` 开关语义）
- 协议预算分配逻辑
- 数据处理字段/编码规则
