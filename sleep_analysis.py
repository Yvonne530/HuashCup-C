import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon
import itertools

# ========================
# 1. 数据导入和预处理
# ========================
# 读取Excel文件
data = pd.read_excel("Problem4.xlsx", sheet_name="Sheet1", header=None)

# 跳过前两行（标题行）
data = data.iloc[2:]

# 创建列名：被试1_Night1, 被试1_Night2, 被试1_Night3, 被试2_Night1, ...
column_names = []
for subj in range(1, 12):
    for night in range(1, 4):
        column_names.append(f"被试{subj}_Night{night}")
data.columns = column_names

# 转换为数值类型
data = data.apply(pd.to_numeric, errors='coerce')

# AASM 睡眠阶段编码
WAKE = 4
N1_N2 = [1,2]  # 如果需要，可单独处理
N3 = 3
REM = 5

# 每条记录时间间隔 (秒)
interval_sec = 30

# ========================
# 2. 计算每个实验的指标
# ========================
def compute_sleep_metrics(series):
    """计算TST, SE, SOL, N3%, REM%, 醒来次数"""
    # 移除可能的NaN值
    series = series.dropna()
    
    total_records = len(series)
    # 总卧床时间
    total_bedtime_min = total_records * interval_sec / 60.0

    # 标记非清醒阶段
    asleep_mask = series != WAKE
    TST_min = asleep_mask.sum() * interval_sec / 60.0
    SE = TST_min / total_bedtime_min * 100.0 if total_bedtime_min > 0 else np.nan

    # SOL: 从开始到第一次非清醒阶段
    if asleep_mask.any():
        first_asleep_idx = asleep_mask.idxmax()  # pandas idx
        SOL_min = first_asleep_idx * interval_sec / 60.0
    else:
        SOL_min = np.nan

    # N3% 和 REM%
    N3_min = (series == N3).sum() * interval_sec / 60.0
    REM_min = (series == REM).sum() * interval_sec / 60.0
    N3_percent = N3_min / TST_min * 100 if TST_min>0 else np.nan
    REM_percent = REM_min / TST_min * 100 if TST_min>0 else np.nan

    # 夜间醒来次数: 清醒段连续数
    wake_seq = (series == WAKE).astype(int)
    wake_diff = wake_seq.diff().fillna(0)
    awakenings = (wake_diff == 1).sum()

    return pd.Series({
        "TST": TST_min,
        "SE": SE,
        "SOL": SOL_min,
        "N3%": N3_percent,
        "REM%": REM_percent,
        "Awakenings": awakenings
    })

# 对所有列计算指标
metrics_df = data.apply(compute_sleep_metrics).T

# ========================
# 3. 整理为重复测量数据
# ========================
# 假设 Night 1 = 环境A, Night 2 = 环境B, Night 3 = 环境C
# 转置为 被试 × 环境 的形式
def reshape_for_rm(metrics_df, metric_name):
    subjects = 11
    envs = ["A", "B", "C"]
    arr = np.zeros((subjects, len(envs)))
    for i in range(subjects):
        for j in range(len(envs)):
            col_name = f"被试{i+1}_Night{j+1}"
            arr[i,j] = metrics_df.loc[col_name, metric_name]
    return arr

# ========================
# 4. Friedman 检验 + Wilcoxon 事后检验
# ========================
def friedman_wilcoxon(metric_name):
    # 组织数据: subjects × conditions
    arr = reshape_for_rm(metrics_df, metric_name)

    # Friedman 检验
    stat, p = friedmanchisquare(arr[:,0], arr[:,1], arr[:,2])
    print(f"\n指标: {metric_name}")
    print(f"Friedman statistic={stat:.4f}, p-value={p:.4f}")

    # 事后 Wilcoxon 两两比较
    print("Wilcoxon 事后检验:")
    envs = ["A", "B", "C"]
    for i,j in itertools.combinations(range(3),2):
        stat_w, p_w = wilcoxon(arr[:,i], arr[:,j])
        print(f"  {envs[i]} vs {envs[j]}: statistic={stat_w}, p-value={p_w:.4f}")

# 执行分析
for metric in ["TST", "SE", "SOL", "N3%", "REM%", "Awakenings"]:
    friedman_wilcoxon(metric)