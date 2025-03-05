import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
import requests
from evaluate_strategy import evaluate_performance  # 假设这是一个独立的评估模块

# 设置随机种子以确保结果可重复
np.random.seed(42)

# FMP API 配置
API_KEY = "qhylk6wN8OUWTgmLddldoMRPCo59NmBU"
BASE_URL = "https://financialmodelingprep.com/api/v3/historical-price-full/GOOG"


# ==================================================
# 📌 1. 从 FMP API 获取数据
# ==================================================
def fetch_stock_data(symbol, api_key, start_date=None, end_date=None):
    url = f"{BASE_URL}?apikey={api_key}"
    if start_date and end_date:
        url += f"&from={start_date}&to={end_date}"
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"API 请求失败，状态码: {response.status_code}")
    data = response.json()
    if 'historical' not in data or not data['historical']:
        raise ValueError("未获取到历史数据！")

    # 转换为 DataFrame
    df = pd.DataFrame(data['historical'])
    df['date'] = pd.to_datetime(df['date'])
    df = df.rename(columns={'date': 'Date', 'close': 'Close'})
    df = df[['Date', 'Close']].set_index('Date').sort_index()
    return df


# 获取数据（从 2024-01-01 到 2025-02-24）
数据 = fetch_stock_data("GOOG", API_KEY, "2024-01-01", "2025-02-24")
数据 = 数据.ffill()  # 使用前向填充处理缺失值（移除警告）

# 定义关键参数
训练天数 = min(120, len(数据) - 30)  # 训练数据至少120天
预测天数 = 30  # 预测未来30天
结束日期 = pd.Timestamp("2025-01-02")  # 固定的训练结束日期

# 检查数据是否足够
if len(数据) < 训练天数 + 预测天数:
    raise ValueError(f"数据不足！现有: {len(数据)} 天，需: {训练天数 + 预测天数} 天")

# ==================================================
# 📌 2. 准备训练数据
# ==================================================
训练开始日期 = 结束日期 - pd.DateOffset(days=训练天数)
训练数据 = 数据.loc[训练开始日期:结束日期, 'Close']

# 计算日收益率
收益率 = 训练数据.pct_change().dropna().values.flatten()
if len(收益率) == 0:
    raise ValueError("收益率数据为空，无法训练 GARCH 模型！")

# ==================================================
# 📌 3. 训练 GARCH(1,1) 模型
# ==================================================
# 数据缩放：将收益率乘以100以优化 GARCH 参数估计
缩放因子 = 100
收益率_scaled = 收益率 * 缩放因子
garch模型 = arch_model(收益率_scaled, vol='Garch', p=1, q=1, dist='normal', rescale=False)  # 设置 rescale=False 以避免自动缩放
garch拟合 = garch模型.fit(disp="off")  # 关闭训练过程的输出

# 预测未来30天的波动率，并恢复原始规模
预测结果 = garch拟合.forecast(horizon=预测天数)
波动率预测 = np.sqrt(预测结果.variance.values[-1, :]) / 缩放因子  # 除以缩放因子恢复原始规模

# 确保波动率预测的长度足够
if len(波动率预测) < 预测天数:
    波动率预测 = np.full(预测天数, np.mean(波动率预测))

# ==================================================
# 📌 4. 使用几何布朗运动 (GBM) 模拟未来股价
# ==================================================
平均收益率 = 收益率.mean()  # 历史平均收益率（无需缩放，因为用于 GBM）
初始股价 = 训练数据.iloc[-1]  # 使用最后一天的收盘价作为初始值
模拟股价 = [初始股价]

# 模拟未来30天的股价
for t in range(预测天数):
    波动率 = 波动率预测[t]  # 当天的波动率
    随机扰动 = np.random.normal(0, 1)  # 生成标准正态分布的随机数
    新股价 = 模拟股价[-1] * np.exp((平均收益率 - 0.5 * 波动率 ** 2) + 波动率 * 随机扰动)  # GBM公式
    模拟股价.append(新股价)

预测股价 = np.array(模拟股价[1:])  # 去掉初始股价，仅保留预测结果

# ==================================================
# 📌 5. 获取真实股价用于对比
# ==================================================
测试开始日期 = 结束日期 + pd.DateOffset(days=1)
测试结束日期 = 结束日期 + pd.DateOffset(days=预测天数)
测试数据 = 数据.loc[测试开始日期:测试结束日期, 'Close']

# 如果测试数据不足30天，用NaN填充
if len(测试数据) < 预测天数:
    测试数据 = pd.Series(
        np.concatenate([测试数据.values, [np.nan] * (预测天数 - len(测试数据))]),
        index=pd.date_range(start=测试开始日期, periods=预测天数)
    )

# ==================================================
# 📌 6. 绘制预测股价与真实股价对比图
# ==================================================
plt.figure(figsize=(12, 6))
plt.plot(测试数据.index, 测试数据.values, label='真实股价', color='green', marker='o')
plt.plot(pd.date_range(start=测试开始日期, periods=预测天数), 预测股价,
         label='预测股价', color='red', linestyle='dashed', marker='x')
plt.title("股价预测: GARCH + GBM")
plt.xlabel("日期")
plt.ylabel("股价")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.show()