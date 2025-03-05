import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
import requests

# ==================================================
# 📌 1. 加载并预处理数据
# ==================================================
file_path = '/Users/quyou/Desktop/毕业设计/GOOG_stock_data.csv'
data = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
data = data.ffill()  # 处理缺失值

# ==================================================
# 📌 2. 获取市净率（PB）数据（从 FMP 获取）
# ==================================================
def get_pb_ratio_fmp(symbol, api_key):
    url = f"https://financialmodelingprep.com/api/v3/ratios/{symbol}?apikey={api_key}"
    try:
        response = requests.get(url)
        data = response.json()
        if isinstance(data, list) and len(data) > 0:
            pb_ratio = data[0].get('priceToBookRatio', None)
            if pb_ratio is not None:
                print(f"✅ {symbol} 的市净率 (PB Ratio): {pb_ratio}")
                return pb_ratio
        print(f"⚠️ 无法从 FMP 获取 {symbol} 的市净率 (PB Ratio)")
        return None
    except Exception as e:
        print(f"❌ 获取 PB Ratio 失败: {e}")
        return None

# 设置 FMP API 密钥
api_key = 'qhylk6wN8OUWTgmLddldoMRPCo59NmBU'

# 获取 GOOG 的市净率 (PB Ratio)
symbol = 'GOOG'
pb_ratio = get_pb_ratio_fmp(symbol, api_key)

# 如果未获取到市净率，则使用默认值 1
if pb_ratio is None:
    pb_ratio = 1  # 默认值

# 将市净率 (PB) 添加到数据中
data['PB'] = pb_ratio

# ==================================================
# 📌 3. 选取最近 6 个月数据
# ==================================================
train_days = 180
future_steps = 30
end_date = data.index[-1]
train_start_date = end_date - pd.DateOffset(days=train_days)
train_data = data.loc[train_start_date:end_date]

# 计算对数收益率
returns = np.log(train_data['Close'] / train_data['Close'].shift(1)).dropna().values.flatten()
returns_scaled = returns * 100  # 放大 100 倍，防止 GARCH 训练不稳定

# ==================================================
# 📌 4. 训练 GARCH(1,1) 并预测未来股价
# ==================================================
garch_model = arch_model(returns_scaled, vol='Garch', p=1, q=1, dist='normal')
garch_fit = garch_model.fit(disp="off")
forecast = garch_fit.forecast(horizon=future_steps)
vol_forecast = np.sqrt(forecast.variance.values[-1, :]) / 100  # 还原尺度

# 使用几何布朗运动 (GBM) 生成未来股价
mu = returns.mean()
S0 = train_data['Close'].iloc[-1]
np.random.seed(42)

simulated_prices = [S0]
for t in range(future_steps):
    sigma = max(0.05, vol_forecast[t])  # 确保波动率足够大
    epsilon = np.random.normal(0, 1)
    S_t = simulated_prices[-1] * np.exp((mu - 0.5 * sigma**2) + sigma * epsilon)
    simulated_prices.append(S_t)

predicted_prices = np.array(simulated_prices[1:])
predicted_returns = np.diff(predicted_prices) / predicted_prices[:-1]

# 计算未来 30 天的预期收益率
factor_price = predicted_returns.mean()

# ==================================================
# 📌 5. 计算市净率因子（PB Ratio）
# ==================================================
# 假设我们使用当前的 PB 比率作为因子
factor_pe = 1 / pb_ratio  # 低 PB 赋予更高权重（类似低 PE 更具投资价值）

# ==================================================
# 📌 6. 计算动量因子（Momentum）
# ==================================================
momentum_window = 20  # 计算过去 20 天的收益率
factor_momentum = (train_data['Close'].iloc[-1] / train_data['Close'].iloc[-momentum_window]) - 1

# ==================================================
# 📌 7. 计算综合因子评分
# ==================================================
# 归一化各因子权重
w_price, w_pe, w_momentum = 0.5, 0.3, 0.2  # 可调整权重

factor_score = (
    w_price * factor_price +
    w_pe * factor_pe +
    w_momentum * factor_momentum
)

# 输出选股结果
print(f"📊 综合因子评分: {factor_score:.4f}")
if factor_score > 0.01:
    print("✅ 建议买入该股票")
else:
    print("❌ 建议不买入")

# ==================================================
# 📌 8. 绘制未来股价与因子评分
# ==================================================
plt.figure(figsize=(12, 6))

# 绘制未来股价预测
plt.subplot(2, 1, 1)
plt.plot(predicted_prices, label="Predicted Stock Prices", color='blue', linestyle='dashed', marker='x')
plt.title("Predicted Stock Prices using GBM")
plt.xlabel("Days")
plt.ylabel("Price")
plt.legend()

# 绘制各因子的贡献值
plt.subplot(2, 1, 2)
plt.bar(["Price Factor", "PE Factor", "Momentum Factor"], [factor_price, factor_pe, factor_momentum], color=['blue', 'green', 'red'])
plt.title("Factors Used in Multi-Factor Model")
plt.ylabel("Factor Value")
plt.show()