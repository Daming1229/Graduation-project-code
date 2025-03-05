import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model

# ==================================================
# 📌 1. 加载数据
# ==================================================
file_path = '/Users/quyou/Desktop/毕业设计/GOOG_stock_data.csv'
data = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')

# 📌 2. 填充缺失值（前向填充）
data = data.fillna(method='ffill')

# ==================================================
# 📌 3. 确保有足够的训练数据
# ==================================================
train_days = min(120, len(data) - 30)  # 训练集最少 120 天
future_steps = 30  # 预测未来 30 天

if len(data) < train_days + future_steps:
    raise ValueError(f"数据不足！可用: {len(data)}, 需要: {train_days + future_steps}")

# ==================================================
# 📌 4. 选择训练数据
# ==================================================
end_date = pd.Timestamp("2025-01-02")  # 固定预测结束日期
train_start_date = end_date - pd.DateOffset(days=train_days)
train_data = data.loc[train_start_date:end_date, 'Close']  # 选取训练数据

# ==================================================
# 📌 5. 计算收益率 (GARCH 需要用到)
# ==================================================
returns = train_data.pct_change().dropna().values.flatten()  # 计算对数收益率

if len(returns) == 0:
    raise ValueError("收益率数据为空，无法训练 GARCH 模型！")

# ==================================================
# 📌 6. 训练 GARCH(1,1) 模型
# ==================================================
garch_model = arch_model(returns, vol='Garch', p=1, q=1, dist='normal')
garch_fit = garch_model.fit(disp="off")  # 关闭训练输出信息

# ==================================================
# 📌 7. 预测未来 30 天的波动率
# ==================================================
forecast = garch_fit.forecast(horizon=future_steps)
vol_forecast = np.sqrt(forecast.variance.values[-1, :])  # 取波动率的平方根

# 确保波动率预测长度足够
if len(vol_forecast) < future_steps:
    vol_forecast = np.full(future_steps, np.mean(vol_forecast))

# ==================================================
# 📌 8. 使用几何布朗运动 (GBM) 预测未来价格
# ==================================================
mu = returns.mean()  # 计算历史平均收益率
S0 = train_data.iloc[-1]  # 设定初始股价（最后一天的收盘价）

np.random.seed(42)
simulated_prices = [S0]

# 依次预测未来 30 天的股价
for t in range(future_steps):
    sigma = vol_forecast[t] if t < len(vol_forecast) else vol_forecast[-1]  # 使用波动率预测值
    epsilon = np.random.normal(0, 1)  # 生成标准正态随机数
    S_t = simulated_prices[-1] * np.exp((mu - 0.5 * sigma**2) + sigma * epsilon)  # GBM 计算公式
    simulated_prices.append(S_t)

predicted_prices = np.array(simulated_prices[1:])  # 移除初始股价，仅保留预测值

# ==================================================
# 📌 9. 获取真实股价（用于对比）
# ==================================================
test_data = data.loc[end_date + pd.DateOffset(days=1):end_date + pd.DateOffset(days=future_steps), 'Close']

# 确保 test_data 长度匹配
if len(test_data) < future_steps:
    test_data = np.concatenate([test_data.values, [np.nan] * (future_steps - len(test_data))])

# ==================================================
# 📌 10. 绘制 预测 vs 真实股价 对比图
# ==================================================
plt.figure(figsize=(12, 6))
plt.plot(pd.date_range(start=end_date + pd.DateOffset(days=1), periods=future_steps), test_data, label='真实股价', color='green', marker='o')
plt.plot(pd.date_range(start=end_date + pd.DateOffset(days=1), periods=future_steps), predicted_prices, label='预测股价', color='red', linestyle='dashed', marker='x')

plt.title("股票价格预测: GARCH + GBM")
plt.xlabel("日期")
plt.ylabel("股价")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.show()

# ==================================================
# 🚀 **趋势跟踪交易策略**
# ==================================================

class TrendFollowingStrategy:
    def __init__(self, stock_data, model_predictions, initial_capital=10000, risk_per_trade=0.02):
        self.stock_data = stock_data.iloc[:len(model_predictions)]  # 确保数据对齐
        self.model_predictions = model_predictions
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.capital = initial_capital
        self.position = 0  # 当前持仓
        self.trade_history = []
        self.equity_curve = []

    def execute_trade(self, date, price, action):
        """执行交易操作"""
        trade_amount = self.capital * self.risk_per_trade  # 每次交易风险比例
        units = trade_amount / price  # 计算买入的股数

        if action == 'buy' and self.position == 0:
            self.position = units
            self.capital -= units * price
            self.trade_history.append((date, '买入', price, units))

        elif action == 'sell' and self.position > 0:
            self.capital += self.position * price
            self.position = 0
            self.trade_history.append((date, '卖出', price, units))

        # 记录资金变化
        self.equity_curve.append(self.capital + self.position * price)

    def run(self):
        """执行交易策略"""
        for i in range(len(self.stock_data)):
            date = self.stock_data.index[i]
            close_price = self.stock_data['Close'].iloc[i]
            prediction = self.model_predictions[i]

            if prediction == 1 and self.position == 0:
                self.execute_trade(date, close_price, 'buy')
            elif prediction == -1 and self.position > 0:
                self.execute_trade(date, close_price, 'sell')

        return pd.Series(self.equity_curve, index=self.stock_data.index[:len(self.equity_curve)])

    def plot_equity_curve(self):
        """绘制资金曲线"""
        equity_curve = self.run()
        plt.figure(figsize=(10, 6))
        plt.plot(equity_curve, label="资金曲线", color='blue')
        plt.title("交易策略: 资金曲线")
        plt.xlabel("日期")
        plt.ylabel("账户资金")
        plt.legend()
        plt.grid(True)
        plt.show()


# ==================================================
# 📌 12. 生成交易信号 (趋势跟踪)
# ==================================================
model_predictions = np.where(np.diff(predicted_prices) > 0, 1, -1)
model_predictions = np.append(model_predictions, model_predictions[-1])  # 补全长度

# ==================================================
# 📌 13. 运行趋势跟踪策略
# ==================================================
strategy_data = data.loc[end_date + pd.DateOffset(days=1):end_date + pd.DateOffset(days=future_steps)]
strategy = TrendFollowingStrategy(strategy_data, model_predictions)
strategy.plot_equity_curve()