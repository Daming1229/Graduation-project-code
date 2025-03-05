import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model

# ==================================================
# 📌 1. 加载并预处理数据
# ==================================================
file_path = '/Users/quyou/Desktop/毕业设计/GOOG_stock_data.csv'
data = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
data = data.ffill()  # ✅ 处理缺失值

# 选取最近 6 个月数据
train_days = 180
future_steps = 30
end_date = data.index[-1]
train_start_date = end_date - pd.DateOffset(days=train_days)
train_data = data.loc[train_start_date:end_date, 'Close']

# 计算对数收益率
returns = np.log(train_data / train_data.shift(1)).dropna().values.flatten()
returns_scaled = returns * 100  # ✅ 放大 100 倍，防止 GARCH 训练不稳定

# ==================================================
# 📌 2. 训练 GARCH(1,1) 并预测未来波动率
# ==================================================
garch_model = arch_model(returns_scaled, vol='Garch', p=1, q=1, dist='normal')
garch_fit = garch_model.fit(disp="off")

forecast = garch_fit.forecast(horizon=future_steps)
vol_forecast = np.sqrt(forecast.variance.values[-1, :]) / 100  # ✅ 还原尺度

# ==================================================
# 📌 3. 使用几何布朗运动 (GBM) 生成未来股价
# ==================================================
mu = returns.mean()
S0 = train_data.iloc[-1]
np.random.seed(42)

simulated_prices = [S0]
for t in range(future_steps):
    sigma = max(0.05, vol_forecast[t])  # ✅ 确保波动率足够大
    epsilon = np.random.normal(0, 1)
    S_t = simulated_prices[-1] * np.exp((mu - 0.5 * sigma**2) + sigma * epsilon)
    simulated_prices.append(S_t)

predicted_prices = np.array(simulated_prices[1:])
predicted_returns = np.diff(predicted_prices) / predicted_prices[:-1]

# ==================================================
# 📌 4. 风险控制策略（动态调整仓位）
# ==================================================
class RiskControlStrategy:
    def __init__(self, initial_capital=10000, risk_per_trade=0.3, max_position=50, threshold_up=0.002, threshold_down=-0.002):
        """
        initial_capital: 初始资金
        risk_per_trade: 每次交易的资金占比
        max_position: 最大持仓单位
        threshold_up: 预测收益高于该值时增加仓位
        threshold_down: 预测收益低于该值时减少仓位
        """
        self.capital = initial_capital  # 资金
        self.position = 0  # 当前持仓
        self.max_position = max_position  # 持仓上限
        self.risk_per_trade = risk_per_trade  # ✅ 新增这个变量，防止 AttributeError
        self.threshold_up = threshold_up  # 增加仓位的收益阈值
        self.threshold_down = threshold_down  # 减少仓位的收益阈值
        self.equity_curve = []  # 资金曲线
        self.price_curve = []  # 价格曲线

    def execute_trade(self, price, predicted_return):
        """执行交易"""
        trade_amount = self.capital * self.risk_per_trade  # ✅ 交易资金

        if predicted_return > self.threshold_up:
            # 预测收益高，增加仓位
            units = min(self.max_position, (trade_amount / price) * predicted_return * 50)  # ✅ 放大买入量
            self.position += units
            self.capital -= units * price
        elif predicted_return < self.threshold_down and self.position > 0:
            # 预测收益低，减少仓位
            sell_units = self.position * 0.5  # ✅ 逐步减仓
            self.capital += sell_units * price
            self.position -= sell_units

        # 计算总资产（现金 + 持仓价值）
        total_equity = self.capital + self.position * price
        self.equity_curve.append(total_equity)
        self.price_curve.append(price)

    def run(self, predicted_prices, predicted_returns):
        """运行交易策略"""
        for i in range(len(predicted_prices) - 1):
            self.execute_trade(predicted_prices[i], predicted_returns[i])

    def plot_equity_curve(self, strategy_dates):
        """绘制资金曲线"""
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.plot(strategy_dates[:len(self.equity_curve)], self.equity_curve, label="Equity Curve", color='blue', linewidth=2)
        plt.plot(strategy_dates[:len(self.price_curve)], self.price_curve, label="Stock Price", color='gray', linestyle='dashed')

        plt.title("Equity Curve with Risk Control Strategy")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.show()

# ==================================================
# 📌 5. 运行策略并绘制资金曲线
# ==================================================
strategy_dates = pd.date_range(start=end_date + pd.DateOffset(days=1), periods=future_steps)
strategy = RiskControlStrategy()
strategy.run(predicted_prices, predicted_returns)
strategy.plot_equity_curve(strategy_dates)