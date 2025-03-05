import requests
import pandas as pd
import numpy as np
from pmdarima import auto_arima
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Layer, Input
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import tensorflow as tf


# 自定义 Transformer 层
class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential(
            [Dense(ff_dim, activation="relu"), Dense(embed_dim)]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


# 1. 从 FMP 获取历史股价数据
def get_fmp_data(api_key, symbol="GOOG"):
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?apikey={api_key}"
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data["historical"])
    df["date"] = pd.to_datetime(df["date"])
    df = df[["date", "close"]].sort_values("date")
    return df


# 2. 数据预处理：插值生成 5 分钟数据（取最后 100 天）
def interpolate_to_5min(df):
    df = df.tail(100)
    df.set_index("date", inplace=True)
    new_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq="5min")
    df = df.reindex(new_index).interpolate(method="linear")
    df.reset_index(inplace=True)
    df.columns = ["date", "close"]
    return df


# 3. ARIMA 模型预测
def arima_predict(data, steps=6):
    model = auto_arima(data, seasonal=False, suppress_warnings=True, max_p=5, max_q=5, max_d=2)
    model_fit = model.fit(data)
    forecast = model_fit.predict(n_periods=steps)
    return forecast


# 4. LSTM 模型预测（轻量化）
def lstm_predict(data, steps=6, look_back=20):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))

    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i - look_back:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)

    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    model = Sequential()
    model.add(LSTM(20, input_shape=(look_back, 1)))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=5, batch_size=32, verbose=1)

    last_sequence = scaled_data[-look_back:]
    forecast = []
    for _ in range(steps):
        x = np.reshape(last_sequence, (1, look_back, 1))
        pred = model.predict(x, verbose=0)
        forecast.append(pred[0, 0])
        last_sequence = np.append(last_sequence[1:], pred)

    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
    return forecast.flatten()


# 5. Transformer 模型预测（使用 Functional API）
def transformer_predict(data, steps=6, look_back=20):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))

    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i - look_back:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)

    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    inputs = Input(shape=(look_back, 1))
    transformer_output = TransformerBlock(embed_dim=1, num_heads=2, ff_dim=16)(inputs, training=True)
    pooled_output = layers.GlobalAveragePooling1D()(transformer_output)
    outputs = Dense(1)(pooled_output)
    model = Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=5, batch_size=32, verbose=1)

    last_sequence = scaled_data[-look_back:]
    forecast = []
    for _ in range(steps):
        x = np.reshape(last_sequence, (1, look_back, 1))
        pred = model.predict(x, verbose=0)
        forecast.append(pred[0, 0])
        last_sequence = np.append(last_sequence[1:], pred)

    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
    return forecast.flatten()


# 6. 组合预测（纯模型结果，不作假）
def combined_predict(arima_forecast, lstm_forecast, transformer_forecast):
    return (arima_forecast + lstm_forecast + transformer_forecast) / 3


# 7. 基于收益的风险控制策略（严格按你的描述）
def risk_control_strategy(prices, forecast):
    capital = 10000  # 初始资金
    position = 0  # 持仓（股票数量）
    equity_curve = [capital]
    profit_threshold = 0.05  # 收益阈值 5%

    for i in range(len(prices)):
        current_price = prices[i]

        if i < len(forecast):
            future_prices = forecast[i:]
            future_mean = np.mean(future_prices)
            expected_return = (future_mean - current_price) / current_price  # 预期收益率

            print(
                f"Step {i}: Current Price={current_price:.2f}, Future Mean={future_mean:.2f}, Expected Return={expected_return:.4f}, Position={position:.2f}")

            # 增加仓位：预期收益 > 5%
            if expected_return > profit_threshold and position == 0:
                investable_cash = capital  # 全仓买入
                position = investable_cash / current_price
                capital -= investable_cash
                print(f"  Buy: Invested {investable_cash:.2f}, Position={position:.2f}")

            # 减少仓位：预期收益 < 0%
            elif expected_return < 0 and position > 0:
                sell_amount = position * 0.5  # 卖出 50% 仓位
                capital += sell_amount * current_price
                position -= sell_amount
                print(f"  Sell: Sold {sell_amount:.2f}, Capital={capital:.2f}")

            # 其他情况：维持仓位
            else:
                print("  Hold: No action")

        equity = capital + position * current_price
        equity_curve.append(equity)

    # 平滑处理
    equity_curve = pd.Series(equity_curve).rolling(window=2, min_periods=1).mean().tolist()
    print("Equity Curve:", equity_curve[:-1])
    return equity_curve


# 主程序
if __name__ == "__main__":
    api_key = "qhylk6wN8OUWTgmLddldoMRPCo59NmBU"
    symbol = "GOOG"

    # 获取数据
    df = get_fmp_data(api_key, symbol)
    df_5min = interpolate_to_5min(df)
    prices = df_5min["close"].values

    # 分割训练数据和真实值
    train_data = prices[:-6]
    true_values = prices[-6:]

    # ARIMA 预测
    arima_forecast = arima_predict(train_data)

    # LSTM 预测
    lstm_forecast = lstm_predict(train_data)

    # Transformer 预测
    transformer_forecast = transformer_predict(train_data)

    # 组合预测
    combined_forecast = combined_predict(arima_forecast, lstm_forecast, transformer_forecast)

    # 时间轴：0-25 分钟（6 个点）
    time_axis = np.arange(0, 26, 5)

    # 用真实值回测（更贴近实际）
    all_prices = true_values

    # 风险控制策略
    equity_curve = risk_control_strategy(all_prices, combined_forecast)

    # 可视化
    plt.figure(figsize=(14, 10))

    # 上半部分：真实值与预测值对比
    plt.subplot(2, 1, 1)
    plt.plot(time_axis, true_values, label="True Values", marker="o")
    plt.plot(time_axis, combined_forecast, label="Predicted Values", linestyle="--", marker="x")
    plt.title(f"{symbol} 5-Minute Price Prediction (ARIMA + LSTM + Transformer)")
    plt.xlabel("Time (minutes)")
    plt.ylabel("Price")
    plt.legend()

    # 下半部分：资金曲线
    plt.subplot(2, 1, 2)
    plt.plot(time_axis, equity_curve[:-1], label="Equity Curve", color="green", marker="o")
    plt.title("Risk Control Strategy Equity Curve")
    plt.xlabel("Time (minutes)")
    plt.ylabel("Capital")
    plt.ylim(9950, 10050)  # y 轴范围缩小，突出真实变化
    plt.legend()

    plt.tight_layout()
    plt.show()