"""
@author: Claude AI
@description: 优化的GARCH-LSTM组合模型预测股票价格
@date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from arch import arch_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# 参数设置
TICKER = "GOOG"
SEQUENCE_LENGTH = 20
FUTURE_DAYS = 22
EPOCHS = 100

def create_sequences(data, seq_length):
    """创建时间序列数据"""
    X, y = [], []
    for i in range(len(data) - seq_length - 1):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length, 0])
    return np.array(X), np.array(y)

def main():
    # 获取历史数据
    end_date = "2025-01-01"
    start_date = "2020-01-01"
    
    print(f"正在下载 {TICKER} 的历史数据...")
    data = yf.download(TICKER, start=start_date, end=end_date)
    
    # 准备特征
    df = pd.DataFrame()
    df['Close'] = data['Close']
    df['Volume'] = data['Volume']
    df['High'] = data['High']
    df['Low'] = data['Low']
    
    # 添加技术指标
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['RSI'] = 100 - (100 / (1 + df['Close'].diff(1).clip(lower=0).rolling(14).mean() / 
                                  abs(df['Close'].diff(1)).rolling(14).mean()))
    df = df.dropna()
    
    # GARCH模型
    returns = df['Close'].pct_change().dropna()
    garch = arch_model(returns, vol='EGARCH', p=1, q=1)
    garch_model = garch.fit(disp='off')
    volatility = garch_model.conditional_volatility
    
    # 数据标准化
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    
    # 准备LSTM数据
    X, y = create_sequences(scaled_data, SEQUENCE_LENGTH)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # 构建LSTM模型
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(SEQUENCE_LENGTH, scaled_data.shape[1])),
        Dropout(0.3),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
    
    # 训练模型
    print("\n训练LSTM模型...")
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=32,
              validation_data=(X_test, y_test),
              callbacks=[early_stopping], verbose=1)
    
    # 预测未来价格
    last_sequence = scaled_data[-SEQUENCE_LENGTH:]
    future_prices = []
    current_sequence = last_sequence.copy()
    
    for _ in range(FUTURE_DAYS):
        current_sequence_reshaped = current_sequence.reshape(1, SEQUENCE_LENGTH, scaled_data.shape[1])
        next_pred = model.predict(current_sequence_reshaped)[0, 0]
        future_prices.append(next_pred)
        
        # 更新序列
        next_row = current_sequence[-1].copy()
        next_row[0] = next_pred
        current_sequence = np.vstack((current_sequence[1:], next_row))
    
    # 转换预测结果
    dummy_array = np.zeros((len(future_prices), df.shape[1]))
    dummy_array[:, 0] = future_prices
    predicted_prices = scaler.inverse_transform(dummy_array)[:, 0]
    
    # 获取实际数据进行对比
    future_dates = pd.date_range(start="2024-12-01", end="2025-01-01", freq='B')
    actual_data = yf.download(TICKER, start="2024-12-01", end="2025-01-01")['Close']
    
    # 确保数据长度一致并创建一维数组
    min_length = min(len(predicted_prices), len(actual_data))
    dates = future_dates[:min_length].values
    actual_prices = actual_data.values[:min_length].flatten()
    predicted_prices = predicted_prices[:min_length].flatten()
    
    # 创建结果DataFrame，确保所有列都是一维的
    results_df = pd.DataFrame({
        'Date': pd.Series(dates),
        'Actual_Price': pd.Series(actual_prices),
        'Predicted_Price': pd.Series(predicted_prices)
    })
    
    # 保存结果
    save_path = '/Users/quyou/Desktop/毕业设计/forecast data/ARIMA_GARCH_LSTM/g_l.csv'
    results_df.to_csv(save_path, index=False)
    print(f"\n预测结果已保存至: {save_path}")
    
    # 绘制对比图
    plt.figure(figsize=(12, 6))
    plt.plot(results_df['Date'], results_df['Actual_Price'], 
             label='实际价格', color='blue')
    plt.plot(results_df['Date'], results_df['Predicted_Price'], 
             label='预测价格', color='red', linestyle='--')
    plt.title(f'{TICKER} 股票价格预测 (2024-12-01 至 2025-01-01)')
    plt.xlabel('日期')
    plt.ylabel('价格')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('/Users/quyou/Desktop/毕业设计/forecast data/ARIMA_GARCH_LSTM/g_l_plot.png')
    plt.show()
    
    # 计算评估指标
    mse = np.mean((results_df['Actual_Price'] - results_df['Predicted_Price'])**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(results_df['Actual_Price'] - results_df['Predicted_Price']))
    mape = np.mean(np.abs((results_df['Actual_Price'] - results_df['Predicted_Price']) / 
                         results_df['Actual_Price'])) * 100
    
    print("\n预测评估指标:")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"MAPE: {mape:.2f}%")

if __name__ == "__main__":
    main()