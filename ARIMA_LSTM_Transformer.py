import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# 读取GOOG数据
data = pd.read_csv('/Users/quyou/Desktop/毕业设计/GOOG_data_alpha_vantage.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# 获取收盘价
close_prices = data['Close']

# 数据预处理（标准化）
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices.values.reshape(-1, 1))

# 创建训练集和测试集（80%训练，20%测试）
train_size = int(len(scaled_data) * 0.8)
train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]


# 创建简单的数据集类
class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + self.seq_length]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# 模型定义：简化版 Transformer + LSTM
class SimpleTransformerLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, num_heads=2):
        super(SimpleTransformerLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.transformer = nn.Transformer(d_model=hidden_size, nhead=num_heads, num_encoder_layers=num_layers,
                                          num_decoder_layers=num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        transformer_out = self.transformer(lstm_out, lstm_out)
        output = self.fc(transformer_out[:, -1, :])
        return output


# 设置超参数
seq_length = 60
input_size = 1
hidden_size = 64  # 可以调节
num_layers = 2
output_size = 1
learning_rate = 0.001
epochs = 100
batch_size = 32

# 创建数据集和数据加载器
train_dataset = TimeSeriesDataset(train_data, seq_length)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 初始化模型
model = SimpleTransformerLSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                              output_size=output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# 训练模型
def train_model(model, train_loader, epochs):
    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs = inputs.view(-1, seq_length, input_size)
            targets = targets.view(-1, output_size)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")


# 训练模型
train_model(model, train_loader, epochs)


# 预测函数
def predict(model, data, seq_length, steps=30):
    model.eval()
    predictions = []
    input_seq = data[-seq_length:]
    input_seq = torch.tensor(input_seq, dtype=torch.float32).view(1, seq_length, 1)

    with torch.no_grad():
        for _ in range(steps):
            output = model(input_seq)
            predictions.append(output.item())
            input_seq = torch.cat((input_seq[:, 1:, :], output.view(1, 1, 1)), dim=1)

    return predictions


# 预测未来30天
forecast_transformer = predict(model, scaled_data, seq_length, steps=30)

# 生成真实值：人为构造真实值数据（你可以换成实际的真实值）
real_prices = np.linspace(scaled_data[-30], scaled_data[-1], 30)

# 逆缩放预测结果和真实值
final_forecast = scaler.inverse_transform(np.array(forecast_transformer).reshape(-1, 1))
real_prices = scaler.inverse_transform(real_prices.reshape(-1, 1))

# 保存预测和真实值到CSV文件
forecast_dates = pd.date_range(start='2024-12-01', periods=30, freq='D')
forecast_df = pd.DataFrame({
    'Date': forecast_dates,
    'Forecast': final_forecast.flatten(),
    'Real': real_prices.flatten()
})
forecast_df.to_csv('/Users/quyou/Desktop/毕业设计/forecast data/ARIMA_GARCH_LSTM/t_a.csv', index=False)

# 计算 MAE, MSE, RMSE, MAPE
mae = mean_absolute_error(real_prices, final_forecast.flatten())
mse = mean_squared_error(real_prices, final_forecast.flatten())
rmse = np.sqrt(mse)
mape = np.mean(np.abs((real_prices - final_forecast.flatten()) / real_prices)) * 100

print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'MAPE: {mape}')

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(forecast_dates, final_forecast.flatten(), label='预测值')
plt.plot(forecast_dates, real_prices.flatten(), label='真实值', linestyle='--')
plt.legend()
plt.title('2024年12月 GOOG股价预测与真实值对比')
plt.xlabel('日期')
plt.ylabel('股价')
plt.grid(True)
plt.show()