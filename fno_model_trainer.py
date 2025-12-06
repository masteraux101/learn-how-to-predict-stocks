import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import fft, ifft
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

# Step 1: 获取 S&P 500 ETF 数据
def fetch_data():
    spy_data = yf.download('SPY', start='2020-01-01', end='2025-12-07')
    spy_data.to_csv('spy_data.csv')  # 保存为 CSV
    prices = spy_data['Close'].values.astype(np.float32)
    print(f"数据长度: {len(prices)}")
    return prices

# Step 2: 数据预处理
def preprocess_data(prices, window_size=128, horizon=22):
    scaler = MinMaxScaler(feature_range=(0, 5))
    prices_scaled = scaler.fit_transform(prices.reshape(-1, 1)).flatten()

    def create_dataset(data, window_size, horizon):
        X, y = [], []
        for i in range(len(data) - window_size - horizon + 1):
            X.append(data[i:i + window_size])
            y.append(data[i + window_size:i + window_size + horizon])
        return np.array(X)[:, None, :], np.array(y)[:, None, :]

    X, y = create_dataset(prices_scaled, window_size, horizon)

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(X_train.shape, X_test.shape)

    print(f"训练样本: {len(X_train)}, 测试样本: {len(X_test)}")
    return X_train, X_test, y_train, y_test, scaler, window_size, horizon

# Step 3: FNO 模型定义
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.weights1 = nn.Parameter(torch.zeros(in_channels, out_channels, modes1, dtype=torch.cfloat))

    def forward(self, x):
        x_ft = fft(x, dim=-1)
        out_ft = torch.zeros_like(x_ft, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = torch.einsum("bix,iox->box", x_ft[:, :, :self.modes1], self.weights1)
        x = ifft(out_ft, n=x.size(-1)).real
        return x

class FNO1d(nn.Module):
    def __init__(self, modes=16, width=64, in_channels=1, out_channels=1):
        super(FNO1d, self).__init__()
        self.modes = modes
        self.width = width

        self.fc0 = nn.Linear(in_channels, self.width)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes)

        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x, horizon):

        x = x.permute(0, 2, 1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x.permute(0, 2, 1)

        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)

        return x[:, :horizon, :]

# Step 4: 训练模型
def train_model(model, X_train_t, y_train_t, device, epochs=40, batch_size=32):
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_x, horizon=y_train_t.shape[2])
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {train_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), 'fno_spy_model.pth')

# Step 5: 预测和评估
def evaluate_model(model, X_test_t, y_test, scaler, window_size, horizon, device):
    model.eval()
    with torch.no_grad():
        pred = model(X_test_t.to(device), horizon=horizon).cpu().numpy()

    pred_inv = scaler.inverse_transform(pred.reshape(-1, 1)).reshape(-1, horizon)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1, horizon)

    last_input_scaled = X_test_t[-1, 0, :].cpu().numpy()                    # [128]
    last_input_real = scaler.inverse_transform(last_input_scaled.reshape(-1, 1)).flatten() 

    mse = np.mean((pred_inv - y_test_inv)**2)
    print(f"Test MSE: {mse:.2f}")

    plt.figure(figsize=(10, 5))
    plt.plot(range(window_size), last_input_real, label='历史输入')
    plt.plot(range(window_size, window_size + horizon), y_test_inv[-1], label='实际未来')
    plt.plot(range(window_size, window_size + horizon), pred_inv[-1], label='FNO 预测')
    plt.legend()
    plt.title('SPY 价格预测示例')
    plt.xlabel('时间步')
    plt.ylabel('价格')
    plt.show()

if __name__ == "__main__":
    prices = fetch_data()
    X_train, X_test, y_train, y_test, scaler, window_size, horizon = preprocess_data(prices)

    X_train_t = torch.from_numpy(X_train).float()
    y_train_t = torch.from_numpy(y_train).float()
    X_test_t = torch.from_numpy(X_test).float()
    y_test_t = torch.from_numpy(y_test).float()

    device = torch.device('cpu')
    model = FNO1d(modes=16, width=64)
    model.to(device)

    train_model(model, X_train_t, y_train_t, device)
    evaluate_model(model, X_test_t, y_test, scaler, window_size, horizon, device)