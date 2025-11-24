import pandas as pd
import numpy as np
import warnings
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']


warnings.filterwarnings('ignore')

class TSDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).unsqueeze(1)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc   = nn.Linear(hidden_size, 1)      # 输出下一根标准化后的 Close

    def forward(self, x):
        # x: (batch, 60, 6)
        out, _ = self.lstm(x)          # out: (batch, 60, hidden_size)
        out = out[:, -1, :]            # 取最后一根K线的输出
        out = self.fc(out)             # (batch, 1)
        return out


if __name__ == "__main__":
    df = pd.read_csv("SPY_5min_last_60days.csv", parse_dates=True)

    df['Datetime'] = pd.to_datetime(df.iloc[:, 0], utc=True).dt.tz_convert('America/New_York')

    df['Hours'] = df['Datetime'].dt.hour

    df['Hours'] = df['Hours'] - df['Hours'].min() + 1

    print(df)

    df = df.set_index('Datetime')


    print(f"total bars: {df.shape}")

    df['ema20'] = df['Close'].ewm(span=20, adjust=False).mean()


    df = df.dropna()

    feature_columns = ['Open', 'High', 'Low', 'Close', 'ema20', 'Hours']

    df = df[feature_columns]

    data = df.values.astype(np.float32)

    scaler = StandardScaler()

    data_scaled = scaler.fit_transform(data)
    joblib.dump(scaler, 'scaler_ema20_simple_lstm.pkl')

    SEQ_LEN = 30
    X, y = [], []

    for i in range(len(data_scaled) - SEQ_LEN):
        X.append(data_scaled[i:i + SEQ_LEN])
        y.append(data_scaled[i + SEQ_LEN, 3])

    X = np.array(X)
    y = np.array(y)

    sample_dates = df.index[SEQ_LEN:]

    val_days = 6
    test_days = 3

    unique_dates = np.unique(sample_dates.date)

    val_start_date = unique_dates[-(val_days + test_days)]
    test_start_date = unique_dates[-test_days]

    print(val_start_date)

    val_start_idx = sample_dates.get_loc(str(val_start_date)).start
    test_start_idx = sample_dates.get_loc(str(test_start_date)).start

    X_val = X[val_start_idx:test_start_idx]
    y_val = y[val_start_idx:test_start_idx]

    X_test = X[test_start_idx:]
    y_test = y[test_start_idx:]

    X_train = X[:val_start_idx]
    y_train = y[:val_start_idx]




    train_loader = DataLoader(TSDataset(X_train, y_train), batch_size=64, shuffle=True, drop_last=True)
    val_loader   = DataLoader(TSDataset(X_val,   y_val),   batch_size=64, shuffle=False)
    test_loader  = DataLoader(TSDataset(X_test,  y_test),  batch_size=64, shuffle=False)




    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # force cpu training
    device = torch.device("cpu")
    print(f"当前设备: {device}")   # 会输出 mps 或 cpu
    model = LSTMModel(input_size=len(feature_columns)).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    patience = 15
    best_val_loss = float('inf')
    wait = 0
    num_epochs = 300

    train_losses = []
    val_losses = []

    print("开始训练 LSTM...")

    for epoch in range(num_epochs):
        # ---------- 训练 ----------
        model.train()
        epoch_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 防梯度爆炸
            optimizer.step()

            epoch_loss += loss.item()

        train_loss = epoch_loss / len(train_loader)
        train_losses.append(train_loss)

        # ---------- 验证 ----------
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                pred = model(x_batch)
                val_loss += criterion(pred, y_batch).item()

        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)

        # ---------- 早停 ----------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_lstm_model.pth')   # 保存最优模型
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        if (epoch+1) % 20 == 0 or epoch < 10:
            print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Best: {best_val_loss:.6f}")

    # ==================== 4. 加载最优模型 + 测试集评估 ====================
    model.load_state_dict(torch.load('best_lstm_model.pth'))
    model.eval()

    # 预测并反标准化（还原真实价格）

    with torch.no_grad():
        test_pred_scaled = []
        test_true_scaled = []
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            pred = model(x_batch).cpu().numpy()
            test_pred_scaled.extend(pred.flatten())
            test_true_scaled.extend(y_batch.numpy().flatten())

    # 反标准化：只反第3列（Close）
    dummy = np.zeros((len(test_pred_scaled), len(feature_columns)))   # 5列全0
    dummy[:, 3] = test_pred_scaled                 # 只把 Close 列填上预测的标准化值

    test_pred_price = scaler.inverse_transform(dummy)[:, 3]  # 取出真实价格

    dummy = np.zeros((len(test_true_scaled), len(feature_columns)))   # 5列全0
    dummy[:, 3] = test_true_scaled                 # 只把 Close 列填上预测的标准化值

    test_true_price = scaler.inverse_transform(dummy)[:, 3]  # 取出真实价格


    rmse = mean_squared_error(test_true_price, test_pred_price)
    mae  = mean_absolute_error(test_true_price, test_pred_price)

    print(f"\n测试集表现（真实价格）：")
    print(f"RMSE: {rmse:.4f} 美元")
    print(f"MAE : {mae:.4f} 美元")

    # ==================== 5. 可视化最后100根预测 vs 真实 ====================
    plt.figure(figsize=(15,6))
    plt.plot(test_true_price[-100:], label='真实收盘价', linewidth=2)
    plt.plot(test_pred_price[-100:], label='LSTM预测', alpha=0.8)
    plt.title('LSTM 预测 vs 真实价格（最后100根5分钟K线）')
    plt.xlabel('时间步')
    plt.ylabel('SPY 价格')
    plt.legend()
    plt.grid(True)
    plt.show()



