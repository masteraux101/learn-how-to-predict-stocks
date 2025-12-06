import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from torch.utils.data import Dataset, DataLoader
import joblib


# ==================== 参数 ====================
SEQ_LEN   = 20     # k线数量
HORIZON   = 5      # 未来多少根k线
THRESHOLD = 0.002  # 多空阈值，BTC可调到0.003，A股0.006也行


def make_features_and_labels(df, dump=False):

    datetime = pd.to_datetime(df.iloc[:, 0], utc=True).dt.tz_convert('America/New_York')

    hours = datetime.dt.hour

    hours= hours - hours.min() + 1

    close = df['Close']
    ema20 = close.ewm(span=20, adjust=False).mean()

    features = pd.DataFrame({
        'ema20': ema20,
        'slope5': ema20.pct_change(5),               # 中期斜率
        'close': close,
        'hours': hours
    }).fillna(0)


    # 标签：未来HORIZON根收益率是否超过阈值
    future_ret = close.shift(-HORIZON) / close - 1
    labels = (future_ret > THRESHOLD).astype(int)     # 1=未来上涨，做多胜
    valid  = np.abs(future_ret) > THRESHOLD           # 过滤震荡样本

    scaler = RobustScaler()
    X = torch.FloatTensor(scaler.fit_transform(features))

    if dump:
        joblib.dump(scaler, "enhance_ema20_scaler.pkl")

    return X, torch.LongTensor(labels.values), torch.BoolTensor(valid.values), scaler

# ==================== 数据集 ====================
class EMA20Dataset(Dataset):
    def __init__(self, X, labels, valid):
        self.X = X
        self.labels = labels
        self.valid = valid

    def __len__(self):
        return len(self.X) - SEQ_LEN - HORIZON

    def __getitem__(self, idx):
        x = self.X[idx:idx+SEQ_LEN]
        y = self.labels[idx + SEQ_LEN]
        if not self.valid[idx + SEQ_LEN]:
            y = -1                                          # 无效样本标记
        return x, torch.tensor(y, dtype=torch.long)

class MicroLSTM(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 32, num_layers=2, batch_first=True)
        self.fc   = nn.Sequential(
            nn.Linear(32, 8), # 0=做空胜率, 1=做多胜率
            nn.ReLU(),
            nn.Linear(8, 2)
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])   # 只看最后一根

if __name__ == "__main__":

    df = pd.read_csv("SPY_5min_last_60days.csv", parse_dates=True)
    total_len = len(df)
    train_end   = int(total_len * 0.70)
    val_end     = int(total_len * 0.85)

    train_df = df.iloc[:train_end]
    val_df   = df.iloc[train_end:val_end]
    test_df  = df.iloc[val_end:]

    print(f"总数据: {len(df)} 根K线")
    print(f"训练集: {len(train_df)}  ({train_end/len(df)*100:.1f}%)")
    print(f"验证集: {len(val_df)}   ({(val_end-train_end)/len(df)*100:.1f}%)")
    print(f"测试集: {len(test_df)}   ({(len(df)-val_end)/len(df)*100:.1f}%)")

    # 分别生成特征（每段数据独立标准化！更严谨）
    X_train, y_train, v_train, scaler_train = make_features_and_labels(train_df, True)
    X_val,   y_val,   v_val,   _             = make_features_and_labels(val_df)
    X_test,  y_test,  v_test,  _             = make_features_and_labels(test_df)


    train_set = EMA20Dataset(X_train, y_train, v_train)
    val_set   = EMA20Dataset(X_val,   y_val,   v_val)
    test_set  = EMA20Dataset(X_test,  y_test,  v_test)

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_set,   batch_size=128, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_set,  batch_size=128, shuffle=False, drop_last=False)

    device = torch.device("cpu")
    model = MicroLSTM(X_train.shape[-1]).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0
    patience = 0
    max_patience = 20

    for epoch in range(1, 301):
        # ----- 训练 -----
        model.train()
        train_loss = correct = total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            mask = y != -1
            if not mask.any(): continue
            x, y = x[mask], y[mask]

            opt.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()

            train_loss += loss.item()
            correct += (pred.argmax(1) == y).sum().item()
            total += y.size(0)
        train_acc = correct / total

        # ----- 验证 -----
        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                mask = y != -1
                if not mask.any(): continue
                x, y = x[mask], y[mask]
                pred = model(x)
                val_correct += (pred.argmax(1) == y).sum().item()
                val_total += y.size(0)
        val_acc = val_correct / val_total

         # ----- 早停 -----
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_ema20_model.pth")  # 保存最优模型
            patience = 0
        else:
            patience += 1
            if patience >= max_patience:
                print(f"Early stopping at epoch {epoch}")
                break

        if epoch % 10 == 0 or epoch <= 100:
            print(f"Epoch {epoch:3d} | Train Acc: {train_acc*100:5.2f}% | Val Acc: {val_acc*100:5.2f}% {'← BEST' if patience==0 else ''}")

    # ==================== 加载最优模型并在测试集评估 ====================
    model.load_state_dict(torch.load("best_ema20_model.pth"))
    model.eval()
    test_correct = test_total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            mask = y != -1
            if not mask.any(): continue
            x, y = x[mask], y[mask]
            pred = model(x)
            test_correct += (pred.argmax(1) == y).sum().item()
            test_total += y.size(0)

    print(f"\n【最终结果】测试集准确率: {test_correct/test_total*100:.2f}%")

    # ==================== 实时预测最新胜率 ====================
    model.eval()
    with torch.no_grad():
        latest_seq = X_test[-SEQ_LEN:].unsqueeze(0).to(device)  # 用最新一段
        res = model(latest_seq)
        print(res)
        prob = torch.softmax(res, dim=1)[0]
        print(f"\n当前最新信号（未来{HORIZON}根K线）:")
        print(f"做多胜率: {prob[1].item()*100:5.2f}%")
        print(f"做空胜率: {prob[0].item()*100:5.2f}%")
