import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from enchance_lstm_model_trainer import MicroLSTM
import argparse

# ------------------- 参数 -------------------
SEQ_LEN   = 20
HORIZON   = 5
THRESHOLD = 0.002         # 和你训练时保持一致！



# ------------------- 特征生成（和训练时100%一致） -------------------
def make_ema20_features(df):
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

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(features)
    return torch.FloatTensor(X_scaled), scaler

# ------------------- 主程序 -------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('csv_path',type=str)
    parser.add_argument('model_path', type=str)
    parser.add_argument('from_index', type=int)
    parser.add_argument('to_index', type=int)

    args = parser.parse_args()

    csv_path = args.csv_path
    model_path = args.model_path
    from_index = args.from_index
    to_index = args.to_index

    # 1. 读取数据
    print(f"正在读取数据：{csv_path}")
    df = pd.read_csv(csv_path, parse_dates=True)
    df = df.sort_index()
    df = df[['Open','High','Low','Close','Volume','Datetime']].dropna()
    print(f"共加载 {len(df)} 根K线，最后时间：{df.index[-1]}")

    # 2. 生成特征
    X, scaler = make_ema20_features(df)

    # 3. 加载模型
    device = torch.device('mps' if torch.mps.is_available() else 'cpu')
    model = MicroLSTM(X.shape[-1]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"模型加载成功：{model_path}")

    # 4. 取最新一段进行预测
    if len(X) < SEQ_LEN:
        print(f"数据太短，至少需要 {SEQ_LEN} 根K线！")
        sys.exit(1)

    latest_seq = X[from_index:to_index].unsqueeze(0).to(device)   # shape: (1, 60, 4)
    print(df.iloc[to_index])

    with torch.no_grad():
        logits = model(latest_seq)
        prob = torch.softmax(logits, dim=1)[0]          # [做空概率, 做多概率]

    short_rate = prob[0].item() * 100
    long_rate  = prob[1].item() * 100

    print("\n" + "="*50)
    print(f"   实时信号（预测未来 {HORIZON} 根K线）")
    print("="*50)
    print(f"   做多胜率： {long_rate:6.2f}%")
    print(f"   做空胜率： {short_rate:6.2f}%")
    print("="*50)

    if long_rate >= 58:
        print("   强烈看多！建议立即做多")
    elif short_rate >= 58:
        print("   强烈看空！建议立即做空")
    elif long_rate >= 53:
        print("   轻度看多，可轻仓做多")
    elif short_rate >= 53:
        print("   轻度看空，可轻仓做空")
    else:
        print("   震荡行情，建议观望")


