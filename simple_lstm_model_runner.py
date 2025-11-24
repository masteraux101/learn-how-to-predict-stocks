import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from simple_lstm_model_trainer import LSTMModel


device = torch.device('mps' if torch.mps.is_available() else 'cpu')

model = LSTMModel(input_size=5).to(device)          # ← input_size 改成你的特征数
model.load_state_dict(torch.load('best_lstm_model.pth', map_location=device))
model.eval()                                        # 必须 eval！

scaler = joblib.load('scaler_ema20_simple_lstm.pkl')             # ← 你保存时的名字

print("模型和标准化器加载成功！")

# ==================== 3. 实时预测函数（核心！） ====================


def predict_next_close(latest_60_bars: pd.DataFrame):
    """
    输入：最近60根5分钟K线（已经包含 ema20）
    输出：预测的下一根收盘价（真实美元价格）
    """
    # 1. 选特征（和你训练时完全一致！）
    features = latest_60_bars[['Open', 'High', 'Low', 'Close', 'ema20']].values.astype(np.float32)
    # 2. 标准化
    features_scaled = scaler.transform(features)                 # 注意：transform，不是fit！
    # 3. 增加 batch 维度
    x = torch.FloatTensor(features_scaled).unsqueeze(0).to(device)   # (1, 60, 5)
    # 4. 预测（关闭梯度）
    with torch.no_grad():
        pred_scaled = model(x).cpu().item()                      # 标准化后的预测值
    # 5. 反标准化 → 真实价格
    dummy = np.zeros((1, 5))                                     # 5列全0
    dummy[0, 3] = pred_scaled                                    # 只填 Close 列
    pred_price = scaler.inverse_transform(dummy)[0, 3]
    return round(pred_price, 4)

# ==================== 4. 实战使用示例 ====================
# 假设你已经从 yfinance 拿到最新数据
import yfinance as yf
spy = yf.Ticker("SPY")
latest_data = spy.history(period="60d", interval="5m", prepost=False)
latest_data = latest_data.between_time('9:30', '16:00')
latest_data['ema20'] = latest_data['Close'].ewm(span=20, adjust=False).mean()
latest_data = latest_data.dropna()

# 取最后60根
last_60 = latest_data.tail(60)

# 预测！
next_price = predict_next_close(last_60)
current_price = last_60['Close'].iloc[-1]

print(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"最新价格: {current_price:.4f}")
print(f"模型预测下一根(5分钟后)收盘价: {next_price:.4f}")
print(f"预测涨跌: {'↑' if next_price > current_price else '↓'} {abs(next_price - current_price):.4f}")
