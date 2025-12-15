import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import rfft, irfft
import joblib
import matplotlib.pyplot as plt
from fno_model_trainer import FNO1d

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

def load_model_and_scaler():
    """加载训练好的模型和scaler"""
    try:
        # 加载模型配置
        config = joblib.load('fno_spy_model_config.joblib')
        print(f"Model config loaded: {config}")
        
        # 初始化模型
        device = torch.device('mpu' if torch.cuda.is_available() else 'cpu')
        model = FNO1d(modes=config['modes'], width=config['width'])
        model.load_state_dict(torch.load('fno_spy_model.pth', map_location=device))
        model.to(device)
        model.eval()
        print("Model loaded successfully")
        
        # 加载scaler
        scaler = joblib.load('fno_spy_scaler.joblib')
        print("Scaler loaded successfully")
        
        return model, scaler, config, device
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run fno_model_trainer.py first to train and save the model")
        return None, None, None, None


def predict_from_csv(model, scaler, config, device, csv_path='SPY_5min_last_3days.csv'):
    """从CSV文件读取最后的数据点进行预测"""
    df = pd.read_csv(csv_path)
    prices = df['Close'].values.astype(np.float32)
    
    window_size = config['window_size']
    horizon = config['horizon']
    
    # 使用最后的window_size个价格进行预测
    if len(prices) < window_size:
        print(f"Error: CSV data has only {len(prices)} points, need at least {window_size}")
        return None
    
    # 获取最后的window_size个数据
    last_prices = prices[-window_size:]
    
    # 使用相同的scaler进行转换
    last_prices_scaled = scaler.transform(last_prices.reshape(-1, 1)).flatten()
    
    # 准备输入张量
    X = torch.from_numpy(last_prices_scaled).float().unsqueeze(0).unsqueeze(0)  # [1, 1, window_size]
    
    # 进行预测
    with torch.no_grad():
        prediction_scaled = model(X.to(device), horizon=horizon).cpu().numpy()
    
    # 反转缩放
    prediction = scaler.inverse_transform(prediction_scaled.reshape(-1, 1)).flatten()
    
    return last_prices, prediction, window_size, horizon


def predict_next_n_steps(model, scaler, config, device, csv_path='SPY_5min_last_10days.csv'):
    """预测未来N个时间步的股票价格"""
    last_prices, prediction, window_size, horizon = predict_from_csv(
        model, scaler, config, device, csv_path
    )
    
    if prediction is None:
        return
    
    # 获取最后的真实价格
    df = pd.read_csv(csv_path)
    last_price = df['Close'].values[-1]
    
    print(f"\n{'='*60}")
    print(f"FNO Model Prediction Results")
    print(f"{'='*60}")
    print(f"Last price in data: ${last_price:.2f}")
    print(f"Window size: {window_size}, Horizon: {horizon}")
    print(f"\nFuture {horizon} steps prediction:")
    print(f"{'-'*60}")
    
    for i, pred_price in enumerate(prediction[:horizon], 1):
        change = pred_price - last_price
        pct_change = (change / last_price) * 100
        print(f"Step {i:2d}: ${pred_price:.2f} (Change: ${change:+.2f}, {pct_change:+.2f}%)")
    
    # 绘制结果
    plt.figure(figsize=(12, 6))
    
    # 历史价格
    plt.plot(range(window_size), last_prices, 'b-', linewidth=2, label='Historical Data')
    
    # 预测价格
    plt.plot(range(window_size, window_size + horizon), prediction[:horizon], 
             'r--', linewidth=2, marker='o', label='FNO Prediction')
    
    plt.axvline(x=window_size - 1, color='gray', linestyle=':', alpha=0.7)
    plt.xlabel('Time Steps')
    plt.ylabel('Price ($)')
    plt.title('FNO Model: Historical Data and Future Prediction')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('fno_prediction_result.png', dpi=100)
    print(f"\n{'='*60}")
    print("Prediction plot saved as 'fno_prediction_result.png'")
    plt.show()


if __name__ == "__main__":
    print("Loading FNO model and scaler...")
    model, scaler, config, device = load_model_and_scaler()
    
    if model is None:
        exit(1)
    
    print("\nGenerating predictions...")
    predict_next_n_steps(model, scaler, config, device)
