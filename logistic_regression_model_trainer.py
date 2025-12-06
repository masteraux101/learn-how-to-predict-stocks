import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import argparse

# --- 配置参数 (5分钟K线适用) ---
CSV_FILE_PATH = "SPY_5min_last_60days.csv" # 您的连续 K 线数据文件路径
TRAIN_TEST_SPLIT_RATIO = 0.015  # 训练集划分比例
RANDOM_SEED = 42
BARS_1_HOUR = 12   # 1小时动量
BARS_4_HOURS = 48  # 4小时波动性
FEATURE_NAMES = ['Momentum_12_Bar', 'Reversal_1_Bar', 'Volatility_48_Bar', 'Bar_Range_Ratio', 'Bar_Close_Position']

def load_data(file_path):
    """加载 OHLC 数据并按日期排序。"""
    data = pd.read_csv(file_path, index_col='Datetime', parse_dates=True)
    data.sort_index(inplace=True)
    return data[['Open', 'High', 'Low', 'Close']]

def create_features_and_label(df):
    """创建适用于5分钟K线的特征和目标变量。"""
    df_copy = df.copy()

    # 1. 目标变量 (Y): 下一根K线是否上涨 (1:上涨, 0:下跌/持平)
    df_copy['Target_Next_Bar_Up'] = (df_copy['Close'].shift(-1) > df_copy['Close']).astype(int)

    # 2. 核心收益率
    df_copy['Log_Return'] = np.log(df_copy['Close'] / df_copy['Close'].shift(1))

    # --- 特征工程 ---
    df_copy['Momentum_12_Bar'] = df_copy['Log_Return'].rolling(window=BARS_1_HOUR).sum()
    df_copy['Reversal_1_Bar'] = df_copy['Log_Return'].shift(1)
    df_copy['Volatility_48_Bar'] = df_copy['Log_Return'].rolling(window=BARS_4_HOURS).std()

    # OHLC 特征
    range_diff = df_copy['High'] - df_copy['Low']
    df_copy['Bar_Range_Ratio'] = range_diff / df_copy['Close'].shift(1)
    df_copy['Bar_Close_Position'] = np.where(range_diff == 0, 0.5, (df_copy['Close'] - df_copy['Low']) / range_diff)

    # 清理缺失值
    df_copy.dropna(subset=FEATURE_NAMES + ['Target_Next_Bar_Up'], inplace=True)

    X = df_copy[FEATURE_NAMES]
    Y = df_copy['Target_Next_Bar_Up']

    return X, Y

def train_and_predict_latest(X, Y, offset):
    """训练模型，并使用最新的特征向量进行预测。"""

    # 1. 划分数据
    # X_train_full 是训练集（所有数据，不包含用于预测的 Target，但包含用于计算特征的X）
    X_train_full = X.iloc[:-1]
    Y_train_full = Y.iloc[:-1]

    # 2. 进一步划分训练集 (如果需要)
    # 假设我们只使用总训练集的前80%进行模型训练
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_train_full, Y_train_full, test_size=TRAIN_TEST_SPLIT_RATIO, shuffle=False, random_state=RANDOM_SEED
    )

    # 3. 训练标准化器 (仅在 X_train 上拟合)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # 4. 训练逻辑回归模型
    model = LogisticRegression(random_state=RANDOM_SEED, solver='liblinear', penalty='l2')
    model.fit(X_train_scaled, Y_train)

    X_predict = X_test.iloc[[offset]]

    # 5. 准备预测数据
    # 对预测向量进行标准化 (使用训练集的scaler)
    X_predict_scaled = scaler.transform(X_predict)

    # 6. 概率预测
    predicted_proba = model.predict_proba(X_predict_scaled)[:, 1][0]
    latest_date = X_predict.index[-1].strftime('%Y-%m-%d %H:%M:%S')

    return predicted_proba, latest_date

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('train_percent', type=float)
    parser.add_argument('bar_offset', type=int)

    args = parser.parse_args()

    TRAIN_TEST_SPLIT_RATIO = args.train_percent


    print("--- 启动模型训练和最新 K 线预测 ---")

    # 1. 加载数据
    try:
        data = load_data(CSV_FILE_PATH)
    except Exception as e:
        print(f"致命错误：加载数据失败。请检查文件路径和格式是否正确。详细信息: {e}")
        exit()

    # 2. 特征工程和标签定义
    X, Y = create_features_and_label(data)

    # 3. 训练和预测
    predicted_probability, latest_date = train_and_predict_latest(X, Y, args.bar_offset)

    # 4. 最终输出
    print("\n==============================================")
    print(f"基于 {latest_date} 的 K 线收盘数据：")
    print(f"📈 下一根 K 线 (5分钟) 上涨的预测概率 (P(Up)): **{predicted_probability:.2%}**")
    print("==============================================")

    if predicted_probability > 0.60:
        print(">>> 交易信号：看涨 (BUY)")
    elif predicted_probability < 0.40:
        print(">>> 交易信号：看跌 (SELL)")
    else:
        print(">>> 交易信号：观望 (HOLD)")
