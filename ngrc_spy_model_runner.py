import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from datetime import datetime, timedelta
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
warnings.filterwarnings('ignore')


class NGRCModel:
    """Next Generation Reservoir Computing Model
    
    NGRC的核心思想：
    1. 时间延迟嵌入 (Time-delay embedding) 创建虚拟储层
    2. 非线性变换扩展特征空间
    3. 线性读出层进行预测
    """
    
    def __init__(self, delay_length=50, degree=2, ridge_alpha=1e-6):
        """
        Args:
            delay_length: 时间延迟长度
            degree: 多项式特征的度数
            ridge_alpha: Ridge回归的正则化参数
        """
        self.delay_length = delay_length
        self.degree = degree
        self.ridge_alpha = ridge_alpha
        self.scaler = StandardScaler()
        self.readout = Ridge(alpha=ridge_alpha)
        self.is_trained = False
    
    def time_delay_embedding(self, data):
        """时间延迟嵌入 - NGRC的核心组件"""
        n_samples, n_features = data.shape
        
        # 根据数据长度自动调整延迟长度
        if n_samples < self.delay_length:
            # 如果数据长度不够，调整延迟长度为数据长度的一半（最小为5）
            adjusted_delay = max(5, min(n_samples // 2, 20))
            print(f"警告: 数据长度({n_samples})小于设定的延迟长度({self.delay_length})")
            print(f"自动调整延迟长度为: {adjusted_delay}")
            self.delay_length = adjusted_delay
        
        # 创建时间延迟矩阵
        embedded_data = []
        for i in range(n_samples - self.delay_length + 1):
            # 取当前时间窗口的所有特征
            window = data[i:i + self.delay_length].flatten()
            embedded_data.append(window)
        
        return np.array(embedded_data)
    
    def nonlinear_transform(self, embedded_data):
        """非线性变换 - 扩展特征空间"""
        # 1. 原始特征
        features = [embedded_data]
        
        # 2. 二次项 (如果degree >= 2)
        if self.degree >= 2:
            features.append(embedded_data ** 2)
        
        # 3. 交互项 (选择部分，避免维度爆炸)
        if self.degree >= 2:
            n_features = embedded_data.shape[1]
            # 只选择一些重要的交互项
            for i in range(0, min(n_features, 20), 5):  # 每隔5个特征
                for j in range(i+1, min(i+10, n_features)):  # 局部交互
                    features.append((embedded_data[:, i] * embedded_data[:, j]).reshape(-1, 1))
        
        # 4. 三角函数变换
        features.append(np.sin(embedded_data))
        features.append(np.cos(embedded_data))
        
        # 5. 双曲函数
        features.append(np.tanh(embedded_data))
        
        return np.concatenate(features, axis=1)
    
    def create_virtual_reservoir(self, data):
        """创建虚拟储层状态"""
        # 时间延迟嵌入
        embedded = self.time_delay_embedding(data)
        
        # 非线性变换
        reservoir_states = self.nonlinear_transform(embedded)
        
        return reservoir_states
    
    def prepare_training_data(self, data, target_col=3, lookback=1):
        """准备训练数据"""
        reservoir_states = self.create_virtual_reservoir(data)
        
        # 创建目标变量 (预测下一个时间步的收盘价)
        targets = []
        valid_indices = []
        
        for i in range(len(reservoir_states)):
            # 对应的原始数据索引
            original_idx = i + self.delay_length - 1
            # 预测下一个时间步
            if original_idx + lookback < len(data):
                targets.append(data[original_idx + lookback, target_col])
                valid_indices.append(i)
        
        return reservoir_states[valid_indices], np.array(targets)
    
    def fit(self, data):
        """训练NGRC模型"""
        X, y = self.prepare_training_data(data)
        
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        
        # 训练线性读出层
        self.readout.fit(X_scaled, y)
        
        self.is_trained = True
    
    def predict(self, data):
        """预测"""
        if not self.is_trained:
            raise ValueError("模型未训练!")
        
        reservoir_states = self.create_virtual_reservoir(data)
        X_scaled = self.scaler.transform(reservoir_states)
        predictions = self.readout.predict(X_scaled)
        
        return predictions


def load_model_and_predict(csv_file, model_path, n_predictions=20):
    """加载模型并进行预测"""
    print("加载NGRC模型...")
    
    # 加载模型
    ngrc = joblib.load(model_path)
    
    # 加载数据
    print("加载数据...")
    df = pd.read_csv(csv_file)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    
    # 自动检测时间间隔
    time_diff = (df['Datetime'].iloc[1] - df['Datetime'].iloc[0]).total_seconds() / 60
    time_interval_minutes = int(time_diff)
    
    if time_interval_minutes >= 60:
        time_label = f"{time_interval_minutes//60}小时"
    else:
        time_label = f"{time_interval_minutes}分钟"
    
    # 准备特征
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    data = df[features].values
    
    print(f"数据形状: {data.shape}")
    print(f"时间间隔: {time_label}")
    print(f"最后时间: {df['Datetime'].iloc[-1]}")
    print(f"最后收盘价: ${data[-1, 3]:.2f}")
    
    # 预测未来
    print(f"\n预测未来 {n_predictions} 个{time_label}K线...")
    
    # 使用最后的数据进行预测
    current_data = data[-ngrc.delay_length:].copy()
    predictions = []
    
    for step in range(n_predictions):
        # 预测下一步
        reservoir_states = ngrc.create_virtual_reservoir(current_data)
        X_scaled = ngrc.scaler.transform(reservoir_states)
        pred = ngrc.readout.predict(X_scaled)
        predictions.append(pred[-1])  # 取最后一个预测值
        
        # 更新数据窗口
        next_row = current_data[-1].copy()
        next_row[3] = pred[-1]  # 更新Close价格
        
        # 滚动窗口
        current_data = np.vstack([current_data[1:], next_row])
    
    # 生成时间序列
    last_time = df['Datetime'].iloc[-1]
    future_times = [last_time + timedelta(minutes=time_interval_minutes*(i+1)) for i in range(n_predictions)]
    
    return np.array(predictions), future_times, data, df, time_label


def plot_predictions(predictions, future_times, data, df, time_label, n_history=100):
    """绘制预测结果"""
    print("生成预测图表...")
    
    # 历史数据
    hist_data = data[-n_history:]
    hist_times = df['Datetime'].iloc[-n_history:]
    
    # 创建连续的索引
    hist_indices = np.arange(len(hist_data))
    pred_indices = np.arange(len(hist_data), len(hist_data) + len(predictions))
    
    # 绘制图表
    plt.figure(figsize=(15, 10))
    
    # 上图：价格走势
    plt.subplot(2, 1, 1)
    plt.plot(hist_indices, hist_data[:, 3], 'b-', linewidth=2, label='历史收盘价', alpha=0.8)
    plt.fill_between(hist_indices, hist_data[:, 2], hist_data[:, 1], alpha=0.2, color='blue', label='历史价格区间')
    
    plt.plot(pred_indices, predictions, 'r-', linewidth=2, label='预测收盘价', marker='o', markersize=4)
    
    plt.axvline(x=len(hist_data)-1, color='green', linestyle='--', linewidth=2, alpha=0.7, label='预测起点')
    
    # 设置x轴标签 - 选择关键时间点显示
    x_ticks = []
    x_labels = []
    
    # 历史数据的时间标签
    step = max(1, len(hist_data) // 5)  # 最多显示5个历史时间点
    for i in range(0, len(hist_data), step):
        x_ticks.append(i)
        x_labels.append(hist_times.iloc[i].strftime('%m-%d %H:%M'))
    
    # 预测数据的时间标签
    pred_step = max(1, len(predictions) // 4)  # 最多显示4个预测时间点
    for i in range(0, len(predictions), pred_step):
        x_ticks.append(len(hist_data) + i)
        x_labels.append(future_times[i].strftime('%m-%d %H:%M'))
    
    # 最后一个预测点
    if len(predictions) > 0:
        x_ticks.append(pred_indices[-1])
        x_labels.append(future_times[-1].strftime('%m-%d %H:%M'))
    
    plt.xticks(x_ticks, x_labels, rotation=45)
    plt.xlabel('时间')
    plt.ylabel('价格 ($)')
    plt.title(f'NGRC模型 - SPY {time_label}走势预测')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 下图：价格变化
    plt.subplot(2, 1, 2)
    price_changes = np.diff(predictions, prepend=hist_data[-1, 3])
    colors = ['green' if x > 0 else 'red' for x in price_changes]
    
    bars = plt.bar(pred_indices, price_changes, color=colors, alpha=0.7, edgecolor='black')
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    # 设置x轴标签与上图一致
    pred_x_ticks = []
    pred_x_labels = []
    for i in range(0, len(predictions), max(1, len(predictions)//6)):
        pred_x_ticks.append(len(hist_data) + i)
        pred_x_labels.append(f"步骤{i+1}")
    
    plt.xticks(pred_x_ticks, pred_x_labels, rotation=45)
    plt.xlabel('预测步数')
    plt.ylabel('价格变化 ($)')
    plt.title('预测价格变化')
    plt.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签 - 只在部分位置添加，避免过密
    for i, (idx, pred, change) in enumerate(zip(pred_indices, predictions, price_changes)):
        if i % max(1, len(predictions)//8) == 0:  # 每隔几个点标注一次
            height = change
            plt.text(idx, height,
                    f'${pred:.2f}',
                    ha='center', va='bottom' if height > 0 else 'top', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('ngrc_spy_predictions.png', dpi=300, bbox_inches='tight')
    print("预测图表已保存: ngrc_spy_predictions.png")
    plt.show()


def main():
    print("╔════════════════════════════════════════════════════════════════════════════════╗")
    print("║           Next Generation Reservoir Computing - SPY预测运行器                  ║")
    print("╚════════════════════════════════════════════════════════════════════════════════╝\n")
    
    # 可以根据需要修改不同时间间隔的数据文件
    # 例如：'SPY_5min_last_60days.csv', 'SPY_30min_data.csv', 'SPY_1h_data.csv'
    data_file = 'SPY_30min_last_30days.csv'
    
    # 进行预测
    predictions, future_times, data, df, time_label = load_model_and_predict(
        data_file,
        'ngrc_spy_model.joblib',
        n_predictions=20
    )
    
    # 打印预测结果
    print("\n" + "="*60)
    print(f"未来20个{time_label}K线预测结果")
    print("="*60)
    
    for i, (time, price) in enumerate(zip(future_times, predictions)):
        change = price - data[-1, 3] if i == 0 else price - predictions[i-1]
        direction = "↗" if change > 0 else "↘" if change < 0 else "→"
        print(f"{i+1:2d}. 第{i+1}步预测 | "
              f"${price:7.2f} | {direction} {change:+6.2f}")
    
    # 绘制结果
    plot_predictions(predictions, future_times, data, df, time_label)
    
    # 趋势分析
    total_change = predictions[-1] - data[-1, 3]
    trend = "上升" if total_change > 0 else "下降" if total_change < 0 else "平稳"
    
    print(f"\n趋势分析:")
    print(f"当前价格: ${data[-1, 3]:.2f}")
    print(f"预测结束价格: ${predictions[-1]:.2f}")
    print(f"总变化: {total_change:+.2f} ({total_change/data[-1, 3]*100:+.2f}%)")
    print(f"预测趋势: {trend}")
    
    print("\n预测完成!")


if __name__ == "__main__":
    main()
