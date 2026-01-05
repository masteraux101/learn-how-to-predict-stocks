import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib

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
        print("创建虚拟储层...")
        original_delay = self.delay_length
        X, y = self.prepare_training_data(data)
        
        if self.delay_length != original_delay:
            print(f"延迟长度已从 {original_delay} 调整为 {self.delay_length}")
        
        print(f"虚拟储层维度: {X.shape}")
        print(f"目标变量数量: {len(y)}")
        
        # 标准化特征
        print("标准化特征...")
        X_scaled = self.scaler.fit_transform(X)
        
        # 训练线性读出层
        print("训练线性读出层...")
        self.readout.fit(X_scaled, y)
        
        self.is_trained = True
        print("NGRC模型训练完成!")
    
    def predict(self, data):
        """预测"""
        if not self.is_trained:
            raise ValueError("模型未训练!")
        
        reservoir_states = self.create_virtual_reservoir(data)
        X_scaled = self.scaler.transform(reservoir_states)
        predictions = self.readout.predict(X_scaled)
        
        return predictions


def load_and_prepare_data(csv_file):
    """加载和准备SPY数据"""
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
    
    # 根据数据长度自动调整延迟长度
    data_length = len(df)
    if data_length >= 1000:
        optimal_delay_length = 50  # 长数据用50
    elif data_length >= 500:
        optimal_delay_length = 30  # 中长数据用30
    elif data_length >= 200:
        optimal_delay_length = 20  # 中等数据用20
    elif data_length >= 100:
        optimal_delay_length = 15  # 较短数据用15
    elif data_length >= 50:
        optimal_delay_length = 10  # 短数据用10
    else:
        optimal_delay_length = max(5, data_length // 5)  # 很短数据用最小值
    
    # 选择关键特征
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    data = df[features].values
    
    print(f"数据形状: {data.shape}")
    print(f"时间间隔: {time_label}")
    print(f"建议延迟长度: {optimal_delay_length}")
    print(f"时间范围: {df['Datetime'].min()} 到 {df['Datetime'].max()}")
    
    return data, df, time_label, optimal_delay_length


def split_data(data, test_ratio=0.2):
    """分割训练和测试数据"""
    n_total = len(data)
    n_train = int(n_total * (1 - test_ratio))
    
    train_data = data[:n_train]
    test_data = data[n_train:]
    
    print(f"总数据长度: {n_total}")
    print(f"训练集长度: {len(train_data)}")
    print(f"测试集长度: {len(test_data)}")
    
    return train_data, test_data


def evaluate_model(model, test_data):
    """评估模型性能"""
    print("\n评估模型...")
    
    # 准备测试数据
    X_test, y_test = model.prepare_training_data(test_data)
    
    # 预测
    y_pred = model.predict(test_data)
    
    # 确保长度一致
    min_len = min(len(y_test), len(y_pred))
    y_test = y_test[:min_len]
    y_pred = y_pred[:min_len]
    
    # 计算指标
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"测试集性能:")
    print(f"  MSE:  {mse:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE:  {mae:.6f}")
    print(f"  R²:   {r2:.6f}")
    
    return y_test, y_pred, {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}


def plot_results(y_test, y_pred, metrics, time_label):
    """绘制结果图表"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Next Generation Reservoir Computing - SPY {time_label}预测结果', fontsize=16, fontweight='bold')
    
    # 1. 预测 vs 实际
    ax1 = axes[0, 0]
    time_steps = np.arange(len(y_test))
    ax1.plot(time_steps, y_test, label='实际值', linewidth=2, alpha=0.8)
    ax1.plot(time_steps, y_pred, label='预测值', linewidth=2, alpha=0.8)
    ax1.set_xlabel('时间步')
    ax1.set_ylabel('价格')
    ax1.set_title('预测 vs 实际值')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 散点图
    ax2 = axes[0, 1]
    ax2.scatter(y_test, y_pred, alpha=0.5, s=20)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='完美预测线')
    ax2.set_xlabel('实际值')
    ax2.set_ylabel('预测值')
    ax2.set_title(f'预测精度 (R² = {metrics["r2"]:.4f})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 预测误差
    errors = y_pred - y_test
    ax3 = axes[1, 0]
    ax3.plot(time_steps, errors, label='预测误差', color='red', linewidth=1, alpha=0.7)
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax3.fill_between(time_steps, errors, 0, alpha=0.3, color='red')
    ax3.set_xlabel('时间步')
    ax3.set_ylabel('误差')
    ax3.set_title('预测误差')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 误差分布
    ax4 = axes[1, 1]
    ax4.hist(errors, bins=50, edgecolor='black', alpha=0.7, color='lightblue')
    ax4.axvline(x=np.mean(errors), color='red', linestyle='--', linewidth=2, 
                label=f'均值: {np.mean(errors):.4f}')
    ax4.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax4.set_xlabel('误差')
    ax4.set_ylabel('频率')
    ax4.set_title(f'误差分布 (RMSE = {metrics["rmse"]:.4f})')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('ngrc_spy_results.png', dpi=300, bbox_inches='tight')
    print("结果图表已保存: ngrc_spy_results.png")
    plt.show()


def predict_future(model, data, n_steps=20):
    """预测未来走势"""
    print(f"\n预测未来 {n_steps} 步...")
    
    # 使用最后的数据作为起点
    current_data = data[-model.delay_length:].copy()
    predictions = []
    
    for step in range(n_steps):
        # 预测下一步
        pred = model.predict(current_data.reshape(1, -1) if len(current_data.shape) == 1 
                            else current_data)
        predictions.append(pred[0])
        
        # 更新数据 - 简单地重复最后一行但替换Close价格
        next_row = current_data[-1].copy()
        next_row[3] = pred[0]  # Close价格在索引3
        
        # 滚动窗口
        current_data = np.vstack([current_data[1:], next_row])
    
    return np.array(predictions)


def plot_future_predictions(data, df, future_preds, time_label):
    """绘制未来预测"""
    # 取最后100个点用于展示
    n_history = min(100, len(data))
    hist_data = data[-n_history:]
    hist_times = df['Datetime'].iloc[-n_history:]
    
    # 创建连续的索引，避免时间缺口问题
    hist_indices = np.arange(len(hist_data))
    pred_indices = np.arange(len(hist_data), len(hist_data) + len(future_preds))
    
    # 绘制
    plt.figure(figsize=(14, 8))
    plt.plot(hist_indices, hist_data[:, 3], 'b-', linewidth=2, label='历史收盘价', marker='o', markersize=3)
    plt.plot(pred_indices, future_preds, 'r-', linewidth=2, label='预测收盘价', marker='s', markersize=4)
    
    plt.axvline(x=len(hist_data)-1, color='green', linestyle='--', linewidth=2, alpha=0.7, label='预测起点')
    
    # 设置x轴标签 - 选择关键时间点显示
    x_ticks = []
    x_labels = []
    
    # 历史数据的时间标签
    step = max(1, len(hist_data) // 5)  # 最多显示5个历史时间点
    for i in range(0, len(hist_data), step):
        x_ticks.append(i)
        x_labels.append(hist_times.iloc[i].strftime('%m-%d %H:%M'))
    
    # 预测数据的标签 - 只显示步骤编号
    pred_step = max(1, len(future_preds) // 4)  # 最多显示4个预测点
    for i in range(0, len(future_preds), pred_step):
        x_ticks.append(len(hist_data) + i)
        x_labels.append(f'预测{i+1}')
    
    # 最后一个预测点
    if len(future_preds) > 0:
        x_ticks.append(pred_indices[-1])
        x_labels.append(f'预测{len(future_preds)}')
    
    plt.xticks(x_ticks, x_labels, rotation=45)
    plt.xlabel('时间点')
    plt.ylabel('价格 ($)')
    plt.title(f'NGRC模型 - SPY未来{time_label}走势预测')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ngrc_future_predictions.png', dpi=300, bbox_inches='tight')
    print("未来预测图表已保存: ngrc_future_predictions.png")
    plt.show()


def main():
    print("╔════════════════════════════════════════════════════════════════════════════════╗")
    print("║               Next Generation Reservoir Computing - SPY预测                   ║")
    print("╚════════════════════════════════════════════════════════════════════════════════╝\n")
    
    # 加载数据
    data, df, time_label, optimal_delay_length = load_and_prepare_data('SPY_30min_last_30days.csv')
    
    # 分割数据
    train_data, test_data = split_data(data)
    print(f"训练集大小: {train_data.shape}")
    print(f"测试集大小: {test_data.shape}")
    
    # 验证数据长度是否足够
    min_required_length = optimal_delay_length + 10  # 至少需要延迟长度+10个样本
    if len(train_data) < min_required_length:
        print(f"警告: 训练数据太少! 需要至少 {min_required_length} 个样本，当前只有 {len(train_data)} 个")
        optimal_delay_length = max(5, len(train_data) // 3)  # 进一步减少延迟长度
        print(f"自动调整延迟长度为: {optimal_delay_length}")
    
    # 创建和训练NGRC模型
    print("\n" + "="*50)
    print("训练NGRC模型")
    print("="*50)
    
    ngrc = NGRCModel(delay_length=optimal_delay_length, degree=2, ridge_alpha=1e-4)
    ngrc.fit(train_data)
    
    # 保存模型
    joblib.dump(ngrc, 'ngrc_spy_model.joblib')
    print("模型已保存: ngrc_spy_model.joblib")
    
    # 评估模型
    y_test, y_pred, metrics = evaluate_model(ngrc, test_data)
    
    # 绘制结果
    plot_results(y_test, y_pred, metrics, time_label)
    
    # 预测未来
    future_preds = predict_future(ngrc, data, n_steps=20)
    print(f"未来20步预测: {future_preds}")
    
    # 绘制未来预测
    plot_future_predictions(data, df, future_preds, time_label)
    
    print("\n" + "="*50)
    print("NGRC训练和预测完成!")
    print("="*50)


if __name__ == "__main__":
    main()
