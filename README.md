## 预测学习

- 使用LSTM，内含state，防止RNN梯度爆炸或者消失

- 增加特征，ema，小时数

- 预测出现恒等映射的情况，仅仅是时间序列的偏移

<img width="2456" height="1064" alt="image" src="https://github.com/user-attachments/assets/39eb1caa-4a16-4c47-8413-9b68ecef7304" />

- 修改模型，使其预测做多做空胜率，加入softmax，避免模型收敛至50%偷懒

- `simple_lstm_model_trainer.py` 是简单的lstm网络训练脚本，预测收盘价
- `simple_lstm_model_runner.py` 是简单的lstm运行脚本
- `enchance_lstm_model_trainer.py` 是lstm输出胜率的修正
- `enchance_lstm_model_runner.py` lstm输出胜率的运行脚本

<img width="472" height="411" alt="image" src="https://github.com/user-attachments/assets/6903fd19-6fee-4f3e-b5e7-e2366080d919" />

- 预测胜率的模型出现一个倾向：盘中前期倾向做多，盘中后期倾向做空
  - 如果去掉时序标记进行训练，模型会非常不稳定
 
- 新增FNO，傅立叶神经算子
  - 本身就是对序列进行离散傅立叶变换，然后低通滤波一下，最后ifft会原始序列
  - 同时单独全连接一下序列，把计算的两个结果相加，激活一下，重复此过程

- 超参数：
  - MinMaxScaler：是shrink到0-1还是0-10000？
    - 我感觉扩大原有的区间，比如400-600的数据范围扩展到0-10000，实际上是给训练增加精度，算的会慢一点，但是精度会高一点，看起来会更加拟合
  - 低通滤波调节：n序列会离散出n个频率，其中n // 2是正频率，你需要决定保留哪些？
    - 保留的多，那么高频的的细节就越少
    - 保存的少，模型会尝试放大低频的振幅尝试去拟合
<img width="2400" height="1336" alt="image" src="https://github.com/user-attachments/assets/efc52f45-f581-4df6-bfbb-db3083dd07d2" />

     

