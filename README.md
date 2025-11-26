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

