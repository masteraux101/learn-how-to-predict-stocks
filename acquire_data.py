import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


spy = yf.Ticker("SPY")

# 直接下载最近60天的5分钟数据（包含预盘和盘后）
df = spy.history(period="60d", interval="5m", prepost=False)   # prepost=True 包含盘前盘后

# 如果你只想要常规交易时段（9:30-16:00），可以加参数
# df = spy.history(period="60d", interval="5m", prepost=False)

print(f"总共获取到 {len(df)} 根5分钟K线")
print(f"时间范围：{df.index[0]}  →  {df.index[-1]}")

# 重置索引方便查看
df_reset = df.reset_index()
print(df_reset.head(10))

# 保存到 csv（可选）
df.to_csv("SPY_5min_last_60days.csv")
