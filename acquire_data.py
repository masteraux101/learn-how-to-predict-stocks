import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('when', help="last n days")

args = parser.parse_args()

when = args.when

spy = yf.Ticker("SPY")

df = spy.history(period=f"{when}d", interval="5m", prepost=False)



print(f"总共获取到 {len(df)} 根5分钟K线")
print(f"时间范围：{df.index[0]}  →  {df.index[-1]}")

# 重置索引方便查看
df_reset = df.reset_index()
print(df_reset.head(10))

# 保存到 csv（可选）
df.to_csv(f"SPY_5min_last_{when}days.csv")
