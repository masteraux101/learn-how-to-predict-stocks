import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import argparse
import time


def fetch_stock_data_chunked(ticker_symbol, days, interval_minutes, chunk_days=8):
    """
    分片获取股票数据，解决yfinance 1分钟K线最多8天的限制
    
    Args:
        ticker_symbol: 股票代码 (如 "SPY")
        days: 需要获取的总天数
        interval_minutes: K线周期（分钟）
        chunk_days: 每次获取的天数 (默认8天)
    
    Returns:
        拼接后的DataFrame
    """
    spy = yf.Ticker(ticker_symbol)
    dataframes = []
    
    # 计算需要获取的天数范围
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # 分片获取数据
    current_date = start_date
    chunk_num = 0
    
    while current_date < end_date:
        chunk_num += 1
        # 计算本次获取的时间段
        chunk_end = min(current_date + timedelta(days=chunk_days), end_date)
        days_in_chunk = (chunk_end - current_date).days
        
        print(f"[分片 {chunk_num}] 获取数据: {current_date.strftime('%Y-%m-%d')} 到 {chunk_end.strftime('%Y-%m-%d')} ({days_in_chunk} 天)...")
        
        try:
            # 获取该分片的数据
            df_chunk = spy.history(
                start=current_date, 
                end=chunk_end, 
                interval=f"{interval_minutes}m", 
                prepost=False
            )
            
            if len(df_chunk) > 0:
                dataframes.append(df_chunk)
                print(f"  ✓ 成功获取 {len(df_chunk)} 根K线")
            else:
                print(f"  ⚠ 该分片无数据")
            
            # 避免请求过于频繁
            time.sleep(1)
            
        except Exception as e:
            print(f"  ✗ 获取失败: {str(e)}")
        
        # 移动到下一个分片
        current_date = chunk_end
    
    if not dataframes:
        raise ValueError("未能获取任何数据")
    
    # 拼接所有分片数据并去重
    df_combined = pd.concat(dataframes, ignore_index=False)
    df_combined = df_combined[~df_combined.index.duplicated(keep='first')]
    df_combined = df_combined.sort_index()
    
    return df_combined


parser = argparse.ArgumentParser(description="获取SPY股票K线数据（支持超过8天的1分钟K线）")

parser.add_argument('when', type=int, help="过去n天的数据")
parser.add_argument('minutes', type=int, help="K线周期（分钟）")
parser.add_argument('--chunk-days', type=int, default=8, help="每次获取的天数，默认8天")

args = parser.parse_args()

when = args.when
minutes = args.minutes
chunk_days = args.chunk_days

print(f"开始获取数据...")
print(f"股票代码: SPY")
print(f"获取周期: {when} 天")
print(f"K线周期: {minutes} 分钟")
print(f"分片大小: {chunk_days} 天")
print("-" * 50)

try:
    df = fetch_stock_data_chunked("SPY", when, minutes, chunk_days)
    
    print("-" * 50)
    print(f"✓ 总共获取到 {len(df)} 根{minutes}分钟K线")
    print(f"✓ 时间范围: {df.index[0]} → {df.index[-1]}")
    
    # 重置索引方便查看
    df_reset = df.reset_index()
    print("\n数据预览（前10行）:")
    print(df_reset.head(10))
    
    # 保存到 csv
    output_file = f"SPY_{minutes}min_last_{when}days.csv"
    df.to_csv(output_file)
    print(f"\n✓ 数据已保存到: {output_file}")
    
except Exception as e:
    print(f"✗ 错误: {str(e)}")
