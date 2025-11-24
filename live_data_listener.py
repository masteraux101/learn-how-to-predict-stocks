import yfinance as yf
from datetime import datetime
import pandas as pd
import argparse


start_hour = 22
start_minute = 30

end_hour = 5
end_minute = 0

prepost = False

collect_bars = pd.DataFrame(
    columns =[
    "Time",
    "Open",
    "Close",
    "High",
    "Low"],
)

pre_bar_key = -1

interval = 5



def message_handler(message):

    global pre_bar_key, collect_bars

    current_time = datetime.fromtimestamp(int(message['time'])/ 1000)

    price = float(message['price'])

    print(f"current {message['id']} price is {price:.2f}, now: {current_time}")

    current_wd = current_time.weekday()
    current_h = current_time.hour
    current_m = current_time.minute

    is_live = (current_h >= start_hour and current_m >= start_minute) or (current_h < 5)


    is_live = current_wd < 5 and is_live

    if prepost:
        is_live = True

    if is_live:
        print("stock market opened")
    else:
        print("stock market not opened")

    if is_live:
        bar_key = current_h * 60 + current_m
        bar_key = bar_key // interval * interval
        if bar_key not in collect_bars["Time"].values:
            if pre_bar_key > 0:
                collect_bars.loc[pre_bar_key, "Close"] = price
            collect_bars.loc[bar_key] = {
                "Time": bar_key,
                "Open": price,
                "Close": price,
                "High": price,
                "Low": price
            }
        else:
            pre_bar_key = bar_key
            if price > collect_bars.at[bar_key, "High"]:
                collect_bars.loc[bar_key, "High"] = price

            if price < collect_bars.at[bar_key, "Low"]:
                collect_bars.loc[bar_key, "Low"] = price


        collect_bars.dropna()
        print(collect_bars)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("symbol", type=str)
    parser.add_argument("interval", type=int, default=5)
    parser.add_argument("prepost", type=bool)

    args = parser.parse_args()
    interval = args.interval
    prepost = args.prepost


    with yf.WebSocket() as ws:
        ws.subscribe([args.symbol])
        ws.listen(message_handler)



