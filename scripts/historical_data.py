## Historical Data Collection Script. Used to collect historical data for model training/eval.
# %%
import oandapyV20
import oandapyV20.endpoints.pricing as pricing
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.accounts as accounts
import pandas as pd
import datetime
# %%
ACCESS_TOKEN = "ADD_ACCESS_TOKEN"
client = oandapyV20.API(access_token=ACCESS_TOKEN, environment="live")
# %%
start_date = "2024-09-01T00:00:00Z"
start_datetime = datetime.datetime.strptime(start_date, "%Y-%m-%dT%H:%M:%SZ")
current_datetime = datetime.datetime.now()
candlestick_df = pd.DataFrame()
# %%
while start_datetime < current_datetime:

    params = {
        "granularity": "S5",   # 5-second candles
        "count": 5000,         # Number of data points to retrieve
        "from": start_date,    # Fetch data from a specific date onwards
    }

    instrument = "USD_JPY"
    r = instruments.InstrumentsCandles(instrument=instrument, params=params)

    response = client.request(r)

    candles = response.get("candles", [])
    print(f"Fetched {len(candles)} candles starting from {start_date}")

    if len(candles) == 1:
        print("Reached the current time")
        break

    columns = ['time', 'open', 'high', 'low', 'close', 'volume']
    candlestick_data = []

    for candle in candles:
        candlestick_data.append({
            'time': candle.get("time"),
            'open': float(candle.get("mid", {}).get("o", 0)),
            'high': float(candle.get("mid", {}).get("h", 0)),
            'low': float(candle.get("mid", {}).get("l", 0)),
            'close': float(candle.get("mid", {}).get("c", 0)),
            'volume': candle.get("volume", 0),
        })
    # Add the fetched data to the DataFrame
    new_df = pd.DataFrame(candlestick_data)
    candlestick_df = pd.concat([candlestick_df, new_df], ignore_index=True)

    # Update the start_datetime to the last candle's time for the next iteration
    if candles:
        last_candle_time = candles[-1]["time"]
        
        # Trim the nanoseconds by removing the extra characters after microseconds
        last_candle_time_trimmed = last_candle_time[:26] + "Z"  # Keep only up to microseconds
        
        # Parse the trimmed datetime string
        start_datetime = datetime.datetime.strptime(last_candle_time_trimmed, "%Y-%m-%dT%H:%M:%S.%fZ")

    # Update start_date to the new start_datetime
    start_date = start_datetime.strftime("%Y-%m-%dT%H:%M:%SZ")
# %%
candlestick_df['time'] = pd.to_datetime(candlestick_df['time'])
# %%
print(candlestick_df)
# %%
candlestick_df.to_csv('data_USD_JPY_01-09-2024-now_S5.csv', index=False)