## Model Inference Script. Used for live/demo trading.

''' 1. Load appropriate model and model_params
    2. Update ACCOUNT_ID and ACCESS_TOKEN 
    3. Adjust specficied variables (instrument, interval time, features etc.) based on the loaded model
'''
# %%
import oandapyV20
import oandapyV20.endpoints.pricing as pricing
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.accounts as accounts
import pandas as pd
import numpy as np
import datetime
import tensorflow as tf
import yfinance as yf
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import layers
from sklearn.preprocessing import MinMaxScaler
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.trades as trades
import pickle
import time
# %%
ACCOUNT_ID = "123-456-78901234-012"
ACCESS_TOKEN = "abcd-efgh"
client = oandapyV20.API(access_token=ACCESS_TOKEN, environment="practice")
# %%
model_path = "../models/28-11-2024/model_D_v1.keras"
model = keras.models.load_model(model_path)
file = open("../models/28-11-2024/params_D.pkl",'rb')
model_params = pickle.load(file)
# %%
instrument = "USD_JPY" # ADJUST BASED ON MODEL
def get_current_data():
    start_time = datetime.datetime.isoformat(datetime.datetime.now(datetime.timezone.utc))
    params = {
        "granularity": "M30", # ADJUST BASED ON MODEL
        "count": 64, # ADJUST BASED ON MODEL
        "to": start_time,
    }
    r = instruments.InstrumentsCandles(instrument=instrument, params=params)
    response = client.request(r)

    candles = response.get("candles", [])
    print(f"Fetched {len(candles)} till now")
    candlestick_data = []

    for candle in candles:
        candlestick_data.append({
            'time': candle.get("time"),
            'close': float(candle.get("mid", {}).get("c", 0)),
            'volume': candle.get("volume", 0), # ADJUST BASED ON MODEL
        })
    df = pd.DataFrame(candlestick_data)
    df['time'] = pd.to_datetime(df['time'])

    data = df.set_index('time')
    data['returns'] = np.log(data['close'] / data['close'].shift())
    return data
# %%
def get_prediction(data): # ADJUST PARAMETERS (features, window size, lag etc) BASED ON MODEL
    window = 48
    df = data.copy()
    df['dir'] = np.where(df['returns'] > 0, 1, 0)
    df['sma_12'] = df['returns'].rolling(12).mean()
    df['sma_26'] = df['returns'].rolling(26).mean()
    df['vol_12'] = df['volume'].rolling(12).mean()
    df['vol_26'] = df['volume'].rolling(26).mean()
    df['boll'] = (df['returns'] - df['returns'].rolling(window).mean()) / df['returns'].rolling(window).std()
    df['min'] = (df['returns'] - df['returns'].rolling(window).min()) / df['returns'].rolling(window).min()
    df['max'] = (df['returns'].rolling(window).max() - df['returns']) / df['returns'].rolling(window).max()
    df.dropna(inplace = True)

    lags = 3
    cols = []
    features = ['dir','sma_12','sma_26', 'vol_12', 'vol_26', 'boll', 'min', 'max']
    for f in features:
        for lag in range(1, lags + 1):
            col = "{}_lag_{}".format(f,lag)
            df[col] = df[f].shift(lag)
            cols.append(col)
    df.dropna(inplace = True)
   
    df_norm = (df - model_params['mu']) / model_params['std']
    pred = model.predict(df_norm[cols])[-1][0]
    return pred
# %%
def place_order(client, account_id, units, instrument):
    """
    Places a market order.
    
    Args:
        client (oandapyV20.API): The OANDA API client.
        account_id (str): The account ID.
        units (int): Number of units for the order (positive = long, negative = short).
        instrument (str): The trading instrument (e.g., 'EUR_USD').
    """
    order_data = {
        "order": {
            "instrument": instrument,
            "units": str(units),  # Must be a string
            "type": "MARKET",
            "positionFill": "DEFAULT",
        }
    }
    try:
        r_order = orders.OrderCreate(account_id, data=order_data)
        response = client.request(r_order)
        print(f"Order placed: {response}")
    except oandapyV20.exceptions.V20Error as e:
        print(f"API Error when placing order: {e}")
# %%
def close_trade(client, account_id, trade_id):
    """
    Closes an open trade.
    
    Args:
        client (oandapyV20.API): The OANDA API client.
        account_id (str): The account ID.
        trade_id (str): The ID of the trade to close.
    """
    try:
        r_close = trades.TradeClose(account_id, tradeID=trade_id)
        response = client.request(r_close)
        print(f"Closed trade {trade_id}: {response}")
    except oandapyV20.exceptions.V20Error as e:
        print(f"API Error when closing trade: {e}")
# %%
def manage_position(client, account_id, position, units, instrument):
    """
    Manages a single trade position: long, short, or neutral.
    
    Args:
        client (oandapyV20.API): The OANDA API client.
        account_id (str): The account ID.
        position (int): Desired position (-1 = short, 0 = neutral, 1 = long).
        units (int): Number of units for the trade (positive value).
        instrument (str): The trading instrument (e.g., 'EUR_USD').
    """
    try:
        # Fetch open trades
        r_trades = trades.TradesList(account_id)
        response = client.request(r_trades)
        open_trades = response.get("trades", [])

        # Determine current position
        current_position = 0
        if open_trades:
            trade = open_trades[0]  # Assume only one trade is active
            current_position = 1 if int(trade["currentUnits"]) > 0 else -1

        # Manage position logic
        if position == 0:
            # Neutral: Close any open position
            if current_position != 0:
                close_trade(client, account_id, open_trades[0]["id"])
            print("Position set to neutral. No active trades.")
        
        elif position == -1:
            # Short position
            if current_position == 1:
                # Close long trade before opening a short position
                close_trade(client, account_id, open_trades[0]["id"])
                place_order(client, account_id, -units, instrument)
                print("Closed long position and opened a short position.")
            elif current_position == 0:
                # No trade: Open a short position
                place_order(client, account_id, -units, instrument)
                print("Opened a short position.")
            else:
                print("Short position already active. No action taken.")
        
        elif position == 1:
            # Long position
            if current_position == -1:
                # Close short trade before opening a long position
                close_trade(client, account_id, open_trades[0]["id"])
                place_order(client, account_id, units, instrument)
                print("Closed short position and opened a long position.")
            elif current_position == 0:
                # No trade: Open a long position
                place_order(client, account_id, units, instrument)
                print("Opened a long position.")
            else:
                print("Long position already active. No action taken.")

    except oandapyV20.exceptions.V20Error as e:
        print(f"API Error: {e}")
    except Exception as ex:
        print(f"Unexpected Error: {ex}")
# %%
def trade():
    data = get_current_data()
    pred = get_prediction(data)
    if (pred > 0.52):
        manage_position(client,ACCOUNT_ID,1,10000,instrument=instrument)
    elif (pred < 0.48):
        manage_position(client,ACCOUNT_ID,-1,10000,instrument=instrument)
    else:
        manage_position(client,ACCOUNT_ID,0,10000,instrument=instrument)
    print(pred)
# %%
while True:
    trade()
    time.sleep(30 * 60) # ADJUST BASED ON MODEL