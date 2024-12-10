## Model Development Script
# %%
import oandapyV20
import oandapyV20.endpoints.pricing as pricing
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.accounts as accounts
import pandas as pd
import datetime
import oandapyV20
import oandapyV20.endpoints.pricing as pricing
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.accounts as accounts
import pandas as pd
import numpy as np
import datetime
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.regularizers import l1, l2
from keras.optimizers import Adam
# %%
ACCESS_TOKEN = "abcd-efgh"
client = oandapyV20.API(access_token=ACCESS_TOKEN, environment="practice")
# %%
''' Load data from csv, or collect from scratch below. '''

start_date = "2024-01-01T00:00:00Z"
start_datetime = datetime.datetime.strptime(start_date, "%Y-%m-%dT%H:%M:%SZ")
current_datetime = datetime.datetime.now()
candlestick_df = pd.DataFrame()
# %%
while start_datetime < current_datetime:

    params = {
        "granularity": "M15",   # 1-hour candles
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

    columns = ['time','close']
    # columns = ['time', 'open', 'high', 'low', 'close', 'volume']

    candlestick_data = []

    for candle in candles:
        candlestick_data.append({
            'time': candle.get("time"),
            # 'open': float(candle.get("mid", {}).get("o", 0)),
            # 'high': float(candle.get("mid", {}).get("h", 0)),
            # 'low': float(candle.get("mid", {}).get("l", 0)),
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
data = candlestick_df.set_index('time')
data.info()
# %%
data['returns'] = np.log(data['close'] / data['close'].shift())
data
# %%
window = 48
# %%
''' Feature Engineering '''
df = data.copy()
df['dir'] = np.where(df['returns'] > 0, 1, 0)

df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
df['macd_line'] = df['close'].ewm(span=26, adjust=False).mean() - df['close'].ewm(span=12, adjust=False).mean()
df['signal_line'] = df['macd_line'].ewm(span=9, adjust=False).mean()
df['min'] = (df['returns'] - df['returns'].rolling(window).min()) / df['returns'].rolling(window).min()
df['max'] = (df['returns'].rolling(window).max() - df['returns']) / df['returns'].rolling(window).max()
# df['sma_12'] = df['returns'].rolling(12).mean()
# df['sma_26'] = df['returns'].rolling(26).mean()
# df['vol_12'] = df['volume'].rolling(12).mean()
# df['vol_26'] = df['volume'].rolling(26).mean()
# df['boll'] = (df['returns'] - df['returns'].rolling(window).mean()) / df['returns'].rolling(window).std()
# df['mom'] = df['returns'].rolling(15).mean()
# df['vol'] = df['returns'].rolling(window).std()

df.dropna(inplace = True)
# %%
''' Feature Lags '''
lags = 3
# %%
cols = []
features = ['dir', 'close', 'macd_line', 'signal_line', 'ema_200', 'min', 'max']
# %%
for f in features:
  for lag in range(1, lags + 1):
    col = "{}_lag_{}".format(f,lag)
    df[col] = df[f].shift(lag)
    cols.append(col)
df.dropna(inplace = True)
# %%
df.info()
# %%
''' Shuffle df for training/evaluation '''
df_shuffled = df.sample(frac=1)
df_shuffled
# %%
split = int(len(df_shuffled)*0.8)
# %%
train = df_shuffled.iloc[:split].copy()
train
# %%
test = df_shuffled.iloc[split:].copy()
test
# %%
''' Feature Scaling (Normal Standardisation)'''
train[cols]
mu, std = train.mean(), train.std()
# %%
train_s = (train - mu) / std
train_s.describe()
# %%
''' Creating and Fitting the DNN model '''
def set_seeds(seed = 100):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def cw(df):
    c0, c1 = np.bincount(df["dir"])
    w0 = (1/c0) * (len(df)) / 2
    w1 = (1/c1) * (len(df)) / 2
    return {0:w0, 1:w1}
# %%
optimizer = Adam(learning_rate = 0.00001)
# %%
def create_model(hl = 2, hu = 100, dropout = False, rate = 0.3, regularize = False,
                 reg = l1(0.0005), optimizer = optimizer, input_dim = None):
    if not regularize:
        reg = None
    model = Sequential()
    model.add(Dense(hu, input_dim = input_dim, activity_regularizer = reg ,activation = "relu"))
    if dropout:
        model.add(Dropout(rate, seed = 100))
    for layer in range(hl):
        model.add(Dense(hu, activation = "relu", activity_regularizer = reg))
        if dropout:
            model.add(Dropout(rate, seed = 100))
    model.add(Dense(1, activation = "sigmoid"))
    model.compile(loss = "binary_crossentropy", optimizer = optimizer, metrics = ["accuracy"])
    return model
# %%
set_seeds(100)
model = create_model(hl = 2, hu = 100, dropout=True, input_dim = len(cols))
# %%
model.summary()
# %%
model.fit(x = train_s[cols], y = train["dir"], epochs = 100, verbose = True, validation_split=0.2, shuffle=True, class_weight = cw(train))
# %%
model.evaluate(x = train_s[cols], y = train["dir"])
# %%
''' Visualise model predictions '''
pred = model.predict(train_s[cols])
plt.hist(pred, bins=50)
plt.show()
# %%
''' Out-Sample Predicition and Backtesting '''
test_s = (test - mu) / std # use train set parameters
model.evaluate(test_s[cols], test['dir'])
# %%
''' Evaluation functions '''
def strategy_df(x, y):
  evaluate_df = test.copy()
  evaluate_df['proba'] = model.predict(test_s[cols])
  evaluate_df['position'] = np.where(evaluate_df['proba'] < x, -1, np.nan) # short where proba < 0.47
  evaluate_df['position'] = np.where(evaluate_df['proba'] > y, 1, evaluate_df.position) # long where proba > 0.53
  evaluate_df['position'] = np.where(evaluate_df['position'].isna(), 0, evaluate_df['position'])

  return evaluate_df

def plot_returns(df):
  df['strategy'] = df['position'] * df['returns']
  df['creturns'] = df['returns'].cumsum().apply(np.exp)
  df['cstrategy'] = df['strategy'].cumsum().apply(np.exp)

  # Set up the period index
  df['period'] = range(1, len(df) + 1)

  # Plot with custom x-axis
  ax = df.plot(x='period', y=['creturns', 'cstrategy'], figsize=(12, 8))
  ax.set_xlabel("Period (15m)")
  ax.set_ylabel("Cumulative Returns")
  ax.set_title("Strategy vs Market Returns")

  plt.legend(["Market Returns", "Strategy Returns"])
  plt.show()
# %%
eval_df = strategy_df(0.48, 0.52)
eval_df['strategy'] = eval_df['position'] * eval_df['returns']
# %%
plot_returns(eval_df)
# %%
''' Net Returns (Before Fees) '''
print(eval_df.iloc[-1]['cstrategy'])
# %%
''' Saving model and parameters '''
model.save('../models/10-12-2024/model_E_v1.keras')
params = {"mu":mu, "std":std}
pickle.dump(params, open("../models/10-12-2024/params.pkl", "wb"))