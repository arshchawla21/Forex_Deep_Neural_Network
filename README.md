# Forex Trading using Deep Neural Networks

This repository contains a deep neural network (DNN) model for predicting forex commodity price movements and a trading bot that uses the model with the OANDA API for live trading. The bot is designed to trade forex pairs such as USD/JPY, leveraging historical candlestick data and machine learning to make informed trading decisions.

_**Note:** This project is for educational purposes only, and caution should be exercised when/if live trading._

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Setup Instructions](#setup-instructions)
4. [Usage](#usage)
5. [Model Details](#model-details)
6. [Backtesting and Performance](#backtesting-and-performance)
7. [Live Trading](#live-trading)
8. [Example](#example)

---

## Overview

This project combines financial data analysis, machine learning, and automated trading. The DNN model predicts price direction using engineered features and a custom architecture. The trading bot integrates the model with the OANDA API for real-time trading.

---

## Features

- **Data Collection**: Uses OANDA API to fetch historical candlestick data.
- **Feature Engineering**: Includes MACD, EMA, and custom lagged features.
- **DNN Model**: Binary classification for price movement direction.
- **Backtesting**: Evaluates strategy performance on historical data.
- **Live Trading**: Executes trades in real-time based on model predictions.

---

## Setup Instructions

### Prerequisites
- Python 3.8+
- OANDA Practice Account (API access)
- Required Python packages:
  ```bash
  pip install tensorflow keras matplotlib pandas numpy oandapyV20

---

## Usage

### Training the Model
1. Use the model.py script to train the DNN model.
2. Model parameters and training results are saved in the models/ directory.
### Running the Trading Bot
1. Use the model_inference.py script for live trading.
2. Adjust the instrument and parameters in the script as per your model's training setup.

---

## Model Details

* Inputs: Historical price data, MACD, EMA, SMA, Volume, Min, Max, lagged features.
* Architecture: Fully connected layers with dropout and optional L1 regularization.
* Optimizer: Adam with a learning rate of 1e-5.
* Loss Function: Binary cross-entropy.
* Outputs: Probability of upward price movement.

---

## Backtesting and Performance

* Backtesting is implemented in `model.py` using a trading strategy based on model probabilities.
* Results include cumulative returns and strategy visualization.

---

## Live Trading

* The trading bot fetches real-time data using the OANDA API.
* Executes long/short positions based on model predictions.
* Handles trading parameters like granularity, thresholds, and lot sizes.

---

## Example

A Model was trained on the USD/JPY instrument with 15-minute candles. Historical data was collected from 2024-01-01, with features EMA_200, MACD_line, SIGNAL_line, min and max. Results were as follows:

![image](https://github.com/user-attachments/assets/eda436ac-e454-44ee-8ddf-7181748d65e0)

![model_e_returns](https://github.com/user-attachments/assets/3a81ab44-8449-4f8f-88cc-a47e3f54fa3d)

Across the randomised two-month period, the model achieved a net return (before fees) of 3.398%, marginally exceeding buy-hold returns.



