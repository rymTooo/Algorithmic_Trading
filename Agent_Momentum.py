from utils import TradingAgent
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import deque
import random
import tensorflow as tf
from tensorflow.keras import models, layers

class Agent_Momentum(TradingAgent):
    """Momentum Trading Agent."""
    def __init__(self, lookback):
        super().__init__()
        self.lookback = lookback

    def generate_signals(self, data):
        if len(data) < self.lookback:
            return 0  # Not enough data to calculate momentum

        # Calculate momentum as the rate of change (ROC)
        momentum = data['close'].pct_change(self.lookback).iloc[-1]

        # Generate signals based on momentum
        if momentum > 0:
            return 1  # Buy (upward momentum)
        elif momentum < 0:
            return -1  # Sell (downward momentum)
        else:
            return 0  # Hold

class Agent_TrendFollowing(TradingAgent):
    """Trend Following Trading Agent."""
    def __init__(self, lookback):
        super().__init__()
        self.short_window = lookback  # Use lookback for short_window
        self.long_window = lookback * 3  # Example, long_window could be 3x lookback or another logic

    def generate_signals(self, data):
        if len(data) < self.long_window:
            return 0  # Not enough data to calculate trend

        short_ma = data['close'].rolling(window=self.short_window).mean().iloc[-1]
        long_ma = data['close'].rolling(window=self.long_window).mean().iloc[-1]

        # Generate signals based on moving averages
        if short_ma > long_ma:
            return 1  # Buy (uptrend)
        elif short_ma < long_ma:
            return -1  # Sell (downtrend)
        else:
            return 0  # Hold (neutral trend)

class Agent_RiskOnRiskOff(TradingAgent):
    """Risk-On/Risk-Off Trading Agent."""
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def generate_signals(self, data):
        if len(data) < 1:
            return 0  # Not enough data to assess risk

        price_change = data['close'].pct_change().iloc[-1]

        # Risk-on if price change exceeds threshold (bullish)
        if price_change > self.threshold:
            return 1  # Buy
        elif price_change < -self.threshold:
            return -1  # Sell
        else:
            return 0  # Hold

class Agent_Arbitrage(TradingAgent):
    """Arbitrage Trading Agent."""
    def __init__(self, spread_threshold):
        super().__init__()
        self.spread_threshold = spread_threshold

    def generate_signals(self, data):
        if len(data) < 2:
            return 0  # Not enough data for arbitrage calculation

        price_1 = data['close'].iloc[-2]
        price_2 = data['close'].iloc[-1]
        spread = abs(price_1 - price_2)

        # If the spread is greater than the threshold, execute arbitrage trade
        if spread > self.spread_threshold:
            return 1 if price_1 < price_2 else -1  # Buy if first price is lower, sell if higher
        else:
            return 0  # Hold
class Agent_BlackSwan(TradingAgent):
    """Black Swan Catching Agent."""
    def __init__(self, z_score_threshold):
        super().__init__()
        self.z_score_threshold = z_score_threshold

    def generate_signals(self, data):
        if len(data) < 50:
            return 0  # Not enough data to calculate Z-score

        # Calculate Z-score for price volatility
        mean = data['close'].mean()
        std_dev = data['close'].std()
        z_score = (data['close'].iloc[-1] - mean) / std_dev

        # If Z-score exceeds threshold, we have a black swan event
        if abs(z_score) > self.z_score_threshold:
            return -1 if z_score > 0 else 1  # Sell if positive z-score, buy if negative
        else:
            return 0  # Hold
class Agent_MarketTiming(TradingAgent):
    """Market Timing Trading Agent."""
    def __init__(self, trend_threshold, volatility_threshold):
        super().__init__()
        self.trend_threshold = trend_threshold
        self.volatility_threshold = volatility_threshold

    def generate_signals(self, data):
        if len(data) < 2:
            return 0  # Not enough data to assess market conditions

        price_change = data['close'].pct_change().iloc[-1]
        volatility = data['close'].std()

        # Buy when the market is trending upwards and volatility is low
        if price_change > self.trend_threshold and volatility < self.volatility_threshold:
            return 1  # Buy
        elif price_change < -self.trend_threshold or volatility > self.volatility_threshold:
            return -1  # Sell
        else:
            return 0  # Hold
class Agent_InverseVolatility(TradingAgent):
    """Inverse Volatility Trading Agent."""
    def __init__(self, lookback):
        super().__init__()
        self.lookback = lookback

    def generate_signals(self, data):
        if len(data) < self.lookback:
            return 0  # Not enough data to calculate volatility

        # Calculate rolling volatility as the standard deviation of returns
        returns = data['close'].pct_change()
        volatility = returns.rolling(window=self.lookback).std().iloc[-1]

        # Generate signals based on inverse volatility
        if volatility < 0.01:  # Low volatility, high risk, hold
            return 0
        else:  # High volatility, consider taking trades
            return 1 if returns.iloc[-1] > 0 else -1

class Agent_MR_Stat(TradingAgent):
    """Mean Reversion Trading Agent."""
    def __init__(self, lookback=300, z_score_threshold=2):
        super().__init__()
        self.lookback = lookback
        self.z_score_threshold = z_score_threshold

    def generate_signals(self, data):
        if len(data) < self.lookback:
            return 0  # Not enough data

        rolling_mean = data['close'].rolling(self.lookback).mean()
        rolling_std = data['close'].rolling(self.lookback).std()
        z_scores = (data['close'] - rolling_mean) / rolling_std

        if z_scores.iloc[-1] > self.z_score_threshold:
            return -1  # Overbought, sell
        elif z_scores.iloc[-1] < -self.z_score_threshold:
            return 1   # Oversold, buy
        return 0

class Agent_XGBoost(TradingAgent):
    """XGBoost-based Trading Agent."""
    def __init__(self):
        super().__init__()
        self.model = xgb.XGBClassifier()

    def train(self, data):
        data['returns'] = data['close'].pct_change()
        data.dropna(inplace=True)

        features = data[['open', 'high', 'low', 'close', 'volume']]
        labels = (data['returns'] > 0).astype(int)

        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"XGBoost Model Accuracy: {accuracy * 100:.2f}%")

    def generate_signals(self, data):
        if len(data) < 5:
            return 0  # Not enough data for a prediction

        latest_data = data[['open', 'high', 'low', 'close', 'volume']].iloc[-1]
        prediction = self.model.predict([latest_data])[0]
        return 1 if prediction == 1 else -1

    

class Agent_BollingerBands(TradingAgent):
    """Trading Agent for Bollinger Bands Mean Reversion."""
    def __init__(self, lookback):
        super().__init__()
        self.lookback = lookback
        self.trigger = 0  # To avoid multiple signals in quick succession

    def generate_signals(self, data):
        if len(data) < self.lookback:
            return 0  # Not enough data to calculate Bollinger Bands

        # Calculate the rolling mean and standard deviation
        mean = data['close'].rolling(window=self.lookback).mean().iloc[-1]
        std_dev = data['close'].rolling(window=self.lookback).std().iloc[-1]

        # Define the upper and lower bands
        upper_band = mean + (2 * std_dev)
        lower_band = mean - (2 * std_dev)

        # Get the current price
        current_price = data['close'].iloc[-1]

        # Generate signals
        if current_price > upper_band and self.trigger >= 0:
            self.trigger = -1
            return -1  # Sell
        elif current_price < lower_band and self.trigger <= 0:
            self.trigger = 1
            return 1  # Buy
        else:
            return 0  # Hold
class Agent_MACrossover(TradingAgent):
    """Trading Agent for Moving Average Crossover."""
    def __init__(self, short_lookback, long_lookback):
        super().__init__()
        self.short_lookback = short_lookback
        self.long_lookback = long_lookback
        self.trigger = 0

    def generate_signals(self, data):
        if len(data) < max(self.short_lookback, self.long_lookback):
            return 0  # Not enough data to calculate moving averages

        # Calculate moving averages
        short_ma = data['close'].rolling(window=self.short_lookback).mean().iloc[-1]
        long_ma = data['close'].rolling(window=self.long_lookback).mean().iloc[-1]

        # Generate signals
        if short_ma > long_ma and self.trigger >= 0:
            self.trigger = -1
            return 1  # Buy
        elif short_ma < long_ma and self.trigger <= 0:
            self.trigger = 1
            return -1  # Sell
        else:
            return 0  # Hold
class Agent_Donchian(TradingAgent):
    """Trading Agent for Donchian Channel Breakout."""
    def __init__(self, lookback):
        super().__init__()
        self.lookback = lookback
        self.trigger = 0

    def generate_signals(self, data):
        if len(data) < self.lookback:
            return 0  # Not enough data to calculate Donchian channels

        # Calculate Donchian Channel
        upper_band = data['high'].rolling(window=self.lookback).max().iloc[-1]
        lower_band = data['low'].rolling(window=self.lookback).min().iloc[-1]
        current_price = data['close'].iloc[-1]

        # Generate signals
        if current_price > upper_band and self.trigger >= 0:
            self.trigger = -1
            return 1  # Buy
        elif current_price < lower_band and self.trigger <= 0:
            self.trigger = 1
            return -1  # Sell
        else:
            return 0  # Hold
class Agent_RSI(TradingAgent):
    """Trading Agent for RSI-based Momentum."""
    def __init__(self, rsi_period, oversold=30, overbought=70):
        super().__init__()
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.trigger = 0

    def generate_signals(self, data):
        if len(data) < self.rsi_period:
            return 0  # Not enough data to calculate RSI

        # Calculate RSI
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=self.rsi_period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        # Generate signals
        if rsi.iloc[-1] < self.oversold and self.trigger >= 0:
            self.trigger = -1
            return 1  # Buy
        elif rsi.iloc[-1] > self.overbought and self.trigger <= 0:
            self.trigger = 1
            return -1  # Sell
        else:
            return 0  # Hold

