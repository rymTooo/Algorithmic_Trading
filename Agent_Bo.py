from utils import TradingAgent

class Agent_Bo(TradingAgent):
    def __init__(self, lookback, multiplier=2):
        super().__init__()
        self.lookback = lookback
        self.multiplier = multiplier

    def generate_signals(self, data):
        if len(data) < self.lookback:
            return 0  # Not enough data for calculation
        
        # Calculate Bollinger Bands
        rolling_mean = data['close'].rolling(window=self.lookback).mean()
        rolling_std = data['close'].rolling(window=self.lookback).std()
        upper_band = rolling_mean + (rolling_std * self.multiplier)
        lower_band = rolling_mean - (rolling_std * self.multiplier)

        # Get the most recent price and band values
        recent_price = data['close'].iloc[-1]
        recent_upper = upper_band.iloc[-1]
        recent_lower = lower_band.iloc[-1]

        # Generate signals
        if recent_price < recent_lower:
            return 1  # Buy Signal
        elif recent_price > recent_upper:
            return -1  # Sell Signal
        else:
            return 0  # Hold Signal