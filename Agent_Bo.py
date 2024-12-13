from utils import TradingAgent

class Agent_Bo(TradingAgent):
    def __init__(self, lookback):
        super().__init__()
        self.lookback = lookback
        self.trigger = 0

    def generate_signals(self, data):
        recent_high = data['high'].rolling(self.lookback).max()
        recent_low = data['low'].rolling(self.lookback).min()
        
        # Generate signals
        if data['close'].iloc[-1] > recent_high.iloc[-1] and self.trigger <= 0:
            self.trigger = 1  
            return 1  # Buy signal
        elif data['close'].iloc[-1] < recent_low.iloc[-1] and self.trigger >= 0:
            self.trigger = -1  
            return -1  # Sell signal
        else:
            return 0  # Hold signal