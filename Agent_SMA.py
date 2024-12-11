from utils import TradingAgent


class Agent_SMA(TradingAgent):
    """Trading Agent for 1-minute interval."""
    def __init__(self, short_window, medium_window, long_window):
        super().__init__()
        self.short_window = short_window
        self.medium_window = medium_window
        self.long_window = long_window
        self.trigger = 0
    def generate_signals(self, data):
        if len(data) < self.long_window:
            return 0  # Not enough data to calculate the long SMA

        short_sma = data['close'].rolling(window=self.short_window).mean().iloc[-1]
        medium_sma = data['close'].rolling(window=self.medium_window).mean().iloc[-1]
        long_sma = data['close'].rolling(window=self.long_window).mean().iloc[-1]

        if short_sma > medium_sma and short_sma > long_sma and self.trigger <= 0:
            self.trigger = 1
            return 1  # Buy signal
        elif short_sma < medium_sma and short_sma < long_sma and self.trigger >= 0:
            self.trigger = -1
            return -1  # Sell signal
        else:
            return 0  # Hold