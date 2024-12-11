from utils import TradingAgent


class Agent_MR(TradingAgent):
    """Trading Agent for 1-minute interval."""
    def __init__(self, lookback):
        super().__init__()
        self.lookback = lookback
        self.trigger = 0
    def generate_signals(self, data):

        if len(data) < self.lookback:
            return 0  # Not enough data to calculate moving average

        # Calculate the moving average and standard deviation
        mean = data['close'].iloc[-self.lookback:].mean()
        std_dev = data['close'].iloc[-self.lookback:].std()

        # Get the current price
        current_price = data['close'].iloc[-1]

        # Define thresholds for mean reversion
        upper_threshold = mean + std_dev
        lower_threshold = mean - std_dev

        # Generate signals
        if current_price > upper_threshold and self.trigger <= 0:
            self.trigger = 1
            return -1  # Sell
        elif current_price < lower_threshold and self.trigger <= 0:
            self.trigger = 1
            return 1  # Buy
        else:
            return 0  # Hold