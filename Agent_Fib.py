from utils import TradingAgent
class Agent_Fib(TradingAgent):
    def __init__(self, lookback):
        super().__init__()
        self.lookback = lookback
        self.trigger = 0
    
    def calculate_fibonacci_levels(self, high, low):
        """
        Calculate Fibonacci retracement levels based on a high and low price.
        :param high: High price during the lookback period.
        :param low: Low price during the lookback period.
        :return: Dictionary of Fibonacci retracement levels.
        """
        diff = high - low
        return {
            '0%': high,
            '23.6%': high - 0.236 * diff,
            '38.2%': high - 0.382 * diff,
            '50%': high - 0.5 * diff,
            '61.8%': high - 0.618 * diff,
            '100%': low
        }

    def generate_signals(self, data):
        if len(data) < self.lookback:
            return 0  # Not enough data to calculate levels

        # Identify the high and low in the lookback period
        high = data['close'].iloc[-self.lookback:].max()
        low = data['close'].iloc[-self.lookback:].min()

        # Calculate Fibonacci retracement levels
        fib_levels = self.calculate_fibonacci_levels(high, low)
        current_price = data['close'].iloc[-1]

        # Generate signals based on price nearing support or resistance levels
        tolerance = 0.005  # 0.5% tolerance around levels
        for level_name, level_price in fib_levels.items():
            if abs(current_price - level_price) / level_price < tolerance:
                if current_price > fib_levels['50%'] and self.trigger >= 0: # Price above 50%, treat as resistance
                    self.trigger = -1  
                    return -1  # Sell
                elif current_price < fib_levels['50%'] and self.trigger <= 0: # Price below 50%, treat as support
                    self.trigger = 1  
                    return 1  # Buy

        return 0  # Hold if no significant signal