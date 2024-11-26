from utils import TradingAgent


class Agent_1m(TradingAgent):
    """Trading Agent for 1-minute interval."""
    def generate_signals(self, data):
        """Generate random signals for 1-minute interval."""
        import random
        return random.choice([-1, 0, 1]) # Randomly decide to Buy, Sell, or