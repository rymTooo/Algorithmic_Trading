from utils import TradingAgent


class Agent_4h(TradingAgent):
    """Trading Agent for 1-minute interval."""
    def generate_signals(self, data):
        """Generate random signals for 4-hours interval."""
        import random
        return random.choice([-1, 0, 1]) # Randomly decide to Buy, Sell, or