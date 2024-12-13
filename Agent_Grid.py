from utils import TradingAgent

class Agent_Grid(TradingAgent):
    def __init__(self, lower_bound, upper_bound, grid_size):
        super().__init__()
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.grid_size = grid_size
        self.grid_levels = self._calculate_grid()

    def _calculate_grid(self):
        """Calculate grid levels between lower and upper bounds."""
        import numpy as np
        return np.linspace(self.lower_bound, self.upper_bound, self.grid_size)

    def generate_signals(self, data):
        """
        Generate trading signals based on grid levels.
        signal: 1 = Buy, -1 = Sell, 0 = Hold
        """
        current_price = data['close'].iloc[-1]  # Get the most recent price

        if any(current_price <= level for level in self.grid_levels):  # Price falls to a grid level
            return 1  # Buy signal
        elif any(current_price >= level for level in self.grid_levels):  # Price rises to a grid level
            return -1  # Sell signal
        else:
            return 0  # Hold signal   