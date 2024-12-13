from utils import TradingAgent

class Agent_o(TradingAgent):
    def __init__(self):
        super().__init__()
        self.trigger = 0

    def generate_signals(self, data):       
        # Generate signals
        if data['close'].iloc[-1] < data['close'].iloc[0] and self.trigger <= 0:
            self.trigger = 1  
            return 1  # Buy signal
        elif data['close'].iloc[-1] > data['close'].iloc[0] and self.trigger >= 0:
            self.trigger = -1  
            return -1  # Sell signal
        else:
            return 0  # Hold signal