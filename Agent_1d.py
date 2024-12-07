from utils import TradingAgent
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

class Agent_1d(TradingAgent):
    """Trading Agent for 1-day interval."""
    def __init__(self, initial_cash=100000):
        super().__init__(initial_cash)
        self.past_data = pd.DataFrame(columns=("Open","High","Low","Close","Volume"))

    def update(self, data):
        data = data.rename({
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })
        self.past_data = pd.concat([self.past_data, data.to_frame().T], ignore_index=True)


    def generate_signals(self, data): # data = 1 day close price at a time:  open high low close volume
        """Generate random signals for 1-day interval. \n
        signal: 1 = Buy, -1 = Sell, other = Do nothing"""
        self.update(data)
        self.past_data["ma"] = self.calculate_ma(self.past_data, 20)
        self.past_data["rsi"] = self.calculate_rsi(self.past_data)
        # print(self.past_data)

    def plot_past_data(self, data, timestamp):
        plot_num = 1
        have_ma = False
        have_rsi = False
        if "rsi" in data.columns:
            have_rsi = True
            plot_num += 1
        if "ma" in data.columns:
            have_ma = True
        # Ensure the timestamp passed is datetime and set it as index
        timestamp = pd.to_datetime(timestamp)  # Convert to datetime if needed
        data['timestamp'] = timestamp
        # Convert the 'timestamp' column to datetime format
        data['timestamp'] = pd.to_datetime(data['timestamp'])

        # Set the 'timestamp' column as the index
        data.set_index('timestamp', inplace=True)

        # Create subplots
        fig, axs = plt.subplots(plot_num, 1, figsize=(10, 8), sharex=True)

        # Plot price
        axs[0].plot(timestamp, data['Close'], label='Close')
        if have_ma:
            axs[0].plot(timestamp, data['ma'], label='Moving Average', color='orange')
        axs[0].set_title('Prices')
        axs[0].legend()

        # Plot RSI
        if have_rsi:
            axs[1].plot(timestamp, data['rsi'], label='RSI', color='purple')
            axs[1].set_title('RSI')
            axs[1].axhline(y=70, color='red', linestyle='--', label='Overbought (70)')
            axs[1].axhline(y=30, color='green', linestyle='--', label='Oversold (30)')
            axs[1].legend()

        # Add grid and labels
        for ax in axs:
            ax.grid(True)
            ax.set_ylabel('Value')

        # Format x-axis as dates
        axs[-1].set_xlabel('Date')
        axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        axs[-1].xaxis.set_major_locator(mdates.AutoDateLocator())

        # Rotate date labels for better readability
        plt.setp(axs[-1].xaxis.get_majorticklabels(), rotation=45)

        # Show plot
        plt.tight_layout()
        plt.show()