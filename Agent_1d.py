from utils import TradingAgent
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

class Agent_1d(TradingAgent):
    """Trading Agent for 1-day interval."""
    def __init__(self, initial_cash=100000):
        super().__init__(initial_cash)
        self.past_data = pd.DataFrame(columns=("Open","High","Low","Close","Volume"))
        self.trade_signal = []

    def update(self, data):
        data = data.rename({
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })
        self.past_data = pd.concat([self.past_data, data.to_frame().T], ignore_index=True)


    def generate_signals(self, data : pd.DataFrame): # data = 1 day close price at a time:  open high low close volume
        """Generate random signals for 1-day interval. \n
        signal: 1 = Buy, -1 = Sell, other = Do nothing"""
        signal = 0
        self.update(data)
        self.past_data["ma_20"] = self.calculate_ma(self.past_data, 20)
        self.past_data["ma_40"] = self.calculate_ma(self.past_data, 40) # calculating like this is efficient af but whatever
        self.past_data["rsi"] = self.calculate_rsi(self.past_data)
        self.calculate_macd(self.past_data)
        macd_cut_up = False
        macd_cut_down = False
        if len(self.past_data) >= 2:
            macd_cut_up = (self.past_data['MACD'].iloc[-1] > self.past_data['Signal Line'].iloc[-1]) & (self.past_data['MACD'].iloc[-2] <= self.past_data['Signal Line'].iloc[-2])
            macd_cut_down = (self.past_data['MACD'].iloc[-1] < self.past_data['Signal Line'].iloc[-1]) & (self.past_data['MACD'].iloc[-2] >= self.past_data['Signal Line'].iloc[-2])
        if macd_cut_up:
            signal = 1
        elif macd_cut_down:
            signal = -1
        else:
            signal = 0
        self.trade_signal.append(signal)
        return signal


    def plot_past_data(self, data : pd.DataFrame, timestamp:pd.DataFrame):
        # Ensure the timestamp passed is datetime and set it as index
        timestamp = pd.to_datetime(timestamp)  # Convert to datetime if needed
        data['timestamp'] = timestamp
        # Convert the 'timestamp' column to datetime format
        data['timestamp'] = pd.to_datetime(data['timestamp'])

        # Set the 'timestamp' column as the index
        data.set_index('timestamp', inplace=True)

        if len(self.trade_signal) == len(data):
            data['trade_signal'] = self.trade_signal
        print(self.trade_signal)


        # Create subplots
        fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        # Plot price
        axs[0].plot(timestamp, data['Close'], label='Close')
        axs[0].plot(timestamp, data['ma_20'], label='Moving Average', color='orange')
        axs[0].plot(timestamp, data['ma_40'], label='Moving Average', color='green')
        # Plot buy signals
        buy_signals = data[data['trade_signal'] == 1]
        print(buy_signals)
        axs[0].scatter(buy_signals.index, buy_signals['Close'], marker='^', color='green', label='Buy', zorder=5)

        # Plot sell signals
        sell_signals = data[data['trade_signal'] == -1]
        axs[0].scatter(sell_signals.index, sell_signals['Close'], marker='v', color='red', label='Sell', zorder=5)
        axs[0].set_title('Prices')
        axs[0].legend()

        # Plot RSI
        axs[1].plot(timestamp, data['rsi'], label='RSI', color='purple')
        axs[1].set_title('RSI')
        axs[1].axhline(y=70, color='red', linestyle='--', label='Overbought (70)')
        axs[1].axhline(y=30, color='green', linestyle='--', label='Oversold (30)')
        axs[1].legend()

        # Plot MACD and Signal Line
        axs[2].plot(timestamp, data['MACD'], label='MACD', color='blue')
        axs[2].plot(timestamp, data['Signal Line'], label='Signal Line', color='orange')
        axs[2].set_title('MACD')
        axs[2].legend()

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