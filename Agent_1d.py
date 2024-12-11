from utils import TradingAgent
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from constant import *

class Agent_1d(TradingAgent):
    """Trading Agent for 1-day interval."""
    def __init__(self, initial_cash=100000):
        super().__init__(initial_cash)
        self.past_data = pd.DataFrame(columns=("Open","High","Low","Close","Volume"))
        self.trade_signal = []
        self.macd_trade_signal = []
        self.sto_trade_signal = []

    def update(self, data):
        data = data.rename({
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })
        self.past_data = pd.concat([self.past_data, data.to_frame().T], ignore_index=True)

    def macd_strat(self):
        macd_signal = 0
        macd_cut_down = False
        macd_cut_up = False
        if len(self.past_data) >= 2:
            macd_cut_up = (self.past_data['MACD'].iloc[-1] > self.past_data['Signal Line'].iloc[-1]) & (self.past_data['MACD'].iloc[-2] <= self.past_data['Signal Line'].iloc[-2])
            macd_cut_down = (self.past_data['MACD'].iloc[-1] < self.past_data['Signal Line'].iloc[-1]) & (self.past_data['MACD'].iloc[-2] >= self.past_data['Signal Line'].iloc[-2])
            if macd_cut_up:
                macd_signal = 1
            elif macd_cut_down and self.trend != 1:
                # print("sell at trend = ", self.trend)
                macd_signal = -1
            else:
                macd_signal = 0
        else:
            macd_signal = 0

        return macd_signal
    
    def macd_strat2(self):
        macd_signal = 0
        if len(self.past_data) >= 3:
            macd_change = (self.past_data['MACD'].iloc[-1] - self.past_data['MACD'].iloc[-3]) / self.past_data['MACD'].iloc[-3]
            macd_cut_down = (self.past_data['MACD'].iloc[-1] < self.past_data['Signal Line'].iloc[-1]) & (self.past_data['MACD'].iloc[-2] >= self.past_data['Signal Line'].iloc[-2])
            if macd_change > 0.05  and self.position == 0:
                macd_signal = 1
            elif macd_change < -0.04 and self.trend != 1 and self.position != 0:
                macd_signal = -1
            else:
                macd_signal = 0
        else:
            macd_signal = 0

        return macd_signal

    def sell_if_drop_strat(self, drop_threshold = 0.02, macd_threshold = 0.02):
        drop = ( self.past_data['Close'].iloc[-1] - self.max_position ) / self.max_position
        if len(self.past_data) > 3:
            macd_condition = (self.past_data['MACD'].iloc[-1] - self.past_data['MACD'].iloc[-2])/self.past_data['MACD'].iloc[-2] < -macd_threshold
        else:
            macd_condition = True
        if drop < -drop_threshold and macd_condition:
            return True
        elif drop < -(drop_threshold + 0.05):
            return True
        else:
            return False
        

    def sto_strat(self, bottom_threshold = 20, top_threshold = 80):
        stochastic_buy = False
        stochastic_sell = False
        if len(self.past_data) >= 3:
            stochastic_buy = (self.past_data['%D'].iloc[-2] <= bottom_threshold) & (self.past_data['%D'].iloc[-1] > bottom_threshold) & (self.past_data['%D'].iloc[-1] - self.past_data['%D'].iloc[-2] > 3)
            stochastic_sell = (self.past_data['%D'].iloc[-2] >= top_threshold + 2) & (self.past_data['%D'].iloc[-1] < top_threshold) & (self.past_data['%D'].iloc[-2] - self.past_data['%D'].iloc[-1] > 3)
            if stochastic_buy:
                sto_signal = 1
            elif stochastic_sell:
                sto_signal = -1
            else:
                sto_signal = 0
        else:
            sto_signal = 0

        return sto_signal




    def generate_signals(self, data : pd.DataFrame): # data = 1 day close price at a time:  open high low close volume
        """Generate random signals for 1-day interval. \n
        signal: 1 = Buy, -1 = Sell, other = Do nothing"""
        signal = 0
        self.update(data)
        if self.position != 0:
            self.max_position = max(self.max_position, data['close'])
        self.calcalte_return(self.past_data)
        latest_ma_20 = self.calculate_ma(self.past_data).iloc[-1]
        self.past_data.loc[self.past_data.index[-1], "ma_20"] = latest_ma_20
        latest_ma_40 = self.calculate_ma(self.past_data,40).iloc[-1]
        self.past_data.loc[self.past_data.index[-1], "ma_40"] = latest_ma_40
        latest_ma_80 = self.calculate_ma(self.past_data,80).iloc[-1]
        self.past_data.loc[self.past_data.index[-1], "ma_80"] = latest_ma_80
        self.trend, self.is_positive_trend = self.calculate_trend(self.past_data[["ma_40"]])
        latest_rsi = self.calculate_rsi2(self.past_data).iloc[-1]
        self.past_data.loc[self.past_data.index[-1], "rsi"] = latest_rsi
        self.calculate_macd2(self.past_data)
        self.calculate_stochastic2(self.past_data)
        self.calculate_adx2(self.past_data)
        # self.past_data.drop(['Open', 'High', 'Low'], axis=1, inplace=True)
        self.past_data.to_csv('./data/your_file.csv', index=False)
        sto_signal = 0
        macd_signal = 0

        
        
        # MACD strat For upward trend / sideway
        if self.trend != -1: 
            macd_signal = self.macd_strat()
            self.macd_trade_signal.append(macd_signal)
        else:
            self.macd_trade_signal.append(0)

        #Stochastic Strat for side way handling
        
        # elif trend == 0:
        sto_signal = self.sto_strat()
        self.sto_trade_signal.append(sto_signal)

        signal = macd_signal


        # cut loss section
        if self.position != 0:
            if self.sell_if_drop_strat(drop_threshold=DAY_DROP_THRESHOLD, macd_threshold=DAY_MACD_DROP):
                signal = -1

            if self.cut_loss(self.past_data["Close"].iloc[-1], self.bought_price):
                signal = -1
                self.trade_signal.append(signal)
                return signal

        if self.position == 0 and signal == -1:
            signal = 0
        if self.position != 0 and signal == 1:
            signal = 0


        self.trade_signal.append(signal)
        return signal
    
    def calculate_trend(self,ma):
        """
        return 1, 0, -1 for positive, sideway, correction
        """
        if len(ma) >= 3:
            slope = ((ma.iloc[-1]- ma.iloc[-3])/ma.iloc[-3]).iloc[0]
            is_positive = slope > 0
            if slope > DAY_POSITIVE_TREND_THRESHOLD:
                return 1, is_positive
            elif slope > DAY_SIDEWAY_TREND_THRESHOLD:
                return 0, is_positive
            else:
                return -1, is_positive
        if len(ma) == 2:
            slope = ((ma.iloc[-1]- ma.iloc[-2])/ma.iloc[-2]).iloc[0]
            is_positive = slope > 0
            if slope > DAY_POSITIVE_TREND_THRESHOLD:
                return 1, is_positive
            elif slope > DAY_SIDEWAY_TREND_THRESHOLD:
                return 0, is_positive
            else:
                return -1, is_positive
        else:
            return 0, False
        
    def cut_loss(self, current_price, bought_price):
        if self.position != 0:
            loss = (current_price - bought_price)/ bought_price
            if loss <= -DAY_CUTLOSS:
                return True
            else:
                return False
        else:
            return False





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

        # print(len(self.macd_trade_signal))
        # print(len(data))
        if len(self.macd_trade_signal) == len(data):
            data['macd_trade_signal'] = self.macd_trade_signal

        if len(self.sto_trade_signal) == len(data):
            data['sto_trade_signal'] = self.sto_trade_signal

        buy_indices = data[data['trade_signal'] == 1].index
        sell_indices = data[data['trade_signal'] == -1].index

        pairs = []
        for buy in buy_indices:
            sell = sell_indices[sell_indices > buy]
            if not sell.empty:
                sell = sell[0]
                pairs.append((buy, sell))

        


        # Create subplots
        fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        # Plot price
        axs[0].plot(timestamp, data['Close'], label='Close')
        axs[0].plot(timestamp, data['ma_20'], label='Moving Average 20', color='orange')
        axs[0].plot(timestamp, data['ma_40'], label='Moving Average 40', color='green')

        # # Plot signals
        # buy_signals = data[data['trade_signal'] == 1]
        # axs[0].scatter(buy_signals.index, buy_signals['Close'], marker='^', color='green', label='Buy', zorder=5)
        # sell_signals = data[data['trade_signal'] == -1]
        # axs[0].scatter(sell_signals.index, sell_signals['Close'], marker='v', color='red', label='Sell', zorder=5)

        # Plot signals
        buy_signals = data[data['macd_trade_signal'] == 1]
        axs[0].scatter(buy_signals.index, buy_signals['Close'], marker='^', color='blue', label='Buy', zorder=5)
        sell_signals = data[data['macd_trade_signal'] == -1]
        axs[0].scatter(sell_signals.index, sell_signals['Close'], marker='v', color='orange', label='Sell', zorder=5)

        # Plot signals
        # buy_signals = data[data['sto_trade_signal'] == 1]
        # axs[0].scatter(buy_signals.index, buy_signals['Close'], marker='^', color='blue', label='Buy', zorder=5)
        # sell_signals = data[data['sto_trade_signal'] == -1]
        # axs[0].scatter(sell_signals.index, sell_signals['Close'], marker='v', color='orange', label='Sell', zorder=5)

        # plot box for profit visualization
        for buy, sell in pairs:
            profit = data.loc[sell, 'Close'] - data.loc[buy, 'Close']
            color = 'green' if profit > 0 else 'red'

            # Create a rectangle from buy to sell
            rect = Rectangle(
                (buy, data.loc[buy, 'Close']),  # Bottom-left corner (x, y)
                width=sell - buy,  # Width spans from buy to sell
                height=data.loc[sell, 'Close'] - data.loc[buy, 'Close'],  # Height spans price change
                color=color,
                alpha=0.3,
            )
            axs[0].add_patch(rect)

        axs[0].set_title('Prices')
        axs[0].legend()

        # # Plot RSI
        # axs[1].plot(timestamp, data['rsi'], label='RSI', color='purple')
        # axs[1].set_title('RSI')
        # axs[1].axhline(y=70, color='red', linestyle='--', label='Overbought (70)')
        # axs[1].axhline(y=30, color='green', linestyle='--', label='Oversold (30)')
        # axs[1].legend()

        # Plot Stochastic Oscillator (%K and %D)
        axs[1].plot(data.index, data['%K'], label='%K', color='blue')
        axs[1].plot(data.index, data['%D'], label='%D', color='orange')
        axs[1].set_title('Stochastic Oscillator')
        axs[1].axhline(y=80, color='red', linestyle='--', label='Overbought (80)')
        axs[1].axhline(y=20, color='green', linestyle='--', label='Oversold (20)')
        axs[1].legend()

        # # Plot MACD and Signal Line
        # axs[1].plot(timestamp, data['MACD'], label='MACD', color='blue')
        # axs[1].plot(timestamp, data['Signal Line'], label='Signal Line', color='orange')
        # axs[1].set_title('MACD')
        # axs[1].legend()

        # Plot ADX
        axs[2].plot(timestamp, data['ADX'], label='ADX', color='red')
        # axs[2].plot(timestamp, data['DX'], label='DX', color='green')
        axs[2].axhline(y=25, color='red', linestyle='--', label='threshold')
        axs[2].set_title('ADX')
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