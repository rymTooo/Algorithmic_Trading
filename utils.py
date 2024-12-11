import pandas as pd
import requests

class TradingAgent:
    """Base class for trading agents."""
    def __init__(self, initial_cash=100000):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.position = 0 # Number of units of the asset
        self.history = [] # To store trade history


    def generate_signals(self, data):
        """Generate trading signals. To be implemented by subclasses."""
        raise NotImplementedError
    

    def trade(self, price, signal):
        """Execute trades based on the signal.
        \n signal: 1 = Buy, -1 = Sell, other = Do nothing"""
        if signal == 1:  # Buy
            if self.cash > 0:  # Ensure there's cash to buy
                units_to_buy = self.cash / price
                self.position += units_to_buy
                self.cash = 0
                self.history.append(f"Buy at {price:.2f}, units: {units_to_buy:.6f}")
        elif signal == -1:  # Sell
            if self.position > 0:  # Ensure there's a position to sell
                self.cash += self.position * price
                self.history.append(f"Sell at {price:.2f}, position: {self.position:.6f}")
                self.position = 0
        else:  # Hold: Do nothing
            self.history.append(f"Hold at {price:.2f}")


    def get_portfolio_value(self, price):
        """Calculate total portfolio value."""
        return self.cash + (self.position * price)

    def calculate_macd(self, data, short_window=12, long_window=26, signal_window=9):
        data['ShortEMA'] = data['Close'].ewm(span=short_window, adjust=False).mean()
        data['LongEMA'] = data['Close'].ewm(span=long_window, adjust=False).mean()
        data['MACD'] = data['ShortEMA'] - data['LongEMA']
        data['Signal Line'] = data['MACD'].ewm(span=signal_window, adjust=False).mean()
        return data
    
    def calculate_rsi(self, data, window=14):
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_ma(self, data, window=20, column='Close'):
        """
        Calculate the moving average (MA) for the specified column in the dataset.

        Returns:
            pd.Series: A series containing the moving average values.
        """
        # if want to modify to return last value only
        # ma = data[column].rolling(window=window).mean()
        # return ma.iloc[-1]
        data[f'MA_{window}'] = data[column].rolling(window=window,min_periods=1).mean()
        return data

    def calculate_stochastic(self, data, lookback_window=14):
        """
        Calculates the Stochastic Oscillator (%K and %D) for the given data.

        Parameters:
        - data: pandas DataFrame containing 'High', 'Low', and 'Close' columns.
        - lookback_window: The number of periods for calculating the Stochastic Oscillator.

        Returns:
        - data: pandas DataFrame with added '%K' and '%D' columns.
        """
        # Calculate the lowest low and highest high over the lookback period
        data['Lowest Low'] = data['Low'].rolling(window=lookback_window,min_periods=1).min()
        data['Highest High'] = data['High'].rolling(window=lookback_window,min_periods=1).max()

        # Calculate the %K line
        data['%K'] = 100 * ((data['Close'] - data['Lowest Low']) / 
                            (data['Highest High'] - data['Lowest Low']))

        # Calculate the %D line (3-period simple moving average of %K)
        data['%D'] = data['%K'].rolling(window=3).mean()

        # Drop intermediate columns to keep the DataFrame clean
        data.drop(['Lowest Low', 'Highest High'], axis=1, inplace=True)

        return data
        

def fetch_historical_data(symbol, interval, limit=1000):
    """Fetch historical data from Binance API.
    \nReturn Dataframe with ['open', 'high', 'low', 'close', 'volume'] columns"""

    url = f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}'
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
    'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
    'taker_buy_quote_asset_volume', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df


def calculate_performance_metrics(initial_cash, final_value):
    """Calculate performance metrics."""
    total_return = (final_value - initial_cash) / initial_cash
    return {"Total Return": total_return}