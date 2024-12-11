import pandas as pd
import requests

class TradingAgent:
    """Base class for trading agents."""
    def __init__(self, initial_cash=100000):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.position = 0 # Number of units of the asset
        self.history = [] # To store trade history
        self.bought_price = 0
        self.max_position = 0


    def generate_signals(self, data):
        """Generate trading signals. To be implemented by subclasses."""
        raise NotImplementedError
    

    def trade(self, price, signal):
        """Execute trades based on the signal.
        \n signal: 1 = Buy, -1 = Sell, other = Do nothing"""
        if signal == 1: # Buy
            self.position += self.cash / price
            self.bought_price  = price
            self.cash = 0
            self.history.append(f"Buy at {price}")
        elif signal == -1: # Sell
            self.max_position = 0
            self.cash += self.position * price
            self.position = 0
            self.history.append(f"Sell at {price}")
        # Hold: Do nothing
        self.history.append(f"Hold at {price}")


    def get_portfolio_value(self, price):
        """Calculate total portfolio value."""
        return self.cash + (self.position * price)

    def calculate_macd(self, data, short_window=12, long_window=26, signal_window=9):
        data['ShortEMA'] = data['Close'].ewm(span=short_window, adjust=False).mean()
        data['LongEMA'] = data['Close'].ewm(span=long_window, adjust=False).mean()
        data['MACD'] = data['ShortEMA'] - data['LongEMA']
        data['Signal Line'] = data['MACD'].ewm(span=signal_window, adjust=False).mean()
        data.drop(['ShortEMA', 'LongEMA'], axis=1, inplace=True)
        return data
    
    def calculate_macd2(self, data, short_window=12, long_window=26, signal_window=9):
        # Determine the largest window size
        largest_window = max(short_window, long_window, signal_window)
        
        # Trim the data to the largest window size
        if len(data) < largest_window*2:
            recent_data = data.copy()
        else:
            recent_data = data.iloc[-(largest_window*2):].copy()
        
        # Calculate the MACD and Signal Line on the trimmed data
        recent_data['ShortEMA'] = recent_data['Close'].ewm(span=short_window, adjust=False).mean()
        recent_data['LongEMA'] = recent_data['Close'].ewm(span=long_window, adjust=False).mean()
        recent_data['MACD'] = recent_data['ShortEMA'] - recent_data['LongEMA']
        recent_data['Signal Line'] = recent_data['MACD'].ewm(span=signal_window, adjust=False).mean()
        
        # Get the latest MACD and Signal Line values
        latest_macd = recent_data['MACD'].iloc[-1]
        latest_signal = recent_data['Signal Line'].iloc[-1]
        
        # Append the latest MACD and Signal Line to the original columns
        data.at[data.index[-1], 'MACD'] = latest_macd
        data.at[data.index[-1], 'Signal Line'] = latest_signal
        
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
    
    def calculate_rsi2(self, data, window=14):
        if len(data) >= window*2:
            recent_data = data.iloc[-(window*2):].copy()
        else:
            recent_data = data.copy()
        delta = recent_data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_ma(self, data, window=20):
        """
        Calculate the moving average (MA) for the specified column in the dataset.

        Returns:
            pd.Series: A series containing the moving average values.
        """
        # if want to modify to return last value only
        # ma = data[column].rolling(window=window).mean()
        # return ma.iloc[-1]
        if len(data) >= window:
            recent_data = data.iloc[-(window):].copy()
        else:
            recent_data = data.copy()

        ma = recent_data['Close'].rolling(window=window,min_periods=1).mean()
        return ma

    def calculate_stochastic(self, data, lookback_window=14):
        """
        Calculates the Stochastic Oscillator (%K and %D) for the given data.

        Parameters:
        - data: pandas DataFrame containing 'High', 'Low', and 'Close' columns.
        - lookback_window: The number of periods for calculating the Stochastic Oscillator.

        Returns:
        - data: pandas DataFrame with added '%K' and '%D' columns.
        """

        data['Lowest Low'] = data['Low'].rolling(window=lookback_window,min_periods=1).min()
        data['Highest High'] = data['High'].rolling(window=lookback_window,min_periods=1).max()
        data['%K'] = 100 * ((data['Close'] - data['Lowest Low']) / 
                            (data['Highest High'] - data['Lowest Low']))
        data['%D'] = data['%K'].rolling(window=3).mean()

        data.drop(['Lowest Low', 'Highest High'], axis=1, inplace=True)

        return data
    
    def calculate_stochastic2(self, data, lookback_window=14):
        """
        Calculates the Stochastic Oscillator (%K and %D) for the given data.

        Parameters:
        - data: pandas DataFrame containing 'High', 'Low', and 'Close' columns.
        - lookback_window: The number of periods for calculating the Stochastic Oscillator.

        Returns:
        - data: pandas DataFrame with added '%K' and '%D' columns.
        """
        if len(data) < lookback_window*2:
            recent_data = data.copy()
        else:
            recent_data = data.iloc[-(lookback_window*2):].copy()

        recent_data['Lowest Low'] = recent_data['Low'].rolling(window=lookback_window,min_periods=1).min()
        recent_data['Highest High'] = recent_data['High'].rolling(window=lookback_window,min_periods=1).max()
        recent_data['%K'] = 100 * ((recent_data['Close'] - recent_data['Lowest Low']) / 
                            (recent_data['Highest High'] - recent_data['Lowest Low']))
        recent_data['%D'] = recent_data['%K'].rolling(window=3).mean()

        # recent_data.drop(['Lowest Low', 'Highest High'], axis=1, inplace=True)

        data.at[data.index[-1], '%K'] = recent_data["%K"].iloc[-1]
        data.at[data.index[-1], '%D'] = recent_data["%D"].iloc[-1]

        return data
    
    def calculate_adx2(self, data, lookback_window=14):
        """
        Calculates the Average Directional Index (ADX) for the given data. To use ADX above 25 mean trending, below mean sideway

        Parameters:
        - data: pandas DataFrame containing 'High', 'Low', and 'Close' columns.
        - lookback_window: The number of periods for calculating ADX.

        Returns:
        - data: pandas DataFrame with added '+DI', '-DI', and 'ADX' columns.
        """
        if len(data) < lookback_window*2:
            recent_data = data.copy()
        else:
            recent_data = data.iloc[-(lookback_window*2):].copy()

        # Calculate True Range (TR)
        recent_data['TR'] = recent_data[['High', 'Low', 'Close']].apply(
            lambda row: max(row['High'] - row['Low'], 
                            abs(row['High'] - row['Close']), 
                            abs(row['Low'] - row['Close'])),
            axis=1
        )

        # Calculate +DM and -DM
        recent_data['+DM'] = (recent_data['High'] - recent_data['High'].shift(1)).clip(lower=0)
        recent_data['-DM'] = (recent_data['Low'].shift(1) - recent_data['Low']).clip(lower=0)

        # Replace NaN values for the first row
        recent_data.fillna(0, inplace=True)

        # Smooth the True Range (ATR), +DM, and -DM over the lookback period
        recent_data['ATR'] = recent_data['TR'].rolling(window=lookback_window,min_periods=5).mean()
        recent_data['+DI'] = 100 * (recent_data['+DM'].rolling(window=lookback_window,min_periods=5).mean() / recent_data['ATR'])
        recent_data['-DI'] = 100 * (recent_data['-DM'].rolling(window=lookback_window,min_periods=5).mean() / recent_data['ATR'])

        # Calculate the Directional Movement Index (DX)
        recent_data['DX'] = (abs(recent_data['+DI'] - recent_data['-DI']) / (recent_data['+DI'] + recent_data['-DI'])) * 100

        # Calculate the Average Directional Index (ADX)
        recent_data['ADX'] = recent_data['DX'].rolling(window=lookback_window,min_periods=5).mean()

        # Drop intermediate columns for cleanliness
        # data.drop(['TR', '+DM', '-DM', 'ATR', '+DI', '-DI'], axis=1, inplace=True)

        data.at[data.index[-1], 'DX'] = recent_data["DX"].iloc[-1]
        data.at[data.index[-1], 'ADX'] = recent_data["ADX"].iloc[-1]

        return data
    
    def calculate_adx(self, data, lookback_window=14):
        """
        Calculates the Average Directional Index (ADX) for the given data. To use ADX above 25 mean trending, below mean sideway

        Parameters:
        - data: pandas DataFrame containing 'High', 'Low', and 'Close' columns.
        - lookback_window: The number of periods for calculating ADX.

        Returns:
        - data: pandas DataFrame with added '+DI', '-DI', and 'ADX' columns.
        """
        # Calculate True Range (TR)
        data['TR'] = data[['High', 'Low', 'Close']].apply(
            lambda row: max(row['High'] - row['Low'], 
                            abs(row['High'] - row['Close']), 
                            abs(row['Low'] - row['Close'])),
            axis=1
        )

        # Calculate +DM and -DM
        data['+DM'] = (data['High'] - data['High'].shift(1)).clip(lower=0)
        data['-DM'] = (data['Low'].shift(1) - data['Low']).clip(lower=0)

        # Replace NaN values for the first row
        data.fillna(0, inplace=True)

        # Smooth the True Range (ATR), +DM, and -DM over the lookback period
        data['ATR'] = data['TR'].rolling(window=lookback_window,min_periods=5).mean()
        data['+DI'] = 100 * (data['+DM'].rolling(window=lookback_window,min_periods=5).mean() / data['ATR'])
        data['-DI'] = 100 * (data['-DM'].rolling(window=lookback_window,min_periods=5).mean() / data['ATR'])

        # Calculate the Directional Movement Index (DX)
        data['DX'] = (abs(data['+DI'] - data['-DI']) / (data['+DI'] + data['-DI'])) * 100

        # Calculate the Average Directional Index (ADX)
        data['ADX'] = data['DX'].rolling(window=lookback_window,min_periods=5).mean()

        # Drop intermediate columns for cleanliness
        data.drop(['TR', '+DM', '-DM', 'ATR', '+DI', '-DI'], axis=1, inplace=True)

        return data
    
    def calcalte_return(self, data):
        data['percentage_change'] = data['Close'].pct_change() * 100

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