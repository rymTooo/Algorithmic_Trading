import matplotlib.pyplot as plt
from Agent_1m import Agent_1m
from Agent_1h import Agent_1h
from Agent_4h import Agent_4h
from Agent_1d import Agent_1d
from Agent_SMA import Agent_SMA
from Agent_MR import Agent_MR
from Agent_Fib import Agent_Fib
from Agent_Bo import Agent_Bo
from Agent_Grid import Agent_Grid
from utils import fetch_historical_data, calculate_performance_metrics


def backtest(agent, data):
    """Backtest a trading agent."""
    for timestamp, row in data.iterrows():
        current_data = data.loc[:timestamp]
        price = row['close']
        signal = agent.generate_signals(current_data)
        agent.trade(price, signal)
    return agent.get_portfolio_value(data['close'].iloc[-1])

def plot_historical_data(dataframes, timeframes, title="BTCUSDT Historical Data"):
    plt.figure(figsize=(15, 10))
    
    for i in range(len(dataframes)):
        df = dataframes[i]
        tf = timeframes[i]
        plt.subplot(2, 2, i+1)  # Create a 2x2 grid of plots
        plt.plot(df['close'], label=f'Close Price ({tf})', linewidth=1)
        plt.title(f'{tf} Timeframe')
        plt.xlabel('Timestamp')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid()
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to fit the title
    plt.show()

# Fetch data for different intervals
df_1m = fetch_historical_data('BTCUSDT', '1m')
df_1h = fetch_historical_data('BTCUSDT', '1h')
df_4h = fetch_historical_data('BTCUSDT', '4h')
df_1d = fetch_historical_data('BTCUSDT', '1d')

# print(df_1m.size)
# print(df_1h.size)
# print(df_4h.size)
# print(df_1d.size)

dataframes = [df_1m, df_1h, df_4h, df_1d]
timeframes = ['1 Minute', '1 Hour', '4 Hour', '1 Day']
# plot_historical_data(dataframes, timeframes)

# Initialize agents
agent_1m = Agent_1m()
agent_1h = Agent_1h()
agent_4h = Agent_4h()
agent_1d = Agent_1d()

agent_1m_SMA= Agent_SMA(short_window=10, medium_window=20, long_window=50)
agent_1h_SMA= Agent_SMA(short_window=10, medium_window=20, long_window=50)
agent_4h_SMA= Agent_SMA(short_window=10, medium_window=20, long_window=50)
agent_1d_SMA= Agent_SMA(short_window=50, medium_window=100, long_window=200)

agent_1m_MR = Agent_MR(lookback=10)
agent_1h_MR = Agent_MR(lookback=20)
agent_4h_MR = Agent_MR(lookback=50)
agent_1d_MR = Agent_MR(lookback=200)

agent_1m_Fib = Agent_Fib(lookback=10)
agent_1h_Fib = Agent_Fib(lookback=20)
agent_4h_Fib = Agent_Fib(lookback=50)
agent_1d_Fib = Agent_Fib(lookback=200)

agent_1m_Bo = Agent_Bo(lookback=10, multiplier=1.5)
agent_1h_Bo = Agent_Bo(lookback=20, multiplier=1.5)
agent_4h_Bo = Agent_Bo(lookback=50, multiplier=1.5)
agent_1d_Bo = Agent_Bo(lookback=200, multiplier=1.5)

agent_1m_Grid = Agent_Grid(lower_bound = 99000, upper_bound = 100000, grid_size = 15)
agent_1h_Grid = Agent_Grid(lower_bound = 70000, upper_bound = 100000, grid_size = 15)
agent_4h_Grid = Agent_Grid(lower_bound = 50000, upper_bound = 100000, grid_size = 15)
agent_1d_Grid = Agent_Grid(lower_bound = 20000, upper_bound = 100000, grid_size = 15)

# Backtest each agent
# portfolio_value_1m = backtest(agent_1m, df_1m)
# portfolio_value_1h = backtest(agent_1h, df_1h)
# portfolio_value_4h = backtest(agent_4h, df_4h)
# portfolio_value_1d = backtest(agent_1d, df_1d)

# portfolio_value_1m_SMA = backtest(agent_1m_SMA, df_1m)
# portfolio_value_1h_SMA = backtest(agent_1h_SMA, df_1h)
# portfolio_value_4h_SMA = backtest(agent_4h_SMA, df_4h)
# portfolio_value_1d_SMA = backtest(agent_1d_SMA, df_1d)

# portfolio_value_1m_MR = backtest(agent_1m_MR, df_1m)
# portfolio_value_1h_MR = backtest(agent_1h_MR, df_1h)
# portfolio_value_4h_MR = backtest(agent_4h_MR, df_4h)
# portfolio_value_1d_MR = backtest(agent_1d_MR, df_1d)

# portfolio_value_1m_Fib = backtest(agent_1m_Fib, df_1m)
# portfolio_value_1h_Fib = backtest(agent_1h_Fib, df_1h)
# portfolio_value_4h_Fib = backtest(agent_4h_Fib, df_4h)
# portfolio_value_1d_Fib = backtest(agent_1d_Fib, df_1d)

portfoloio_value_1m_Bo = backtest(agent_1m_Bo, df_1m)
portfoloio_value_1h_Bo = backtest(agent_1h_Bo, df_1h)
portfoloio_value_4h_Bo = backtest(agent_4h_Bo, df_4h)
portfoloio_value_1d_Bo = backtest(agent_1d_Bo, df_1d)


portfolio_value_1m_Grid = backtest(agent_1m_Grid, df_1m)
portfolio_value_1h_Grid = backtest(agent_1h_Grid, df_1h)
portfolio_value_4h_Grid = backtest(agent_4h_Grid, df_4h)
portfolio_value_1d_Grid = backtest(agent_1d_Grid, df_1d)

# Print results
# print(f"Portfolio Value for 1m Interval: {portfolio_value_1m}")
# print(f"Portfolio Value for 1h Interval: {portfolio_value_1h}")
# print(f"Portfolio Value for 4h Interval: {portfolio_value_4h}")
# print(f"Portfolio Value for 1d Interval: {portfolio_value_1d}")

# print(f"Portfolio Value for 1m SMA Interval: {portfolio_value_1m_SMA}")
# print(f"Portfolio Value for 1h SMA Interval: {portfolio_value_1h_SMA}")
# print(f"Portfolio Value for 4h SMA Interval: {portfolio_value_4h_SMA}")
# print(f"Portfolio Value for 1d SMA Interval: {portfolio_value_1d_SMA}")

# print(f"Portfolio Value for 1m MR Interval: {portfolio_value_1m_MR}")
# print(f"Portfolio Value for 1h MR Interval: {portfolio_value_1h_MR}")
# print(f"Portfolio Value for 4h MR Interval: {portfolio_value_4h_MR}")
# print(f"Portfolio Value for 1d MR Interval: {portfolio_value_1d_MR}")

# print(f"Portfolio Value for 1m Fib Interval: {portfolio_value_1m_Fib}")
# print(f"Portfolio Value for 1h Fib Interval: {portfolio_value_1h_Fib}")
# print(f"Portfolio Value for 4h Fib Interval: {portfolio_value_4h_Fib}")
# print(f"Portfolio Value for 1d Fib Interval: {portfolio_value_1d_Fib}")

print(f"Portfolio Value for 1m Bo Interval: {portfoloio_value_1m_Bo}")
print(f"Portfolio Value for 1h Bo Interval: {portfoloio_value_1h_Bo}")
print(f"Portfolio Value for 4h Bo Interval: {portfoloio_value_4h_Bo}")
print(f"Portfolio Value for 1d Bo Interval: {portfoloio_value_1d_Bo}")

print(f"Portfolio Value for 1m Grid Interval: {portfolio_value_1m_Grid}")
print(f"Portfolio Value for 1h Grid Interval: {portfolio_value_1h_Grid}")
print(f"Portfolio Value for 4h Grid Interval: {portfolio_value_4h_Grid}")
print(f"Portfolio Value for 1d Grid Interval: {portfolio_value_1d_Grid}")

# print(f"Portfolio Return for 1m Interval: ",calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_1m)["Total Return"]*100, " %")
# print(f"Portfolio Return for 1h Interval: ",calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_1h)["Total Return"]*100, " %")
# print(f"Portfolio Return for 4h Interval: ",calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_4h)["Total Return"]*100, " %")
# print(f"Portfolio Return for 1d Interval: ",calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_1d)["Total Return"]*100, " %")

# print(f"Portfolio Return for 1m SMA Interval: ",calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_1m_SMA)["Total Return"]*100, " %")
# print(f"Portfolio Return for 1h SMA Interval: ",calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_1h_SMA)["Total Return"]*100, " %")
# print(f"Portfolio Return for 4h SMA Interval: ",calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_4h_SMA)["Total Return"]*100, " %")
# print(f"Portfolio Return for 1d SMA Interval: ",calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_1d_SMA)["Total Return"]*100, " %")

# print(f"Portfolio Return for 1m MR Interval: ",calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_1m_MR)["Total Return"]*100, " %")
# print(f"Portfolio Return for 1h MR Interval: ",calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_1h_MR)["Total Return"]*100, " %")
# print(f"Portfolio Return for 4h MR Interval: ",calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_4h_MR)["Total Return"]*100, " %")
# print(f"Portfolio Return for 1d MR Interval: ",calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_1d_MR)["Total Return"]*100, " %")

# print(f"Portfolio Return for 1m Fib Interval: ",calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_1m_Fib)["Total Return"]*100, " %")
# print(f"Portfolio Return for 1h Fib Interval: ",calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_1h_Fib)["Total Return"]*100, " %")
# print(f"Portfolio Return for 4h Fib Interval: ",calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_4h_Fib)["Total Return"]*100, " %")
# print(f"Portfolio Return for 1d Fib Interval: ",calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_1d_Fib)["Total Return"]*100, " %")

print(f"Portfolio Return for 1m Bo Interval: ",calculate_performance_metrics(initial_cash=100000, final_value=portfoloio_value_1m_Bo)["Total Return"]*100, " %")
print(f"Portfolio Return for 1h Bo Interval: ",calculate_performance_metrics(initial_cash=100000, final_value=portfoloio_value_1h_Bo)["Total Return"]*100, " %")
print(f"Portfolio Return for 4h Bo Interval: ",calculate_performance_metrics(initial_cash=100000, final_value=portfoloio_value_4h_Bo)["Total Return"]*100, " %")
print(f"Portfolio Return for 1d Bo Interval: ",calculate_performance_metrics(initial_cash=100000, final_value=portfoloio_value_1d_Bo)["Total Return"]*100, " %")

print(f"Portfolio Return for 1m Grid Interval: ",calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_1m_Grid)["Total Return"]*100, " %")
print(f"Portfolio Return for 1h Grid Interval: ",calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_1h_Grid)["Total Return"]*100, " %")
print(f"Portfolio Return for 4h Grid Interval: ",calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_4h_Grid)["Total Return"]*100, " %")
print(f"Portfolio Return for 1d Grid Interval: ",calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_1d_Grid)["Total Return"]*100, " %")

# print(agent_1m_MR.history)