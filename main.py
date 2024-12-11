import matplotlib.pyplot as plt
from Agent_1m import Agent_1m
from Agent_1h import Agent_1h
from Agent_4h import Agent_4h
from Agent_1d import Agent_1d
from Agent_SMA import Agent_SMA
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
agent_1d_SMA= Agent_SMA(short_window=10, medium_window=20, long_window=50)

# Backtest each agent
portfolio_value_1m = backtest(agent_1m, df_1m)
portfolio_value_1h = backtest(agent_1h, df_1h)
portfolio_value_4h = backtest(agent_4h, df_4h)
portfolio_value_1d = backtest(agent_1d, df_1d)

portfolio_value_1m_SMA = backtest(agent_1m_SMA, df_1m)
portfolio_value_1h_SMA = backtest(agent_1h_SMA, df_1h)
portfolio_value_4h_SMA = backtest(agent_4h_SMA, df_4h)
portfolio_value_1d_SMA = backtest(agent_1d_SMA, df_1d)

# Print results
print(f"Portfolio Value for 1m Interval: {portfolio_value_1m}")
print(f"Portfolio Value for 1h Interval: {portfolio_value_1h}")
print(f"Portfolio Value for 4h Interval: {portfolio_value_4h}")
print(f"Portfolio Value for 1d Interval: {portfolio_value_1d}")

print(f"Portfolio Value for 1m SMA Interval: {portfolio_value_1m_SMA}")
print(f"Portfolio Value for 1h SMA Interval: {portfolio_value_1h_SMA}")
print(f"Portfolio Value for 4h SMA Interval: {portfolio_value_4h_SMA}")
print(f"Portfolio Value for 1d SMA Interval: {portfolio_value_1d_SMA}")

print(f"Portfolio Return for 1m Interval: ",calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_1m)["Total Return"]*100, " %")
print(f"Portfolio Return for 1h Interval: ",calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_1h)["Total Return"]*100, " %")
print(f"Portfolio Return for 4h Interval: ",calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_4h)["Total Return"]*100, " %")
print(f"Portfolio Return for 1d Interval: ",calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_1d)["Total Return"]*100, " %")

print(f"Portfolio Return for 1m SMA Interval: ",calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_1m_SMA)["Total Return"]*100, " %")
print(f"Portfolio Return for 1h SMA Interval: ",calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_1h_SMA)["Total Return"]*100, " %")
print(f"Portfolio Return for 4h SMA Interval: ",calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_4h_SMA)["Total Return"]*100, " %")
print(f"Portfolio Return for 1d SMA Interval: ",calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_1d_SMA)["Total Return"]*100, " %")

# print(agent_1m_SMA.history)