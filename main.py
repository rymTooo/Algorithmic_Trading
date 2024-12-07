from Agent_1m import Agent_1m
from Agent_1h import Agent_1h
from Agent_4h import Agent_4h
from Agent_1d import Agent_1d
from utils import fetch_historical_data, calculate_performance_metrics


def backtest(agent, data):
    """Backtest a trading agent."""
    for timestamp, row in data.iterrows():
        price = row['close'] # Use the closing price for trades
        signal = agent.generate_signals(row)
        agent.trade(price, signal)
    return agent.get_portfolio_value(data['close'].iloc[-1])


# Fetch data for different intervals
df_1m = fetch_historical_data('BTCUSDT', '1m')
df_1h = fetch_historical_data('BTCUSDT', '1h')
df_4h = fetch_historical_data('BTCUSDT', '4h')
df_1d = fetch_historical_data('BTCUSDT', '1d')


# Initialize agents
agent_1m = Agent_1m()
agent_1h = Agent_1h()
agent_4h = Agent_4h()
agent_1d = Agent_1d()


# Backtest each agent
portfolio_value_1m = backtest(agent_1m, df_1m)
portfolio_value_1h = backtest(agent_1h, df_1h)
portfolio_value_4h = backtest(agent_4h, df_4h)
portfolio_value_1d = backtest(agent_1d, df_1d)

agent_1d.plot_past_data(agent_1d.past_data, df_1d.index)

# Print results
print(f"Portfolio Value for 1m Interval: {portfolio_value_1m}")
print(f"Portfolio Value for 1h Interval: {portfolio_value_1h}")
print(f"Portfolio Value for 4h Interval: {portfolio_value_4h}")
print(f"Portfolio Value for 1d Interval: {portfolio_value_1d}")

print(f"Portfolio Return for 1m Interval: ",calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_1m)["Total Return"]*100, " %")
print(f"Portfolio Return for 1h Interval: ",calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_1h)["Total Return"]*100, " %")
print(f"Portfolio Return for 4h Interval: ",calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_4h)["Total Return"]*100, " %")
print(f"Portfolio Return for 1d Interval: ",calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_1d)["Total Return"]*100, " %")

