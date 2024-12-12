import matplotlib.pyplot as plt
from Agent_1m import Agent_1m
from Agent_1h import Agent_1h
from Agent_4h import Agent_4h
from Agent_1d import Agent_1d
from Agent_SMA import Agent_SMA
from Agent_MR import Agent_MR
from DQN_Agent import DQN_Agent
from utils import fetch_historical_data, calculate_performance_metrics
from Agent_Momentum import Agent_BollingerBands, Agent_Donchian, Agent_MACrossover, Agent_MR_Stat, Agent_Momentum,Agent_Arbitrage,Agent_BlackSwan,Agent_InverseVolatility,Agent_MarketTiming, Agent_RSI,Agent_RiskOnRiskOff,Agent_TrendFollowing, Agent_XGBoost


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

agent_1m_MR = Agent_MR(lookback=10)
agent_1h_MR = Agent_MR(lookback=10)
agent_4h_MR = Agent_MR(lookback=30)
agent_1d_MR = Agent_MR(lookback=30)



# Backtest each agent
# portfolio_value_1m = backtest(agent_1m, df_1m)
# portfolio_value_1h = backtest(agent_1h, df_1h)
# portfolio_value_4h = backtest(agent_4h, df_4h)
# portfolio_value_1d = backtest(agent_1d, df_1d)

# portfolio_value_1m_SMA = backtest(agent_1m_SMA, df_1m)
# portfolio_value_1h_SMA = backtest(agent_1h_SMA, df_1h)
# portfolio_value_4h_SMA = backtest(agent_4h_SMA, df_4h)
# portfolio_value_1d_SMA = backtest(agent_1d_SMA, df_1d)

portfolio_value_1m_MR = backtest(agent_1m_MR, df_1m)
portfolio_value_1h_MR = backtest(agent_1h_MR, df_1h)
portfolio_value_4h_MR = backtest(agent_4h_MR, df_4h)
portfolio_value_1d_MR = backtest(agent_1d_MR, df_1d)

threshold = 20
z_score_threshold = 2
trend_threshold = 0.03
volatility_threshold = 1.5
lookback = 200

# Initialize agents for each strategy
agent_momentum_1m = Agent_Momentum(20)
agent_trendfollowing_1m = Agent_TrendFollowing(20)
agent_riskonriskoff_1m = Agent_RiskOnRiskOff(threshold)
agent_arbitrage_1m = Agent_Arbitrage(0.01)
agent_blackswan_1m = Agent_BlackSwan(z_score_threshold)
agent_markettiming_1m = Agent_MarketTiming(trend_threshold,volatility_threshold)
agent_inversevolatility_1m = Agent_InverseVolatility(20)

agent_mr_stat_1m = Agent_MR_Stat(lookback=600, z_score_threshold=2)
agent_mr_stat_1h = Agent_MR_Stat(lookback=200, z_score_threshold=2)
agent_mr_stat_4h = Agent_MR_Stat(lookback=200, z_score_threshold=2)
agent_mr_stat_1d = Agent_MR_Stat(lookback=200, z_score_threshold=2)

# agent_xgboost_1m = Agent_XGBoost()
# agent_xgboost_1h = Agent_XGBoost()
# agent_xgboost_4h = Agent_XGBoost()
# agent_xgboost_1d = Agent_XGBoost()

# # Example training for XGBoost
# agent_xgboost_1m.train(df_1m)
# agent_xgboost_1h.train(df_1h)
# agent_xgboost_4h.train(df_4h)
# agent_xgboost_1d.train(df_1d)

# Backtest for 1m Interval
portfolio_value_momentum_1m = backtest(agent_momentum_1m, df_1m)
portfolio_value_trendfollowing_1m = backtest(agent_trendfollowing_1m, df_1m)
portfolio_value_riskonriskoff_1m = backtest(agent_riskonriskoff_1m, df_1m)
portfolio_value_arbitrage_1m = backtest(agent_arbitrage_1m, df_1m)
portfolio_value_blackswan_1m = backtest(agent_blackswan_1m, df_1m)
portfolio_value_markettiming_1m = backtest(agent_markettiming_1m, df_1m)
portfolio_value_inversevolatility_1m = backtest(agent_inversevolatility_1m, df_1m)

# Repeat the same for 1h, 4h, 1d intervals
agent_momentum_1h = Agent_Momentum(50)
agent_trendfollowing_1h = Agent_TrendFollowing(50)
agent_riskonriskoff_1h = Agent_RiskOnRiskOff(threshold)
agent_arbitrage_1h = Agent_Arbitrage(0.01)
agent_blackswan_1h = Agent_BlackSwan(z_score_threshold)
agent_markettiming_1h = Agent_MarketTiming(trend_threshold,volatility_threshold)
agent_inversevolatility_1h = Agent_InverseVolatility(50)

portfolio_value_momentum_1h = backtest(agent_momentum_1h, df_1h)
portfolio_value_trendfollowing_1h = backtest(agent_trendfollowing_1h, df_1h)
portfolio_value_riskonriskoff_1h = backtest(agent_riskonriskoff_1h, df_1h)
portfolio_value_arbitrage_1h = backtest(agent_arbitrage_1h, df_1h)
portfolio_value_blackswan_1h = backtest(agent_blackswan_1h, df_1h)
portfolio_value_markettiming_1h = backtest(agent_markettiming_1h, df_1h)
portfolio_value_inversevolatility_1h = backtest(agent_inversevolatility_1h, df_1h)

# Repeat for 4h
agent_momentum_4h = Agent_Momentum(100)
agent_trendfollowing_4h = Agent_TrendFollowing(100)
agent_riskonriskoff_4h = Agent_RiskOnRiskOff(threshold)
agent_arbitrage_4h = Agent_Arbitrage(0.01)
agent_blackswan_4h = Agent_BlackSwan(z_score_threshold)
agent_markettiming_4h = Agent_MarketTiming(trend_threshold,volatility_threshold)
agent_inversevolatility_4h = Agent_InverseVolatility(100)

portfolio_value_momentum_4h = backtest(agent_momentum_4h, df_4h)
portfolio_value_trendfollowing_4h = backtest(agent_trendfollowing_4h, df_4h)
portfolio_value_riskonriskoff_4h = backtest(agent_riskonriskoff_4h, df_4h)
portfolio_value_arbitrage_4h = backtest(agent_arbitrage_4h, df_4h)
portfolio_value_blackswan_4h = backtest(agent_blackswan_4h, df_4h)
portfolio_value_markettiming_4h = backtest(agent_markettiming_4h, df_4h)
portfolio_value_inversevolatility_4h = backtest(agent_inversevolatility_4h, df_4h)

# Repeat for 1d
agent_momentum_1d = Agent_Momentum(lookback)
agent_trendfollowing_1d = Agent_TrendFollowing(lookback)
agent_riskonriskoff_1d = Agent_RiskOnRiskOff(threshold)
agent_arbitrage_1d = Agent_Arbitrage(0.01)
agent_blackswan_1d = Agent_BlackSwan(z_score_threshold)
agent_markettiming_1d = Agent_MarketTiming(trend_threshold,volatility_threshold)
agent_inversevolatility_1d = Agent_InverseVolatility(lookback)

portfolio_value_momentum_1d = backtest(agent_momentum_1d, df_1d)
portfolio_value_trendfollowing_1d = backtest(agent_trendfollowing_1d, df_1d)
portfolio_value_riskonriskoff_1d = backtest(agent_riskonriskoff_1d, df_1d)
portfolio_value_arbitrage_1d = backtest(agent_arbitrage_1d, df_1d)
portfolio_value_blackswan_1d = backtest(agent_blackswan_1d, df_1d)
portfolio_value_markettiming_1d = backtest(agent_markettiming_1d, df_1d)
portfolio_value_inversevolatility_1d = backtest(agent_inversevolatility_1d, df_1d)

# Backtest for each interval
portfolio_value_mr_stat_1m = backtest(agent_mr_stat_1m, df_1m)
portfolio_value_mr_stat_1h = backtest(agent_mr_stat_1h, df_1h)
portfolio_value_mr_stat_4h = backtest(agent_mr_stat_4h, df_4h)
portfolio_value_mr_stat_1d = backtest(agent_mr_stat_1d, df_1d)

# portfolio_value_xgboost_1m = backtest(agent_xgboost_1m, df_1m)
# portfolio_value_xgboost_1h = backtest(agent_xgboost_1h, df_1h)
# portfolio_value_xgboost_4h = backtest(agent_xgboost_4h, df_4h)
# portfolio_value_xgboost_1d = backtest(agent_xgboost_1d, df_1d)

# Print Portfolio Values for each strategy
print(f"Portfolio Value for Momentum 1m: {portfolio_value_momentum_1m}")
print(f"Portfolio Value for Trend Following 1m: {portfolio_value_trendfollowing_1m}")
print(f"Portfolio Value for Risk-on/Risk-off 1m: {portfolio_value_riskonriskoff_1m}")
print(f"Portfolio Value for Arbitrage 1m: {portfolio_value_arbitrage_1m}")
print(f"Portfolio Value for Black Swan 1m: {portfolio_value_blackswan_1m}")
print(f"Portfolio Value for Market Timing 1m: {portfolio_value_markettiming_1m}")
print(f"Portfolio Value for Inverse Volatility 1m: {portfolio_value_inversevolatility_1m}")

print(f"Portfolio Value for 1m MR Interval: {portfolio_value_1m_MR}")
print(f"Portfolio Value for 1h MR Interval: {portfolio_value_1h_MR}")
print(f"Portfolio Value for 4h MR Interval: {portfolio_value_4h_MR}")
print(f"Portfolio Value for 1d MR Interval: {portfolio_value_1d_MR}")

# Print Portfolio Values for each strategy
print(f"Portfolio Value for Momentum 1m: {portfolio_value_momentum_1m}")
print(f"Portfolio Value for Trend Following 1m: {portfolio_value_trendfollowing_1m}")
print(f"Portfolio Value for Risk-on/Risk-off 1m: {portfolio_value_riskonriskoff_1m}")
print(f"Portfolio Value for Arbitrage 1m: {portfolio_value_arbitrage_1m}")
print(f"Portfolio Value for Black Swan 1m: {portfolio_value_blackswan_1m}")
print(f"Portfolio Value for Market Timing 1m: {portfolio_value_markettiming_1m}")
print(f"Portfolio Value for Inverse Volatility 1m: {portfolio_value_inversevolatility_1m}")

# Print Portfolio Values for 1h interval
print(f"Portfolio Value for Momentum 1h: {portfolio_value_momentum_1h}")
print(f"Portfolio Value for Trend Following 1h: {portfolio_value_trendfollowing_1h}")
print(f"Portfolio Value for Risk-on/Risk-off 1h: {portfolio_value_riskonriskoff_1h}")
print(f"Portfolio Value for Arbitrage 1h: {portfolio_value_arbitrage_1h}")
print(f"Portfolio Value for Black Swan 1h: {portfolio_value_blackswan_1h}")
print(f"Portfolio Value for Market Timing 1h: {portfolio_value_markettiming_1h}")
print(f"Portfolio Value for Inverse Volatility 1h: {portfolio_value_inversevolatility_1h}")

# Print Portfolio Values for 4h interval
print(f"Portfolio Value for Momentum 4h: {portfolio_value_momentum_4h}")
print(f"Portfolio Value for Trend Following 4h: {portfolio_value_trendfollowing_4h}")
print(f"Portfolio Value for Risk-on/Risk-off 4h: {portfolio_value_riskonriskoff_4h}")
print(f"Portfolio Value for Arbitrage 4h: {portfolio_value_arbitrage_4h}")
print(f"Portfolio Value for Black Swan 4h: {portfolio_value_blackswan_4h}")
print(f"Portfolio Value for Market Timing 4h: {portfolio_value_markettiming_4h}")
print(f"Portfolio Value for Inverse Volatility 4h: {portfolio_value_inversevolatility_4h}")

# Print Portfolio Values for 1d interval
print(f"Portfolio Value for Momentum 1d: {portfolio_value_momentum_1d}")
print(f"Portfolio Value for Trend Following 1d: {portfolio_value_trendfollowing_1d}")
print(f"Portfolio Value for Risk-on/Risk-off 1d: {portfolio_value_riskonriskoff_1d}")
print(f"Portfolio Value for Arbitrage 1d: {portfolio_value_arbitrage_1d}")
print(f"Portfolio Value for Black Swan 1d: {portfolio_value_blackswan_1d}")
print(f"Portfolio Value for Market Timing 1d: {portfolio_value_markettiming_1d}")
print(f"Portfolio Value for Inverse Volatility 1d: {portfolio_value_inversevolatility_1d}")

print(f"Portfolio Return for 1m MR Interval: ",calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_1m_MR)["Total Return"]*100, " %")
print(f"Portfolio Return for 1h MR Interval: ",calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_1h_MR)["Total Return"]*100, " %")
print(f"Portfolio Return for 4h MR Interval: ",calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_4h_MR)["Total Return"]*100, " %")
print(f"Portfolio Return for 1d MR Interval: ",calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_1d_MR)["Total Return"]*100, " %")

# Momentum Strategy
print(f"Portfolio Return for 1m Momentum Interval: ", calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_momentum_1m)["Total Return"]*100, " %")
print(f"Portfolio Return for 1h Momentum Interval: ", calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_momentum_1h)["Total Return"]*100, " %")
print(f"Portfolio Return for 4h Momentum Interval: ", calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_momentum_4h)["Total Return"]*100, " %")
print(f"Portfolio Return for 1d Momentum Interval: ", calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_momentum_1d)["Total Return"]*100, " %")

# Trend Following Strategy
print(f"Portfolio Return for 1m Trend Following Interval: ", calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_trendfollowing_1m)["Total Return"]*100, " %")
print(f"Portfolio Return for 1h Trend Following Interval: ", calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_trendfollowing_1h)["Total Return"]*100, " %")
print(f"Portfolio Return for 4h Trend Following Interval: ", calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_trendfollowing_4h)["Total Return"]*100, " %")
print(f"Portfolio Return for 1d Trend Following Interval: ", calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_trendfollowing_1d)["Total Return"]*100, " %")

# Risk-on/Risk-off Strategy
print(f"Portfolio Return for 1m Risk-on/Risk-off Interval: ", calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_riskonriskoff_1m)["Total Return"]*100, " %")
print(f"Portfolio Return for 1h Risk-on/Risk-off Interval: ", calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_riskonriskoff_1h)["Total Return"]*100, " %")
print(f"Portfolio Return for 4h Risk-on/Risk-off Interval: ", calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_riskonriskoff_4h)["Total Return"]*100, " %")
print(f"Portfolio Return for 1d Risk-on/Risk-off Interval: ", calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_riskonriskoff_1d)["Total Return"]*100, " %")

# Arbitrage Strategy
print(f"Portfolio Return for 1m Arbitrage Interval: ", calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_arbitrage_1m)["Total Return"]*100, " %")
print(f"Portfolio Return for 1h Arbitrage Interval: ", calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_arbitrage_1h)["Total Return"]*100, " %")
print(f"Portfolio Return for 4h Arbitrage Interval: ", calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_arbitrage_4h)["Total Return"]*100, " %")
print(f"Portfolio Return for 1d Arbitrage Interval: ", calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_arbitrage_1d)["Total Return"]*100, " %")

# Black Swan Strategy
print(f"Portfolio Return for 1m Black Swan Interval: ", calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_blackswan_1m)["Total Return"]*100, " %")
print(f"Portfolio Return for 1h Black Swan Interval: ", calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_blackswan_1h)["Total Return"]*100, " %")
print(f"Portfolio Return for 4h Black Swan Interval: ", calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_blackswan_4h)["Total Return"]*100, " %")
print(f"Portfolio Return for 1d Black Swan Interval: ", calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_blackswan_1d)["Total Return"]*100, " %")

# Market Timing Strategy
print(f"Portfolio Return for 1m Market Timing Interval: ", calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_markettiming_1m)["Total Return"]*100, " %")
print(f"Portfolio Return for 1h Market Timing Interval: ", calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_markettiming_1h)["Total Return"]*100, " %")
print(f"Portfolio Return for 4h Market Timing Interval: ", calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_markettiming_4h)["Total Return"]*100, " %")
print(f"Portfolio Return for 1d Market Timing Interval: ", calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_markettiming_1d)["Total Return"]*100, " %")

# Inverse Volatility Strategy
print(f"Portfolio Return for 1m Inverse Volatility Interval: ", calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_inversevolatility_1m)["Total Return"]*100, " %")
print(f"Portfolio Return for 1h Inverse Volatility Interval: ", calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_inversevolatility_1h)["Total Return"]*100, " %")
print(f"Portfolio Return for 4h Inverse Volatility Interval: ", calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_inversevolatility_4h)["Total Return"]*100, " %")
print(f"Portfolio Return for 1d Inverse Volatility Interval: ", calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_inversevolatility_1d)["Total Return"]*100, " %")

# Print portfolio values for MR_Stat
print(f"Portfolio Value for Mean Reversion (Advanced Stats) 1m: {portfolio_value_mr_stat_1m}")
print(f"Portfolio Value for Mean Reversion (Advanced Stats) 1h: {portfolio_value_mr_stat_1h}")
print(f"Portfolio Value for Mean Reversion (Advanced Stats) 4h: {portfolio_value_mr_stat_4h}")
print(f"Portfolio Value for Mean Reversion (Advanced Stats) 1d: {portfolio_value_mr_stat_1d}")

# # Print portfolio values for XGBoost
# print(f"Portfolio Value for XGBoost 1m: {portfolio_value_xgboost_1m}")
# print(f"Portfolio Value for XGBoost 1h: {portfolio_value_xgboost_1h}")
# print(f"Portfolio Value for XGBoost 4h: {portfolio_value_xgboost_4h}")
# print(f"Portfolio Value for XGBoost 1d: {portfolio_value_xgboost_1d}")

# Evaluate portfolio returns for MR_Stat
print(f"Portfolio Return for 1m MR_Stat Interval: ", 
      calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_mr_stat_1m)["Total Return"] * 100, " %")
print(f"Portfolio Return for 1h MR_Stat Interval: ", 
      calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_mr_stat_1h)["Total Return"] * 100, " %")
print(f"Portfolio Return for 4h MR_Stat Interval: ", 
      calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_mr_stat_4h)["Total Return"] * 100, " %")
print(f"Portfolio Return for 1d MR_Stat Interval: ", 
      calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_mr_stat_1d)["Total Return"] * 100, " %")

# # Evaluate portfolio returns for XGBoost
# print(f"Portfolio Return for 1m XGBoost Interval: ", 
#       calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_xgboost_1m)["Total Return"] * 100, " %")
# print(f"Portfolio Return for 1h XGBoost Interval: ", 
#       calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_xgboost_1h)["Total Return"] * 100, " %")
# print(f"Portfolio Return for 4h XGBoost Interval: ", 
#       calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_xgboost_4h)["Total Return"] * 100, " %")
# print(f"Portfolio Return for 1d XGBoost Interval: ", 
#       calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_xgboost_1d)["Total Return"] * 100, " %")

# Define agents for different intervals
agent_bollinger_1m = Agent_BollingerBands(750)
agent_bollinger_1h = Agent_BollingerBands(75)
agent_bollinger_4h = Agent_BollingerBands(250)
agent_bollinger_1d = Agent_BollingerBands(100)

agent_macrossover_1m = Agent_MACrossover(short_lookback=10, long_lookback=100)
agent_macrossover_1h = Agent_MACrossover(short_lookback=20, long_lookback=100)
agent_macrossover_4h = Agent_MACrossover(short_lookback=50, long_lookback=400)
agent_macrossover_1d = Agent_MACrossover(short_lookback=100, long_lookback=600)

agent_donchian_1m = Agent_Donchian(5)
agent_donchian_1h = Agent_Donchian(20)
agent_donchian_4h = Agent_Donchian(50)
agent_donchian_1d = Agent_Donchian(100)

agent_rsi_1m = Agent_RSI(rsi_period=7, oversold=40, overbought=60)
agent_rsi_1h = Agent_RSI(rsi_period=14, oversold=35, overbought=65)
agent_rsi_4h = Agent_RSI(rsi_period=21, oversold=20, overbought=80)
agent_rsi_1d = Agent_RSI(rsi_period=28, oversold=25, overbought=75)

# Backtesting for all models and intervals
portfolio_value_bollinger_1m = backtest(agent_bollinger_1m, df_1m)
portfolio_value_bollinger_1h = backtest(agent_bollinger_1h, df_1h)
portfolio_value_bollinger_4h = backtest(agent_bollinger_4h, df_4h)
portfolio_value_bollinger_1d = backtest(agent_bollinger_1d, df_1d)

portfolio_value_macrossover_1m = backtest(agent_macrossover_1m, df_1m)
portfolio_value_macrossover_1h = backtest(agent_macrossover_1h, df_1h)
portfolio_value_macrossover_4h = backtest(agent_macrossover_4h, df_4h)
portfolio_value_macrossover_1d = backtest(agent_macrossover_1d, df_1d)

portfolio_value_donchian_1m = backtest(agent_donchian_1m, df_1m)
portfolio_value_donchian_1h = backtest(agent_donchian_1h, df_1h)
portfolio_value_donchian_4h = backtest(agent_donchian_4h, df_4h)
portfolio_value_donchian_1d = backtest(agent_donchian_1d, df_1d)

portfolio_value_rsi_1m = backtest(agent_rsi_1m, df_1m)
portfolio_value_rsi_1h = backtest(agent_rsi_1h, df_1h)
portfolio_value_rsi_4h = backtest(agent_rsi_4h, df_4h)
portfolio_value_rsi_1d = backtest(agent_rsi_1d, df_1d)

# Evaluate portfolio returns for each model and interval
print("--- Portfolio Returns ---")

# Bollinger Bands
print(f"Portfolio Return for Bollinger Bands 1m: ", 
      calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_bollinger_1m)["Total Return"] * 100, "%")
print(f"Portfolio Return for Bollinger Bands 1h: ", 
      calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_bollinger_1h)["Total Return"] * 100, "%")
print(f"Portfolio Return for Bollinger Bands 4h: ", 
      calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_bollinger_4h)["Total Return"] * 100, "%")
print(f"Portfolio Return for Bollinger Bands 1d: ", 
      calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_bollinger_1d)["Total Return"] * 100, "%")

# Moving Average Crossover
print(f"Portfolio Return for MACrossover 1m: ", 
      calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_macrossover_1m)["Total Return"] * 100, "%")
print(f"Portfolio Return for MACrossover 1h: ", 
      calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_macrossover_1h)["Total Return"] * 100, "%")
print(f"Portfolio Return for MACrossover 4h: ", 
      calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_macrossover_4h)["Total Return"] * 100, "%")
print(f"Portfolio Return for MACrossover 1d: ", 
      calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_macrossover_1d)["Total Return"] * 100, "%")

# Donchian Channel
print(f"Portfolio Return for Donchian 1m: ", 
      calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_donchian_1m)["Total Return"] * 100, "%")
print(f"Portfolio Return for Donchian 1h: ", 
      calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_donchian_1h)["Total Return"] * 100, "%")
print(f"Portfolio Return for Donchian 4h: ", 
      calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_donchian_4h)["Total Return"] * 100, "%")
print(f"Portfolio Return for Donchian 1d: ", 
      calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_donchian_1d)["Total Return"] * 100, "%")

# RSI Momentum
print(f"Portfolio Return for RSI 1m: ", 
      calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_rsi_1m)["Total Return"] * 100, "%")
print(f"Portfolio Return for RSI 1h: ", 
      calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_rsi_1h)["Total Return"] * 100, "%")
print(f"Portfolio Return for RSI 4h: ", 
      calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_rsi_4h)["Total Return"] * 100, "%")
print(f"Portfolio Return for RSI 1d: ", 
      calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value_rsi_1d)["Total Return"] * 100, "%")

# lookback_periods = [1,2,5, 10, 25, 50, 75, 100, 200,250,500,750,1000]

# # Initialize agents with different lookback values
# agents = {
#     "1m": [Agent_BollingerBands(period) for period in lookback_periods],
#     "1h": [Agent_BollingerBands(period) for period in lookback_periods],
#     "4h": [Agent_BollingerBands(period) for period in lookback_periods],
#     "1d": [Agent_BollingerBands(period) for period in lookback_periods]
# }

# # Backtest the agents for each timeframe and calculate portfolio returns
# portfolio_values = {
#     "1m": [],
#     "1h": [],
#     "4h": [],
#     "1d": []
# }

# for timeframe, agents_list in agents.items():
#     for agent in agents_list:
#         # Apply Bollinger Bands to respective timeframe data
#         df = globals()[f"df_{timeframe}"]  # Dynamically fetch the correct dataframe
#         portfolio_value = backtest(agent, df)
#         portfolio_values[timeframe].append(portfolio_value)

# # Calculate performance metrics for each timeframe and lookback period
# for timeframe, values in portfolio_values.items():
#     for i, portfolio_value in enumerate(values):
#         lookback = lookback_periods[i]
#         performance = calculate_performance_metrics(initial_cash=100000, final_value=portfolio_value)
#         total_return = performance["Total Return"] * 100
#         print(f"Portfolio Return for Bollinger Bands {timeframe} (Lookback: {lookback}): {total_return:.2f}%")