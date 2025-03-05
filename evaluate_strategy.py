
import numpy as np
import pandas as pd


def calculate_sharpe_ratio(equity_curve):
    """计算夏普比率"""
    equity_curve = pd.Series(equity_curve)  # 转换为 Pandas Series
    returns = equity_curve.pct_change().dropna()  # 计算收益率
    if len(returns) == 0:  # 如果没有收益率数据，返回0
        return 0.0
    excess_returns = returns - np.mean(returns)  # 计算超额收益
    sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)  # 年化夏普比率
    return sharpe_ratio


def evaluate_performance(equity_curve, actual_prices, trade_history):
    """
    评估策略表现，返回多个指标
    """
    # 将 actual_prices 和 equity_curve 转换为 Pandas Series
    actual_prices = pd.Series(actual_prices)
    equity_curve = pd.Series(equity_curve)

    # 1. 计算最终账户价值
    final_value = equity_curve.iloc[-1] if len(equity_curve) > 0 else 0

    # 2. 计算最大回撤
    if len(equity_curve) > 1:  # 确保有足够的数据
        rolling_max = np.maximum.accumulate(equity_curve)
        drawdowns = 1 - equity_curve / rolling_max
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
    else:
        max_drawdown = 0.0

    # 3. 计算夏普比率
    sharpe_ratio = calculate_sharpe_ratio(equity_curve)

    # 4. 计算胜率
    successful_trades = sum(1 for trade in trade_history if
                            trade[1] == '卖出' and trade[2] > trade_history[trade_history.index(trade) - 1][2])
    win_rate = successful_trades / len(trade_history) if len(trade_history) > 0 else 0

    # 5. 计算信息比率
    benchmark_returns = actual_prices.pct_change().dropna().values
    strategy_returns = equity_curve.pct_change().dropna().values
    if len(strategy_returns) > 0 and len(benchmark_returns) > 0:
        min_len = min(len(strategy_returns), len(benchmark_returns))
        tracking_error = np.std(strategy_returns[:min_len] - benchmark_returns[:min_len])
        information_ratio = np.mean(
            strategy_returns[:min_len] - benchmark_returns[:min_len]) / tracking_error if tracking_error != 0 else 0
    else:
        information_ratio = 0.0

    # 6. 计算 Alpha 值
    if len(strategy_returns) > 0 and len(benchmark_returns) > 0:
        alpha = np.mean(strategy_returns[:min_len]) - np.mean(benchmark_returns[:min_len])
    else:
        alpha = 0.0

    return {
        '最终账户价值': final_value,
        '最大回撤': max_drawdown,
        '夏普比率': sharpe_ratio,
        '胜率': win_rate,
        '信息比率': information_ratio,
        'Alpha': alpha
    }