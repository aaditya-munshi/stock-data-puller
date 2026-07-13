from __future__ import annotations

import calendar
import math
import os
from collections.abc import Sequence
from datetime import datetime
from typing import Any, Mapping

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency
    plt = None

try:
    import pandas as pd
    import yfinance as yf
except ImportError:  # pragma: no cover - optional dependency in test env
    pd = None
    yf = None


def get_valid_days(year: int, month: int) -> list[int]:
    """Return a list of valid day numbers for a given year and month."""
    if not 1 <= month <= 12:
        raise ValueError("Month must be between 1 and 12")
    return list(range(1, calendar.monthrange(year, month)[1] + 1))


def parse_date_parts(year: int, month: int, day: int) -> tuple[int, int, int]:
    """Validate a gregorian date and return the normalized tuple."""
    if not 1 <= month <= 12:
        raise ValueError("Month must be between 1 and 12")
    valid_days = get_valid_days(year, month)
    if day not in valid_days:
        raise ValueError(f"Day {day} is invalid for {year}-{month:02d}")
    return year, month, day


def prompt_for_selection(prompt_text: str, options: Sequence[Any], default: Any = None) -> Any:
    """Prompt the user to choose one option from a numbered list."""
    if not options:
        raise ValueError("At least one option is required")

    print(prompt_text)
    for index, option in enumerate(options, start=1):
        print(f"    {index}. {option}")

    while True:
        raw_value = input(f"Select an option [default: {default}]: ").strip()
        if not raw_value and default is not None:
            return default
        if raw_value.isdigit():
            selected_index = int(raw_value) - 1
            if 0 <= selected_index < len(options):
                return options[selected_index]
        if raw_value in {str(option) for option in options}:
            return next(option for option in options if str(option) == raw_value)
        print("Please enter a number from the list above.")


def _coerce_prices(values: Sequence[float] | Any) -> list[float]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, (str, bytes)):
        raise TypeError("Expected a sequence of prices")
    if not isinstance(values, Sequence):
        raise TypeError("Expected a sequence of prices")
    return [float(item) for item in values]


def calculate_metrics(prices: Sequence[float] | Any) -> dict[str, float]:
    """Compute a small set of downside-resilience metrics for a price series."""
    price_series = _coerce_prices(prices)
    if len(price_series) < 2:
        raise ValueError("At least two prices are required")

    first_price = price_series[0]
    last_price = price_series[-1]
    if first_price <= 0:
        raise ValueError("Prices must be positive")

    returns = [
        (price_series[index] / price_series[index - 1]) - 1.0
        for index in range(1, len(price_series))
    ]
    average_return = sum(returns) / len(returns) if returns else 0.0
    variance = sum((value - average_return) ** 2 for value in returns) / len(returns) if returns else 0.0
    volatility = math.sqrt(variance) if variance else 0.0
    total_return = (last_price / first_price) - 1.0

    peak_value = first_price
    max_drawdown = 0.0
    for price in price_series:
        peak_value = max(peak_value, price)
        drawdown = (price / peak_value) - 1.0
        max_drawdown = min(max_drawdown, drawdown)

    sharpe_ratio = average_return / volatility if volatility else 0.0
    stability_score = 1.0 - min(volatility * 2.0, 1.0)

    score = (max(total_return, 0.0) * 0.4) + ((1.0 + max_drawdown) * 0.4) + (max(sharpe_ratio, 0.0) * 0.2)
    score -= volatility * 0.1

    return {
        "total_return": total_return,
        "volatility": volatility,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe_ratio,
        "stability_score": stability_score,
        "score": score,
    }


def moving_average(prices: Sequence[float] | Any, window: int) -> list[float]:
    """Return a simple moving average series (NaN-free, shorter than input by window-1)."""
    price_series = _coerce_prices(prices)
    if window < 1:
        raise ValueError("window must be at least 1")
    if len(price_series) < window:
        return []
    return [
        sum(price_series[index - window + 1 : index + 1]) / window
        for index in range(window - 1, len(price_series))
    ]


def calculate_rsi(prices: Sequence[float] | Any, period: int = 14) -> float | None:
    """Compute the Relative Strength Index (RSI) for the most recent `period` changes."""
    price_series = _coerce_prices(prices)
    if len(price_series) < period + 1:
        return None

    changes = [price_series[i] - price_series[i - 1] for i in range(-period, 0)]
    gains = [change for change in changes if change > 0]
    losses = [-change for change in changes if change < 0]

    average_gain = sum(gains) / period
    average_loss = sum(losses) / period

    if average_loss == 0:
        return 100.0
    relative_strength = average_gain / average_loss
    return 100.0 - (100.0 / (1.0 + relative_strength))


def generate_buy_signal(
    prices: Sequence[float] | Any,
    short_window: int = 20,
    long_window: int = 50,
    rsi_period: int = 14,
) -> dict[str, Any]:
    """Score a price series on trend and momentum to suggest BUY/HOLD/SELL.

    Combines a moving-average crossover (trend) with RSI (momentum) into a
    single -1..1 signal score. This is a heuristic, not a guarantee of
    future performance.
    """
    price_series = _coerce_prices(prices)
    if len(price_series) < long_window:
        raise ValueError(f"At least {long_window} prices are required")

    short_ma = moving_average(price_series, short_window)[-1]
    long_ma = moving_average(price_series, long_window)[-1]
    trend_strength = (short_ma - long_ma) / long_ma  # positive => uptrend

    rsi = calculate_rsi(price_series, rsi_period)

    # Momentum score: RSI > 70 (overbought) pulls score down, < 30 (oversold) pulls it up.
    if rsi is None:
        momentum_score = 0.0
    elif rsi >= 70:
        momentum_score = -((rsi - 70) / 30)
    elif rsi <= 30:
        momentum_score = (30 - rsi) / 30
    else:
        momentum_score = (50 - rsi) / 20 * -1  # mild pull toward the trend signal near neutral

    trend_score = max(min(trend_strength * 10, 1.0), -1.0)
    signal_score = max(min((trend_score * 0.65) + (momentum_score * 0.35), 1.0), -1.0)

    if signal_score >= 0.3:
        recommendation = "BUY"
    elif signal_score <= -0.3:
        recommendation = "SELL"
    else:
        recommendation = "HOLD"

    return {
        "recommendation": recommendation,
        "signal_score": signal_score,
        "short_ma": short_ma,
        "long_ma": long_ma,
        "trend_strength": trend_strength,
        "rsi": rsi,
    }


def rank_stocks(stock_data: Mapping[str, Sequence[float] | Any]) -> list[dict[str, Any]]:
    """Rank stocks by downside resilience and consistency."""
    ranked: list[dict[str, Any]] = []
    for ticker, values in stock_data.items():
        metrics = calculate_metrics(values)
        ranked.append(
            {
                "ticker": ticker,
                "score": metrics["score"],
                "total_return": metrics["total_return"],
                "max_drawdown": metrics["max_drawdown"],
                "volatility": metrics["volatility"],
                "sharpe_ratio": metrics["sharpe_ratio"],
                "stability_score": metrics["stability_score"],
            }
        )

    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked


def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> Any:
    """Fetch historical data for a ticker using Yahoo Finance."""
    if yf is None:
        raise RuntimeError("yfinance is required to fetch stock data")

    data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
    if data.empty:
        raise ValueError(f"No data returned for {ticker}")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data


def build_summary(ticker: str, data: Any) -> dict[str, Any]:
    """Create a compact summary for a ticker using the close price series."""
    close_prices = data["Close"].tolist() if hasattr(data, "__getitem__") and "Close" in data.columns else data
    metrics = calculate_metrics(close_prices)
    return {
        "ticker": ticker,
        "start_date": data.index[0].strftime("%Y-%m-%d") if hasattr(data.index[0], "strftime") else str(data.index[0]),
        "end_date": data.index[-1].strftime("%Y-%m-%d") if hasattr(data.index[-1], "strftime") else str(data.index[-1]),
        "metrics": metrics,
    }


def plot_price_history(data: Any, ticker: str, output_path: str = "stock_analysis.png") -> str:
    """Save a simple price and drawdown chart for the history."""
    if plt is None:
        return output_path

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    close_prices = data["Close"]

    axes[0].plot(close_prices.index, close_prices.values, color="royalblue", linewidth=1.5)
    axes[0].set_title(f"{ticker} closing price")
    axes[0].grid(True, alpha=0.3)

    cumulative_max = close_prices.cummax()
    drawdown = (close_prices / cumulative_max) - 1.0
    axes[1].fill_between(drawdown.index, drawdown.values, 0, color="tomato", alpha=0.35)
    axes[1].set_title("Drawdown")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def main() -> None:
    """Run a small interactive CLI for stock analysis."""
    print("==================================================")
    print("Stock Data Analyzer")
    print("==================================================")
    print("This tool ranks stocks by downside resilience and consistency, not guaranteed profit.")

    ticker = input("Enter ticker symbol (e.g., AAPL, TSLA, GOOGL): ").strip().upper() or "AAPL"

    print("Select start date")
    start_year = int(input("Start year [default: 2023]: ") or "2023")
    start_month = int(input("Start month [default: 1]: ") or "1")
    start_day = int(input("Start day [default: 1]: ") or "1")
    parse_date_parts(start_year, start_month, start_day)

    print("Select end date")
    end_year = int(input("End year [default: 2026]: ") or "2026")
    end_month = int(input("End month [default: 1]: ") or "1")
    end_day = int(input("End day [default: 1]: ") or "1")
    parse_date_parts(end_year, end_month, end_day)

    start_date = datetime(start_year, start_month, start_day).strftime("%Y-%m-%d")
    end_date = datetime(end_year, end_month, end_day).strftime("%Y-%m-%d")

    print(f"Downloading data for {ticker}...")
    data = fetch_stock_data(ticker, start_date, end_date)
    summary = build_summary(ticker, data)
    print(f"{ticker} Stock Data ({summary['start_date']} to {summary['end_date']}):")
    print(data.head())

    metrics = summary["metrics"]
    print("\nRisk-aware summary:")
    print(f"  Total return: {metrics['total_return'] * 100:.2f}%")
    print(f"  Max drawdown: {metrics['max_drawdown'] * 100:.2f}%")
    print(f"  Volatility: {metrics['volatility'] * 100:.2f}%")
    print(f"  Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"  Resilience score: {metrics['score']:.3f}")

    close_prices = data["Close"].tolist()
    if len(close_prices) >= 50:
        signal = generate_buy_signal(close_prices)
        print("\nBuy signal (trend + momentum heuristic, not financial advice):")
        print(f"  Recommendation: {signal['recommendation']}")
        print(f"  Signal score: {signal['signal_score']:.2f} (-1 sell .. +1 buy)")
        print(f"  20d MA vs 50d MA: {signal['short_ma']:.2f} vs {signal['long_ma']:.2f}")
        print(f"  RSI(14): {signal['rsi']:.1f}" if signal["rsi"] is not None else "  RSI(14): n/a")
    else:
        print("\nNot enough history (need 50+ trading days) to compute a buy signal.")

    chart_path = plot_price_history(data, ticker)
    print(f"Saved chart to {os.path.abspath(chart_path)}")


if __name__ == "__main__":
    main()
