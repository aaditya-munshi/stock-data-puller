import unittest
from unittest.mock import patch

from stock_analyzer import (
    calculate_metrics,
    calculate_rsi,
    generate_buy_signal,
    get_valid_days,
    moving_average,
    parse_date_parts,
    prompt_for_selection,
    rank_stocks,
)


class DateSelectionTests(unittest.TestCase):
    def test_get_valid_days_for_leap_year_february(self):
        self.assertEqual(get_valid_days(2024, 2), list(range(1, 30)))

    def test_parse_date_parts_rejects_invalid_day(self):
        with self.assertRaises(ValueError):
            parse_date_parts(2023, 2, 30)

    def test_prompt_for_selection_uses_numbered_menu(self):
        with patch("builtins.input", side_effect=["2", ""]):
            self.assertEqual(prompt_for_selection("Pick one", [10, 20], 10), 20)

    def test_rank_stocks_uses_downside_resilience(self):
        data = {
            "Stable": [100, 102, 101, 105, 108],
            "Volatile": [100, 90, 95, 110, 80],
        }

        ranked = rank_stocks(data)

        self.assertEqual(ranked[0]["ticker"], "Stable")
        self.assertGreater(ranked[0]["score"], ranked[1]["score"])

    def test_calculate_metrics_reports_negative_drawdown(self):
        metrics = calculate_metrics([100, 90, 95, 110, 80])
        self.assertLess(metrics["max_drawdown"], 0.0)


class BuySignalTests(unittest.TestCase):
    def test_moving_average_uses_trailing_window(self):
        self.assertEqual(moving_average([1, 2, 3, 4, 5], 3), [2.0, 3.0, 4.0])

    def test_moving_average_returns_empty_when_insufficient_data(self):
        self.assertEqual(moving_average([1, 2], 3), [])

    def test_calculate_rsi_is_100_when_all_gains(self):
        prices = list(range(1, 20))
        self.assertEqual(calculate_rsi(prices, period=14), 100.0)

    def test_calculate_rsi_returns_none_with_insufficient_history(self):
        self.assertIsNone(calculate_rsi([1, 2, 3], period=14))

    def test_generate_buy_signal_flags_uptrend_as_buy(self):
        prices = [100 + index * 1.5 for index in range(60)]
        signal = generate_buy_signal(prices)
        self.assertEqual(signal["recommendation"], "BUY")
        self.assertGreater(signal["signal_score"], 0)

    def test_generate_buy_signal_flags_downtrend_as_sell(self):
        prices = [200 - index * 1.5 for index in range(60)]
        signal = generate_buy_signal(prices)
        self.assertEqual(signal["recommendation"], "SELL")
        self.assertLess(signal["signal_score"], 0)

    def test_generate_buy_signal_requires_long_window_history(self):
        with self.assertRaises(ValueError):
            generate_buy_signal([100] * 10)


if __name__ == "__main__":
    unittest.main()
