"""Information-boundary tests, including 48 h features and forecast vintages."""
from dataclasses import replace
import unittest

import numpy as np

from microgrid.data import DataBundle, canonical_intervals, load_data, parse_time_minutes
from microgrid.forecast import ForecastEngine


def synthetic(days=45):
    rng = np.random.default_rng(14)
    dates = np.datetime64("2025-01-01") + np.arange(days).astype("timedelta64[D]")
    slot = np.arange(144)
    solar = np.maximum(0, np.sin((slot - 36) * np.pi / 72)) * 5000
    load = 4200 + 400 * np.cos(slot * 2 * np.pi / 144)[None, :] + rng.normal(0, 50, (days, 144))
    pv = np.maximum(0, solar[None, :] * rng.uniform(.7, 1.3, (days, 1)))
    price = np.exp(-.5 + .3 * np.sin(slot * 2 * np.pi / 144)[None, :] + rng.normal(0, .03, (days, 144)))
    issued = np.zeros((days, 4, 24))
    for d in range(days):
        for issue in range(4):
            for h in range(24):
                issued[d, issue, h] = solar[(issue * 36 + (h + 1) * 6 - 1) % 144]
    return DataBundle(dates, load, pv, price, load[:7].mean(0), solar,
                      price[:7].mean(0), issued)


class ForecastTests(unittest.TestCase):
    def test_parser_and_template_clock(self):
        self.assertEqual(parse_time_minutes("0:00+1"), 1440)
        self.assertEqual(canonical_intervals()[0], "00:00-00:10")
        self.assertEqual(canonical_intervals()[-1], "23:50-24:00")

    def test_future_actual_cannot_change_forecast(self):
        data = synthetic()
        origin = 32 * 144 + 43
        changed = replace(data, load=data.load.copy(), pv=data.pv.copy(), price=data.price.copy())
        changed.load.reshape(-1)[origin:] *= 3
        changed.pv.reshape(-1)[origin:] *= .1
        changed.price.reshape(-1)[origin:] *= 4
        for family in ("naive", "ridge", "boost"):
            first = ForecastEngine(data, family).forecast(origin, 288, True, True)
            second = ForecastEngine(changed, family).forecast(origin, 288, True, True)
            np.testing.assert_array_equal(first.net_s, second.net_s, err_msg=family)
            np.testing.assert_array_equal(first.price_s, second.price_s, err_msg=family)
            self.assertLessEqual(first.provenance["residual_actual_end_exclusive"], origin)

    def test_unpublished_forecast_cannot_change_result(self):
        data = synthetic()
        origin = 32 * 144 + 43  # after 06:00, before 12:00
        changed = replace(data, forecasts=data.forecasts.copy())
        changed.forecasts[32, 2:] += 70000
        changed.forecasts[33:] += 80000
        first = ForecastEngine(data).forecast(origin, 288, True, True)
        second = ForecastEngine(changed).forecast(origin, 288, True, True)
        np.testing.assert_array_equal(first.net_s, second.net_s)
        self.assertEqual(first.provenance["forecast_issue"], 32 * 144 + 36)

    def test_issued_hourly_alignment_and_allowed_issues(self):
        data = synthetic()
        data.forecasts[0, 0] = np.arange(1, 25) * 600
        data.forecasts[0, 1] = np.arange(1, 25) * 900
        data.prior_pv[-1] = 0
        first = ForecastEngine(data).forecast(0, 12, True)
        np.testing.assert_allclose(first.mean_pv, np.arange(1, 13) * 100)
        engine = ForecastEngine(data)
        updated = engine.forecast(36, 6, True, allowed_issues=(0, 6))
        baseline = engine.forecast(36, 6, True, allowed_issues=(0,))
        self.assertEqual(updated.mean_pv[-1], 900)
        self.assertEqual(baseline.mean_pv[-1], 4200)
        self.assertEqual(baseline.provenance["forecast_issue"], 0)

    def test_lazy_february_start_is_explicit_and_finite(self):
        engine = ForecastEngine(synthetic(), "ridge")
        result = engine.forecast(31 * 144, 288, True, True)
        self.assertEqual(result.net_s.shape, (12, 288))
        self.assertTrue(np.isfinite(result.net_s).all())
        self.assertTrue((result.price_s > 0).all())
        self.assertGreater(engine.diagnostics()["naive_initialization_record_count"], 0)
        self.assertEqual(len(engine.audit_records), 1)

    def test_fixed_price_problem_cannot_use_attachment4(self):
        data = synthetic()
        changed = replace(data, price=data.price * np.linspace(.01, 20, data.n_steps).reshape(data.price.shape))
        for family in ("naive", "ridge", "boost"):
            first = ForecastEngine(data, family).forecast(31 * 144, 288, False, False)
            second = ForecastEngine(changed, family).forecast(31 * 144, 288, False, False)
            np.testing.assert_array_equal(first.net_s, second.net_s, err_msg=family)
            np.testing.assert_array_equal(first.price_s, second.price_s, err_msg=family)

    def test_one_scenario_is_the_point_forecast(self):
        result = ForecastEngine(synthetic()).forecast(31 * 144 + 41, 288, True, True, 1)
        np.testing.assert_allclose(result.net_s[0], (result.mean_load - result.mean_pv) / 6)
        np.testing.assert_allclose(result.price_s[0], result.mean_price)
        self.assertEqual(result.provenance["residual_block_count"], 0)

    def test_historical_block_has_one_forecast_origin(self):
        data = synthetic()
        changed = replace(data, load=data.load.copy(), pv=data.pv.copy(), price=data.price.copy())
        changed.load[21:] *= 2
        changed.pv[21:] *= .1
        changed.price[21:] *= 4
        first, second = ForecastEngine(data), ForecastEngine(changed)
        first._set_price_mode(True); second._set_price_mode(True)
        a, b = first._historical_record(21), second._historical_record(21)
        self.assertEqual(a.shape, (432, 3))
        np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
