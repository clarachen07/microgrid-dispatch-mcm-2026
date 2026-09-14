"""Causal forecasts and finite historical residual scenarios.

Every actual observation is indexed by the interval it describes. An origin of
``i`` can inspect observations ``[:i]`` only. No current-interval actual is an
optimization input. Historical forecast records are either predictions made at
their original origin or explicitly labelled causal naive reconstructions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import time

import numpy as np

from .data import DataBundle


class _RidgeModel:
    """Small standardized ridge solve with explicitly checked finite arrays.

    Explicit contractions avoid spurious floating-point status warnings from
    the platform's NumPy/Accelerate matrix multiplication implementation.
    """

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        from scipy.linalg import solve
        self.center = x.mean(axis=0)
        self.scale = x.std(axis=0)
        self.scale[self.scale < 1e-12] = 1.
        z = (x - self.center) / self.scale
        self.intercept = y.mean(axis=0)
        gram = np.einsum("ni,nj->ij", z, z, optimize=False)
        gram.flat[::len(gram) + 1] += 30.
        cross = np.einsum("ni,nj->ij", z, y - self.intercept, optimize=False)
        self.coef = solve(gram, cross, assume_a="pos", check_finite=True)
        if not np.isfinite(self.coef).all():
            raise FloatingPointError("non-finite ridge coefficients")

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.center) / self.scale
        return np.einsum("ni,ij->nj", z, self.coef, optimize=False) + self.intercept


@dataclass
class ForecastResult:
    net_s: np.ndarray
    price_s: np.ndarray
    mean_load: np.ndarray
    mean_pv: np.ndarray
    mean_price: np.ndarray
    provenance: dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


class ForecastEngine:
    """Daily fitted point models and 12 past, completely observed error blocks.

    Ridge/boost learn corrections to the same causal seasonal baseline. A
    second-day forecast uses the origin's last observed day, never the target
    day's unobserved preceding day. Price is modelled in logarithms.
    """

    def __init__(self, data: DataBundle, family: str = "naive", seed: int = 2026):
        if family not in {"naive", "ridge", "boost"}:
            raise ValueError("family must be naive, ridge, or boost")
        self.data = data
        self.family = family
        self.seed = int(seed)
        self._records: dict[int, np.ndarray] = {}
        self._record_sources: dict[int, str] = {}
        self._daily: dict[int, np.ndarray] = {}
        self._baseline_cache: dict[tuple[int, int], np.ndarray] = {}
        self._day_fit: int | None = None
        self._model: Any = None
        self._price_mode: bool | None = None
        # Fixed-tariff problems may not use Attachment 4, even indirectly as
        # a feature for load/PV. The source is selected on the first query.
        self._price_daily = np.broadcast_to(data.prior_price, data.price.shape)
        self._price_flat = np.tile(data.prior_price, data.n_days)
        self._calls = 0
        self._max_actual_end = 0
        self.audit_records: list[dict[str, Any]] = []
        self._start_weekday = int((data.start_date - np.datetime64("1970-01-05", "D")).astype(int) % 7)

    def _weekday(self, day: int) -> int:
        return (self._start_weekday + day) % 7

    def _prior(self) -> np.ndarray:
        return np.column_stack((self.data.prior_load, self.data.prior_pv,
                                np.log(np.maximum(self.data.prior_price, 1e-6))))

    def _actual(self, start: int, end: int) -> np.ndarray:
        return np.column_stack((self.data.flat_load[start:end], self.data.flat_pv[start:end],
                                np.log(np.maximum(self._price_flat[start:end], 1e-6))))

    def _set_price_mode(self, variable_price: bool) -> None:
        if self._price_mode == variable_price:
            return
        self._price_mode = variable_price
        self._price_daily = (self.data.price if variable_price else
                             np.broadcast_to(self.data.prior_price, self.data.price.shape))
        self._price_flat = self.data.flat_price if variable_price else np.tile(self.data.prior_price, self.data.n_days)
        self._records.clear(); self._record_sources.clear(); self._daily.clear()
        self._baseline_cache.clear(); self._day_fit = None; self._model = None

    def _baseline_day(self, origin_day: int, target_day: int) -> np.ndarray:
        key = (origin_day, target_day)
        if key in self._baseline_cache:
            return self._baseline_cache[key]
        prior = self._prior()
        if origin_day <= 0:
            result = prior.copy()
        else:
            end = min(origin_day, self.data.n_days)
            begin = max(0, end - 28)
            recent_days = np.arange(max(0, end - 7), end)
            same_days = np.array([d for d in range(begin, end)
                                  if self._weekday(d) == self._weekday(target_day)], dtype=int)
            load_recent = self.data.load[recent_days].mean(axis=0)
            pv_recent = self.data.pv[recent_days].mean(axis=0)
            lp_recent = np.log(self._price_daily[recent_days]).mean(axis=0)
            if not len(same_days):
                same_days = recent_days
            load_same = self.data.load[same_days].mean(axis=0)
            lp_same = np.log(self._price_daily[same_days]).mean(axis=0)
            result = np.column_stack((
                .70 * load_same + .20 * load_recent + .10 * prior[:, 0],
                .85 * pv_recent + .15 * prior[:, 1],
                .60 * lp_same + .30 * lp_recent + .10 * prior[:, 2],
            ))
        self._baseline_cache[key] = result
        return result

    def _features(self, origin_day: int, target_day: int) -> tuple[np.ndarray, np.ndarray]:
        baseline = self._baseline_day(origin_day, target_day)
        prior = self._prior()
        last = self._actual((origin_day - 1) * 144, origin_day * 144) if origin_day > 0 else prior
        lag7_day = target_day - 7
        lag7 = self._actual(lag7_day * 144, (lag7_day + 1) * 144) if 0 <= lag7_day < origin_day else prior
        slot = np.arange(144)
        angle = 2 * np.pi * (slot + 1) / 144
        seasonal = 2 * np.pi * target_day / 365.25
        weekday = np.zeros((144, 7)); weekday[:, self._weekday(target_day)] = 1
        scaled = lambda x: x / np.array([5000., 5000., 1.])
        features = np.column_stack((
            scaled(baseline), scaled(last), scaled(lag7), scaled(prior),
            np.sin(angle), np.cos(angle), np.sin(2 * angle), np.cos(2 * angle),
            np.full(144, np.sin(seasonal)), np.full(144, np.cos(seasonal)),
            np.full(144, target_day - origin_day), weekday,
        ))
        return features, baseline

    def _fit(self, day: int) -> None:
        if self._day_fit == day:
            return
        self._day_fit, self._model = day, None
        if self.family == "naive" or day < 7:
            return
        features, targets = [], []
        for origin_day in range(max(1, day - 60), day):
            for ahead in (0, 1):
                target_day = origin_day + ahead
                if target_day >= day:
                    continue
                x, baseline = self._features(origin_day, target_day)
                y = self._actual(target_day * 144, (target_day + 1) * 144)
                features.append(x)
                targets.append((y - baseline) / np.array([5000., 5000., 1.]))
        if not features:
            return
        x, y = np.vstack(features), np.vstack(targets)
        if self.family == "ridge":
            self._model = _RidgeModel()
            self._model.fit(x, y)
        else:
            from sklearn.ensemble import HistGradientBoostingRegressor
            self._model = []
            # Subsampling time points bounds the daily fit cost without looking
            # at any test-day observation; all 144 points are still predicted.
            x, y = x[::2], y[::2]
            for j in range(3):
                model = HistGradientBoostingRegressor(
                    max_iter=60, max_leaf_nodes=15, min_samples_leaf=36,
                    learning_rate=.08, l2_regularization=5., early_stopping=False,
                    random_state=self.seed,
                )
                model.fit(x, y[:, j]); self._model.append(model)

    @staticmethod
    def _physical(transformed: np.ndarray) -> np.ndarray:
        if not np.isfinite(transformed).all():
            raise FloatingPointError("non-finite point forecast")
        result = transformed.copy()
        result[:, :2] = np.maximum(result[:, :2], 0)
        result[:, 2] = np.exp(np.clip(result[:, 2], -9, 3))
        return result

    def _ensure_day(self, day: int) -> np.ndarray:
        if day in self._daily:
            return self._daily[day]
        start = time.perf_counter()
        self._fit(day)
        parts = []
        # Three daily profiles cover a 48 h query issued after midnight.
        for target_day in (day, day + 1, day + 2):
            x, baseline = self._features(day, target_day)
            pred = baseline.copy()
            if self._model is not None:
                correction = (np.column_stack([m.predict(x) for m in self._model])
                              if isinstance(self._model, list) else self._model.predict(x))
                pred += correction * np.array([5000., 5000., 1.])
            parts.append(self._physical(pred))
        result = np.vstack(parts)
        self._daily[day] = result
        self._records[day] = result
        self._record_sources[day] = self.family if self._model is not None else "naive"
        self.audit_records.append({
            "day": day, "family": self.family,
            "effective_family": self._record_sources[day],
            "training_actual_end_exclusive": day * 144,
            "training_days_max": min(day, 60),
            "variable_price": self._price_mode,
            "fit_and_predict_seconds": time.perf_counter() - start,
        })
        return result

    def _historical_record(self, day: int) -> np.ndarray:
        if day not in self._records:
            # Lazy Feb-1 start: do not fit the selected February model on
            # January targets and then pretend those were January predictions.
            self._records[day] = np.vstack([
                self._physical(self._baseline_day(day, target_day))
                for target_day in (day, day + 1, day + 2)
            ])
            self._record_sources[day] = "causal_naive_initialization"
        return self._records[day]

    def _latest_issue(self, origin: int, allowed: tuple[int, ...]) -> tuple[int, int, int] | None:
        if not allowed:
            return None
        day, phase = divmod(origin, 144)
        for candidate_day in (day, day - 1):
            if not 0 <= candidate_day < self.data.n_days:
                continue
            for hour in sorted(allowed, reverse=True):
                issue_origin = candidate_day * 144 + hour * 6
                if issue_origin <= origin:
                    return candidate_day, hour // 6, issue_origin
        return None

    def _official_curve(self, origin: int, horizon: int, allowed: tuple[int, ...],
                        fallback: np.ndarray) -> tuple[np.ndarray, int | None, np.ndarray]:
        latest = self._latest_issue(origin, allowed)
        result = fallback.copy()
        mask = np.zeros(horizon, dtype=bool)
        if latest is None:
            return result, None, mask
        day, issue_slot, issue_origin = latest
        endpoints = origin + np.arange(1, horizon + 1)
        targets = issue_origin + 6 * np.arange(1, 25)
        available_future = targets > origin
        if not available_future.any():
            return result, issue_origin, mask
        anchor = self.data.flat_pv[origin - 1] if origin > 0 else self.data.prior_pv[-1]
        xp = np.r_[origin, targets[available_future]]
        yp = np.r_[anchor, self.data.forecasts[day, issue_slot, available_future]]
        mask = endpoints <= targets[-1]
        result[mask] = np.interp(endpoints[mask], xp, yp)
        return np.maximum(result, 0), issue_origin, mask

    def _residuals(self, origin: int, horizon: int, count: int,
                   use_official: bool, allowed: tuple[int, ...]) -> tuple[np.ndarray, dict[str, Any]]:
        day, phase = divmod(origin, 144)
        if count == 1:
            return np.zeros((1, horizon, 3)), {
                "residual_block_count": 0, "residual_actual_end_exclusive": 0,
                "residual_initialization": "point_forecast_ablation_zero_residual",
            }
        # Entire historical target block must have ended by the current origin.
        candidates = [d for d in range(max(0, day - 60), day)
                      if d * 144 + phase + horizon <= origin]
        errors = np.zeros((count, horizon, 3))
        if not candidates:
            return errors, {"residual_block_count": 0, "residual_actual_end_exclusive": 0,
                            "residual_initialization": "zero_residual_cold_start"}
        # Fixed evenly-spaced historical blocks keep the scenario selection
        # deterministic, cover recent history, and avoid jitter between calls.
        selected = np.asarray(candidates)[np.round(np.linspace(0, len(candidates) - 1, count)).astype(int)]
        max_end = 0
        init = False
        for s, hist_day in enumerate(selected):
            start = int(hist_day) * 144 + phase
            end = start + horizon
            historical_prediction = self._historical_record(int(hist_day))[phase:phase + horizon].copy()
            if use_official:
                historical_prediction[:, 1], _, _ = self._official_curve(
                    start, horizon, allowed, historical_prediction[:, 1])
            actual = self._actual(start, end)
            errors[s, :, :2] = actual[:, :2] - historical_prediction[:, :2]
            errors[s, :, 2] = actual[:, 2] - np.log(np.maximum(historical_prediction[:, 2], 1e-6))
            max_end = max(max_end, end)
            init |= self._record_sources[int(hist_day)] == "causal_naive_initialization"
        return errors, {"residual_block_count": len(np.unique(selected)),
                        "residual_actual_end_exclusive": max_end,
                        "residual_initialization": "contains_causal_naive_initialization" if init else "recorded_origin_predictions"}

    def forecast(self, origin: int, horizon: int, use_pv_forecasts: bool = False,
                 variable_price: bool = False, scenario_count: int = 12,
                 allowed_issues: tuple[int, ...] | None = None) -> ForecastResult:
        origin, horizon, scenario_count = int(origin), int(horizon), int(scenario_count)
        if not 0 <= origin <= self.data.n_steps:
            raise ValueError("origin outside observed year")
        if not 1 <= horizon <= 288 or scenario_count < 1:
            raise ValueError("horizon must be 1..288 and scenario_count must be positive")
        allowed = (0, 6, 12, 18) if allowed_issues is None else tuple(sorted(set(allowed_issues)))
        if not set(allowed).issubset({0, 6, 12, 18}):
            raise ValueError("allowed_issues must contain only 0, 6, 12, 18")
        self._set_price_mode(bool(variable_price))
        day, phase = divmod(origin, 144)
        point = self._ensure_day(day)[phase:phase + horizon].copy()
        latest_issue = None
        if use_pv_forecasts:
            point[:, 1], latest_issue, _ = self._official_curve(origin, horizon, allowed, point[:, 1])
        if not variable_price:
            point[:, 2] = self.data.prior_price[(origin + np.arange(horizon)) % 144]
        errors, residual_audit = self._residuals(origin, horizon, scenario_count, use_pv_forecasts, allowed)
        loads = np.maximum(point[None, :, 0] + errors[:, :, 0], 0)
        pvs = np.maximum(point[None, :, 1] + errors[:, :, 1], 0)
        prices = (np.exp(np.clip(np.log(point[None, :, 2]) + errors[:, :, 2], -9, 3))
                  if variable_price else np.broadcast_to(point[None, :, 2], (scenario_count, horizon)).copy())
        provenance = {
            "origin": origin, "actual_history_end": origin,
            "actual_history_end_exclusive": origin,
            "maximum_actual_index": origin - 1,
            "training_actual_end_exclusive": day * 144,
            "forecast_issue": latest_issue,
            "forecast_issue_origin": latest_issue,
            "allowed_issues": list(allowed), "family": self.family,
            "horizon": horizon, "scenario_count": scenario_count,
            "future_price_known": not variable_price,
            "residual_horizon_policy": "same historical midnight origin predicts entire block; official residual uses historical query origin",
            "historical_price_source": "attachment4" if variable_price else "attachment1_fixed_daily_profile",
            **residual_audit,
        }
        if residual_audit["residual_actual_end_exclusive"] > origin or (latest_issue is not None and latest_issue > origin):
            raise AssertionError("causal information boundary violated")
        self._calls += 1; self._max_actual_end = max(self._max_actual_end, origin)
        return ForecastResult((loads - pvs) / 6, prices, point[:, 0], point[:, 1], point[:, 2], provenance)

    def diagnostics(self) -> dict[str, Any]:
        return {"family": self.family, "seed": self.seed, "forecast_calls": self._calls,
                "maximum_actual_history_end_exclusive": self._max_actual_end,
                "daily_fits": self.audit_records,
                "historical_record_count": len(self._records),
                "naive_initialization_record_count": sum(v == "causal_naive_initialization" for v in self._record_sources.values()),
                "interval_information_policy": "actual[:origin]; next interval actual excluded",
                "residual_policy": "whole past blocks mature before current origin; joint additive power and log-price errors"}
