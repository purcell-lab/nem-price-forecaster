"""
Per-hour-of-day isotonic calibration of PD7DAY RRP forecasts.

Background
----------
PD7DAY is a predispatch model run — it systematically over-predicts price
during low-demand (overnight/solar) hours and under-predicts during spike
events.  A single global calibration curve (one PAV fit across all hours) fails
because it sees a mixture of very different price distributions and applies a
single monotone mapping, resulting in residual time-of-day bias.

Our fix: fit 24 separate Pool Adjacent Violators (PAV / isotonic regression)
curves, one per hour-of-day (0..23) in NEM time.  Each curve maps predicted
RRP → expected realised RRP for that hour bucket.

Key design decisions
--------------------
1. Recency weighting.  We weight each (predicted, actual) pair by
       w = exp(-age_days / half_life_days)
   so stale observations (e.g., last summer) contribute exponentially less
   than recent ones.  This prevents a hot-summer calibration from warping
   winter forecasts.

2. Import vs export calibration.  EXPORT calibration MUST be fitted against
   the EXPORT (feed-in) price time series, NOT the import price.  Using the
   wrong series doubles overnight export values because the export price
   (wholesale only) is systematically lower during solar hours.  Use separate
   Pd7DayCalibrator instances if your retailer reports different import and
   export spot prices.

3. Plausibility cap.  After calibration, values are clamped to
   [-0.10, plausibility_cap_AUD_per_kWh].  The lower bound allows modest
   negative prices (common in SA1/VIC1 during solar oversupply) without
   propagating extreme market floor (-$1000/MWh) artefacts to retail prices.

4. Minimum observations guard.  Calibration is only activated once at least
   `min_observations` (hour, predicted, actual) triplets have been collected.
   Before that, a linear fallback (multiply by 1.0) is returned.

Pure numpy — no scipy or sklearn dependency.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

_LOGGER = logging.getLogger(__name__)

_HOURS_PER_DAY = 24
# Absolute floor for calibrated prices ($/kWh) — allows realistic negatives
_PRICE_FLOOR_PER_KWH = -0.10

# ---------------------------------------------------------------------------
# Adjacency-blend default
# ---------------------------------------------------------------------------
# Each calibrated price is blended with the calibrated prices from the
# adjacent hours (hour-1 and hour+1).  The formula is:
#
#   blended = (calibrate(hour) + alpha * calibrate(hour-1) + alpha * calibrate(hour+1))
#             / (1 + 2 * alpha)
#
# alpha=0.0  → pure per-hour curve (no blending, maximum raw MAE accuracy)
# alpha=0.5  → equal contribution from self + half-weight neighbours
#              (default: smooths PAV hour-boundary discontinuities; see note)
#
# alpha=0.0 wins marginally on raw MAE, but alpha=0.5 produces fewer
# hour-boundary jumps and better forecast persistence (smoother, more
# stable forecasts) for a small raw-accuracy cost.
# Default α=0.5 is the stability-adjusted optimum.  Change via
# SIDECAR_CALIBRATOR_ADJACENCY_ALPHA.
_ADJACENT_HOUR_BLEND_WEIGHT: float = 0.5


@dataclass
class CalibrationObservation:
    """One (predicted_rrp, actual_rrp, hour_of_day, observation_datetime) data point."""
    predicted_rrp_per_mwh: float
    actual_rrp_per_mwh: float
    hour_of_day: int  # 0..23 in NEM time (UTC+10)
    observed_at: datetime  # aware datetime, used for recency weighting


class IsotonicCalibratorPerHour:
    """
    Holds observations and fits per-hour PAV isotonic regression.

    Usage
    -----
    calibrator = IsotonicCalibratorPerHour(
        recency_half_life_days=30,
        min_observations=14,
        plausibility_cap_dollars_per_kwh=5.0,
    )

    # Feed historical (predicted, actual) pairs — call repeatedly as new
    # actuals arrive from the NEMWeb TRADINGPRICE / Amber feed.
    calibrator.add_observation(predicted_rrp, actual_rrp, hour_of_day, timestamp)

    # Apply calibration to a raw PD7DAY RRP ($/MWh) for a given hour
    calibrated_kwh = calibrator.calibrate(raw_rrp_per_mwh, hour_of_day)
    """

    def __init__(
        self,
        recency_half_life_days: float = 30.0,
        min_observations: int = 14,
        plausibility_cap_dollars_per_kwh: float = 5.0,
        adjacent_blend_weight: float = _ADJACENT_HOUR_BLEND_WEIGHT,
    ) -> None:
        self._recency_half_life_days = recency_half_life_days
        self._min_observations = min_observations
        self._plausibility_cap_dollars_per_kwh = plausibility_cap_dollars_per_kwh
        # Adjacency blending: weight of adjacent-hour curves mixed with the
        # target hour's curve.  0.0 = pure per-hour (no blend), 0.5 = default.
        self._adjacent_blend_weight = adjacent_blend_weight

        # Per-hour storage: lists of (predicted_mwh, actual_mwh, weight)
        # Weights are computed lazily at fit time from the stored timestamps.
        self._observations: list[CalibrationObservation] = []

        # Cached fitted curves (invalidated on new observation).
        # Keyed by observation count at fit time — a new observation sets
        # _fitted_curves to None via add_observation(); _get_curve_for_hour()
        # refits and stores the count so repeat calibrate() calls within the
        # same predict cycle (same observation set) reuse the cache.
        self._fitted_curves: Optional[list[Optional[_IsotonicCurve]]] = None
        self._fit_observation_count: int = -1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_observation(
        self,
        predicted_rrp_per_mwh: float,
        actual_rrp_per_mwh: float,
        hour_of_day: int,
        observed_at: datetime,
    ) -> None:
        """Record one (predicted, actual) price pair and invalidate the cache.

        Silently drops observations with NaN/infinite predicted or actual values
        to prevent downstream PAV failures.  A debug log is emitted when data
        is dropped so this is diagnosable.
        """
        if not (0 <= hour_of_day < _HOURS_PER_DAY):
            raise ValueError(f"hour_of_day must be 0..23, got {hour_of_day}")

        predicted_float = float(predicted_rrp_per_mwh)
        actual_float = float(actual_rrp_per_mwh)

        if not math.isfinite(predicted_float) or not math.isfinite(actual_float):
            _LOGGER.debug(
                "Dropping calibration observation with non-finite values: "
                "predicted=%.3g, actual=%.3g, hour=%d",
                predicted_float,
                actual_float,
                hour_of_day,
            )
            return

        self._observations.append(
            CalibrationObservation(
                predicted_rrp_per_mwh=predicted_float,
                actual_rrp_per_mwh=actual_float,
                hour_of_day=int(hour_of_day),
                observed_at=observed_at,
            )
        )
        self._fitted_curves = None  # invalidate

    def add_observations_bulk(
        self, observations: list[CalibrationObservation]
    ) -> None:
        """Add multiple observations at once."""
        self._observations.extend(observations)
        self._fitted_curves = None

    def calibrate(
        self,
        raw_rrp_per_mwh: float,
        hour_of_day: int,
        reference_time: Optional[datetime] = None,
    ) -> float:
        """
        Map *raw_rrp_per_mwh* (PD7DAY $/MWh) to a calibrated retail price in $/kWh.

        Steps:
          1. Apply the per-hour PAV isotonic curve (if fitted).
          2. Convert $/MWh → $/kWh (divide by 1000).
          3. Clamp to [_PRICE_FLOOR_PER_KWH, plausibility_cap].

        If there are insufficient observations, returns raw_rrp_per_mwh / 1000
        (direct conversion, no calibration applied).
        """
        if not (0 <= hour_of_day < _HOURS_PER_DAY):
            raise ValueError(f"hour_of_day must be 0..23, got {hour_of_day}")

        if reference_time is None:
            from datetime import timezone
            reference_time = datetime.now(timezone.utc)

        curve = self._get_curve_for_hour(hour_of_day, reference_time)
        if curve is None:
            # Fallback: direct conversion (no blending when uncalibrated)
            calibrated_mwh = raw_rrp_per_mwh
        elif self._adjacent_blend_weight == 0.0:
            # Pure per-hour curve — no blending overhead
            calibrated_mwh = curve.predict(raw_rrp_per_mwh)
        else:
            # Adjacency blend: (target + alpha*prev + alpha*next) / (1 + 2*alpha)
            prev_hour = (hour_of_day - 1) % _HOURS_PER_DAY
            next_hour = (hour_of_day + 1) % _HOURS_PER_DAY
            prev_curve = self._get_curve_for_hour(prev_hour, reference_time)
            next_curve = self._get_curve_for_hour(next_hour, reference_time)

            calibrated_target = curve.predict(raw_rrp_per_mwh)
            calibrated_prev = (
                prev_curve.predict(raw_rrp_per_mwh) if prev_curve is not None
                else calibrated_target
            )
            calibrated_next = (
                next_curve.predict(raw_rrp_per_mwh) if next_curve is not None
                else calibrated_target
            )
            blend_alpha = self._adjacent_blend_weight
            calibrated_mwh = (
                calibrated_target
                + blend_alpha * calibrated_prev
                + blend_alpha * calibrated_next
            ) / (1.0 + 2.0 * blend_alpha)

        calibrated_kwh = calibrated_mwh / 1000.0
        return float(
            np.clip(
                calibrated_kwh,
                _PRICE_FLOOR_PER_KWH,
                self._plausibility_cap_dollars_per_kwh,
            )
        )

    @property
    def observation_count(self) -> int:
        return len(self._observations)

    @property
    def is_calibrated(self) -> bool:
        """True once enough observations exist to use the fitted curves."""
        return len(self._observations) >= self._min_observations

    def per_hour_observation_counts(self) -> list[int]:
        """Return a 24-element list of how many observations exist per hour."""
        counts = [0] * _HOURS_PER_DAY
        for observation in self._observations:
            counts[observation.hour_of_day] += 1
        return counts

    # ------------------------------------------------------------------
    # Internal fitting
    # ------------------------------------------------------------------

    def _get_curve_for_hour(
        self, hour_of_day: int, reference_time: datetime
    ) -> Optional["_IsotonicCurve"]:
        """Return the fitted isotonic curve for *hour_of_day*, fitting if needed.

        The 24-curve fit is expensive.  We cache it keyed on the observation
        count: a single refit per predict cycle (once per new observation batch)
        instead of one refit per slot.  _fitted_curves is set to None by
        add_observation() / add_observations_bulk() on mutation.
        """
        current_count = len(self._observations)
        if current_count < self._min_observations:
            return None

        # Refit only when the observation set has changed (new obs added)
        if self._fitted_curves is None or self._fit_observation_count != current_count:
            self._fitted_curves = self._fit_all_hours(reference_time)
            self._fit_observation_count = current_count

        return self._fitted_curves[hour_of_day]

    def _fit_all_hours(
        self, reference_time: datetime
    ) -> list[Optional["_IsotonicCurve"]]:
        """
        Fit one PAV isotonic curve per hour-of-day.

        For hours with fewer than 2 observations, returns None (fallback to
        linear pass-through for those hours).
        """
        # Compute per-observation recency weight
        reference_timestamp = reference_time.timestamp()
        decay_constant = self._recency_half_life_days * 86400.0  # seconds

        per_hour_predicted: list[list[float]] = [[] for _ in range(_HOURS_PER_DAY)]
        per_hour_actual: list[list[float]] = [[] for _ in range(_HOURS_PER_DAY)]
        per_hour_weights: list[list[float]] = [[] for _ in range(_HOURS_PER_DAY)]

        for observation in self._observations:
            age_seconds = reference_timestamp - observation.observed_at.timestamp()
            weight = math.exp(-max(0.0, age_seconds) / decay_constant)
            hour = observation.hour_of_day
            per_hour_predicted[hour].append(observation.predicted_rrp_per_mwh)
            per_hour_actual[hour].append(observation.actual_rrp_per_mwh)
            per_hour_weights[hour].append(weight)

        curves: list[Optional[_IsotonicCurve]] = []
        for hour_index in range(_HOURS_PER_DAY):
            predicted_values = np.array(per_hour_predicted[hour_index], dtype=np.float64)
            actual_values = np.array(per_hour_actual[hour_index], dtype=np.float64)
            weights = np.array(per_hour_weights[hour_index], dtype=np.float64)

            if len(predicted_values) < 2:
                curves.append(None)
                continue

            try:
                curve = _fit_isotonic_curve(predicted_values, actual_values, weights)
                curves.append(curve)
            except Exception as fit_error:
                _LOGGER.warning(
                    "PAV fit failed for hour %d: %s; using pass-through for this hour",
                    hour_index,
                    fit_error,
                )
                curves.append(None)

        return curves


# ---------------------------------------------------------------------------
# Pool Adjacent Violators (PAV) isotonic regression — pure numpy
# ---------------------------------------------------------------------------

@dataclass
class _IsotonicCurve:
    """
    Fitted isotonic curve stored as (sorted_x, fitted_y) breakpoint arrays.

    Prediction uses linear interpolation between breakpoints, with constant
    extrapolation beyond the training range.
    """
    sorted_x: np.ndarray   # predicted RRP values, sorted ascending
    fitted_y: np.ndarray   # isotonically fitted actual RRP values


def _fit_isotonic_curve(
    predicted: np.ndarray,
    actual: np.ndarray,
    weights: np.ndarray,
) -> _IsotonicCurve:
    """
    Fit a weighted isotonic (non-decreasing) regression via Pool Adjacent
    Violators.

    Minimises sum_i w_i * (y_i - f(x_i))^2 subject to f being non-decreasing,
    where x_i are the predicted values and y_i are the actual values.

    Returns an _IsotonicCurve suitable for interpolated prediction.
    """
    # Sort by predicted value
    sort_order = np.argsort(predicted)
    sorted_predicted = predicted[sort_order]
    sorted_actual = actual[sort_order]
    sorted_weights = weights[sort_order]

    # PAV algorithm: merge adjacent violating blocks into weighted averages
    # Each block is (weighted_sum_y, total_weight, list_of_indices)
    blocks: list[tuple[float, float, list[int]]] = []

    for data_index in range(len(sorted_predicted)):
        new_block_y_sum = sorted_actual[data_index] * sorted_weights[data_index]
        new_block_weight = sorted_weights[data_index]
        new_block_indices = [data_index]

        # Merge with previous block if it violates isotonicity (its mean > new mean)
        while blocks:
            prev_y_sum, prev_weight, prev_indices = blocks[-1]
            if (prev_y_sum / prev_weight) > (new_block_y_sum / new_block_weight):
                # Violation: pool the two blocks
                new_block_y_sum += prev_y_sum
                new_block_weight += prev_weight
                new_block_indices = prev_indices + new_block_indices
                blocks.pop()
            else:
                break

        blocks.append((new_block_y_sum, new_block_weight, new_block_indices))

    # Build the fitted_y array from the pooled block means
    fitted_y = np.empty(len(sorted_predicted), dtype=np.float64)
    for block_y_sum, block_weight, block_indices in blocks:
        block_mean = block_y_sum / block_weight
        for data_index in block_indices:
            fitted_y[data_index] = block_mean

    return _IsotonicCurve(sorted_x=sorted_predicted, fitted_y=fitted_y)


def _predict_isotonic(curve: _IsotonicCurve, query_value: float) -> float:
    """
    Predict the calibrated value for *query_value* using *curve*.

    Uses np.interp which performs linear interpolation between breakpoints and
    constant extrapolation (clamp to boundary) outside the training range.
    """
    return float(np.interp(query_value, curve.sorted_x, curve.fitted_y))


# Attach predict method to the dataclass
_IsotonicCurve.predict = lambda self, query_value: _predict_isotonic(self, query_value)
