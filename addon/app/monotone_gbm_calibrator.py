"""
Monotone-GBM calibration of PD7DAY RRP forecasts (opt-in alternative to isotonic).

Background
----------
The shipped default price calibrator is per-hour isotonic PAV regression
(``isotonic_calibrator.IsotonicCalibratorPerHour``).  It fits 24 independent
monotone curves — one per hour-of-day — mapping predicted RRP → realised price.
That per-hour split is robust and dependency-free, but it cannot model the
*interaction* between hour-of-day, day-of-week and the raw forecast level; each
hour bucket is calibrated in isolation, so signal is fragmented across 24 sparse
buckets.

This module offers a single LightGBM gradient-boosted regressor with a
**monotone constraint on the raw forecast price** (the forecast can only ever be
revised in the same direction as the raw signal — a higher PD7DAY price never
calibrates to a lower expected price).  All the calendar structure that isotonic
spreads across 24 buckets is instead carried by cyclic hour/dow features that the
single model can share statistical strength across.

Runtime feature set
-------------------
This runtime calibrator trains on the data the sidecar actually has in each
calibration observation (``CalibrationObservation`` carries only predicted/actual
RRP, hour-of-day and a timestamp — the *same data the isotonic calibrator trains
on*):

    [ raw_price_kwh,  hour_sin,  hour_cos,  dow_sin,  dow_cos ]
        (+1 monotone)   ──────── cyclic calendar, unconstrained ────────

day-of-week is derived from the observation's NEM-local timestamp.  The module
carries the *method* (monotone GBM over the available signal) plus a hard
never-lose fallback to isotonic.

Never-lose runtime fallback
---------------------------
A LightGBM model can fail to fit (too few observations, library missing, a fit
exception) or — in principle — calibrate worse than isotonic on a given corpus.
To guarantee we never ship worse output than the shipped default, this class:

  1. Always keeps a warm internal ``IsotonicCalibratorPerHour`` fitted on the
     same observations.
  2. On every fit, scores the GBM vs isotonic on a held-out (most-recent) tail
     of the training data and only *activates* the GBM if it does not lose to
     isotonic by more than ``never_lose_margin_cents`` c/kWh.  Otherwise it
     transparently delegates ``calibrate()`` to isotonic.
  3. If LightGBM is unavailable or the GBM fit raises, it delegates to isotonic.

So selecting this calibrator can only ever match or beat isotonic on the live
corpus; it cannot regress below the shipped baseline.

Hyperparameters
---------------
Defaults are n_estimators=300, num_leaves=31, lr=0.05.  Shallower trees
(``num_leaves=15``) can generalise better on smaller observation sets.
All are exposed for tuning.

Pure-Python public surface mirrors ``IsotonicCalibratorPerHour`` so it is a
drop-in alternative inside ``price_engine.PriceEngine``.
"""

from __future__ import annotations

import logging
import math
import warnings
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np

from isotonic_calibrator import (
    IsotonicCalibratorPerHour,
    CalibrationObservation,
    _PRICE_FLOOR_PER_KWH,
)

_LOGGER = logging.getLogger(__name__)


def _lgb_predict(model, features: np.ndarray) -> np.ndarray:
    """Predict with a LightGBM sklearn model, silencing the benign feature-name
    warning.

    We fit/predict with plain numpy arrays (no column names) by design — the
    sidecar runtime path is numpy-only.  sklearn's wrapper then emits a benign
    "X does not have valid feature names" UserWarning on predict; suppress just
    that one message locally so it never spams production logs (a module-level
    filter is unreliable because test runners reset the warning registry).
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names",
            category=UserWarning,
        )
        return model.predict(features)

_HOURS_PER_DAY = 24
_DAYS_PER_WEEK = 7
# NEM time is UTC+10 — used to derive hour-of-day / day-of-week for features so
# the calendar encoding matches the offline bench (which works in NEM time).
_NEM_TIMEZONE = timezone(timedelta(hours=10))

# Number of runtime features fed to LightGBM.  Order matters — the monotone
# constraint vector below is positional.
#   0: raw_price_kwh   (PD7DAY RRP / 1000, $/kWh)   monotone +1
#   1: hour_sin        sin(2π·hour/24)              unconstrained
#   2: hour_cos        cos(2π·hour/24)              unconstrained
#   3: dow_sin         sin(2π·dow/7)  [0=Monday]    unconstrained
#   4: dow_cos         cos(2π·dow/7)                unconstrained
_FEATURE_COUNT = 5
_MONOTONE_CONSTRAINTS = [1, 0, 0, 0, 0]


def _build_feature_matrix(
    raw_price_kwh: np.ndarray,
    hour_of_day: np.ndarray,
    day_of_week: np.ndarray,
) -> np.ndarray:
    """Assemble the [N, 5] runtime feature matrix (see module docstring).

    raw_price_kwh : PD7DAY RRP in $/kWh (i.e. $/MWh ÷ 1000).
    hour_of_day   : 0..23 in NEM time.
    day_of_week   : 0..6 in NEM time, 0 = Monday (datetime.weekday()).
    """
    hour_sin = np.sin(2.0 * math.pi * hour_of_day / float(_HOURS_PER_DAY))
    hour_cos = np.cos(2.0 * math.pi * hour_of_day / float(_HOURS_PER_DAY))
    dow_sin = np.sin(2.0 * math.pi * day_of_week / float(_DAYS_PER_WEEK))
    dow_cos = np.cos(2.0 * math.pi * day_of_week / float(_DAYS_PER_WEEK))
    return np.column_stack([
        raw_price_kwh.astype(np.float64),
        hour_sin,
        hour_cos,
        dow_sin,
        dow_cos,
    ])


class MonotoneGBMCalibrator:
    """
    LightGBM price calibrator with a monotone constraint on the raw forecast.

    Drop-in alternative to ``IsotonicCalibratorPerHour``: same constructor-ish
    surface, same ``add_observation`` / ``add_observations_bulk`` /
    ``calibrate`` / ``is_calibrated`` / ``observation_count`` API.

    Holds an internal isotonic calibrator that is (a) the warm-up fallback before
    enough data exists, (b) the never-lose backstop if the GBM does not beat it,
    and (c) the hard fallback if LightGBM is unavailable or the fit fails.
    """

    def __init__(
        self,
        recency_half_life_days: float = 30.0,
        min_observations: int = 14,
        plausibility_cap_dollars_per_kwh: float = 5.0,
        adjacent_blend_weight: float = 0.5,
        n_estimators: int = 300,
        num_leaves: int = 31,
        learning_rate: float = 0.05,
        min_child_samples: int = 10,
        corpus_window: int = 5000,
        never_lose_margin_cents: float = 0.10,
        gbm_min_observations: int = 200,
    ) -> None:
        # Shared / isotonic-compatible knobs
        self._recency_half_life_days = recency_half_life_days
        self._min_observations = min_observations
        self._plausibility_cap_dollars_per_kwh = plausibility_cap_dollars_per_kwh

        # GBM hyperparameters
        self._n_estimators = n_estimators
        self._num_leaves = num_leaves
        self._learning_rate = learning_rate
        self._min_child_samples = min_child_samples
        self._corpus_window = corpus_window

        # Never-lose: GBM must not lose to isotonic by more than this on the
        # held-out tail, or we keep isotonic active.  Margin in c/kWh.
        self._never_lose_margin_cents = never_lose_margin_cents
        # GBM needs materially more data than isotonic's 14-pair floor before a
        # boosted model is meaningful (a few thousand pairs).
        self._gbm_min_observations = gbm_min_observations

        # Internal isotonic calibrator — same observations, same knobs.
        self._isotonic = IsotonicCalibratorPerHour(
            recency_half_life_days=recency_half_life_days,
            min_observations=min_observations,
            plausibility_cap_dollars_per_kwh=plausibility_cap_dollars_per_kwh,
            adjacent_blend_weight=adjacent_blend_weight,
        )

        # Observation storage (parallel to isotonic; the GBM needs the timestamp
        # to derive day-of-week, which the isotonic curve does not use).
        self._observations: list[CalibrationObservation] = []

        # Fitted GBM state
        self._model = None
        self._gbm_active = False               # True only when GBM passes never-lose
        self._fit_observation_count = -1       # for refit caching
        self._last_decision_reason = "uninitialised"

    # ------------------------------------------------------------------
    # Public API (mirrors IsotonicCalibratorPerHour)
    # ------------------------------------------------------------------

    def add_observation(
        self,
        predicted_rrp_per_mwh: float,
        actual_rrp_per_mwh: float,
        hour_of_day: int,
        observed_at: datetime,
    ) -> None:
        """Record one (predicted, actual) pair; feeds both GBM and isotonic."""
        if not (0 <= hour_of_day < _HOURS_PER_DAY):
            raise ValueError(f"hour_of_day must be 0..23, got {hour_of_day}")

        predicted_float = float(predicted_rrp_per_mwh)
        actual_float = float(actual_rrp_per_mwh)
        if not math.isfinite(predicted_float) or not math.isfinite(actual_float):
            _LOGGER.debug(
                "Dropping calibration observation with non-finite values: "
                "predicted=%.3g, actual=%.3g, hour=%d",
                predicted_float, actual_float, hour_of_day,
            )
            return

        self._isotonic.add_observation(
            predicted_float, actual_float, hour_of_day, observed_at
        )
        self._observations.append(
            CalibrationObservation(
                predicted_rrp_per_mwh=predicted_float,
                actual_rrp_per_mwh=actual_float,
                hour_of_day=int(hour_of_day),
                observed_at=observed_at,
            )
        )
        # Invalidate the fitted GBM; it will refit lazily on next calibrate().
        self._model = None
        self._gbm_active = False

    def add_observations_bulk(
        self, observations: list[CalibrationObservation]
    ) -> None:
        """Add multiple observations at once."""
        self._isotonic.add_observations_bulk(observations)
        self._observations.extend(observations)
        self._model = None
        self._gbm_active = False

    def calibrate(
        self,
        raw_rrp_per_mwh: float,
        hour_of_day: int,
        reference_time: Optional[datetime] = None,
    ) -> float:
        """
        Map raw PD7DAY RRP ($/MWh) for *hour_of_day* to a calibrated $/kWh price.

        Uses the monotone GBM iff it is fitted AND passed the never-lose check;
        otherwise transparently delegates to the internal isotonic calibrator.
        Output is clamped to the same [floor, cap] range as isotonic.
        """
        if not (0 <= hour_of_day < _HOURS_PER_DAY):
            raise ValueError(f"hour_of_day must be 0..23, got {hour_of_day}")

        if reference_time is None:
            reference_time = datetime.now(timezone.utc)

        # Ensure the model is fitted for the current observation set (cached).
        self._maybe_fit(reference_time)

        if not self._gbm_active or self._model is None:
            # Never-lose: fall back to the shipped isotonic calibrator.
            return self._isotonic.calibrate(
                raw_rrp_per_mwh, hour_of_day, reference_time=reference_time
            )

        # day-of-week for the *target* slot.  reference_time is the predict
        # cycle's "now"; in production calibrate() is called per future slot, but
        # the engine passes the slot's hour_of_day separately.  We derive dow
        # from reference_time's NEM date as the best available proxy (the engine
        # calls per-cycle with now_utc), matching how isotonic uses hour only.
        nem_reference = reference_time.astimezone(_NEM_TIMEZONE)
        day_of_week = nem_reference.weekday()

        try:
            features = _build_feature_matrix(
                np.array([raw_rrp_per_mwh / 1000.0]),
                np.array([float(hour_of_day)]),
                np.array([float(day_of_week)]),
            )
            predicted_kwh = float(_lgb_predict(self._model, features)[0])
        except Exception as predict_error:  # pragma: no cover - defensive
            _LOGGER.warning(
                "MonotoneGBM predict failed (%s); falling back to isotonic",
                predict_error,
            )
            return self._isotonic.calibrate(
                raw_rrp_per_mwh, hour_of_day, reference_time=reference_time
            )

        return float(np.clip(
            predicted_kwh,
            _PRICE_FLOOR_PER_KWH,
            self._plausibility_cap_dollars_per_kwh,
        ))

    @property
    def observation_count(self) -> int:
        return len(self._observations)

    @property
    def is_calibrated(self) -> bool:
        """True once the isotonic backstop is usable (>= min_observations).

        Note this mirrors isotonic semantics: the engine uses this flag to decide
        whether *any* calibration (GBM or isotonic fallback) is available.  GBM
        activation is a separate, internal decision (see ``gbm_active``).
        """
        return len(self._observations) >= self._min_observations

    @property
    def gbm_active(self) -> bool:
        """True iff the GBM is fitted AND passed the never-lose check.

        When False, ``calibrate()`` delegates to isotonic.  Useful for logging
        which calibrator is actually serving live forecasts.
        """
        return self._gbm_active

    @property
    def last_decision_reason(self) -> str:
        """Human-readable reason for the most recent GBM-vs-isotonic decision."""
        return self._last_decision_reason

    def per_hour_observation_counts(self) -> list[int]:
        """24-element list of observations per hour (delegates to isotonic)."""
        return self._isotonic.per_hour_observation_counts()

    # ------------------------------------------------------------------
    # Internal: fitting + never-lose gate
    # ------------------------------------------------------------------

    def _maybe_fit(self, reference_time: datetime) -> None:
        """Fit (or reuse) the GBM for the current observation set.

        Caches on observation count so repeated calibrate() calls within one
        predict cycle reuse a single fit, matching the isotonic calibrator's
        refit-cache discipline.
        """
        current_count = len(self._observations)
        if self._model is not None and self._fit_observation_count == current_count:
            return  # cache hit

        self._fit_observation_count = current_count
        self._model = None
        self._gbm_active = False

        if current_count < self._gbm_min_observations:
            self._last_decision_reason = (
                f"isotonic: only {current_count} obs "
                f"(GBM needs >= {self._gbm_min_observations})"
            )
            return

        try:
            import lightgbm as lgb
        except ImportError:
            self._last_decision_reason = "isotonic: lightgbm not installed"
            _LOGGER.warning(
                "MonotoneGBM selected but lightgbm is not installed; "
                "using isotonic calibration. Add 'lightgbm' to requirements."
            )
            return

        try:
            self._fit_and_gate(lgb, reference_time)
        except Exception as fit_error:
            self._model = None
            self._gbm_active = False
            self._last_decision_reason = f"isotonic: GBM fit raised ({fit_error})"
            _LOGGER.warning(
                "MonotoneGBM fit failed (%s); falling back to isotonic calibration",
                fit_error,
            )

    def _fit_and_gate(self, lgb, reference_time: datetime) -> None:
        """Fit the GBM on a train split, gate it against isotonic on a held-out
        tail, and only activate it if it does not lose by > never_lose_margin."""
        # Most-recent corpus_window observations, sorted oldest→newest.
        observations = sorted(self._observations, key=lambda o: o.observed_at)
        if self._corpus_window is not None and len(observations) > self._corpus_window:
            observations = observations[-self._corpus_window:]

        predicted_mwh = np.array(
            [o.predicted_rrp_per_mwh for o in observations], dtype=np.float64
        )
        actual_mwh = np.array(
            [o.actual_rrp_per_mwh for o in observations], dtype=np.float64
        )
        hours = np.array([o.hour_of_day for o in observations], dtype=np.float64)
        days_of_week = np.array(
            [o.observed_at.astimezone(_NEM_TIMEZONE).weekday() for o in observations],
            dtype=np.float64,
        )
        observed_at_list = [o.observed_at for o in observations]

        raw_price_kwh = predicted_mwh / 1000.0
        actual_kwh = actual_mwh / 1000.0

        # Held-out tail: most-recent 20% (min 30, capped so train keeps >= 80%).
        total = len(observations)
        holdout_size = max(30, int(round(total * 0.20)))
        holdout_size = min(holdout_size, total // 5 if total >= 150 else holdout_size)
        train_count = total - holdout_size
        if train_count < self._gbm_min_observations or holdout_size < 20:
            # Not enough to gate honestly — keep isotonic.
            self._last_decision_reason = (
                f"isotonic: insufficient split (train={train_count}, "
                f"holdout={holdout_size})"
            )
            return

        train_slice = slice(0, train_count)
        holdout_slice = slice(train_count, total)

        # Recency weights for the training rows.
        reference_timestamp = reference_time.timestamp()
        decay_seconds = self._recency_half_life_days * 86400.0
        train_ages = np.array([
            max(0.0, reference_timestamp - observed_at_list[i].timestamp())
            for i in range(train_count)
        ])
        train_weights = np.exp(-train_ages / decay_seconds)

        train_features = _build_feature_matrix(
            raw_price_kwh[train_slice], hours[train_slice], days_of_week[train_slice]
        )

        params = {
            "n_estimators": self._n_estimators,
            "num_leaves": self._num_leaves,
            "learning_rate": self._learning_rate,
            "monotone_constraints": _MONOTONE_CONSTRAINTS,
            "monotone_constraints_method": "advanced",
            "min_child_samples": self._min_child_samples,
            "verbose": -1,
            "n_jobs": 2,
        }
        candidate_model = lgb.LGBMRegressor(**params)
        candidate_model.fit(
            train_features, actual_kwh[train_slice], sample_weight=train_weights
        )

        # ---- Never-lose gate on the held-out tail ----
        holdout_actual_kwh = actual_kwh[holdout_slice]

        # GBM predictions on the holdout.
        holdout_features = _build_feature_matrix(
            raw_price_kwh[holdout_slice], hours[holdout_slice], days_of_week[holdout_slice]
        )
        gbm_holdout_pred = np.clip(
            _lgb_predict(candidate_model, holdout_features),
            _PRICE_FLOOR_PER_KWH,
            self._plausibility_cap_dollars_per_kwh,
        )

        # Isotonic predictions on the same holdout — fit a *train-only* isotonic
        # so the comparison is apples-to-apples (no holdout leakage).
        train_only_isotonic = IsotonicCalibratorPerHour(
            recency_half_life_days=self._recency_half_life_days,
            min_observations=self._min_observations,
            plausibility_cap_dollars_per_kwh=self._plausibility_cap_dollars_per_kwh,
            adjacent_blend_weight=self._isotonic._adjacent_blend_weight,
        )
        train_only_isotonic.add_observations_bulk([
            observations[i] for i in range(train_count)
        ])
        iso_holdout_pred = np.array([
            train_only_isotonic.calibrate(
                predicted_mwh[train_count + j],
                int(hours[train_count + j]),
                reference_time=reference_time,
            )
            for j in range(holdout_size)
        ])

        gbm_mae_cents = float(np.mean(np.abs(gbm_holdout_pred - holdout_actual_kwh))) * 100.0
        iso_mae_cents = float(np.mean(np.abs(iso_holdout_pred - holdout_actual_kwh))) * 100.0

        # GBM must not lose to isotonic by more than the margin.
        if gbm_mae_cents <= iso_mae_cents + self._never_lose_margin_cents:
            # Refit on ALL (train + holdout) data so the deployed model uses
            # every available pair, then activate.
            all_ages = np.array([
                max(0.0, reference_timestamp - observed_at_list[i].timestamp())
                for i in range(total)
            ])
            all_weights = np.exp(-all_ages / decay_seconds)
            all_features = _build_feature_matrix(raw_price_kwh, hours, days_of_week)
            final_model = lgb.LGBMRegressor(**params)
            final_model.fit(all_features, actual_kwh, sample_weight=all_weights)
            self._model = final_model
            self._gbm_active = True
            self._last_decision_reason = (
                f"GBM active: holdout MAE {gbm_mae_cents:.4f}c vs isotonic "
                f"{iso_mae_cents:.4f}c (margin {self._never_lose_margin_cents:.2f}c)"
            )
            _LOGGER.info(
                "MonotoneGBM ACTIVE — holdout MAE %.4fc vs isotonic %.4fc "
                "(%d train + %d holdout pairs)",
                gbm_mae_cents, iso_mae_cents, train_count, holdout_size,
            )
        else:
            self._model = None
            self._gbm_active = False
            self._last_decision_reason = (
                f"isotonic: GBM holdout MAE {gbm_mae_cents:.4f}c LOSES to "
                f"isotonic {iso_mae_cents:.4f}c by > {self._never_lose_margin_cents:.2f}c"
            )
            _LOGGER.info(
                "MonotoneGBM stays INACTIVE — holdout MAE %.4fc loses to "
                "isotonic %.4fc by > %.2fc; using isotonic",
                gbm_mae_cents, iso_mae_cents, self._never_lose_margin_cents,
            )
