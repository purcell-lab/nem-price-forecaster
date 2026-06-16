"""
House-load forecaster — Darts LightGBM, direct multi-output.

Accuracy-focused positioning
-----------------------------
This forecaster uses a Darts LightGBM model with `output_chunk_length=48`
(24-hour output blocks).  Darts automatically recurses over multiple blocks to
produce the full planning horizon (e.g., 6 days = 6 × 48-slot recursive passes).
Using 48-slot output chunks rather than a single giant output is ~7× faster to
train AND scores better in backtests — the model avoids extreme LightGBM tree
expansion at the cost of a small number of recursive steps (6 for a 6-day horizon).
Each 48-slot block is still a direct multi-output sub-forecast, so error
accumulation is bounded per block rather than unlimited.

If you need a lighter-weight alternative (e.g., on a Raspberry Pi), EMHASS has
its own built-in house-load forecaster that uses a Lasso model and requires no
extra dependencies.  This project's niche is *maximum accuracy* — users who
want lightweight should use EMHASS's built-in forecaster instead.

Model architecture
------------------
    LightGBMModel(
        lags=96,                                            # 2-day history window (96 × 30 min)
        output_chunk_length=48,                             # 24-h output block; Darts auto-recurses
        lags_past_covariates=[-48, -96],                    # 1-day and 2-day prior calendar values
        lags_future_covariates=list(range(-(96-1), 48+1)), # all future steps (calendar + weather)
        verbose=-1,
    )

Covariate design (past + future split)
---------------------------------------
PAST covariates: calendar features (sin/cos of hour/dow/month, is_weekend) at
  lag positions [-48, -96].  These give the model the same-time-yesterday and
  same-time-two-days-ago calendar context — proven to improve load accuracy.

FUTURE covariates: calendar features + Open-Meteo weather for the full forecast
  horizon.  Weather is forecastable forward, so it belongs in future covariates
  rather than past.  Including it here lets the model see tomorrow's forecast
  temperature/cloud/radiation/wind/humidity at every target slot — which is the
  correct conditioning for forward-looking load prediction.

  Weather variables (all from Open-Meteo forecast + archive APIs, free, no key):
    temperature_2m (°C)          — heating/cooling load driver
    cloud_cover (%)              — solar self-consumption, comfort-cooling
    shortwave_radiation (W/m²)   — GHI solar signal (correlates with AC load)
    wind_speed_10m (m/s)         — comfort, some load correlation
    relative_humidity_2m (%)     — comfort-driven load

  When weather is unavailable (API down), the model trains and predicts with
  calendar-only future covariates — zero-pad weather columns are used so the
  model's feature dimension stays consistent.

Pitfalls (Darts API, as of v0.29–0.44)
----------------------------------------
1. `sample_weight` must be passed as a `darts.TimeSeries`, not a numpy array.
   Passing an ndarray raises "input series must be of type TimeSeries".
2. `add_encoders={'cyclic': {'future': [...]}}` is silently ignored for
   LightGBMModel (it does not support future covariates by default).  Use
   explicit `lags_past_covariates` + `lags_future_covariates` instead.
3. `future_covariates` must extend `output_chunk_length` steps beyond the
   end of the training series; build the future covariate series with padding.
4. Darts strips timezone info from DatetimeIndex; reconstruct UTC datetimes
   from slot positions on the way out.

Persistence
-----------
The fitted Darts model is saved via darts.models.LightGBMModel.save() (pickle).
The coordinator calls save_model() after each retrain and load_model() on startup.
Model file path: <ha_config>/.storage/nem_price_forecaster_{region}_load_model.pkl

EMHASS wiring
-------------
    load_power_forecast: >
      {{ state_attr('sensor.nem_price_forecaster_nsw1_load_forecast', 'forecast')
         | tojson }}
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import numpy as np

_LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FORECAST_SLOT_MINUTES = 30         # Match EMHASS / PD7DAY half-hour slots
_WATTS_MIN_PLAUSIBILITY = 0.0       # No negative load
_WATTS_MAX_PLAUSIBILITY = 20000.0   # 20 kW hard ceiling
_RECENCY_BOOST_DAYS = 7             # Last N days get 3× sample weight
_RECENCY_BOOST_FACTOR = 3.0
_DEFAULT_OUTPUT_CHUNK_LENGTH = 48   # 24 h single-shot blocks; Darts auto-recurses to horizon
_DEFAULT_LAGS = 96                  # 2 days × 48 slots/day input window
_DEFAULT_FORECAST_HORIZON_HOURS = 144  # 6 days = 288 slots
_MIN_TRAINING_SAMPLES = 288         # ~3 days minimum to train
_MODEL_FILE_SUFFIX = "_load_model.pkl"

# Calendar covariate lag indices (past positions relative to each target slot)
# We look back 1 day (lag=-48) and 2 days (lag=-96) so the model sees the
# typical load for the same time on previous days.
_CALENDAR_LAG_POSITIONS = [-48, -96]

# Weather columns in the future covariate matrix (order is significant — must
# match the order in _build_weather_covariate_values_for_load()).
_WEATHER_COLUMN_NAMES = [
    "temperature_2m",
    "cloud_cover",
    "shortwave_radiation",
    "wind_speed_10m",
    "relative_humidity_2m",
]
_N_WEATHER_COLUMNS = len(_WEATHER_COLUMN_NAMES)

# NOTE: _FUTURE_COVARIATE_LAGS has been removed.
# In Darts 0.40+, lags_future_covariates=None DISABLES future covariates entirely
# (the model errors when future_covariates are passed).  The correct value is an
# explicit list covering -(lags-1) to +output_chunk_length so that the model sees
# the covariate at every position in the forecast window.  This is now computed
# per-instance inside DartsLightGBMLoadForecaster.train() from self._lags and
# self._output_chunk_length.


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------

@dataclass
class LoadObservation:
    """One measured house-load data point (30-minute average, watts)."""
    interval_start_utc: datetime  # tz-aware, UTC
    load_watts: float


@dataclass
class LoadForecastSlot:
    """One 30-minute load forecast slot."""
    interval_start_utc: datetime  # tz-aware, UTC
    load_watts: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "datetime": self.interval_start_utc.isoformat(),
            "load_power": round(self.load_watts, 1),
        }


# ---------------------------------------------------------------------------
# Main forecaster class
# ---------------------------------------------------------------------------

class DartsLightGBMLoadForecaster:
    """
    Darts LightGBM house-load forecaster with direct multi-output prediction.

    Training
    --------
    Call `train(observations)` with a list of LoadObservation covering at least
    3 days (~288 half-hourly samples).  Nightly retraining is recommended.

    Prediction
    ----------
    Call `forecast(recent_observations, ...)` to produce a list of
    LoadForecastSlot covering the configured horizon.

    `recent_observations` must cover at least `lags` × 30 minutes of history
    (default: 96 × 30 min = 2 days) so the lag window can be fully populated.

    Model persistence
    -----------------
    Call `save_model(path)` after training and `load_model(path)` on startup.
    """

    def __init__(
        self,
        lags: int = _DEFAULT_LAGS,
        output_chunk_length: int = _DEFAULT_OUTPUT_CHUNK_LENGTH,
        forecast_horizon_hours: int = _DEFAULT_FORECAST_HORIZON_HOURS,
        use_weather_covariates: bool = True,
    ) -> None:
        self._lags = lags
        self._output_chunk_length = output_chunk_length
        self._forecast_horizon_hours = forecast_horizon_hours
        # Weather covariates are ENABLED by default — temperature, cloud cover,
        # radiation, wind, and humidity are all forecastable forward and materially
        # improve load accuracy.  Disable with use_weather_covariates=False for
        # environments where Open-Meteo is permanently unreachable.
        self._use_weather_covariates = use_weather_covariates

        self._model: Any = None  # darts LightGBMModel, set after train()
        self._is_trained: bool = False
        self._training_observation_count: int = 0
        # Set True after a successful train() with weather in the covariate set;
        # used to ensure predict() uses matching feature dimensions.
        self._training_has_weather: bool = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        observations: list[LoadObservation],
        weather_history: Optional[dict[datetime, Any]] = None,
    ) -> None:
        """
        Fit the Darts LightGBM model on *observations*.

        observations must be sorted ascending by interval_start_utc at a
        uniform 30-minute cadence.  Gaps are forward-filled.

        weather_history: optional dict mapping naive-UTC datetime → WeatherSlot
          (or dict with keys temperature_2m / cloud_cover / shortwave_radiation /
          wind_speed_10m / relative_humidity_2m).  Fetched from the Open-Meteo
          archive API by LoadEngine before calling train().  Used as FUTURE
          covariates — weather is forecastable, so it belongs in lags_future_covariates
          rather than past covariates.  When None or empty, the model trains with
          calendar-only future covariates (fully backward-compatible).

        At least _MIN_TRAINING_SAMPLES (288, ~3 days) are required.
        """
        try:
            from darts import TimeSeries  # type: ignore[import-untyped]
            from darts.models import LightGBMModel  # type: ignore[import-untyped]
            import pandas as pd  # type: ignore[import-untyped]
            import warnings
            warnings.filterwarnings("ignore", category=UserWarning)
        except ImportError as import_error:
            _LOGGER.error(
                "DartsLightGBMLoadForecaster: darts package not available: %s.  "
                "Install with: pip install darts  (also needs lightgbm package)",
                import_error,
            )
            return

        if len(observations) < _MIN_TRAINING_SAMPLES:
            _LOGGER.warning(
                "DartsLightGBMLoadForecaster: only %d observations (need %d); skipping fit",
                len(observations),
                _MIN_TRAINING_SAMPLES,
            )
            return

        # Build regularised 30-min grid (tz-naive — Darts strips TZ anyway)
        regular_times, load_values, sample_weights = self._build_regular_series(observations)

        if len(regular_times) < _MIN_TRAINING_SAMPLES:
            _LOGGER.warning(
                "DartsLightGBMLoadForecaster: after regularisation only %d slots; skipping fit",
                len(regular_times),
            )
            return

        import pandas as pd  # noqa: F811
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            times_index = pd.DatetimeIndex(regular_times)
            load_ts = TimeSeries.from_times_and_values(times_index, load_values)

            # sample_weight must be a TimeSeries (not ndarray) in Darts >= 0.29
            sample_weight_ts = TimeSeries.from_times_and_values(times_index, sample_weights)

            # ---- Past covariates ----
            # Calendar features (sin/cos hour/dow/month + is_weekend) at lag positions
            # [-48, -96] (same-time-yesterday, same-time-two-days-ago).
            # Using explicit past covariates avoids the add_encoders silent-ignore bug
            # in Darts v0.29+.
            calendar_values = self._build_calendar_covariate_values(regular_times)
            n_calendar_features = calendar_values.shape[1]
            past_covariate_ts = TimeSeries.from_times_and_values(
                times_index,
                calendar_values,
                columns=[f"cal_{col_idx}" for col_idx in range(n_calendar_features)],
            )

            # ---- Future covariates ----
            # Calendar + weather for the full training + output_chunk_length extension.
            # Future covariates must extend output_chunk_length steps BEYOND the end of
            # the training series (Darts hard requirement for direct multi-output models).
            future_tail_times = [
                regular_times[-1] + timedelta(minutes=30 * (slot_idx + 1))
                for slot_idx in range(self._output_chunk_length + 2)
            ]
            all_future_times = regular_times + future_tail_times
            future_calendar_values = self._build_calendar_covariate_values(all_future_times)

            has_weather = (
                self._use_weather_covariates
                and weather_history is not None
                and len(weather_history) > 0
            )

            if has_weather:
                weather_cov_values = _build_weather_covariate_values_for_load(
                    all_future_times, weather_history  # type: ignore[arg-type]
                )
                future_cov_values = np.column_stack([future_calendar_values, weather_cov_values])
            else:
                future_cov_values = future_calendar_values

            future_covariate_ts = TimeSeries.from_times_and_values(
                pd.DatetimeIndex(all_future_times),
                future_cov_values,
                columns=[f"fut_{col_idx}" for col_idx in range(future_cov_values.shape[1])],
            )

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")

                # Build the future-covariate lag range from the model's configured
                # values.  range(-(lags-1), output_chunk_length+1) covers every
                # position in the forecast window — from one step before the first
                # target up to the last target slot.  This matches the working
                # config from backtest/backtest_3way.py (line ~623).
                future_covariate_lag_range = list(
                    range(-(self._lags - 1), self._output_chunk_length + 1)
                )
                model = LightGBMModel(
                    lags=self._lags,
                    output_chunk_length=self._output_chunk_length,
                    lags_past_covariates=_CALENDAR_LAG_POSITIONS,
                    lags_future_covariates=future_covariate_lag_range,
                    verbose=-1,
                    n_estimators=200,
                    learning_rate=0.05,
                    num_leaves=31,
                    min_child_samples=20,
                )
                model.fit(
                    load_ts,
                    past_covariates=past_covariate_ts,
                    future_covariates=future_covariate_ts,
                    sample_weight=sample_weight_ts,
                )

        except Exception as fit_error:
            _LOGGER.error("DartsLightGBMLoadForecaster fit failed: %s", fit_error)
            return

        self._model = model
        self._is_trained = True
        self._training_observation_count = len(observations)
        self._training_has_weather = bool(has_weather)
        _LOGGER.info(
            "DartsLightGBMLoadForecaster trained: %d obs, lags=%d, output_chunk=%d, "
            "weather_covariates=%s",
            len(observations),
            self._lags,
            self._output_chunk_length,
            self._training_has_weather,
        )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def forecast(
        self,
        recent_observations: list[LoadObservation],
        weather_forecast: Optional[dict[datetime, Any]] = None,
        horizon_hours: Optional[int] = None,
    ) -> list[LoadForecastSlot]:
        """
        Produce a horizon-length load forecast starting from the next 30-min boundary.

        recent_observations: at least `lags` × 30-min of history, sorted ascending.
        weather_forecast: naive-UTC datetime → WeatherSlot dict for the forecast window
          (and ideally also the input window for the past-covariate padding range).
          Fetched from Open-Meteo forecast API by LoadEngine.  Pass None to predict
          without weather (falls back to calendar-only future covariates).
        horizon_hours: override the configured default (144 h = 6 days).
        """
        effective_horizon_hours = horizon_hours or self._forecast_horizon_hours
        num_slots = int(effective_horizon_hours * 60 / _FORECAST_SLOT_MINUTES)

        now_utc = datetime.now(timezone.utc)
        minutes_into_slot = now_utc.minute % 30
        forecast_origin_utc = now_utc - timedelta(
            minutes=minutes_into_slot,
            seconds=now_utc.second,
            microseconds=now_utc.microsecond,
        )
        forecast_start_utc = forecast_origin_utc + timedelta(minutes=30)

        if not self._is_trained or self._model is None:
            return self._fallback_forecast(recent_observations, forecast_start_utc, num_slots)

        try:
            return self._darts_forecast(
                recent_observations,
                forecast_start_utc,
                num_slots,
                weather_forecast,
            )
        except Exception as darts_error:
            _LOGGER.warning(
                "DartsLightGBMLoadForecaster: darts predict failed (%s); using fallback",
                darts_error,
            )
            return self._fallback_forecast(recent_observations, forecast_start_utc, num_slots)

    def _darts_forecast(
        self,
        recent_observations: list[LoadObservation],
        forecast_start_utc: datetime,
        num_slots: int,
        weather_forecast: Optional[dict[datetime, Any]],
    ) -> list[LoadForecastSlot]:
        """Run the Darts model and return clamped LoadForecastSlot list."""
        import pandas as pd
        from darts import TimeSeries
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            # ---- Input series (tz-naive for Darts) ----
            if len(recent_observations) < self._lags:
                # Pad left with the earliest known value
                pad_count = self._lags - len(recent_observations)
                pad_start = (
                    recent_observations[0].interval_start_utc - timedelta(minutes=30 * pad_count)
                    if recent_observations
                    else forecast_start_utc - timedelta(minutes=30 * self._lags)
                )
                pad_value = recent_observations[0].load_watts if recent_observations else 1000.0
                padded_observations = [
                    LoadObservation(
                        interval_start_utc=pad_start + timedelta(minutes=30 * idx),
                        load_watts=pad_value,
                    )
                    for idx in range(pad_count)
                ] + recent_observations
            else:
                padded_observations = recent_observations

            tail_times = [obs.interval_start_utc.replace(tzinfo=None) for obs in padded_observations]
            tail_values = np.array([obs.load_watts for obs in padded_observations], dtype=np.float64)
            input_ts = TimeSeries.from_times_and_values(
                pd.DatetimeIndex(tail_times), tail_values
            )

            # ---- Past covariates ----
            # Calendar features for the input window only.
            # lags_past_covariates=[-48, -96] instructs Darts to look back 1 and 2 days
            # from each target slot, giving same-time-yesterday context.
            calendar_past = self._build_calendar_covariate_values(tail_times)
            past_covariate_ts = TimeSeries.from_times_and_values(
                pd.DatetimeIndex(tail_times), calendar_past
            )

            # ---- Future covariates ----
            # Calendar + weather for input window + output_chunk_length extension.
            # Darts requires the future covariate series to span both the input and
            # the forecast output steps.
            forecast_extra_times = [
                tail_times[-1] + timedelta(minutes=30 * (slot_idx + 1))
                for slot_idx in range(self._output_chunk_length + 2)
            ]
            all_future_cov_times = tail_times + forecast_extra_times
            future_calendar_values = self._build_calendar_covariate_values(all_future_cov_times)

            # Use weather if the model was trained with it AND forecast weather is available
            use_weather_now = (
                self._training_has_weather
                and weather_forecast is not None
                and len(weather_forecast) > 0
            )

            if use_weather_now:
                future_weather_values = _build_weather_covariate_values_for_load(
                    all_future_cov_times, weather_forecast  # type: ignore[arg-type]
                )
                future_cov_values = np.column_stack([future_calendar_values, future_weather_values])
            elif self._training_has_weather:
                # Model expects weather columns but none available — pad with zeros
                _LOGGER.debug(
                    "DartsLightGBMLoadForecaster: model trained with weather but "
                    "no forecast weather available; padding weather columns with zeros"
                )
                zero_weather = np.zeros((len(all_future_cov_times), _N_WEATHER_COLUMNS))
                future_cov_values = np.column_stack([future_calendar_values, zero_weather])
            else:
                future_cov_values = future_calendar_values

            future_covariate_ts = TimeSeries.from_times_and_values(
                pd.DatetimeIndex(all_future_cov_times), future_cov_values
            )

            # Predict — output_chunk_length slots in a single pass
            predict_steps = max(num_slots, 1)
            raw_prediction = self._model.predict(
                predict_steps,
                series=input_ts,
                past_covariates=past_covariate_ts,
                future_covariates=future_covariate_ts,
            )

        # Extract values
        predicted_values = raw_prediction.to_series().values
        if predicted_values.ndim > 1:
            predicted_values = predicted_values.flatten()

        # Build output slots aligned to forecast_start_utc
        forecast_slots: list[LoadForecastSlot] = []
        for slot_index in range(min(num_slots, len(predicted_values))):
            slot_start_utc = forecast_start_utc + timedelta(
                minutes=slot_index * _FORECAST_SLOT_MINUTES
            )
            raw_watts = float(predicted_values[slot_index])
            clamped_watts = float(
                np.clip(raw_watts, _WATTS_MIN_PLAUSIBILITY, _WATTS_MAX_PLAUSIBILITY)
            )
            forecast_slots.append(
                LoadForecastSlot(interval_start_utc=slot_start_utc, load_watts=clamped_watts)
            )

        # Extend with fallback if model produced fewer slots than requested
        if len(forecast_slots) < num_slots:
            fallback_slots = self._fallback_forecast(
                recent_observations,
                forecast_start_utc + timedelta(
                    minutes=len(forecast_slots) * _FORECAST_SLOT_MINUTES
                ),
                num_slots - len(forecast_slots),
            )
            forecast_slots.extend(fallback_slots)

        return forecast_slots

    # ------------------------------------------------------------------
    # Fallback — rolling mean per (hour, dow)
    # ------------------------------------------------------------------

    def _fallback_forecast(
        self,
        recent_observations: list[LoadObservation],
        forecast_start_utc: datetime,
        num_slots: int,
    ) -> list[LoadForecastSlot]:
        """
        Fallback when the model is not yet trained: rolling mean per
        (hour-of-day, day-of-week).  Uses available recent history.
        """
        bucket_sums: dict[tuple[int, int], float] = {}
        bucket_counts: dict[tuple[int, int], int] = {}
        global_sum = 0.0
        global_count = 0

        for obs in recent_observations:
            hour = obs.interval_start_utc.hour
            day_of_week = obs.interval_start_utc.weekday()
            bucket_key = (hour, day_of_week)
            bucket_sums[bucket_key] = bucket_sums.get(bucket_key, 0.0) + obs.load_watts
            bucket_counts[bucket_key] = bucket_counts.get(bucket_key, 0) + 1
            global_sum += obs.load_watts
            global_count += 1

        global_mean = (global_sum / global_count) if global_count > 0 else 1000.0

        forecast_slots: list[LoadForecastSlot] = []
        for slot_index in range(num_slots):
            slot_start_utc = forecast_start_utc + timedelta(
                minutes=slot_index * _FORECAST_SLOT_MINUTES
            )
            hour = slot_start_utc.hour
            day_of_week = slot_start_utc.weekday()
            bucket_key = (hour, day_of_week)

            if bucket_key in bucket_sums:
                prediction = bucket_sums[bucket_key] / bucket_counts[bucket_key]
            else:
                hour_sum = sum(bucket_sums[key] for key in bucket_sums if key[0] == hour)
                hour_count = sum(bucket_counts[key] for key in bucket_counts if key[0] == hour)
                prediction = (hour_sum / hour_count) if hour_count > 0 else global_mean

            prediction = float(
                np.clip(prediction, _WATTS_MIN_PLAUSIBILITY, _WATTS_MAX_PLAUSIBILITY)
            )
            forecast_slots.append(
                LoadForecastSlot(interval_start_utc=slot_start_utc, load_watts=prediction)
            )

        return forecast_slots

    # ------------------------------------------------------------------
    # Model persistence
    # ------------------------------------------------------------------

    def save_model(self, model_file_path: str) -> bool:
        """Persist the fitted model and a companion metadata JSON.

        The metadata file (<model>.meta.json) stores _training_has_weather so
        load_model() can reconstruct the correct covariate shape after restart
        — avoids a feature-dimension mismatch at predict time.

        Returns True on success, False on failure.
        """
        if not self._is_trained or self._model is None:
            _LOGGER.debug("DartsLightGBMLoadForecaster: no trained model to save")
            return False

        try:
            import json as _json
            os.makedirs(os.path.dirname(model_file_path), exist_ok=True)
            self._model.save(model_file_path)
            meta_path = model_file_path + ".meta.json"
            with open(meta_path, "w", encoding="utf-8") as meta_file:
                _json.dump(
                    {
                        "training_has_weather": self._training_has_weather,
                        "training_observation_count": self._training_observation_count,
                        "lags": self._lags,
                        "output_chunk_length": self._output_chunk_length,
                    },
                    meta_file,
                )
            _LOGGER.info("DartsLightGBMLoadForecaster: model saved to %s", model_file_path)
            return True
        except Exception as save_error:
            _LOGGER.warning("DartsLightGBMLoadForecaster: save failed: %s", save_error)
            return False

    def load_model(self, model_file_path: str) -> bool:
        """Restore a previously saved model.

        Also loads the companion metadata file (if present) to restore
        _training_has_weather — critical for matching the covariate shape
        the trained model expects at predict time.

        Returns True on success.  If the file is not found or corrupt,
        returns False and the model stays untrained — the fallback will be
        used until the next nightly retrain.
        """
        if not os.path.exists(model_file_path):
            _LOGGER.debug(
                "DartsLightGBMLoadForecaster: no saved model at %s", model_file_path
            )
            return False

        try:
            import json as _json
            from darts.models import LightGBMModel
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                loaded_model = LightGBMModel.load(model_file_path)
            self._model = loaded_model
            self._is_trained = True
            # Restore training metadata for correct covariate dimension at predict time
            meta_path = model_file_path + ".meta.json"
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, encoding="utf-8") as meta_file:
                        meta = _json.load(meta_file)
                    self._training_has_weather = bool(meta.get("training_has_weather", False))
                    self._training_observation_count = int(
                        meta.get("training_observation_count", 0)
                    )
                except Exception as meta_error:
                    _LOGGER.debug(
                        "Load model meta read failed (%s); assuming no weather", meta_error
                    )
                    self._training_has_weather = False
            else:
                # Legacy save — assume no weather to avoid dimension mismatch
                self._training_has_weather = False
            _LOGGER.info(
                "DartsLightGBMLoadForecaster: model restored from %s "
                "(training_has_weather=%s)",
                model_file_path,
                self._training_has_weather,
            )
            return True
        except Exception as load_error:
            _LOGGER.warning(
                "DartsLightGBMLoadForecaster: failed to load model from %s: %s.  "
                "Will retrain on next nightly cycle.",
                model_file_path,
                load_error,
            )
            return False

    # ------------------------------------------------------------------
    # Internal series-building helpers
    # ------------------------------------------------------------------

    def _build_regular_series(
        self, observations: list[LoadObservation]
    ) -> tuple[list[datetime], np.ndarray, np.ndarray]:
        """
        Convert observations to a gapless 30-min grid (tz-naive for Darts).

        Gaps (missing slots) are forward-filled.
        Returns (tz_naive_times, load_values, sample_weights).
        """
        if not observations:
            return [], np.array([]), np.array([])

        observations_sorted = sorted(observations, key=lambda obs: obs.interval_start_utc)

        obs_by_time: dict[datetime, float] = {}
        for obs in observations_sorted:
            obs_by_time[obs.interval_start_utc] = obs.load_watts

        start_utc = observations_sorted[0].interval_start_utc
        end_utc = observations_sorted[-1].interval_start_utc
        total_slots = int((end_utc - start_utc).total_seconds() / 1800) + 1

        times: list[datetime] = []
        load_values: list[float] = []
        last_known_value = observations_sorted[0].load_watts
        now_utc = datetime.now(timezone.utc)
        now_timestamp = now_utc.timestamp()
        recency_cutoff_seconds = _RECENCY_BOOST_DAYS * 86400.0

        for slot_index in range(total_slots):
            slot_utc = start_utc + timedelta(minutes=30 * slot_index)
            # Strip TZ (Darts works tz-naive internally)
            times.append(slot_utc.replace(tzinfo=None))

            if slot_utc in obs_by_time:
                slot_value = obs_by_time[slot_utc]
                last_known_value = slot_value
            else:
                slot_value = last_known_value  # forward-fill

            load_values.append(slot_value)

        # Vectorised weight computation: compute all slot timestamps, compare to
        # recency cutoff in one numpy operation (avoids per-slot Python datetime arithmetic)
        slot_timestamps = np.array([
            (start_utc + timedelta(minutes=30 * slot_index)).timestamp()
            for slot_index in range(total_slots)
        ], dtype=np.float64)
        age_seconds = now_timestamp - slot_timestamps
        weights = np.where(age_seconds <= recency_cutoff_seconds, _RECENCY_BOOST_FACTOR, 1.0)

        return times, np.array(load_values, dtype=np.float64), weights

    @staticmethod
    def _build_calendar_covariate_values(naive_times: list[datetime]) -> np.ndarray:
        """
        Build a (n_times × 7) array of circular calendar features from tz-naive datetimes.

        Columns: sin_hour, cos_hour, sin_dow, cos_dow, sin_month, cos_month, is_weekend

        Vectorised implementation: extracts all datetime components into arrays
        first, then applies numpy trig functions in bulk — ~10× faster than the
        per-element Python loop for >1000 time-points (e.g., 90-day training series).
        """
        if not naive_times:
            return np.empty((0, 7), dtype=np.float64)

        hours = np.array([slot_time.hour for slot_time in naive_times], dtype=np.float64)
        dows = np.array([slot_time.weekday() for slot_time in naive_times], dtype=np.float64)
        months = np.array([slot_time.month for slot_time in naive_times], dtype=np.float64)

        sin_hour = np.sin(2 * math.pi * hours / 24)
        cos_hour = np.cos(2 * math.pi * hours / 24)
        sin_dow = np.sin(2 * math.pi * dows / 7)
        cos_dow = np.cos(2 * math.pi * dows / 7)
        sin_month = np.sin(2 * math.pi * (months - 1) / 12)
        cos_month = np.cos(2 * math.pi * (months - 1) / 12)
        is_weekend = (dows >= 5).astype(np.float64)

        return np.column_stack([sin_hour, cos_hour, sin_dow, cos_dow, sin_month, cos_month, is_weekend])

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def training_observation_count(self) -> int:
        return self._training_observation_count

    @property
    def model_name(self) -> str:
        return "Darts-LightGBM-Direct"


# ---------------------------------------------------------------------------
# Backward-compat alias (coordinator imports HouseLoadForecaster)
# ---------------------------------------------------------------------------

HouseLoadForecaster = DartsLightGBMLoadForecaster


# ---------------------------------------------------------------------------
# Model file path helper
# ---------------------------------------------------------------------------

def model_storage_path(ha_config_dir: str, region: str) -> str:
    """Return the canonical .storage path for the load model for *region*."""
    filename = f"nem_price_forecaster_{region.lower()}{_MODEL_FILE_SUFFIX}"
    return os.path.join(ha_config_dir, ".storage", filename)


# ---------------------------------------------------------------------------
# Weather covariate builder (module-level, shared by train + forecast paths)
# ---------------------------------------------------------------------------

def _build_weather_covariate_values_for_load(
    naive_times: list[datetime],
    weather_map: dict[datetime, Any],
) -> np.ndarray:
    """
    Build a (n_times × 5) array of weather features from a naive-UTC datetime map.

    Columns (matching _WEATHER_COLUMN_NAMES):
      0  temperature_celsius           (°C)
      1  cloud_cover_percent           (0–100)
      2  shortwave_radiation_wm2       (W/m²)
      3  wind_speed_ms                 (m/s)
      4  relative_humidity_percent     (0–100)

    The *weather_map* values may be WeatherSlot objects (with corresponding
    attributes) or plain dicts with keys matching the _WEATHER_COLUMN_NAMES list.
    Missing slots are forward-filled from the last known value.
    """
    if not naive_times or not weather_map:
        return np.zeros((len(naive_times), _N_WEATHER_COLUMNS), dtype=np.float64)

    # Default fallback values (mid-range temperate)
    last_temp      = 20.0
    last_cloud     = 50.0
    last_radiation = 0.0
    last_wind      = 3.0
    last_humidity  = 60.0

    result_rows: list[list[float]] = []

    for slot_time in naive_times:
        slot_weather = weather_map.get(slot_time)
        if slot_weather is not None:
            if isinstance(slot_weather, dict):
                last_temp      = float(slot_weather.get("temperature_2m", last_temp))
                last_cloud     = float(slot_weather.get("cloud_cover", last_cloud))
                last_radiation = float(slot_weather.get("shortwave_radiation", last_radiation))
                last_wind      = float(slot_weather.get("wind_speed_10m", last_wind))
                last_humidity  = float(slot_weather.get("relative_humidity_2m", last_humidity))
            else:
                # WeatherSlot object
                last_temp      = float(slot_weather.temperature_celsius)
                last_cloud     = float(slot_weather.cloud_cover_percent)
                last_radiation = float(slot_weather.shortwave_radiation_wm2)
                last_wind      = float(slot_weather.wind_speed_ms)
                last_humidity  = float(slot_weather.relative_humidity_percent)

        result_rows.append(
            [last_temp, last_cloud, last_radiation, last_wind, last_humidity]
        )

    return np.array(result_rows, dtype=np.float64)


# ---------------------------------------------------------------------------
# Calendar / weather feature helpers (kept for tests & backtest scripts)
# ---------------------------------------------------------------------------

def _calendar_features(interval_utc: datetime) -> list[float]:
    """
    Return a 7-element calendar feature vector for *interval_utc*.

    Features (circular encoding, all in [-1, 1]):
      sin_hour, cos_hour, sin_dow, cos_dow, sin_month, cos_month, is_weekend
    """
    hour = interval_utc.hour
    day_of_week = interval_utc.weekday()
    month = interval_utc.month

    sin_hour = math.sin(2 * math.pi * hour / 24)
    cos_hour = math.cos(2 * math.pi * hour / 24)
    sin_dow = math.sin(2 * math.pi * day_of_week / 7)
    cos_dow = math.cos(2 * math.pi * day_of_week / 7)
    sin_month = math.sin(2 * math.pi * (month - 1) / 12)
    cos_month = math.cos(2 * math.pi * (month - 1) / 12)
    is_weekend = 1.0 if day_of_week >= 5 else 0.0

    return [sin_hour, cos_hour, sin_dow, cos_dow, sin_month, cos_month, is_weekend]


def _degree_features(
    temperature_celsius: float,
    heating_setpoint_celsius: float = 18.0,
    cooling_setpoint_celsius: float = 26.0,
) -> list[float]:
    """
    Return (heating_degree_hours, cooling_degree_hours) for a 30-min slot.

    heating_degree_hours = max(0, setpoint − temperature) × 0.5
    cooling_degree_hours = max(0, temperature − setpoint) × 0.5
    """
    heating_degree_hours = max(0.0, heating_setpoint_celsius - temperature_celsius) * 0.5
    cooling_degree_hours = max(0.0, temperature_celsius - cooling_setpoint_celsius) * 0.5
    return [heating_degree_hours, cooling_degree_hours]


def build_features_with_weather(
    interval_utc: datetime,
    lag_values: list[float],
    temperature_celsius: Optional[float] = None,
    heating_setpoint_celsius: float = 18.0,
    cooling_setpoint_celsius: float = 26.0,
) -> list[float]:
    """
    Combine calendar + (optional) weather + lag features for a single slot.

    Retained for tests.  The Darts model uses explicit covariate series internally.
    temperature_celsius=None means no weather features.
    """
    features = _calendar_features(interval_utc)
    if temperature_celsius is not None:
        features += _degree_features(
            temperature_celsius, heating_setpoint_celsius, cooling_setpoint_celsius
        )
    features += lag_values
    return features
