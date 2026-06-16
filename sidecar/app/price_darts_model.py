"""
Darts LightGBM price forecaster (optional; activated by price_model=darts).

Architecture
-----------
Trains a Darts LightGBM model on historical (actual) NEM wholesale prices using
a SPLIT covariate design:

  PAST covariates (history-only signals, not available in the future):
    - Calendar features (sin/cos hour/dow/month, is_weekend) — at lag positions
      [-48, -96] (same time yesterday, same time two days ago)
    - PD7DAY CONVERGED value — the most-converged (most recent) PD7DAY estimate
      for each past training interval.  See PD7DAY note below.

  FUTURE covariates (known or forecastable for the full horizon):
    - Calendar features (same 7 columns) — always available
    - Open-Meteo weather: temperature_2m, cloud_cover, shortwave_radiation,
      wind_speed_10m, relative_humidity_2m — fetched from the free Open-Meteo
      API (api.open-meteo.com); falls back to calendar-only if unavailable
    - PD7DAY CURRENT RRP — the latest-available PD7DAY forecast for the future
      window.  See PD7DAY note below.

PD7DAY bid-stack note (important)
---------------------------------
PD7DAY is NOT a neutral forecast — it is a converging BID STACK.  Far-out
intervals (6–7 days ahead) have many bids not yet finalised, so the raw RRP
reflects early high/speculative bids.  As the dispatch interval approaches,
more bids are firm and the RRP converges toward the realised spot price.

Consequence: feeding raw PD7DAY as a future covariate introduces a systematic
LEAD-TIME BIAS.  The model would learn different PD7DAY→actual mappings for
day-1 (mostly converged) vs day-7 (mostly speculative).

Our handling:
  1. During TRAINING, for each historical target slot we use only the
     MOST-CONVERGED (i.e. the most recent PD7DAY run before that slot) value.
     This means training always sees well-converged PD7DAY values — not the
     noisy far-out ones.
  2. During PREDICTION, we pass the current PD7DAY file's RRP as-is BUT the
     model has been trained on converged values, so it implicitly learns the
     direction and magnitude of the convergence correction.
  3. Callers should ideally pass only ONE PD7DAY value per target interval
     (the latest/most-converged run); passing multiple runs and averaging
     is also acceptable but unnecessary.

This design is a conservative approximation.  A fuller solution would pass
the lead-time (hours-to-dispatch) as an explicit feature, but that requires
a multi-dimensional target series which LightGBMModel does not support in
Darts v0.29–0.44.

Why use Darts `lags_future_covariates` for weather + calendar?
--------------------------------------------------------------
Darts LightGBMModel supports BOTH lags_past_covariates AND lags_future_covariates
simultaneously.  Using lags_future_covariates (rather than only past covariates)
lets the model see the UPCOMING calendar + weather values directly at each target
slot — not just lagged history.  For weather this is critical: a temperature
forecast for tomorrow directly conditions tomorrow's load-driven price, whereas
the temperature two days ago is only weakly informative.

Persistence
-----------
Model saved via Darts pickle. Path: <data_dir>/price_darts_model.pkl

Notes for the isotonic comparison
----------------------------------
For wholesale price prediction the isotonic calibrator often matches or beats
this Darts model, because it captures the systematic hour-of-day bias directly
(calendar features don't fully capture it).  This model is provided for users
who want to experiment or who have long enough history for the Darts model to
learn the bias itself.
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

_SLOT_MINUTES = 30
_DEFAULT_LAGS = 96                   # 2 days × 48 slots/day
_DEFAULT_OUTPUT_CHUNK_LENGTH = 336   # 7 days single-shot
_PRICE_FLOOR_PER_MWH = -1000.0      # Market floor
_PRICE_CAP_PER_MWH = 16600.0        # MPC (market price cap)
_MIN_TRAINING_SAMPLES = 336         # 1 week minimum
_RECENCY_BOOST_DAYS = 14
_RECENCY_BOOST_FACTOR = 2.0
_MODEL_FILENAME = "price_darts_model.pkl"
_CALENDAR_LAG_POSITIONS = [-48, -96]

# Region(s) that ship a pre-trained Darts price model bundled in the image.
# The bundled model lets a fresh install produce a Darts-backed forecast on day 1
# instead of waiting to self-train once enough live history has accumulated.
# Other regions self-train (or fall back to seasonal-naive) as before.
_BUNDLED_MODEL_REGIONS = {"QLD1"}

# Weather variable column names in the combined covariate array.
# Order must match the column order in _build_weather_covariate_values().
_WEATHER_COLUMNS = [
    "temperature_2m",
    "cloud_cover",
    "shortwave_radiation",
    "wind_speed_10m",
    "relative_humidity_2m",
]
_N_WEATHER_COLUMNS = len(_WEATHER_COLUMNS)

# NOTE: _FUTURE_COVARIATE_LAGS has been removed.
# In Darts 0.40+, lags_future_covariates=None DISABLES future covariates entirely
# (the model errors when future_covariates are passed).  The correct value is an
# explicit list covering -(lags-1) to +output_chunk_length so that the model sees
# the covariate at every position in the forecast window.  This is now computed
# per-instance inside DartsLightGBMPriceForecaster.train() from self._lags and
# self._output_chunk_length.


@dataclass
class PriceObservation:
    """One historical wholesale price data point (30-min, $/MWh)."""
    interval_start_utc: datetime
    rrp_per_mwh: float


class DartsLightGBMPriceForecaster:
    """
    Optional Darts LightGBM wholesale price forecaster.

    Activated when price_model=darts in the sidecar config.
    Falls back to isotonic calibration if not trained or Darts unavailable.

    Covariate design (past + future split):
    ----------------------------------------
    PAST covariates: carry history-only signals the model sees at lag positions.
      - Calendar features (sin/cos hour/dow/month, is_weekend) at lags [-48, -96]
      - PD7DAY converged value at those same lags

    FUTURE covariates: carry signals available for the full forecast horizon.
      - Calendar features (same 7 columns)
      - Open-Meteo weather (temperature, cloud cover, GHI radiation, wind, humidity)
        — 5 columns, fetched from the free forecast API; omitted if unavailable
      - PD7DAY current RRP for the forecast window (the most-converged estimate
        currently available; see PD7DAY bias note in module docstring)

    Darts LightGBMModel supports lags_past_covariates + lags_future_covariates
    simultaneously.  Using lags_future_covariates lets the model see upcoming
    calendar and weather values at each target slot rather than only lagged history.
    """

    def __init__(
        self,
        lags: int = _DEFAULT_LAGS,
        output_chunk_length: int = _DEFAULT_OUTPUT_CHUNK_LENGTH,
        forecast_horizon_hours: int = 168,
        use_weather_covariates: bool = True,
    ) -> None:
        self._lags = lags
        self._output_chunk_length = output_chunk_length
        self._forecast_horizon_hours = forecast_horizon_hours
        self._use_weather_covariates = use_weather_covariates

        self._model: Any = None
        self._is_trained: bool = False
        self._training_observation_count: int = 0
        # Track whether the trained model expects weather columns, so
        # predict() can match the training feature set.
        self._training_has_weather: bool = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        observations: list[PriceObservation],
        pd7day_history: Optional[list[PriceObservation]] = None,
        weather_history: Optional[dict[datetime, Any]] = None,
    ) -> None:
        """
        Fit the Darts LightGBM model on *observations* (actual NEM prices).

        observations:
            list of PriceObservation covering the training window (actual RRP).
            May include AEMO NEMWeb historical data to supplement online-accumulated
            calibration observations — just merge the two lists before calling.

        pd7day_history:
            Optional PD7DAY forecasts aligned to the training window.
            These are treated as past-covariate signals (history-only).
            Pass the MOST-CONVERGED (most recent run) PD7DAY value per target
            slot to avoid the lead-time convergence bias — see module docstring.

        weather_history:
            Optional dict mapping naive-UTC datetime → WeatherSlot (or dict)
            for the training window.  Fetched from Open-Meteo archive API.
            Used as FUTURE covariates because weather is forecastable forward.
            Omit or pass None to train without weather (calendar-only).
        """
        try:
            from darts import TimeSeries
            from darts.models import LightGBMModel
            import pandas as pd
            import warnings
            warnings.filterwarnings("ignore")
        except ImportError as import_error:
            _LOGGER.error(
                "DartsLightGBMPriceForecaster: darts not available: %s", import_error
            )
            return

        if len(observations) < _MIN_TRAINING_SAMPLES:
            _LOGGER.warning(
                "DartsLightGBMPriceForecaster: only %d observations (need %d); skipping",
                len(observations),
                _MIN_TRAINING_SAMPLES,
            )
            return

        regular_times, price_values, sample_weights = self._build_regular_series(observations)

        if len(regular_times) < _MIN_TRAINING_SAMPLES:
            _LOGGER.warning(
                "DartsLightGBMPriceForecaster: after regularisation only %d slots; skipping",
                len(regular_times),
            )
            return

        import pandas as pd
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            times_index = pd.DatetimeIndex(regular_times)
            price_ts = TimeSeries.from_times_and_values(times_index, price_values)
            weight_ts = TimeSeries.from_times_and_values(times_index, sample_weights)

            # ---- Past covariates ----
            # Calendar features (lagged) + converged PD7DAY (lagged)
            calendar_values = _build_calendar_covariate_values(regular_times)

            if pd7day_history and len(pd7day_history) >= len(regular_times) // 2:
                # Use the most-converged PD7DAY value per slot to avoid lead-time bias
                # (see module docstring for full explanation)
                pd7day_past_values = self._align_pd7day_to_times(regular_times, pd7day_history)
                past_cov_values = np.column_stack([calendar_values, pd7day_past_values])
            else:
                past_cov_values = calendar_values

            past_covariate_ts = TimeSeries.from_times_and_values(
                times_index,
                past_cov_values,
                columns=[f"past_{col_idx}" for col_idx in range(past_cov_values.shape[1])],
            )

            # ---- Future covariates ----
            # Calendar + weather (if available) for the full training + future window.
            # The future covariate series must extend output_chunk_length steps BEYOND
            # the end of the training series (Darts requirement for direct multi-output).
            future_tail_times = [
                regular_times[-1] + timedelta(minutes=30 * (slot_idx + 1))
                for slot_idx in range(self._output_chunk_length + 2)
            ]
            all_future_times = regular_times + future_tail_times
            future_calendar_values = _build_calendar_covariate_values(all_future_times)

            has_weather = (
                self._use_weather_covariates
                and weather_history is not None
                and len(weather_history) > 0
            )

            if has_weather:
                weather_values = _build_weather_covariate_values(
                    all_future_times, weather_history  # type: ignore[arg-type]
                )
                future_cov_values = np.column_stack([future_calendar_values, weather_values])
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
                        price_ts,
                        past_covariates=past_covariate_ts,
                        future_covariates=future_covariate_ts,
                        sample_weight=weight_ts,
                    )
            except Exception as fit_error:
                _LOGGER.error("DartsLightGBMPriceForecaster fit failed: %s", fit_error)
                return

        self._model = model
        self._is_trained = True
        self._training_observation_count = len(observations)
        self._training_has_weather = bool(has_weather)
        _LOGGER.info(
            "DartsLightGBMPriceForecaster trained: %d observations, lags=%d, "
            "output_chunk=%d, weather_covariates=%s",
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
        recent_observations: list[PriceObservation],
        forecast_start_utc: datetime,
        num_slots: int,
        pd7day_forecast: Optional[list[PriceObservation]] = None,
        weather_forecast: Optional[dict[datetime, Any]] = None,
    ) -> list[float]:
        """
        Return a list of num_slots predicted RRP values ($/MWh).

        recent_observations: recent actual prices for the input lag window
        forecast_start_utc:  UTC datetime of the first forecast slot
        num_slots:           how many 30-min slots to predict
        pd7day_forecast:     current PD7DAY file's RRP as future covariate
        weather_forecast:    naive-UTC → WeatherSlot dict for the forecast window

        Falls back to an empty list on failure (caller should use isotonic).
        """
        if not self._is_trained or self._model is None:
            return []

        try:
            return self._darts_predict(
                recent_observations,
                forecast_start_utc,
                num_slots,
                pd7day_forecast,
                weather_forecast,
            )
        except Exception as predict_error:
            _LOGGER.warning("DartsLightGBMPriceForecaster predict failed: %s", predict_error)
            return []

    def _darts_predict(
        self,
        recent_observations: list[PriceObservation],
        forecast_start_utc: datetime,
        num_slots: int,
        pd7day_forecast: Optional[list[PriceObservation]],
        weather_forecast: Optional[dict[datetime, Any]],
    ) -> list[float]:
        import pandas as pd
        from darts import TimeSeries
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            # ---- Input series (tz-naive for Darts) ----
            if len(recent_observations) < self._lags:
                pad_count = self._lags - len(recent_observations)
                pad_value = recent_observations[0].rrp_per_mwh if recent_observations else 80.0
                pad_start = (
                    recent_observations[0].interval_start_utc - timedelta(minutes=30 * pad_count)
                    if recent_observations
                    else forecast_start_utc - timedelta(minutes=30 * self._lags)
                )
                padded_obs = [
                    PriceObservation(
                        interval_start_utc=pad_start + timedelta(minutes=30 * pad_idx),
                        rrp_per_mwh=pad_value,
                    )
                    for pad_idx in range(pad_count)
                ] + recent_observations
            else:
                padded_obs = recent_observations

            tail_times = [obs.interval_start_utc.replace(tzinfo=None) for obs in padded_obs]
            tail_values = np.array([obs.rrp_per_mwh for obs in padded_obs], dtype=np.float64)
            input_ts = TimeSeries.from_times_and_values(pd.DatetimeIndex(tail_times), tail_values)

            # ---- Autoregression tail length ----
            # When num_slots > output_chunk_length, Darts auto-regresses and needs
            # BOTH past and future covariates to extend (num_slots - output_chunk_length)
            # steps beyond the input window (it consumes future values of the past
            # covariates at each recursion step).  A few extra slots of slack guard
            # against off-by-one boundary requirements inside Darts.
            autoregression_tail = max(0, num_slots - self._output_chunk_length)

            # ---- Past covariates ----
            # Calendar (lagged) + converged PD7DAY (lagged).  These MUST span the
            # input window AND the autoregression tail — calendar is deterministic
            # forward, and PD7DAY future values come from pd7day_forecast.
            past_cov_times = tail_times + [
                tail_times[-1] + timedelta(minutes=30 * (slot_idx + 1))
                for slot_idx in range(autoregression_tail + 2)
            ]
            calendar_input = _build_calendar_covariate_values(past_cov_times)

            if pd7day_forecast and len(pd7day_forecast) > 0:
                pd7day_input_values = self._align_pd7day_to_times(past_cov_times, pd7day_forecast)
                past_input_cov = np.column_stack([calendar_input, pd7day_input_values])
            else:
                past_input_cov = calendar_input

            past_covariate_ts = TimeSeries.from_times_and_values(
                pd.DatetimeIndex(past_cov_times),
                past_input_cov,
            )

            # ---- Future covariates ----
            # Calendar + weather for the full input + output window.
            # The future covariate series must cover the input window plus
            # output_chunk_length AND the autoregression tail (Darts reads future
            # covariate values at every recursion step beyond output_chunk_length).
            forecast_extra_times = [
                tail_times[-1] + timedelta(minutes=30 * (slot_idx + 1))
                for slot_idx in range(self._output_chunk_length + autoregression_tail + 2)
            ]
            all_future_cov_times = tail_times + forecast_extra_times
            future_calendar_values = _build_calendar_covariate_values(all_future_cov_times)

            # Use weather if the model was trained with it AND forecast weather is available
            use_weather_now = (
                self._training_has_weather
                and weather_forecast is not None
                and len(weather_forecast) > 0
            )

            if use_weather_now:
                future_weather_values = _build_weather_covariate_values(
                    all_future_cov_times, weather_forecast  # type: ignore[arg-type]
                )
                future_cov_values = np.column_stack(
                    [future_calendar_values, future_weather_values]
                )
            elif self._training_has_weather:
                # Model was trained with weather but none available now — pad with zeros
                _LOGGER.debug(
                    "DartsLightGBMPriceForecaster: model trained with weather but "
                    "no weather forecast available; padding weather columns with zeros"
                )
                zero_weather = np.zeros((len(all_future_cov_times), _N_WEATHER_COLUMNS))
                future_cov_values = np.column_stack([future_calendar_values, zero_weather])
            else:
                future_cov_values = future_calendar_values

            future_covariate_ts = TimeSeries.from_times_and_values(
                pd.DatetimeIndex(all_future_cov_times),
                future_cov_values,
            )

            prediction = self._model.predict(
                max(num_slots, 1),
                series=input_ts,
                past_covariates=past_covariate_ts,
                future_covariates=future_covariate_ts,
            )

        raw_values = prediction.to_series().values
        if raw_values.ndim > 1:
            raw_values = raw_values.flatten()

        clamped = [
            float(np.clip(raw_values[slot_idx], _PRICE_FLOOR_PER_MWH, _PRICE_CAP_PER_MWH))
            for slot_idx in range(min(num_slots, len(raw_values)))
        ]
        return clamped

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, model_dir: str) -> bool:
        """Persist the fitted model and a companion metadata JSON.

        The metadata file (<model>.meta.json) stores _training_has_weather so
        load_model() can reconstruct the correct covariate shape without needing
        to retrain — avoids a feature-dimension mismatch after a container restart.
        """
        if not self._is_trained or self._model is None:
            return False
        try:
            import json as _json
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, _MODEL_FILENAME)
            self._model.save(model_path)
            meta_path = model_path + ".meta.json"
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
            _LOGGER.info("Price Darts model saved to %s", model_path)
            return True
        except Exception as save_error:
            _LOGGER.warning("Price Darts model save failed: %s", save_error)
            return False

    def load_model(self, model_dir: str) -> bool:
        """Restore a previously saved Darts price model.

        Also loads the companion metadata file (if present) so that
        _training_has_weather is correctly restored — without it a
        weather-trained model would receive a mismatched covariate
        matrix at predict time (calendar-only vs calendar+weather).
        """
        model_path = os.path.join(model_dir, _MODEL_FILENAME)
        meta_path = model_path + ".meta.json"
        if not os.path.exists(model_path):
            return False
        try:
            from darts.models import LightGBMModel
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                self._model = LightGBMModel.load(model_path)
            self._is_trained = True
            # Restore training metadata so predict() uses the correct feature shape
            if os.path.exists(meta_path):
                try:
                    import json as _json
                    with open(meta_path, encoding="utf-8") as meta_file:
                        meta = _json.load(meta_file)
                    self._training_has_weather = bool(meta.get("training_has_weather", False))
                    self._training_observation_count = int(
                        meta.get("training_observation_count", 0)
                    )
                except Exception as meta_error:
                    _LOGGER.debug(
                        "Price Darts model meta load failed (%s); assuming no weather", meta_error
                    )
                    self._training_has_weather = False
            else:
                # Legacy save (no metadata) — assume no weather to be safe
                # (will trigger zero-padding on predict, not a dimension mismatch)
                self._training_has_weather = False
            _LOGGER.info(
                "Price Darts model loaded from %s (training_has_weather=%s)",
                model_path,
                self._training_has_weather,
            )
            return True
        except Exception as load_error:
            _LOGGER.warning("Price Darts model load failed: %s", load_error)
            return False

    @staticmethod
    def bundled_model_dir(region: str) -> Optional[str]:
        """Directory of the bundled pre-trained model for *region*, or None.

        The bundled model lives next to the app code (app/models/<region_lower>/)
        so it ships inside the Docker image (the Dockerfiles COPY app/ wholesale)
        and is read-only — distinct from the writable <data_dir> the user's own
        self-trained model accumulates in.  Returns the path only if the region
        ships a bundled model AND the model file actually exists on disk.
        """
        if region.upper() not in _BUNDLED_MODEL_REGIONS:
            return None
        candidate_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "models",
            region.lower(),
        )
        if os.path.exists(os.path.join(candidate_dir, _MODEL_FILENAME)):
            return candidate_dir
        return None

    def load_model_with_bundled_fallback(self, model_dir: str, region: str) -> bool:
        """Load the user's self-trained model, falling back to the bundled one.

        Precedence (mirrors the calibrator-seed precedence in observation_store):
          1. The user's own self-trained model in *model_dir* (the writable
             <data_dir>), if present — this is the model the sidecar saved after
             accumulating live history, and always wins.
          2. Otherwise, the region's bundled pre-trained model (read-only, shipped
             in the image) so a fresh install is Darts-backed on day 1.
          3. Otherwise, leave the model untrained (caller self-trains or falls back
             to seasonal-naive) — unchanged behaviour for regions with no bundle.

        Returns True if a model was loaded from either source.
        """
        if self.load_model(model_dir):
            return True
        bundled_dir = self.bundled_model_dir(region)
        if bundled_dir is not None:
            if self.load_model(bundled_dir):
                _LOGGER.info(
                    "Price Darts model: loaded BUNDLED pre-trained model for %s "
                    "from %s (fresh install — day-1 ready)",
                    region.upper(),
                    bundled_dir,
                )
                return True
        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_regular_series(
        self,
        observations: list[PriceObservation],
    ) -> tuple[list[datetime], np.ndarray, np.ndarray]:
        if not observations:
            return [], np.array([]), np.array([])

        sorted_obs = sorted(observations, key=lambda observation: observation.interval_start_utc)
        # Filter NaN / infinite RRP values before building the lookup to prevent
        # forward-fill propagating a bad value through the entire training series.
        obs_by_time: dict[datetime, float] = {
            observation.interval_start_utc: observation.rrp_per_mwh
            for observation in sorted_obs
            if math.isfinite(observation.rrp_per_mwh)
        }

        if not obs_by_time:
            return [], np.array([]), np.array([])

        start_utc = sorted_obs[0].interval_start_utc
        end_utc = sorted_obs[-1].interval_start_utc
        total_slots = int((end_utc - start_utc).total_seconds() / 1800) + 1

        times: list[datetime] = []
        price_values: list[float] = []
        # Seed last_known with the first finite value rather than sorted_obs[0]
        # which may be NaN and would then forward-fill bad data.
        last_known = next(iter(obs_by_time.values()))
        now_utc = datetime.now(timezone.utc)
        recency_cutoff_seconds = _RECENCY_BOOST_DAYS * 86400.0

        for slot_idx in range(total_slots):
            slot_utc = start_utc + timedelta(minutes=30 * slot_idx)
            times.append(slot_utc.replace(tzinfo=None))
            if slot_utc in obs_by_time:
                last_known = obs_by_time[slot_utc]
            price_values.append(last_known)

        slot_timestamps = np.array([
            (start_utc + timedelta(minutes=30 * slot_idx)).timestamp()
            for slot_idx in range(total_slots)
        ], dtype=np.float64)
        age_seconds = now_utc.timestamp() - slot_timestamps
        weights = np.where(age_seconds <= recency_cutoff_seconds, _RECENCY_BOOST_FACTOR, 1.0)

        return times, np.array(price_values, dtype=np.float64), weights

    @staticmethod
    def _align_pd7day_to_times(
        naive_times: list[datetime],
        pd7day_observations: list[PriceObservation],
    ) -> np.ndarray:
        """Map PD7DAY RRP values to naive_times via nearest-or-last-known lookup."""
        pd7day_by_naive: dict[datetime, float] = {}
        for obs in pd7day_observations:
            naive_key = obs.interval_start_utc.replace(tzinfo=None)
            pd7day_by_naive[naive_key] = obs.rrp_per_mwh

        result: list[float] = []
        last_known = 80.0
        for slot_time in naive_times:
            if slot_time in pd7day_by_naive:
                last_known = pd7day_by_naive[slot_time]
            result.append(last_known)

        return np.array(result, dtype=np.float64).reshape(-1, 1)

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def training_observation_count(self) -> int:
        return self._training_observation_count


# ---------------------------------------------------------------------------
# Calendar feature helper (shared with load forecaster pattern)
# ---------------------------------------------------------------------------

def _build_calendar_covariate_values(naive_times: list[datetime]) -> np.ndarray:
    """Build sin/cos calendar features — same 7 columns as load forecaster."""
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

    return np.column_stack(
        [sin_hour, cos_hour, sin_dow, cos_dow, sin_month, cos_month, is_weekend]
    )


def _build_weather_covariate_values(
    naive_times: list[datetime],
    weather_map: dict[datetime, Any],
) -> np.ndarray:
    """
    Build a (n_times × 5) array of weather features from a naive-UTC datetime map.

    Columns (in order, matching _WEATHER_COLUMNS):
      0  temperature_celsius     (°C)
      1  cloud_cover_percent     (0–100)
      2  shortwave_radiation_wm2 (W/m²)
      3  wind_speed_ms           (m/s)
      4  relative_humidity_percent (0–100)

    The *weather_map* keys should be naive (tz-stripped) datetimes at hourly or
    30-min resolution.  Missing slots are forward-filled.  The map values may be
    WeatherSlot objects (with the above attributes) or plain dicts with the same
    keys as _WEATHER_COLUMNS.
    """
    if not naive_times or not weather_map:
        return np.zeros((len(naive_times), _N_WEATHER_COLUMNS), dtype=np.float64)

    # Default fallback values (temperate mid-range)
    last_temp     = 20.0
    last_cloud    = 50.0
    last_radiation = 0.0
    last_wind     = 3.0
    last_humidity = 60.0

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
