"""
Load engine — wraps DartsLightGBMLoadForecaster for background execution.

Architecture (cache-first):
  1. Nightly (cron-scheduled): run_train_cycle() — retrains the Darts model on
     all available load observations (CPU-heavy, runs in executor thread).
  2. Periodic (every predict_interval_seconds): run_predict_cycle() — calls
     model.forecast() on the recent observation window and writes to cache.
  3. Change detection: skip predict if observations and model haven't changed
     since last predict.

POST /load_observation feeds new observations; the engine accumulates them
in the ObservationStore.  No on-request training.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any

from config import SidecarConfig
from forecast_cache import ForecastCache, LoadForecastSlot
from load_forecaster import DartsLightGBMLoadForecaster, LoadObservation, LoadForecastSlot as DartsLoadSlot
from observation_store import ObservationStore
from forecast_resampler import resample_load_slots
from weather_client import build_weather_client_for_region, OpenMeteoClient

_LOGGER = logging.getLogger(__name__)

_MODEL_FILENAME = "load_darts_model.pkl"
_MIN_OBSERVATIONS_FOR_PREDICT = 96   # 2 days × 48 = enough for the lag window
_MIN_OBSERVATIONS_FOR_TRAIN = 288    # ~3 days minimum


class LoadEngine:
    """
    Manages the Darts LightGBM house-load forecaster lifecycle.

    Training is CPU-heavy (~60s on first run).  All calls run in an executor
    thread via the scheduler — never on the async event loop.
    """

    def __init__(
        self,
        config: SidecarConfig,
        cache: ForecastCache,
        store: ObservationStore,
    ) -> None:
        self._config = config
        self._cache = cache
        self._store = store

        self._forecaster = DartsLightGBMLoadForecaster(
            forecast_horizon_hours=config.forecast_horizon_hours,
            use_weather_covariates=config.weather_enabled,
        )

        # Open-Meteo weather client — built once, reused per cycle
        self._weather_client: OpenMeteoClient = build_weather_client_for_region(
            region=config.region,
            latitude_override=config.latitude,
            longitude_override=config.longitude,
        )

        # Change detection: track observation count at last predict
        self._last_predict_observation_count: int = 0

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def restore_model_from_disk(self) -> None:
        """Restore a previously saved Darts model from disk on startup."""
        model_path = self._model_path()
        if os.path.exists(model_path):
            restored = self._forecaster.load_model(model_path)
            if restored:
                _LOGGER.info("LoadEngine: model restored from %s", model_path)
            else:
                _LOGGER.info("LoadEngine: model restore failed; will train on next cycle")
        else:
            _LOGGER.info("LoadEngine: no saved model found; will train when enough data arrives")

    # ------------------------------------------------------------------
    # Training cycle (nightly, called by APScheduler)
    # ------------------------------------------------------------------

    def run_train_cycle(self) -> bool:
        """
        Retrain the Darts model on all available load observations.

        When weather covariates are enabled (config.weather_enabled=True), fetches
        historical weather from the Open-Meteo archive API to cover the full
        training window.  On weather fetch failure the model trains without weather
        (falls back gracefully with a WARNING log).

        Returns True on success, False if skipped or failed.
        This is called from a background thread — safe to block.
        """
        all_observations = self._store.get_load_observations()

        if len(all_observations) < _MIN_OBSERVATIONS_FOR_TRAIN:
            _LOGGER.info(
                "LoadEngine: only %d observations (need %d for training); skipping train",
                len(all_observations),
                _MIN_OBSERVATIONS_FOR_TRAIN,
            )
            return False

        _LOGGER.info(
            "LoadEngine: starting train on %d observations", len(all_observations)
        )

        # Fetch historical weather for the training window (Open-Meteo archive API)
        weather_history_map = None
        if self._config.weather_enabled:
            weather_history_map = self._fetch_weather_for_training(all_observations)

        self._forecaster.train(all_observations, weather_history=weather_history_map)

        if self._forecaster.is_trained:
            self._save_model()
            _LOGGER.info("LoadEngine: training complete; model saved")
            return True

        _LOGGER.warning("LoadEngine: training failed")
        return False

    def _fetch_weather_for_training(
        self,
        observations: list[LoadObservation],
    ) -> "dict | None":
        """
        Fetch Open-Meteo archive weather for the training window and return a
        naive-UTC datetime → WeatherSlot map at 30-min resolution.

        Returns None on error so the caller can train without weather.
        """
        from datetime import date as date_type
        if not observations:
            return None

        sorted_observations = sorted(observations, key=lambda obs: obs.interval_start_utc)
        start_utc = sorted_observations[0].interval_start_utc
        end_utc = sorted_observations[-1].interval_start_utc

        # Open-Meteo archive lag is ~5 days; clip to what's available
        archive_cutoff = datetime.now(timezone.utc) - timedelta(days=6)
        effective_end = min(end_utc, archive_cutoff)
        if start_utc >= effective_end:
            # Recent data only — use forecast API instead
            forecast_slots = self._weather_client.fetch_forecast(hours_ahead=48)
            return self._weather_client.slots_to_30min_map(forecast_slots)

        archive_slots = self._weather_client.fetch_archive(
            start_date=start_utc.date(),
            end_date=effective_end.date(),
        )

        # For the recent gap (last 6 days not in archive), top up from forecast
        if end_utc > archive_cutoff:
            forecast_slots = self._weather_client.fetch_forecast(hours_ahead=48)
        else:
            forecast_slots = []

        all_slots = archive_slots + forecast_slots
        if not all_slots:
            return None

        # Build naive-UTC keyed map at 30-min resolution (forward-fill from hourly)
        weather_map = self._weather_client.slots_to_30min_map(all_slots)
        # Convert tz-aware keys → naive for Darts
        naive_weather_map = {
            slot_time.replace(tzinfo=None): slot_weather
            for slot_time, slot_weather in weather_map.items()
        }
        _LOGGER.info(
            "LoadEngine: fetched %d weather slots for training window",
            len(naive_weather_map),
        )
        return naive_weather_map

    # ------------------------------------------------------------------
    # Predict cycle (periodic, called by APScheduler)
    # ------------------------------------------------------------------

    def run_predict_cycle(self, force: bool = False) -> bool:
        """
        Run load forecast prediction and write to cache.

        Change detection: skip if observation count hasn't changed since last
        predict AND the model is already trained AND force=False.

        Returns True if cache was updated.
        """
        all_observations = self._store.get_load_observations()
        current_count = len(all_observations)

        if current_count < _MIN_OBSERVATIONS_FOR_PREDICT:
            _LOGGER.debug(
                "LoadEngine: only %d observations (need %d for predict); skipping",
                current_count,
                _MIN_OBSERVATIONS_FOR_PREDICT,
            )
            return False

        # Change detection
        if (
            not force
            and current_count == self._last_predict_observation_count
            and self._cache.has_load_forecast
        ):
            _LOGGER.debug(
                "LoadEngine: no new observations (%d); skipping predict", current_count
            )
            return False

        # Use recent 200 observations for the predict input window (last ~4 days)
        recent_observations = all_observations[-200:]

        # Fetch forecast weather for the prediction window (Open-Meteo forecast API)
        weather_forecast_map = None
        if self._config.weather_enabled:
            forecast_weather_slots = self._weather_client.fetch_forecast(
                hours_ahead=self._config.forecast_horizon_hours + 48
            )
            if forecast_weather_slots:
                raw_weather_map = self._weather_client.slots_to_30min_map(forecast_weather_slots)
                # Convert tz-aware keys → naive for Darts
                weather_forecast_map = {
                    slot_time.replace(tzinfo=None): slot_weather
                    for slot_time, slot_weather in raw_weather_map.items()
                }

        raw_darts_slots: list[DartsLoadSlot] = self._forecaster.forecast(
            recent_observations,
            weather_forecast=weather_forecast_map,
        )

        now_utc = datetime.now(timezone.utc)

        # Convert DartsLoadSlot → sidecar LoadForecastSlot
        cache_load_slots: list[LoadForecastSlot] = [
            LoadForecastSlot(
                interval_start=darts_slot.interval_start_utc,
                load_watts=darts_slot.load_watts,
            )
            for darts_slot in raw_darts_slots
        ]

        # Resample to configured period + horizon
        raw_dicts = [
            {"datetime": slot.interval_start.isoformat(), "load_power": slot.load_watts}
            for slot in cache_load_slots
        ]
        resampled = resample_load_slots(
            raw_dicts,
            target_period_minutes=self._config.forecast_period_minutes,
            horizon_hours=self._config.forecast_horizon_hours,
            now_utc=now_utc,
        )

        self._cache.update_load_forecast(
            load_slots=cache_load_slots,
            resampled_slots=resampled,
            model_trained=self._forecaster.is_trained,
            training_observations=self._forecaster.training_observation_count,
        )

        self._last_predict_observation_count = current_count

        _LOGGER.info(
            "LoadEngine: forecast updated — %d raw slots, %d resampled (%dmin), "
            "trained=%s obs=%d",
            len(cache_load_slots),
            len(resampled),
            self._config.forecast_period_minutes,
            self._forecaster.is_trained,
            current_count,
        )
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _model_path(self) -> str:
        return os.path.join(self._config.data_dir, _MODEL_FILENAME)

    def _save_model(self) -> None:
        model_path = self._model_path()
        os.makedirs(self._config.data_dir, exist_ok=True)
        if self._forecaster.is_trained:
            self._forecaster.save_model(model_path)
