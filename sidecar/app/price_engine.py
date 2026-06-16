"""
Price engine — runs PD7DAY fetch + calibration + tariff computation
in a background thread and writes results to ForecastCache.

This is the sidecar's equivalent of the HA coordinator's price logic,
but decoupled from the HA event loop.

Architecture (cache-first, no on-request compute):
  1. Background scheduler calls run_predict_cycle() every predict_interval seconds.
  2. run_predict_cycle() checks the PD7DAY index for a new file (cheap HEAD-like poll).
     If the file hasn't changed AND we have a cached forecast, we skip recompute.
  3. When a new PD7DAY file is detected, we download + parse + calibrate + resample
     and write the result to ForecastCache.
  4. HTTP endpoints read from ForecastCache (sub-ms, no lock contention).

Price model selection:
  price_model=darts_naive_blend (DEFAULT): 50/50 mixture of Darts LightGBM and
    seasonal-naive (same-hour, same-day-of-week, 7 days ago).
    Blending reduces worst-case regime-transition error while matching
    seasonal-naive's average accuracy and beating the Darts model alone.
    Requires no rolling prediction cache — the naive component works immediately.
    The blend weight is tunable via config.naive_blend_weight (default 0.5).
    MAX REACH: unbounded in principle (Darts recurses; naive uses 7-days-ago
    observations); validated accuracy is ≤7 days.
    UPGRADE PATH: an adaptive rolling-regime-selector can be enabled once
    ≥14 days of live predictions are banked (see TODO below).
  price_model=isotonic: uses IsotonicCalibratorPerHour (pure numpy).
    Best when AEMO PD7DAY predispatch is accurate; poor when PD7DAY diverges
    from settlement.
    MAX REACH: bounded by PD7DAY (~7d).  Beyond chain_seam_days, the engine
    chains to darts_naive_blend automatically.
  price_model=darts: uses DartsLightGBMPriceForecaster (requires darts + lightgbm).
    - Requires at least config.darts_price_min_training_days days of accumulated
      actual wholesale prices (fed via the /calibration/import POST endpoint).
    - Falls back to isotonic with a WARNING log when insufficient history exists.
    - When enough history is available: trains the Darts model once per predict
      cycle, produces a full horizon forecast array, and maps each PD7DAY slot
      to the nearest Darts-predicted RRP rather than isotonic calibration.
    - The isotonic calibrator is always kept warm (import/export observations
      are stored regardless of model selection) so switching back to isotonic
      requires no data migration.
    - MAX REACH: unbounded in principle (recursive prediction); validated ≤7d.
  price_model=hybrid: combines isotonic and Darts by lead time.
    - Slots with lead_time <= config.hybrid_crossover_hours → isotonic path.
    - Slots with lead_time > config.hybrid_crossover_hours → Darts path.
    - Optional smooth linear blend across a ±12h band around the crossover
      (config.hybrid_blend_enabled=True, the default) to eliminate the step
      discontinuity at the boundary.  Within the blend band each slot's price
      is a weighted mixture: isotonic * (1 − alpha) + darts * alpha, where
      alpha ramps linearly from 0 to 1 as lead_time goes from crossover−12h
      to crossover+12h.
    - Graceful degradation:
        * If Darts is unavailable (insufficient history or training error):
          isotonic is used for the entire horizon and a WARNING is logged.
        * If isotonic has insufficient observations: the Darts path is used
          for the entire horizon and a WARNING is logged.
    - Both calibrators remain warm regardless of model selection so switching
      models never requires data migration.
    - Default crossover: 120h (tuned for QLD1).  Re-tune on your own data
      before relying on this for other markets.
    - MAX REACH: bounded by PD7DAY for isotonic leg (~7d).  Beyond chain_seam_days,
      the engine chains to darts_naive_blend automatically.

Horizon-aware model chaining:
  When config.forecast_horizon_days > config.chain_seam_days the engine chains:
    - Slots with lead_time_days <= (seam - blend_half): served by primary model,
      tagged model_segment="primary"
    - Slots with lead_time_days >= (seam + blend_half): served by darts_naive_blend,
      tagged model_segment="chain:darts_naive_blend"
    - Slots within the ±blend_window: linearly interpolated, tagged "primary" or
      "chain:darts_naive_blend" based on which side of the seam they fall.
  Chain slots are generated from the darts_naive_blend model run on the same
  predict cycle.  HONESTY: beyond-seam accuracy is UNVALIDATED convenience —
  validated accuracy is ≤7 days.

  Model reach table:
    isotonic        — PD7DAY-bounded, reach ≈ chain_seam_days (default 7d)
    hybrid          — PD7DAY-bounded (isotonic leg), reach ≈ chain_seam_days
    darts           — unbounded in principle; validated ≤7d
    darts_naive_blend — unbounded in principle; validated ≤7d; also the chain continuation

TODO (adaptive rolling-regime-selector):
  Once ≥14 days of live champion predictions are banked in ObservationStore,
  add a rolling-14d MAE comparison (champion vs naive on trailing realised data)
  and switch darts_naive_blend to use naive_blend_weight=1.0 when naive beats
  champion by ≥0.5 c/kWh on the rolling window.  Guards required:
    1. Minimum ≥7 days of recent predictions before enabling
    2. Hysteresis: naive must beat champion by ≥0.5 c/kWh to switch
    3. Coverage check: if trailing realised-price coverage < 80%, keep champion
    4. One-way hysteresis: champion must lead ≥3 days before switching back
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import numpy as np

from config import SidecarConfig
from forecast_cache import ForecastCache, PriceForecastSlot
from isotonic_calibrator import IsotonicCalibratorPerHour, CalibrationObservation
from monotone_gbm_calibrator import MonotoneGBMCalibrator
from observation_store import ObservationStore
from pd7day_client import Pd7DayClient, PriceSlot, Pd7DayForecast, NEM_TIMEZONE
from price_darts_model import (
    DartsLightGBMPriceForecaster,
    PriceObservation,
    _PRICE_FLOOR_PER_MWH,
    _PRICE_CAP_PER_MWH,
)
from tariff import TariffCalculator, parse_tou_bands_from_config
from forecast_resampler import resample_price_slots
from weather_client import build_weather_client_for_region, OpenMeteoClient
from aemo_historical_client import AemoHistoricalClient, PriceHistorySlot

_LOGGER = logging.getLogger(__name__)

_CALIBRATION_RECENCY_HALF_LIFE_DAYS = 30.0
# 30-minute slots per day — used when converting min_training_days → observation count
_SLOTS_PER_DAY = 48

# ---------------------------------------------------------------------------
# Model reach declarations
# ---------------------------------------------------------------------------
# Each entry names the max reach in days for that model's *validated* range.
# Beyond this the chain resolver activates (when horizon > chain_seam_days).
# "unbounded" models have recursive / free prediction but validated accuracy is ≤7d.
# These are informational constants used in logging — the chain seam is controlled
# by config.chain_seam_days, not by these values.
_MODEL_VALIDATED_DAYS: dict[str, float] = {
    "isotonic": 7.0,          # PD7DAY-bounded; validated ≤7d
    "hybrid": 7.0,            # PD7DAY-bounded for isotonic leg; validated ≤7d
    "darts": 7.0,             # unbounded in principle; validated ≤7d
    "darts_naive_blend": 7.0, # unbounded in principle; validated ≤7d (also chain cont.)
}


class PriceEngine:
    """
    Drives PD7DAY polling, isotonic/Darts calibration, and tariff computation.

    Runs in a background executor thread (called by the scheduler).
    Writes results to ForecastCache.  Thread-safe.
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

        self._pd7day_client = Pd7DayClient()

        # Calibrator backend selected by config.calibrator ("isotonic" default,
        # "monotone_gbm" opt-in). Both expose the same API; the GBM keeps a warm
        # isotonic backstop internally so selecting it can never regress below
        # the shipped isotonic baseline (see _build_calibrator).
        self._import_calibrator = self._build_calibrator()
        self._export_calibrator = self._build_calibrator()
        _LOGGER.info(
            "PriceEngine: calibrator backend = %s (isotonic is the default; "
            "monotone_gbm is opt-in with a never-lose isotonic fallback)",
            config.calibrator,
        )

        self._tariff_calculator = self._build_tariff_calculator()

        # Optional Darts price model (used for "darts", "hybrid", and "darts_naive_blend" modes)
        self._darts_price_model: Any = None
        if config.price_model in ("darts", "hybrid", "darts_naive_blend"):
            self._initialise_darts_price_model()

        # Open-Meteo weather client (used if weather_enabled=True)
        self._weather_client: OpenMeteoClient = build_weather_client_for_region(
            region=config.region,
            latitude_override=config.latitude,
            longitude_override=config.longitude,
        )

        # AEMO historical price client for seeding the Darts training set
        self._aemo_client: AemoHistoricalClient = AemoHistoricalClient(
            data_dir=config.data_dir,
            region=config.region,
        )

        # Change detection: skip predict when PD7DAY hasn't changed
        self._last_pd7day_filename: Optional[str] = None

        # Darts price model refit cache — avoid re-training on every 5-min predict cycle.
        # The Darts model is expensive (~1-3 seconds per fit on 17,520 training observations).
        # We only retrain when the training data or PD7DAY file has changed since the last fit.
        # Keyed on (import_observation_count, pd7day_filename); cached forecast dict is
        # reused across cycles where neither changes.
        self._darts_last_fit_observation_count: int = -1
        self._darts_last_fit_pd7day_filename: Optional[str] = None
        self._darts_cached_forecast: dict[datetime, float] = {}

        # Prevent concurrent predict cycles (APScheduler + HTTP trigger can race)
        self._predict_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def restore_calibration_from_store(self) -> None:
        """
        Load persisted calibration observations into the in-memory calibrators.
        Called once on startup before the first predict cycle.
        """
        import_obs = self._store.get_import_observations()
        export_obs = self._store.get_export_observations()

        if import_obs:
            self._import_calibrator.add_observations_bulk(import_obs)
        if export_obs:
            self._export_calibrator.add_observations_bulk(export_obs)

        _LOGGER.info(
            "Price calibration restored: import=%d, export=%d observations",
            len(import_obs),
            len(export_obs),
        )

        # Restore Darts model if configured.  Precedence: the user's own
        # self-trained model in data_dir wins; otherwise fall back to the
        # region's bundled pre-trained model (read-only, shipped in the image)
        # so a fresh install is Darts-backed on day 1.  Regions with no bundle
        # leave the model untrained and self-train on the first predict cycle.
        if self._darts_price_model is not None:
            loaded = self._darts_price_model.load_model_with_bundled_fallback(
                self._config.data_dir, self._config.region
            )
            if loaded:
                _LOGGER.info(
                    "PriceEngine: Darts price model restored "
                    "(is_trained=%s, training_observations=%d)",
                    self._darts_price_model.is_trained,
                    self._darts_price_model.training_observation_count,
                )
            else:
                _LOGGER.info(
                    "PriceEngine: no pre-trained Darts price model available for %s "
                    "(will self-train once enough history accumulates, or use "
                    "seasonal-naive in the blend)",
                    self._config.region,
                )

    # ------------------------------------------------------------------
    # Main compute cycle (called by scheduler)
    # ------------------------------------------------------------------

    def run_predict_cycle(self, force: bool = False) -> bool:
        """
        Fetch PD7DAY, compute calibrated price forecast, write to cache.

        Returns True if forecast was updated, False if skipped (no new data).

        Change detection: if the PD7DAY filename hasn't changed since last run
        AND cache is non-empty AND force=False, skip recompute.

        Thread safety: a non-blocking lock prevents concurrent cycles (APScheduler
        job and HTTP trigger can both call this simultaneously).  A concurrent
        caller gets False (skipped) immediately rather than blocking or racing.
        """
        if not self._predict_lock.acquire(blocking=False):
            _LOGGER.debug("PriceEngine: predict cycle already running; skipping concurrent call")
            return False
        try:
            return self._run_predict_cycle_locked(force=force)
        finally:
            self._predict_lock.release()

    def _run_predict_cycle_locked(self, force: bool = False) -> bool:
        """Inner predict logic — called only while holding _predict_lock."""
        try:
            forecast = self._pd7day_client.fetch_latest(self._config.region)
        except Exception as fetch_error:
            _LOGGER.warning("PriceEngine: PD7DAY fetch failed: %s", fetch_error)
            return False

        if forecast is None:
            _LOGGER.warning("PriceEngine: PD7DAY returned None")
            return False

        # Change detection using the cached filename inside Pd7DayClient
        current_filename = self._pd7day_client._cached_filename
        if (
            not force
            and current_filename is not None
            and current_filename == self._last_pd7day_filename
            and self._cache.has_price_forecast
        ):
            _LOGGER.debug(
                "PriceEngine: PD7DAY unchanged (%s); skipping recompute", current_filename
            )
            return False

        self._last_pd7day_filename = current_filename

        now_utc = datetime.now(timezone.utc)
        price_slots = self._compute_price_slots(forecast, now_utc)

        # Chain extension: when horizon > seam, extend with darts_naive_blend
        chain_seam_hours = self._config.chain_seam_days * 24.0
        chain_needed = (
            self._config.forecast_horizon_hours > chain_seam_hours
            # darts_naive_blend is already its own "unbounded" model — it never
            # needs to chain to itself, but we still run the logic so that slots
            # carry the correct model_segment when horizon ≤ seam.
            # For isotonic/hybrid the PD7DAY slots naturally stop at ~7d, so the
            # chain fills the gap between the last PD7DAY slot and horizon.
        )
        if chain_needed:
            chain_extension_slots = self._build_chain_extension_slots(
                primary_slots=price_slots,
                forecast=forecast,
                now_utc=now_utc,
            )
            price_slots = self._merge_primary_and_chain(
                primary_slots=price_slots,
                chain_slots=chain_extension_slots,
                now_utc=now_utc,
            )
            _LOGGER.info(
                "PriceEngine: chain extension — seam=%.1fd, horizon=%.1fd, "
                "chain_slots=%d (UNVALIDATED beyond %.1fd)",
                self._config.chain_seam_days,
                self._config.forecast_horizon_days,
                len(chain_extension_slots),
                self._config.chain_seam_days,
            )

        # Resample to configured period + horizon
        raw_dicts = [slot.as_dict() for slot in price_slots]
        resampled = resample_price_slots(
            raw_dicts,
            target_period_minutes=self._config.forecast_period_minutes,
            horizon_hours=self._config.forecast_horizon_hours,
            now_utc=now_utc,
        )

        self._cache.update_price_forecast(
            price_slots=price_slots,
            resampled_slots=resampled,
            pd7day_run_datetime=forecast.run_datetime_utc,
            calibration_active=self._import_calibrator.is_calibrated,
            calibration_observations=self._import_calibrator.observation_count,
        )

        chain_count = sum(
            1 for slot in price_slots if slot.model_segment != "primary"
        )
        _LOGGER.info(
            "PriceEngine: forecast updated — %d raw slots (%d primary, %d chain), "
            "%d resampled (%s @ %dmin)",
            len(price_slots),
            len(price_slots) - chain_count,
            chain_count,
            len(resampled),
            current_filename or "?",
            self._config.forecast_period_minutes,
        )
        return True

    # ------------------------------------------------------------------
    # Calibration feed (called via /calibration API endpoint)
    # ------------------------------------------------------------------

    def add_import_calibration_observation(
        self,
        predicted_rrp_per_mwh: float,
        actual_rrp_per_mwh: float,
        hour_of_day: int,
        observed_at: datetime,
    ) -> None:
        """Feed a new (predicted, actual import) observation to the calibrator."""
        self._import_calibrator.add_observation(
            predicted_rrp_per_mwh, actual_rrp_per_mwh, hour_of_day, observed_at
        )
        self._store.add_import_observation(
            predicted_rrp_per_mwh, actual_rrp_per_mwh, hour_of_day, observed_at
        )

    def add_export_calibration_observation(
        self,
        predicted_rrp_per_mwh: float,
        actual_rrp_per_mwh: float,
        hour_of_day: int,
        observed_at: datetime,
    ) -> None:
        """Feed a new (predicted, actual export) observation to the calibrator."""
        self._export_calibrator.add_observation(
            predicted_rrp_per_mwh, actual_rrp_per_mwh, hour_of_day, observed_at
        )
        self._store.add_export_observation(
            predicted_rrp_per_mwh, actual_rrp_per_mwh, hour_of_day, observed_at
        )

    # ------------------------------------------------------------------
    # Internal: price slot computation
    # ------------------------------------------------------------------

    def _compute_price_slots(
        self,
        forecast: Pd7DayForecast,
        now_utc: datetime,
    ) -> list[PriceForecastSlot]:
        """Apply calibration + tariff to every future PD7DAY slot."""
        # For the Darts, hybrid, and darts_naive_blend paths: train once per cycle
        # and build a time-indexed lookup so each slot can be mapped to its predicted
        # RRP without re-training.
        darts_rrp_by_slot_start: dict[datetime, float] = {}
        using_darts = False
        if (
            self._config.price_model in ("darts", "hybrid", "darts_naive_blend")
            and self._darts_price_model is not None
        ):
            darts_rrp_by_slot_start, using_darts = self._build_darts_cycle_forecast(
                forecast, now_utc
            )

        # For the darts_naive_blend path: build a naive RRP lookup (same-hour,
        # same-day-of-week, 7 days ago from the import calibration observations).
        naive_rrp_by_slot_start: dict[datetime, float] = {}
        if self._config.price_model == "darts_naive_blend":
            naive_rrp_by_slot_start = self._build_seasonal_naive_lookup(
                forecast=forecast, now_utc=now_utc
            )

        # For the hybrid path: also check whether isotonic is calibrated.
        isotonic_available = self._import_calibrator.is_calibrated

        # Emit degradation warnings once per cycle (not per slot).
        if self._config.price_model == "hybrid":
            if not using_darts:
                _LOGGER.warning(
                    "PriceEngine [hybrid]: Darts unavailable; using isotonic for entire horizon."
                )
            elif not isotonic_available:
                _LOGGER.warning(
                    "PriceEngine [hybrid]: Isotonic calibrator has insufficient observations; "
                    "using Darts for entire horizon."
                )

        # Half-width of the smooth blend band (hours on each side of the crossover).
        # Sourced from config so it is tunable via SIDECAR_HYBRID_BLEND_WINDOW_HOURS.
        _HYBRID_BLEND_HALF_WIDTH_HOURS = self._config.hybrid_blend_window_hours

        price_slots: list[PriceForecastSlot] = []
        skipped_count = 0

        for price_slot in forecast.slots:
            if price_slot.interval_start_utc < now_utc - timedelta(minutes=30):
                continue

            hour_of_day = price_slot.interval_start_nem.hour

            # ---- Determine which model(s) serve this slot ----
            if self._config.price_model == "darts_naive_blend":
                calibrated_import_kwh, calibrated_export_kwh = (
                    self._compute_naive_blend_slot_price(
                        price_slot=price_slot,
                        now_utc=now_utc,
                        darts_rrp_by_slot_start=darts_rrp_by_slot_start,
                        using_darts=using_darts,
                        naive_rrp_by_slot_start=naive_rrp_by_slot_start,
                        blend_weight=self._config.naive_blend_weight,
                    )
                )
                if calibrated_import_kwh is None:
                    skipped_count += 1
                    continue
            elif self._config.price_model == "hybrid":
                # Use the PD7DAY forecast run time as the lead-time reference so
                # that classification is stable and consistent with the slot
                # timestamps (which are relative to the forecast run, not wall-clock).
                # In production the difference is < 5 min (within one dispatch slot).
                hybrid_lead_reference = forecast.run_datetime_utc
                lead_time_hours = (
                    price_slot.interval_start_utc - hybrid_lead_reference
                ).total_seconds() / 3600.0
                calibrated_import_kwh, calibrated_export_kwh = (
                    self._compute_hybrid_slot_price(
                        price_slot=price_slot,
                        hour_of_day=hour_of_day,
                        lead_time_hours=lead_time_hours,
                        now_utc=now_utc,
                        darts_rrp_by_slot_start=darts_rrp_by_slot_start,
                        using_darts=using_darts,
                        isotonic_available=isotonic_available,
                        blend_half_width_hours=_HYBRID_BLEND_HALF_WIDTH_HOURS,
                    )
                )
                if calibrated_import_kwh is None:
                    skipped_count += 1
                    continue
            elif using_darts and price_slot.interval_start_utc in darts_rrp_by_slot_start:
                # Pure Darts path
                darts_rrp = darts_rrp_by_slot_start[price_slot.interval_start_utc]
                darts_kwh = float(np.clip(darts_rrp, _PRICE_FLOOR_PER_MWH, _PRICE_CAP_PER_MWH)) / 1000.0
                calibrated_import_kwh = float(np.clip(
                    darts_kwh,
                    -0.10,
                    self._config.plausibility_cap_dollars_per_kwh,
                ))
                calibrated_export_kwh = calibrated_import_kwh
            else:
                # Pure isotonic path
                try:
                    calibrated_import_kwh = self._import_calibrator.calibrate(
                        price_slot.rrp_per_mwh, hour_of_day, reference_time=now_utc
                    )
                    calibrated_export_kwh = self._export_calibrator.calibrate(
                        price_slot.rrp_per_mwh, hour_of_day, reference_time=now_utc
                    )
                except Exception as calibration_error:
                    _LOGGER.debug("Calibration error for slot %s: %s",
                                  price_slot.interval_start_utc.isoformat(), calibration_error)
                    skipped_count += 1
                    continue

            try:
                network_rate = self._tariff_calculator.network_rate_for_interval(
                    price_slot.interval_start_nem
                )
                import_price = self._tariff_calculator.compute_import_price(
                    calibrated_import_kwh, price_slot.interval_start_nem
                )
                export_price = self._tariff_calculator.compute_export_price(
                    calibrated_export_kwh, price_slot.interval_start_nem
                )
            except Exception as tariff_error:
                _LOGGER.debug("Tariff error for slot %s: %s",
                              price_slot.interval_start_utc.isoformat(), tariff_error)
                skipped_count += 1
                continue

            price_slots.append(
                PriceForecastSlot(
                    interval_start=price_slot.interval_start_utc,
                    raw_rrp_per_mwh=price_slot.rrp_per_mwh,
                    calibrated_wholesale_kwh=calibrated_import_kwh,
                    import_price_kwh=import_price,
                    export_price_kwh=export_price,
                    network_tou_rate_kwh=network_rate,
                    model_segment="primary",
                )
            )

        if skipped_count > 0:
            _LOGGER.warning(
                "PriceEngine: skipped %d slots due to errors (of %d total)",
                skipped_count,
                len(forecast.slots),
            )

        return price_slots

    def _build_seasonal_naive_lookup(
        self,
        forecast: "Pd7DayForecast",
        now_utc: datetime,
    ) -> dict[datetime, float]:
        """
        Build a slot_utc → actual_rrp_per_mwh lookup for seasonal-naive prediction.

        Seasonal-naive: price at the same (hour-of-day, day-of-week) from exactly
        7 days prior.  We use the import calibration observations as the price series
        (actual_rrp_per_mwh).  For each forecast slot we search for an observation
        within ±15 minutes of (slot_utc − 7 days).

        Falls back gracefully: if no observation is found in the ±15-min window the
        slot is omitted from the lookup; the blend code falls back to Darts-only for
        that slot (or the Darts raw ÷10 fallback if Darts is also unavailable).
        """
        one_week = timedelta(days=7)
        fifteen_minutes = timedelta(minutes=15)

        # Index observations by rounded-to-30min naive UTC for fast lookup.
        import_observations = self._store.get_import_observations()
        obs_by_rounded_utc: dict[datetime, float] = {}
        for calibration_obs in import_observations:
            observed_naive = calibration_obs.observed_at.replace(second=0, microsecond=0)
            # Round down to nearest 30-min boundary
            rounded_minute = (observed_naive.minute // 30) * 30
            rounded_utc = observed_naive.replace(minute=rounded_minute, tzinfo=None)
            if calibration_obs.observed_at.tzinfo is not None:
                rounded_utc = observed_naive.astimezone(timezone.utc).replace(
                    second=0, microsecond=0
                )
                rounded_minute = (rounded_utc.minute // 30) * 30
                rounded_utc = rounded_utc.replace(minute=rounded_minute, second=0, microsecond=0)
            obs_by_rounded_utc[rounded_utc] = calibration_obs.actual_rrp_per_mwh

        naive_lookup: dict[datetime, float] = {}
        for price_slot in forecast.slots:
            if price_slot.interval_start_utc < now_utc - timedelta(minutes=30):
                continue

            target_utc = price_slot.interval_start_utc - one_week
            # Search ±15 min window
            target_rounded = target_utc.replace(second=0, microsecond=0)
            rounded_minute = (target_rounded.minute // 30) * 30
            target_rounded = target_rounded.replace(minute=rounded_minute, second=0, microsecond=0)

            if target_rounded in obs_by_rounded_utc:
                naive_lookup[price_slot.interval_start_utc] = obs_by_rounded_utc[target_rounded]
            else:
                # Try adjacent 30-min bucket
                alt_target = target_rounded + timedelta(minutes=30)
                if alt_target in obs_by_rounded_utc:
                    naive_lookup[price_slot.interval_start_utc] = obs_by_rounded_utc[alt_target]

        _LOGGER.debug(
            "PriceEngine: seasonal-naive lookup built — %d/%d slots covered",
            len(naive_lookup),
            sum(
                1 for slot in forecast.slots
                if slot.interval_start_utc >= now_utc - timedelta(minutes=30)
            ),
        )
        return naive_lookup

    def _compute_naive_blend_slot_price(
        self,
        price_slot: "PriceSlot",
        now_utc: datetime,
        darts_rrp_by_slot_start: dict[datetime, float],
        using_darts: bool,
        naive_rrp_by_slot_start: dict[datetime, float],
        blend_weight: float,
    ) -> tuple[Optional[float], Optional[float]]:
        """
        Compute calibrated import/export price for the darts_naive_blend model.

        Returns (calibrated_import_kwh, calibrated_export_kwh), or (None, None) if
        neither Darts nor naive has coverage for this slot.

        Blend logic:
          - darts_price * (1 − blend_weight) + naive_price * blend_weight
          - If only one component is available, that component is used at full weight.
          - If neither is available, falls back to raw PD7DAY ÷ 10 as a last resort
            (same raw-fallback used by other models during warm-up).
          - blend_weight=0.5 = equal-weight blend (evidence-based default).
        """
        # --- Darts component ---
        darts_rrp: Optional[float] = None
        if using_darts and price_slot.interval_start_utc in darts_rrp_by_slot_start:
            darts_rrp = darts_rrp_by_slot_start[price_slot.interval_start_utc]

        # --- Naive component ---
        naive_rrp: Optional[float] = None
        if price_slot.interval_start_utc in naive_rrp_by_slot_start:
            naive_rrp = naive_rrp_by_slot_start[price_slot.interval_start_utc]

        # --- Blend ---
        if darts_rrp is not None and naive_rrp is not None:
            blended_rrp = darts_rrp * (1.0 - blend_weight) + naive_rrp * blend_weight
        elif darts_rrp is not None:
            blended_rrp = darts_rrp
        elif naive_rrp is not None:
            blended_rrp = naive_rrp
        else:
            # Last resort: raw PD7DAY ÷ 10 (same as warm-up fallback in other modes)
            blended_rrp = price_slot.rrp_per_mwh / 10.0

        blended_kwh = float(
            np.clip(blended_rrp, _PRICE_FLOOR_PER_MWH, _PRICE_CAP_PER_MWH)
        ) / 1000.0
        clamped_kwh = float(
            np.clip(blended_kwh, -0.10, self._config.plausibility_cap_dollars_per_kwh)
        )
        # Import and export are the same in the Darts model (FiT-trained);
        # the tariff layer applies the GST/network delta separately.
        return clamped_kwh, clamped_kwh

    def _compute_hybrid_slot_price(
        self,
        price_slot: "PriceSlot",
        hour_of_day: int,
        lead_time_hours: float,
        now_utc: datetime,
        darts_rrp_by_slot_start: dict[datetime, float],
        using_darts: bool,
        isotonic_available: bool,
        blend_half_width_hours: float,
    ) -> tuple[Optional[float], Optional[float]]:
        """
        Compute calibrated import/export price for a single slot under the hybrid model.

        Returns (calibrated_import_kwh, calibrated_export_kwh), or (None, None) if the
        slot cannot be priced (to be skipped by the caller).

        Routing logic:
          - If Darts is unavailable → isotonic everywhere (degradation mode).
          - If isotonic is uncalibrated → Darts everywhere (degradation mode).
          - Otherwise:
              lead <= crossover − blend_half_width  → pure isotonic weight=1
              lead >= crossover + blend_half_width  → pure Darts weight=1
              otherwise                             → linear blend (if blend enabled),
                                                      or hard switch at crossover (if blend off)
        """
        crossover = self._config.hybrid_crossover_hours
        blend_enabled = self._config.hybrid_blend_enabled

        # ---- Degradation: Darts unavailable → use isotonic everywhere ----
        if not using_darts:
            try:
                import_price = self._import_calibrator.calibrate(
                    price_slot.rrp_per_mwh, hour_of_day, reference_time=now_utc
                )
                export_price = self._export_calibrator.calibrate(
                    price_slot.rrp_per_mwh, hour_of_day, reference_time=now_utc
                )
                return import_price, export_price
            except Exception as isotonic_degradation_error:
                _LOGGER.debug(
                    "Hybrid degradation: isotonic calibration error for slot %s: %s",
                    price_slot.interval_start_utc.isoformat(),
                    isotonic_degradation_error,
                )
                return None, None

        # ---- Degradation: isotonic uncalibrated → use Darts everywhere ----
        if not isotonic_available:
            if price_slot.interval_start_utc not in darts_rrp_by_slot_start:
                return None, None
            darts_rrp = darts_rrp_by_slot_start[price_slot.interval_start_utc]
            darts_kwh = float(np.clip(darts_rrp, _PRICE_FLOOR_PER_MWH, _PRICE_CAP_PER_MWH)) / 1000.0
            clamped = float(np.clip(darts_kwh, -0.10, self._config.plausibility_cap_dollars_per_kwh))
            return clamped, clamped

        # ---- Compute both component prices for potential blending ----
        # Isotonic price
        try:
            isotonic_import_kwh = self._import_calibrator.calibrate(
                price_slot.rrp_per_mwh, hour_of_day, reference_time=now_utc
            )
            isotonic_export_kwh = self._export_calibrator.calibrate(
                price_slot.rrp_per_mwh, hour_of_day, reference_time=now_utc
            )
        except Exception as isotonic_error:
            _LOGGER.debug(
                "Hybrid: isotonic calibration error for slot %s: %s",
                price_slot.interval_start_utc.isoformat(),
                isotonic_error,
            )
            isotonic_import_kwh = None
            isotonic_export_kwh = None

        # Darts price
        darts_import_kwh: Optional[float] = None
        if price_slot.interval_start_utc in darts_rrp_by_slot_start:
            darts_rrp = darts_rrp_by_slot_start[price_slot.interval_start_utc]
            darts_raw_kwh = float(np.clip(darts_rrp, _PRICE_FLOOR_PER_MWH, _PRICE_CAP_PER_MWH)) / 1000.0
            darts_import_kwh = float(np.clip(
                darts_raw_kwh, -0.10, self._config.plausibility_cap_dollars_per_kwh
            ))

        # ---- Determine blend weight for Darts (alpha) ----
        if blend_enabled:
            low_boundary = crossover - blend_half_width_hours
            high_boundary = crossover + blend_half_width_hours
            if lead_time_hours <= low_boundary:
                darts_weight = 0.0          # pure isotonic
            elif lead_time_hours >= high_boundary:
                darts_weight = 1.0          # pure Darts
            else:
                # Linear interpolation across the blend band
                darts_weight = (lead_time_hours - low_boundary) / (2.0 * blend_half_width_hours)
        else:
            # Hard switch at the crossover (no blend)
            darts_weight = 0.0 if lead_time_hours <= crossover else 1.0

        # ---- Mix prices ----
        isotonic_weight = 1.0 - darts_weight

        if darts_weight == 0.0 or darts_import_kwh is None:
            # Pure isotonic region (or Darts lookup miss)
            if isotonic_import_kwh is None:
                return None, None
            return isotonic_import_kwh, isotonic_export_kwh

        if isotonic_weight == 0.0 or isotonic_import_kwh is None:
            # Pure Darts region (or isotonic failed)
            return darts_import_kwh, darts_import_kwh

        # Blend region — both components available
        blended_import = (
            isotonic_weight * isotonic_import_kwh + darts_weight * darts_import_kwh
        )
        blended_export = (
            isotonic_weight * isotonic_export_kwh + darts_weight * darts_import_kwh
        )
        return blended_import, blended_export

    def _build_darts_cycle_forecast(
        self,
        forecast: Pd7DayForecast,
        now_utc: datetime,
    ) -> tuple[dict[datetime, float], bool]:
        """
        Train (or re-use cached) the Darts price model and produce a per-slot RRP
        lookup for the current predict cycle.

        Returns (slot_utc → predicted_rrp_per_mwh dict, success_flag).

        Refit caching: the Darts LightGBM model is expensive to fit (~1-3 seconds
        per cycle on a year of training data).  We skip the re-fit entirely when:
          - The import-observation count has not changed since the last fit, AND
          - The PD7DAY file has not changed since the last fit.
        In that case the cached forecast dict is returned directly, producing the
        same result at effectively zero cost.  A new PD7DAY file or new calibration
        observations invalidate the cache and trigger a full retrain + re-predict.

        Falls back to isotonic (returns empty dict, False) when:
          - There are fewer than darts_price_min_training_days × 48 actual-price
            observations in the calibration store (insufficient history).
          - The Darts train or predict raises any exception.

        The minimum-observations threshold is intentionally higher than the
        model's own _MIN_TRAINING_SAMPLES (7 days) to ensure the LightGBM has
        seen at least two weeks of weekday + weekend patterns before we trust it.
        """
        min_required_observations = (
            self._config.darts_price_min_training_days * _SLOTS_PER_DAY
        )
        import_observations = self._store.get_import_observations()
        actual_observation_count = len(import_observations)

        if actual_observation_count < min_required_observations:
            _LOGGER.warning(
                "PriceEngine: Darts price model selected but only %d actual-price "
                "observations available (need %d = %d days × %d slots/day). "
                "Falling back to isotonic calibration. This warning will clear once "
                "enough (predicted, actual) pairs have been POST-ed to "
                "/calibration/import — see Recipe D in the README.",
                actual_observation_count,
                min_required_observations,
                self._config.darts_price_min_training_days,
                _SLOTS_PER_DAY,
            )
            return {}, False

        current_pd7day_filename = self._pd7day_client._cached_filename

        # Day-1 path: a pre-trained model was loaded (bundled image model or a
        # user model persisted from a previous run) but we have NOT yet fit in
        # this process session (_darts_last_fit_observation_count == -1).  Predict
        # directly with the loaded model instead of triggering an expensive
        # train-from-scratch on the first cycle.  This keeps a fresh install
        # Darts-backed immediately and offline-safe (no AEMO/Open-Meteo fetch
        # required to produce the first forecast).  A subsequent retrain (when the
        # observation count later changes) transparently replaces it with a model
        # fit on the user's own accumulated history.
        no_fit_yet_this_session = self._darts_last_fit_observation_count == -1
        if (
            no_fit_yet_this_session
            and self._darts_price_model is not None
            and self._darts_price_model.is_trained
        ):
            try:
                bundled_forecast = self._predict_with_loaded_darts(
                    import_observations, forecast, now_utc
                )
            except Exception as bundled_predict_error:
                _LOGGER.warning(
                    "PriceEngine: prediction with pre-trained Darts model failed "
                    "(%s); will attempt a fresh train+predict instead.",
                    bundled_predict_error,
                )
                bundled_forecast = {}
            if bundled_forecast:
                self._darts_cached_forecast = bundled_forecast
                self._darts_last_fit_pd7day_filename = current_pd7day_filename
                # Leave _darts_last_fit_observation_count == -1 so that the next
                # cycle with a CHANGED observation count triggers a real retrain
                # on the user's own data; until then this loaded-model forecast is
                # reused via the cache-hit branch above.
                _LOGGER.info(
                    "PriceEngine: produced day-1 forecast from pre-trained Darts "
                    "model (%d slot predictions, no train required)",
                    len(bundled_forecast),
                )
                return bundled_forecast, True

        # Refit-cache check: skip expensive re-train when nothing has changed
        observation_count_unchanged = (
            actual_observation_count == self._darts_last_fit_observation_count
        )
        pd7day_unchanged = (
            current_pd7day_filename is not None
            and current_pd7day_filename == self._darts_last_fit_pd7day_filename
        )
        if (
            observation_count_unchanged
            and pd7day_unchanged
            and self._darts_cached_forecast
            and self._darts_price_model is not None
            and self._darts_price_model.is_trained
        ):
            _LOGGER.debug(
                "PriceEngine: Darts model cache hit "
                "(obs_count=%d, pd7day=%s unchanged) — skipping re-fit",
                actual_observation_count,
                current_pd7day_filename,
            )
            return self._darts_cached_forecast, True

        try:
            result_forecast, success = self._train_and_predict_darts(
                import_observations, forecast, now_utc
            )
        except Exception as darts_error:
            _LOGGER.warning(
                "PriceEngine: Darts price model train/predict failed (%s); "
                "falling back to isotonic calibration for this cycle.",
                darts_error,
            )
            return {}, False

        # Update cache state on successful train+predict
        if success:
            self._darts_last_fit_observation_count = actual_observation_count
            self._darts_last_fit_pd7day_filename = current_pd7day_filename
            self._darts_cached_forecast = result_forecast

        return result_forecast, success

    def _train_and_predict_darts(
        self,
        import_observations: list[CalibrationObservation],
        forecast: Pd7DayForecast,
        now_utc: datetime,
    ) -> tuple[dict[datetime, float], bool]:
        """
        Merge AEMO historical prices + online-calibration observations, fetch
        weather, train the Darts model, run a full-horizon forecast, and return
        a per-slot RRP lookup.

        Training data sources (merged in priority order, newest wins on conflict):
          1. AEMO NEMWeb archive (DISPATCHIS daily ZIPs, cached on disk):
             Provides years of historical 30-min RRP data so the model sees
             full seasonal and structural variation from the first train cycle.
             Controlled by config.aemo_history_days (default 365, 0 = disabled).
          2. Online-calibrated observations (from /calibration/import POSTs):
             Realised prices fed back from the HA integration (Amber or similar).
             These are the most recent and are always given recency-boost weight.

        PD7DAY handling (past covariate, convergence-bias-safe):
          The current PD7DAY file's RRP is passed as a past-covariate signal only.
          The model is trained on ACTUAL prices, not PD7DAY, so PD7DAY acts as a
          feature-enrichment covariate rather than the training target.  See the
          module docstring in price_darts_model.py for the full PD7DAY bias note.

        Weather covariates (future, from Open-Meteo):
          Temperature, cloud cover, GHI radiation, wind, and humidity are fetched
          for the training window (archive API) and forecast window (forecast API).
          These are FUTURE covariates — they are known/forecastable for the full
          horizon and directly condition price spikes (heatwave → peakers, etc.).
        """
        # ---- Step 1: Merge AEMO historical + online observations ----
        aemo_price_observations: list[PriceObservation] = []
        if self._config.aemo_history_days > 0:
            try:
                historical_slots = self._aemo_client.fetch_price_history(
                    days_back=self._config.aemo_history_days
                )
                aemo_price_observations = [
                    PriceObservation(
                        interval_start_utc=hist_slot.interval_start_utc,
                        rrp_per_mwh=hist_slot.rrp_per_mwh,
                    )
                    for hist_slot in historical_slots
                ]
                _LOGGER.info(
                    "PriceEngine: loaded %d AEMO historical slots (%d days back)",
                    len(aemo_price_observations),
                    self._config.aemo_history_days,
                )
            except Exception as aemo_error:
                _LOGGER.warning(
                    "PriceEngine: AEMO historical fetch failed (%s); "
                    "training on online-calibration observations only",
                    aemo_error,
                )

        # Online-calibrated actual prices
        online_price_observations = [
            PriceObservation(
                interval_start_utc=calibration_obs.observed_at,
                rrp_per_mwh=calibration_obs.actual_rrp_per_mwh,
            )
            for calibration_obs in import_observations
        ]

        # Merge: AEMO historical as base, online observations override (most recent wins)
        # Build a time-keyed dict so later entries overwrite earlier ones.
        merged_by_time: dict[datetime, float] = {
            price_obs.interval_start_utc: price_obs.rrp_per_mwh
            for price_obs in aemo_price_observations
        }
        for price_obs in online_price_observations:
            merged_by_time[price_obs.interval_start_utc] = price_obs.rrp_per_mwh

        all_training_observations = [
            PriceObservation(interval_start_utc=slot_time, rrp_per_mwh=rrp_value)
            for slot_time, rrp_value in sorted(merged_by_time.items())
        ]

        _LOGGER.info(
            "PriceEngine: Darts training set = %d slots "
            "(%d AEMO historical + %d online, %d unique after merge)",
            len(all_training_observations),
            len(aemo_price_observations),
            len(online_price_observations),
            len(merged_by_time),
        )

        # ---- Step 2: PD7DAY as past-covariate signal ----
        # Pass the CURRENT (most-converged) PD7DAY estimate for each slot.
        # This avoids the lead-time convergence bias — see price_darts_model.py docstring.
        pd7day_as_price_observations = [
            PriceObservation(
                interval_start_utc=price_slot.interval_start_utc,
                rrp_per_mwh=price_slot.rrp_per_mwh,
            )
            for price_slot in forecast.slots
        ]

        # ---- Step 3: Weather covariates (future, from Open-Meteo) ----
        weather_history_map = None
        weather_forecast_map = None
        if self._config.weather_enabled:
            weather_history_map, weather_forecast_map = self._fetch_weather_for_darts(
                all_training_observations, now_utc
            )

        # ---- Step 4: Train ----
        self._darts_price_model.train(
            all_training_observations,
            pd7day_history=pd7day_as_price_observations,
            weather_history=weather_history_map,
        )

        if not self._darts_price_model.is_trained:
            _LOGGER.warning(
                "PriceEngine: Darts model train() completed but is_trained=False "
                "(internal threshold not met); falling back to isotonic."
            )
            return {}, False

        # ---- Step 5: Predict ----
        tail_size = self._darts_price_model._lags * 2
        recent_tail_observations = all_training_observations[-tail_size:]

        num_forecast_slots = len([
            slot for slot in forecast.slots
            if slot.interval_start_utc >= now_utc - timedelta(minutes=30)
        ])

        predicted_rrp_values = self._darts_price_model.forecast(
            recent_observations=recent_tail_observations,
            forecast_start_utc=now_utc,
            num_slots=max(num_forecast_slots, 1),
            pd7day_forecast=pd7day_as_price_observations,
            weather_forecast=weather_forecast_map,
        )

        if not predicted_rrp_values:
            _LOGGER.warning(
                "PriceEngine: Darts model returned empty forecast; "
                "falling back to isotonic calibration."
            )
            return {}, False

        # Build a time → predicted_rrp lookup aligned to the PD7DAY slots
        darts_rrp_by_slot_start: dict[datetime, float] = {}
        future_slots = [
            slot for slot in forecast.slots
            if slot.interval_start_utc >= now_utc - timedelta(minutes=30)
        ]
        for slot_index, future_slot in enumerate(future_slots):
            if slot_index < len(predicted_rrp_values):
                darts_rrp_by_slot_start[future_slot.interval_start_utc] = (
                    predicted_rrp_values[slot_index]
                )

        _LOGGER.info(
            "PriceEngine: Darts price model forecast produced %d slot predictions "
            "(trained on %d merged observations)",
            len(darts_rrp_by_slot_start),
            len(all_training_observations),
        )
        return darts_rrp_by_slot_start, True

    def _predict_with_loaded_darts(
        self,
        import_observations: list[CalibrationObservation],
        forecast: Pd7DayForecast,
        now_utc: datetime,
    ) -> dict[datetime, float]:
        """
        Produce a per-slot RRP lookup using the ALREADY-LOADED Darts model,
        without retraining.  Used on the first cycle after a bundled (or
        previously-persisted) model is loaded, so a fresh install is Darts-backed
        on day 1 without a train-from-scratch (which would need an AEMO archive
        download + Open-Meteo fetch and several seconds of compute).

        The recent-input tail is built from the available actual-price
        observations (the bundled calibration seed and/or online /calibration
        POSTs).  PD7DAY is passed as the past+future covariate signal exactly as
        in the train+predict path.  Weather is fetched best-effort; if it is
        unavailable the model's predict() zero-pads the weather columns (shape is
        preserved) so the forecast still succeeds offline.

        Returns {} if the model produces no prediction (caller falls back).
        """
        recent_price_observations = [
            PriceObservation(
                interval_start_utc=calibration_obs.observed_at,
                rrp_per_mwh=calibration_obs.actual_rrp_per_mwh,
            )
            for calibration_obs in import_observations
        ]
        recent_price_observations.sort(key=lambda obs: obs.interval_start_utc)

        # PD7DAY current file as the covariate signal (same as the train path).
        pd7day_as_price_observations = [
            PriceObservation(
                interval_start_utc=price_slot.interval_start_utc,
                rrp_per_mwh=price_slot.rrp_per_mwh,
            )
            for price_slot in forecast.slots
        ]

        # Best-effort weather forecast (only used if the loaded model was trained
        # with weather; predict() zero-pads if it is None).
        weather_forecast_map = None
        if self._config.weather_enabled:
            _, weather_forecast_map = self._fetch_weather_for_darts(
                recent_price_observations, now_utc
            )

        tail_size = self._darts_price_model._lags * 2
        recent_tail_observations = recent_price_observations[-tail_size:]

        num_forecast_slots = len([
            slot for slot in forecast.slots
            if slot.interval_start_utc >= now_utc - timedelta(minutes=30)
        ])

        predicted_rrp_values = self._darts_price_model.forecast(
            recent_observations=recent_tail_observations,
            forecast_start_utc=now_utc,
            num_slots=max(num_forecast_slots, 1),
            pd7day_forecast=pd7day_as_price_observations,
            weather_forecast=weather_forecast_map,
        )

        if not predicted_rrp_values:
            return {}

        return self._map_darts_values_to_slots(
            predicted_rrp_values, forecast, now_utc
        )

    @staticmethod
    def _map_darts_values_to_slots(
        predicted_rrp_values: list[float],
        forecast: Pd7DayForecast,
        now_utc: datetime,
    ) -> dict[datetime, float]:
        """Align an ordered list of predicted RRP values to the future PD7DAY slots."""
        darts_rrp_by_slot_start: dict[datetime, float] = {}
        future_slots = [
            slot for slot in forecast.slots
            if slot.interval_start_utc >= now_utc - timedelta(minutes=30)
        ]
        for slot_index, future_slot in enumerate(future_slots):
            if slot_index < len(predicted_rrp_values):
                darts_rrp_by_slot_start[future_slot.interval_start_utc] = (
                    predicted_rrp_values[slot_index]
                )
        return darts_rrp_by_slot_start

    def _fetch_weather_for_darts(
        self,
        training_observations: list[PriceObservation],
        now_utc: datetime,
    ) -> tuple["dict | None", "dict | None"]:
        """
        Fetch Open-Meteo weather for:
          - weather_history_map: naive-UTC dict covering the training window
          - weather_forecast_map: naive-UTC dict covering the future forecast window

        Returns (None, None) on any error — callers fall back to calendar-only.
        """
        try:
            archive_cutoff = now_utc - timedelta(days=6)

            # Training window weather (archive API)
            if training_observations:
                sorted_training = sorted(
                    training_observations,
                    key=lambda obs: obs.interval_start_utc,
                )
                train_start = sorted_training[0].interval_start_utc
                train_end = min(sorted_training[-1].interval_start_utc, archive_cutoff)

                if train_start < train_end:
                    archive_slots = self._weather_client.fetch_archive(
                        start_date=train_start.date(),
                        end_date=train_end.date(),
                    )
                else:
                    archive_slots = []
            else:
                archive_slots = []

            # Forecast window weather (forecast API)
            forecast_slots = self._weather_client.fetch_forecast(
                hours_ahead=self._config.forecast_horizon_hours + 48
            )

            # Build naive-UTC maps at 30-min resolution
            all_slots = archive_slots + forecast_slots
            raw_weather_map = self._weather_client.slots_to_30min_map(all_slots)
            naive_weather_map = {
                slot_time.replace(tzinfo=None): slot_weather
                for slot_time, slot_weather in raw_weather_map.items()
            }

            forecast_raw_map = self._weather_client.slots_to_30min_map(forecast_slots)
            naive_forecast_map = {
                slot_time.replace(tzinfo=None): slot_weather
                for slot_time, slot_weather in forecast_raw_map.items()
            } if forecast_slots else None

            _LOGGER.info(
                "PriceEngine: weather fetched — %d history slots, %d forecast slots",
                len(archive_slots),
                len(forecast_slots),
            )
            return naive_weather_map or None, naive_forecast_map

        except Exception as weather_error:
            _LOGGER.warning(
                "PriceEngine: weather fetch failed (%s); "
                "Darts model will train without weather covariates",
                weather_error,
            )
            return None, None

    def _initialise_darts_price_model(self) -> None:
        """Create the Darts price model object."""
        # For darts_naive_blend: use the same champion config as the v7 backtest
        # (lags=96, output_chunk=48, n_estimators=200, num_leaves=31, lr=0.05).
        # For darts/hybrid: use the full horizon as output_chunk (unchanged behaviour).
        if self._config.price_model == "darts_naive_blend":
            output_chunk_length = 48   # v7 champion — one 24-hour direct-output block
        else:
            output_chunk_length = int(self._config.forecast_horizon_hours * 60 / 30)
        self._darts_price_model = DartsLightGBMPriceForecaster(
            forecast_horizon_hours=self._config.forecast_horizon_hours,
            output_chunk_length=output_chunk_length,
            use_weather_covariates=self._config.weather_enabled,
        )

    def _build_calibrator(self):
        """Construct the configured price calibrator backend.

        Returns an IsotonicCalibratorPerHour (default) or a MonotoneGBMCalibrator
        (opt-in via config.calibrator="monotone_gbm").  Both share the same public
        API — add_observation / add_observations_bulk / calibrate / is_calibrated /
        observation_count — so the rest of the engine is backend-agnostic.

        The monotone-GBM backend wraps an internal isotonic calibrator and only
        serves GBM output when it beats isotonic on a held-out tail; otherwise it
        transparently delegates to isotonic.  So "monotone_gbm" can only ever match
        or beat the shipped isotonic baseline on the live corpus (never-lose).
        """
        if self._config.calibrator == "monotone_gbm":
            return MonotoneGBMCalibrator(
                recency_half_life_days=_CALIBRATION_RECENCY_HALF_LIFE_DAYS,
                min_observations=self._config.calibration_min_observations,
                plausibility_cap_dollars_per_kwh=self._config.plausibility_cap_dollars_per_kwh,
                adjacent_blend_weight=self._config.calibrator_adjacency_alpha,
            )
        # Default: shipped per-hour isotonic PAV calibrator.
        return IsotonicCalibratorPerHour(
            recency_half_life_days=_CALIBRATION_RECENCY_HALF_LIFE_DAYS,
            min_observations=self._config.calibration_min_observations,
            plausibility_cap_dollars_per_kwh=self._config.plausibility_cap_dollars_per_kwh,
            adjacent_blend_weight=self._config.calibrator_adjacency_alpha,
        )

    def _build_tariff_calculator(self) -> TariffCalculator:
        tou_bands = parse_tou_bands_from_config(self._config.tou_bands)
        return TariffCalculator(
            tou_bands=tou_bands,
            fixed_adder_per_kwh=self._config.fixed_adder_per_kwh,
            gst_rate=self._config.gst_rate,
            feed_in_is_wholesale=self._config.feed_in_is_wholesale,
        )

    # ------------------------------------------------------------------
    # Chain resolver — horizon-aware model chaining
    # ------------------------------------------------------------------

    def _build_chain_extension_slots(
        self,
        primary_slots: list[PriceForecastSlot],
        forecast: "Pd7DayForecast",
        now_utc: datetime,
    ) -> list[PriceForecastSlot]:
        """
        Generate darts_naive_blend price slots covering the region beyond the
        chain seam up to config.forecast_horizon_hours from now.

        The extension is generated by running the full darts_naive_blend predict
        pipeline on synthetic PriceSlot stubs whose rrp_per_mwh is set to the
        last known PD7DAY value (or 0.0 if unavailable).  The darts and naive
        components have no structural dependency on PD7DAY so this is sound.

        Returns a list of PriceForecastSlot tagged with
        model_segment="chain:darts_naive_blend".  Slots that already exist in
        primary_slots (within the blend window on the primary side) are NOT
        re-generated here — the blending happens in _merge_primary_and_chain.
        """
        horizon_end_utc = now_utc + timedelta(hours=self._config.forecast_horizon_hours)
        chain_seam_utc = now_utc + timedelta(days=self._config.chain_seam_days)
        blend_half = timedelta(hours=self._config.chain_blend_window_hours)
        chain_start_utc = chain_seam_utc - blend_half

        # Find the last PD7DAY slot to use as the raw_rrp fallback for synthetic slots
        last_pd7day_rrp = 0.0
        if forecast.slots:
            sorted_pd7day_slots = sorted(
                forecast.slots, key=lambda slot: slot.interval_start_utc
            )
            last_pd7day_rrp = sorted_pd7day_slots[-1].rrp_per_mwh

        # Build the set of slot times we need to cover.
        # Use 30-min resolution (native PD7DAY interval) for the synthetic extension.
        slot_times_needed: list[datetime] = []
        # Start from the overlap point (blend window start), stepping 30 min
        step = timedelta(minutes=30)
        candidate_time = chain_start_utc.replace(second=0, microsecond=0)
        # Align to nearest 30-min boundary
        rounded_minute = (candidate_time.minute // 30) * 30
        candidate_time = candidate_time.replace(minute=rounded_minute)

        while candidate_time < horizon_end_utc:
            slot_times_needed.append(candidate_time)
            candidate_time += step

        if not slot_times_needed:
            return []

        # Build fake PriceSlot objects for the darts_naive_blend engine to process.
        # We use the current PD7DAY run's last known RRP as a placeholder.
        # The darts/naive components don't depend on raw rrp, so these values only
        # affect the fallback isotonic path (which doesn't run in darts_naive_blend mode).
        fake_price_slots = [
            PriceSlot(
                interval_start_utc=slot_time,
                interval_start_nem=slot_time.astimezone(NEM_TIMEZONE),
                rrp_per_mwh=last_pd7day_rrp,
            )
            for slot_time in slot_times_needed
        ]

        # Build a synthetic Pd7DayForecast containing the fake slots.
        # The run_datetime is set to now so lead-time calculations are from current time.
        synthetic_forecast = Pd7DayForecast(
            region=self._config.region,
            run_datetime_utc=now_utc,
            slots=fake_price_slots,
        )

        # Run darts_naive_blend on the synthetic slots.
        # We need a temporary engine configuration forced to darts_naive_blend.
        # Rather than reconfiguring the engine, we call the internal helpers directly.

        # Build darts forecast for chain slots (use existing darts model if available)
        chain_darts_rrp_by_slot_start: dict[datetime, float] = {}
        chain_using_darts = False
        if self._darts_price_model is not None:
            try:
                chain_darts_rrp_by_slot_start, chain_using_darts = (
                    self._build_darts_cycle_forecast(synthetic_forecast, now_utc)
                )
            except Exception as chain_darts_error:
                _LOGGER.warning(
                    "PriceEngine [chain]: Darts forecast for chain extension failed (%s); "
                    "chain will use naive-only",
                    chain_darts_error,
                )
        elif self._config.price_model not in ("darts", "hybrid", "darts_naive_blend"):
            # Primary model doesn't use Darts; initialise a temporary one for chain
            _LOGGER.debug(
                "PriceEngine [chain]: primary model=%s has no Darts instance; "
                "chain extension will use naive-only blend",
                self._config.price_model,
            )

        # Build naive lookup for chain slots
        chain_naive_rrp_by_slot_start = self._build_seasonal_naive_lookup(
            forecast=synthetic_forecast, now_utc=now_utc
        )

        chain_slots: list[PriceForecastSlot] = []
        for fake_slot in fake_price_slots:
            # Skip slots before the blend-window start (they're in primary territory)
            # but include the blend zone so _merge_primary_and_chain can interpolate
            import_kwh, export_kwh = self._compute_naive_blend_slot_price(
                price_slot=fake_slot,
                now_utc=now_utc,
                darts_rrp_by_slot_start=chain_darts_rrp_by_slot_start,
                using_darts=chain_using_darts,
                naive_rrp_by_slot_start=chain_naive_rrp_by_slot_start,
                blend_weight=self._config.naive_blend_weight,
            )
            if import_kwh is None:
                continue

            try:
                network_rate = self._tariff_calculator.network_rate_for_interval(
                    fake_slot.interval_start_nem
                )
                import_price = self._tariff_calculator.compute_import_price(
                    import_kwh, fake_slot.interval_start_nem
                )
                export_price = self._tariff_calculator.compute_export_price(
                    export_kwh, fake_slot.interval_start_nem
                )
            except Exception as tariff_error:
                _LOGGER.debug(
                    "PriceEngine [chain]: tariff error for slot %s: %s",
                    fake_slot.interval_start_utc.isoformat(),
                    tariff_error,
                )
                continue

            chain_slots.append(
                PriceForecastSlot(
                    interval_start=fake_slot.interval_start_utc,
                    raw_rrp_per_mwh=last_pd7day_rrp,
                    calibrated_wholesale_kwh=import_kwh,
                    import_price_kwh=import_price,
                    export_price_kwh=export_price,
                    network_tou_rate_kwh=network_rate,
                    model_segment="chain:darts_naive_blend",
                )
            )

        return chain_slots

    def _merge_primary_and_chain(
        self,
        primary_slots: list[PriceForecastSlot],
        chain_slots: list[PriceForecastSlot],
        now_utc: datetime,
    ) -> list[PriceForecastSlot]:
        """
        Merge primary and chain slots into a single sorted list, applying a
        linear blend across the seam window to avoid price discontinuity.

        Blend logic:
          - lead_time_hours <= (seam - blend_half)  → pure primary (weight=1)
          - lead_time_hours >= (seam + blend_half)  → pure chain (weight=1)
          - between                                  → linear interpolation

        Slots that only exist in one set (no overlap partner) are passed through
        as-is.  The output is sorted by interval_start ascending.
        """
        seam_hours = self._config.chain_seam_days * 24.0
        blend_half_hours = self._config.chain_blend_window_hours

        # Index both sets by interval_start for fast lookup
        primary_by_time: dict[datetime, PriceForecastSlot] = {
            slot.interval_start: slot for slot in primary_slots
        }
        chain_by_time: dict[datetime, PriceForecastSlot] = {
            slot.interval_start: slot for slot in chain_slots
        }

        all_times = sorted(
            set(primary_by_time.keys()) | set(chain_by_time.keys())
        )

        merged_slots: list[PriceForecastSlot] = []
        for slot_time in all_times:
            lead_time_hours = (slot_time - now_utc).total_seconds() / 3600.0

            primary_slot = primary_by_time.get(slot_time)
            chain_slot = chain_by_time.get(slot_time)

            if primary_slot is not None and chain_slot is None:
                # Only primary exists (short-lead, well within seam)
                merged_slots.append(primary_slot)
                continue

            if chain_slot is not None and primary_slot is None:
                # Only chain exists (beyond seam + full blend window)
                merged_slots.append(chain_slot)
                continue

            # Both exist — apply blend
            if blend_half_hours <= 0.0:
                # Hard handoff at the seam
                if lead_time_hours <= seam_hours:
                    merged_slots.append(primary_slot)  # type: ignore[arg-type]
                else:
                    merged_slots.append(chain_slot)  # type: ignore[arg-type]
                continue

            low_boundary = seam_hours - blend_half_hours
            high_boundary = seam_hours + blend_half_hours

            if lead_time_hours <= low_boundary:
                chain_weight = 0.0
            elif lead_time_hours >= high_boundary:
                chain_weight = 1.0
            else:
                chain_weight = (
                    (lead_time_hours - low_boundary) / (2.0 * blend_half_hours)
                )
            primary_weight = 1.0 - chain_weight

            if chain_weight == 0.0:
                merged_slots.append(primary_slot)  # type: ignore[arg-type]
                continue
            if chain_weight == 1.0:
                merged_slots.append(chain_slot)  # type: ignore[arg-type]
                continue

            # Blend both slots
            primary_cast: PriceForecastSlot = primary_slot  # type: ignore[assignment]
            chain_cast: PriceForecastSlot = chain_slot      # type: ignore[assignment]

            blended_wholesale = (
                primary_weight * primary_cast.calibrated_wholesale_kwh
                + chain_weight * chain_cast.calibrated_wholesale_kwh
            )
            blended_import = (
                primary_weight * primary_cast.import_price_kwh
                + chain_weight * chain_cast.import_price_kwh
            )
            blended_export = (
                primary_weight * primary_cast.export_price_kwh
                + chain_weight * chain_cast.export_price_kwh
            )
            # Network rate: use primary rate (it's tariff-only, not model-dependent)
            network_rate = primary_cast.network_tou_rate_kwh

            # model_segment: tag as primary in the blend zone (lead < seam), chain beyond
            segment = "primary" if lead_time_hours < seam_hours else "chain:darts_naive_blend"

            merged_slots.append(
                PriceForecastSlot(
                    interval_start=slot_time,
                    raw_rrp_per_mwh=primary_cast.raw_rrp_per_mwh,
                    calibrated_wholesale_kwh=blended_wholesale,
                    import_price_kwh=blended_import,
                    export_price_kwh=blended_export,
                    network_tou_rate_kwh=network_rate,
                    model_segment=segment,
                )
            )

        return merged_slots
