"""
Thread-safe in-memory cache for computed price and load forecasts.

DESIGN INVARIANT: HTTP endpoints NEVER trigger compute — they only read
from this cache.  All compute happens in background scheduler tasks.
This guarantees sub-millisecond endpoint latency regardless of model size.

The cache stores:
  - latest_price_forecast: list of PriceForecastSlot (computed by PriceEngine)
  - latest_load_forecast:  list of LoadForecastSlot (computed by LoadEngine)
  - metadata: run timestamps, model state, PD7DAY run datetime

Both are guarded by a threading.Lock so background-thread writes and
async-thread reads are race-free.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

_LOGGER = logging.getLogger(__name__)


@dataclass
class PriceForecastSlot:
    """
    One price forecast slot as served by /price_forecast.

    All prices are in $/kWh.

    model_segment identifies which model (or chain segment) produced this slot:
      "primary"           — the configured primary model covered this slot
      "chain:darts_naive_blend" — chain resolver extended beyond the primary model's
                                  validated reach using darts_naive_blend
      "chain:seasonal_naive"    — chain resolver used seasonal-naive (unbounded)
    Consumers can filter on this field to distinguish validated from chained slots.
    """
    interval_start: datetime         # UTC, tz-aware
    raw_rrp_per_mwh: float           # unmodified PD7DAY value (0.0 for chain-only slots)
    calibrated_wholesale_kwh: float  # after isotonic / Darts calibration
    import_price_kwh: float          # retail import (calibrated + ToU + fixed + GST)
    export_price_kwh: float          # feed-in (calibrated wholesale, GST-excluded)
    network_tou_rate_kwh: float      # network component only
    model_segment: str = "primary"   # see docstring above

    def as_dict(self) -> dict[str, Any]:
        return {
            "interval_start": self.interval_start.isoformat(),
            "raw_rrp_per_mwh": round(self.raw_rrp_per_mwh, 4),
            "calibrated_wholesale_kwh": round(self.calibrated_wholesale_kwh, 6),
            "import_price_kwh": round(self.import_price_kwh, 6),
            "export_price_kwh": round(self.export_price_kwh, 6),
            "network_tou_rate_kwh": round(self.network_tou_rate_kwh, 6),
            "model_segment": self.model_segment,
        }


@dataclass
class LoadForecastSlot:
    """One load forecast slot as served by /load_forecast."""
    interval_start: datetime  # UTC, tz-aware
    load_watts: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "interval_start": self.interval_start.isoformat(),
            "load_watts": round(self.load_watts, 1),
        }


@dataclass
class ForecastMetadata:
    """Metadata attached to every API response."""
    region: str = ""
    price_model: str = ""
    pd7day_run_datetime: Optional[str] = None
    price_computed_at: Optional[str] = None
    load_computed_at: Optional[str] = None
    price_calibration_active: bool = False
    price_calibration_observations: int = 0
    load_model_trained: bool = False
    load_training_observations: int = 0
    forecast_horizon_hours: int = 168
    forecast_period_minutes: int = 30


class ForecastCache:
    """
    Thread-safe cache for the latest computed price and load forecasts.

    Background scheduler writes via update_price_forecast() / update_load_forecast().
    FastAPI handlers read via get_price_forecast() / get_load_forecast() — these
    never block (lock is held only for microseconds on both sides).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._price_forecast: list[PriceForecastSlot] = []
        self._load_forecast: list[LoadForecastSlot] = []
        self._resampled_price_forecast: list[dict[str, Any]] = []
        self._resampled_load_forecast: list[dict[str, Any]] = []
        self._metadata = ForecastMetadata()

    # ------------------------------------------------------------------
    # Writes (called from background threads)
    # ------------------------------------------------------------------

    def update_price_forecast(
        self,
        price_slots: list[PriceForecastSlot],
        resampled_slots: list[dict[str, Any]],
        pd7day_run_datetime: Optional[datetime],
        calibration_active: bool,
        calibration_observations: int,
    ) -> None:
        """Replace the cached price forecast atomically."""
        computed_at = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._price_forecast = price_slots
            self._resampled_price_forecast = resampled_slots
            self._metadata.pd7day_run_datetime = (
                pd7day_run_datetime.isoformat() if pd7day_run_datetime else None
            )
            self._metadata.price_computed_at = computed_at
            self._metadata.price_calibration_active = calibration_active
            self._metadata.price_calibration_observations = calibration_observations
        _LOGGER.debug(
            "Price forecast cache updated: %d raw slots, %d resampled",
            len(price_slots),
            len(resampled_slots),
        )

    def update_load_forecast(
        self,
        load_slots: list[LoadForecastSlot],
        resampled_slots: list[dict[str, Any]],
        model_trained: bool,
        training_observations: int,
    ) -> None:
        """Replace the cached load forecast atomically."""
        computed_at = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._load_forecast = load_slots
            self._resampled_load_forecast = resampled_slots
            self._metadata.load_computed_at = computed_at
            self._metadata.load_model_trained = model_trained
            self._metadata.load_training_observations = training_observations
        _LOGGER.debug(
            "Load forecast cache updated: %d raw slots, %d resampled",
            len(load_slots),
            len(resampled_slots),
        )

    def update_metadata(self, **kwargs: Any) -> None:
        """Update arbitrary metadata fields atomically."""
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self._metadata, key):
                    setattr(self._metadata, key, value)

    # ------------------------------------------------------------------
    # Reads (called from async FastAPI handlers — never block)
    # ------------------------------------------------------------------

    def get_price_forecast(self) -> tuple[list[PriceForecastSlot], list[dict[str, Any]], ForecastMetadata]:
        """Return (raw_slots, resampled_slots, metadata) snapshot."""
        with self._lock:
            return (
                list(self._price_forecast),
                list(self._resampled_price_forecast),
                _copy_metadata(self._metadata),
            )

    def get_load_forecast(self) -> tuple[list[LoadForecastSlot], list[dict[str, Any]], ForecastMetadata]:
        """Return (raw_slots, resampled_slots, metadata) snapshot."""
        with self._lock:
            return (
                list(self._load_forecast),
                list(self._resampled_load_forecast),
                _copy_metadata(self._metadata),
            )

    def get_metadata(self) -> ForecastMetadata:
        with self._lock:
            return _copy_metadata(self._metadata)

    @property
    def has_price_forecast(self) -> bool:
        with self._lock:
            return len(self._price_forecast) > 0

    @property
    def has_load_forecast(self) -> bool:
        with self._lock:
            return len(self._load_forecast) > 0


def _copy_metadata(metadata: ForecastMetadata) -> ForecastMetadata:
    """Shallow copy of ForecastMetadata (all fields are immutable scalars)."""
    return ForecastMetadata(
        region=metadata.region,
        price_model=metadata.price_model,
        pd7day_run_datetime=metadata.pd7day_run_datetime,
        price_computed_at=metadata.price_computed_at,
        load_computed_at=metadata.load_computed_at,
        price_calibration_active=metadata.price_calibration_active,
        price_calibration_observations=metadata.price_calibration_observations,
        load_model_trained=metadata.load_model_trained,
        load_training_observations=metadata.load_training_observations,
        forecast_horizon_hours=metadata.forecast_horizon_hours,
        forecast_period_minutes=metadata.forecast_period_minutes,
    )
