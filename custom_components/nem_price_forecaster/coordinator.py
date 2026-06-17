"""
DataUpdateCoordinator for NEM Price Forecaster (sidecar mode).

Responsibilities (refactored for sidecar architecture):
  1. Poll the sidecar's /price_forecast and /load_forecast endpoints.
  2. Parse and expose a structured forecast for sensor entities.
  3. Handle sidecar unavailability with HA retry semantics (UpdateFailed).
  4. Provide calibration observation methods (POST to sidecar).

Python 3.14 compatibility: NO darts, NO sklearn, NO scipy imports here.
Only stdlib + numpy + homeassistant builtins.

The sidecar runs all ML compute (PD7DAY fetch, isotonic/Darts calibration,
tariff calculation, load forecast). The coordinator is a thin HTTP client.

Legacy mode (no sidecar):
  When CONF_SIDECAR_URL is not configured, the coordinator falls back to the
  original embedded mode (importing all ML modules directly). This maintains
  backward compatibility for users who have not yet deployed the sidecar.
  NOTE: Legacy embedded mode does NOT work on Python 3.14 if darts is absent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed

from .const import (
    CONF_REGION,
    CONF_FORECAST_HORIZON_HOURS,
    CONF_FORECAST_PERIOD_MINUTES,
    CONF_LOAD_FORECASTER_ENABLED,
    CONF_SIDECAR_URL,
    DEFAULT_FORECAST_HORIZON_HOURS,
    DEFAULT_FORECAST_PERIOD_MINUTES,
    DEFAULT_LOAD_FORECASTER_ENABLED,
    DEFAULT_SIDECAR_URL,
    DEFAULT_UPDATE_INTERVAL_MINUTES,
    CONF_UPDATE_INTERVAL_MINUTES,
    DOMAIN,
)
from .sidecar_client import SidecarClient, SidecarUnavailable

_LOGGER = logging.getLogger(__name__)


@dataclass
class ForecastSlot:
    """
    One price forecast slot as returned by the sidecar /price_forecast endpoint.

    All prices are in $/kWh.
    """
    interval_start_utc: datetime     # UTC, tz-aware — PERIOD-BEGINNING (internal)
    interval_end_utc: datetime       # UTC, tz-aware — PERIOD-ENDING (published)
    raw_rrp_per_mwh: float
    calibrated_wholesale_kwh: float
    import_price_kwh: float
    export_price_kwh: float
    network_tou_rate_kwh: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "interval_start": self.interval_start_utc.isoformat(),
            "interval_end": self.interval_end_utc.isoformat(),
            "import_price": round(self.import_price_kwh, 6),
            "export_price": round(self.export_price_kwh, 6),
            "calibrated_wholesale": round(self.calibrated_wholesale_kwh, 6),
            "raw_rrp_per_mwh": round(self.raw_rrp_per_mwh, 4),
            "network_tou_rate": round(self.network_tou_rate_kwh, 6),
        }


@dataclass
class LoadForecastSlot:
    """One load forecast slot (30-min, watts)."""
    interval_start_utc: datetime
    load_watts: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "datetime": self.interval_start_utc.isoformat(),
            "load_power": round(self.load_watts, 1),
        }


@dataclass
class CoordinatorData:
    """Snapshot of sidecar forecast state, published to all sensor entities."""
    # Parsed price forecast slots (from sidecar raw_forecast)
    forecast_slots: list[ForecastSlot]
    run_datetime_utc: datetime
    region: str
    calibration_observation_count: int
    calibration_is_active: bool
    next_update: datetime
    # Resampled forecasts (from sidecar forecast[])
    resampled_price_forecast: list[dict]
    # Load forecast
    load_forecast_slots: list[LoadForecastSlot]
    resampled_load_forecast: list[dict]
    load_model_name: str
    load_is_trained: bool
    load_training_observations: int
    # Resolution metadata
    forecast_horizon_hours: int
    forecast_period_minutes: int
    # Sidecar health
    sidecar_url: str
    sidecar_reachable: bool


class NemPriceForecastCoordinator(DataUpdateCoordinator[CoordinatorData]):
    """
    Thin coordinator that fetches from the NEM sidecar and exposes sensor data.

    ML compute runs in the sidecar. This coordinator is pure async I/O.
    """

    def __init__(self, hass: HomeAssistant, config_entry: ConfigEntry) -> None:
        self._config_entry = config_entry
        self._region: str = config_entry.data[CONF_REGION]
        self._sidecar_url: str = config_entry.data.get(CONF_SIDECAR_URL, DEFAULT_SIDECAR_URL)

        self._forecast_horizon_hours: int = int(config_entry.data.get(
            CONF_FORECAST_HORIZON_HOURS, DEFAULT_FORECAST_HORIZON_HOURS
        ))
        self._forecast_period_minutes: int = int(config_entry.data.get(
            CONF_FORECAST_PERIOD_MINUTES, DEFAULT_FORECAST_PERIOD_MINUTES
        ))
        self._load_forecaster_enabled: bool = config_entry.data.get(
            CONF_LOAD_FORECASTER_ENABLED, DEFAULT_LOAD_FORECASTER_ENABLED
        )

        self._sidecar_client = SidecarClient(self._sidecar_url)

        # Update cadence: options-flow override > data-entry override > default.
        # The sidecar's price-predict job runs every 5 minutes, so polling more
        # often than ~5 minutes wastes effort; default 15 min gives ~3 companion
        # polls per sidecar refresh cycle.
        merged_config = {**config_entry.data, **config_entry.options}
        update_interval_minutes = int(merged_config.get(
            CONF_UPDATE_INTERVAL_MINUTES, DEFAULT_UPDATE_INTERVAL_MINUTES
        ))
        if update_interval_minutes < 1:
            update_interval_minutes = DEFAULT_UPDATE_INTERVAL_MINUTES
        self._update_interval_minutes = update_interval_minutes

        update_interval = timedelta(minutes=update_interval_minutes)
        super().__init__(
            hass,
            _LOGGER,
            name=f"{DOMAIN}_{self._region}",
            update_interval=update_interval,
        )

    # ------------------------------------------------------------------
    # DataUpdateCoordinator override
    # ------------------------------------------------------------------

    async def _async_update_data(self) -> CoordinatorData:
        """
        Fetch price + load forecasts from the sidecar.

        On SidecarUnavailable, raises UpdateFailed so HA marks sensors
        unavailable and retries.
        """
        now_utc = datetime.now(timezone.utc)

        # --- Price forecast ---
        try:
            price_response = await self._sidecar_client.async_get_price_forecast()
        except SidecarUnavailable as fetch_error:
            raise UpdateFailed(
                f"Sidecar unavailable at {self._sidecar_url}: {fetch_error}"
            ) from fetch_error

        forecast_slots = _parse_price_slots(price_response.get("raw_forecast", []))
        resampled_price = price_response.get("forecast", [])

        calibration_observations = int(price_response.get("calibration_observations", 0))
        calibration_active = bool(price_response.get("calibration_active", False))
        pd7day_run_str = price_response.get("pd7day_run_datetime")
        pd7day_run_utc = _parse_iso_or_now(pd7day_run_str)

        # --- Load forecast (optional) ---
        load_slots: list[LoadForecastSlot] = []
        resampled_load: list[dict] = []
        load_is_trained = False
        load_training_obs = 0

        if self._load_forecaster_enabled:
            try:
                load_response = await self._sidecar_client.async_get_load_forecast()
                if load_response is not None:
                    load_slots = _parse_load_slots(load_response.get("raw_forecast", []))
                    resampled_load = load_response.get("forecast", [])
                    load_is_trained = bool(load_response.get("model_trained", False))
                    load_training_obs = int(load_response.get("training_observations", 0))
            except SidecarUnavailable as load_error:
                _LOGGER.warning(
                    "Load forecast fetch failed (non-fatal): %s", load_error
                )

        next_update_utc = now_utc + timedelta(minutes=self._update_interval_minutes)

        return CoordinatorData(
            forecast_slots=forecast_slots,
            run_datetime_utc=pd7day_run_utc,
            region=self._region,
            calibration_observation_count=calibration_observations,
            calibration_is_active=calibration_active,
            next_update=next_update_utc,
            resampled_price_forecast=resampled_price,
            load_forecast_slots=load_slots,
            resampled_load_forecast=resampled_load,
            load_model_name="Darts-LightGBM-Direct",
            load_is_trained=load_is_trained,
            load_training_observations=load_training_obs,
            forecast_horizon_hours=self._forecast_horizon_hours,
            forecast_period_minutes=self._forecast_period_minutes,
            sidecar_url=self._sidecar_url,
            sidecar_reachable=True,
        )

    # ------------------------------------------------------------------
    # Calibration feed (forwarded to sidecar via HTTP POST)
    # ------------------------------------------------------------------

    async def async_add_import_calibration_observation(
        self,
        predicted_rrp_per_mwh: float,
        actual_import_rrp_per_mwh: float,
        hour_of_day_nem: int,
        observed_at: datetime,
    ) -> None:
        """
        Forward a (predicted, actual import) calibration observation to the sidecar.
        Best-effort (non-fatal if sidecar is unavailable).
        """
        await self._sidecar_client.async_post_import_calibration(
            predicted_rrp_per_mwh,
            actual_import_rrp_per_mwh,
            hour_of_day_nem,
            observed_at,
        )

    async def async_add_export_calibration_observation(
        self,
        predicted_rrp_per_mwh: float,
        actual_export_rrp_per_mwh: float,
        hour_of_day_nem: int,
        observed_at: datetime,
    ) -> None:
        """Forward an export calibration observation to the sidecar."""
        await self._sidecar_client.async_post_export_calibration(
            predicted_rrp_per_mwh,
            actual_export_rrp_per_mwh,
            hour_of_day_nem,
            observed_at,
        )

    async def async_add_load_observation(
        self,
        interval_start_utc: datetime,
        load_watts: float,
    ) -> None:
        """Forward a load observation to the sidecar."""
        await self._sidecar_client.async_post_load_observation(
            interval_start_utc, load_watts
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def async_close(self) -> None:
        """Close the HTTP session. Called on config entry unload."""
        await self._sidecar_client.async_close()


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def _parse_price_slots(raw_list: list[dict[str, Any]]) -> list[ForecastSlot]:
    """Parse the raw_forecast[] from /price_forecast into ForecastSlot objects."""
    slots: list[ForecastSlot] = []
    for raw_item in raw_list:
        try:
            interval_start = _parse_iso_or_now(raw_item.get("interval_start"))
            # interval_end is the PERIOD-ENDING (published) stamp.  Fall back to
            # start + 30 min (the native PD7DAY period) for older sidecars that
            # don't yet emit interval_end.
            interval_end_raw = raw_item.get("interval_end")
            if interval_end_raw is not None:
                interval_end = _parse_iso_or_now(interval_end_raw)
            else:
                interval_end = interval_start + timedelta(minutes=30)
            slots.append(
                ForecastSlot(
                    interval_start_utc=interval_start,
                    interval_end_utc=interval_end,
                    raw_rrp_per_mwh=float(raw_item.get("raw_rrp_per_mwh", 0.0)),
                    calibrated_wholesale_kwh=float(
                        raw_item.get("calibrated_wholesale_kwh", 0.0)
                    ),
                    import_price_kwh=float(raw_item.get("import_price_kwh", 0.0)),
                    export_price_kwh=float(raw_item.get("export_price_kwh", 0.0)),
                    network_tou_rate_kwh=float(raw_item.get("network_tou_rate_kwh", 0.0)),
                )
            )
        except (KeyError, ValueError, TypeError) as parse_error:
            _LOGGER.debug("Skipping malformed price slot: %s", parse_error)
    return slots


def _parse_load_slots(raw_list: list[dict[str, Any]]) -> list[LoadForecastSlot]:
    """Parse the raw_forecast[] from /load_forecast into LoadForecastSlot objects."""
    slots: list[LoadForecastSlot] = []
    for raw_item in raw_list:
        try:
            interval_start = _parse_iso_or_now(raw_item.get("interval_start"))
            slots.append(
                LoadForecastSlot(
                    interval_start_utc=interval_start,
                    load_watts=float(raw_item.get("load_watts", 0.0)),
                )
            )
        except (KeyError, ValueError, TypeError) as parse_error:
            _LOGGER.debug("Skipping malformed load slot: %s", parse_error)
    return slots


def _parse_iso_or_now(iso_string: Optional[str]) -> datetime:
    if iso_string is None:
        return datetime.now(timezone.utc)
    try:
        parsed = datetime.fromisoformat(iso_string.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except (ValueError, AttributeError):
        return datetime.now(timezone.utc)
