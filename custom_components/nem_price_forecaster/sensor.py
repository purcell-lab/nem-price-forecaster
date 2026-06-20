"""
Sensor entities for NEM Price Forecaster.

Exposes three sensor entities per config entry:

  1. sensor.nem_price_forecaster_{region}_import_price
       state = current 30-min import price ($/kWh, GST-inclusive)
       attribute "forecast" = list of dicts for all future 30-min slots

  2. sensor.nem_price_forecaster_{region}_export_price
       state = current 30-min export price ($/kWh, GST-excluded)
       attribute "forecast" = list of dicts for all future 30-min slots

  3. sensor.nem_price_forecaster_{region}_load_forecast
       state = current 30-min house load forecast (W)
       attribute "forecast" = list of {"datetime":..., "load_power":...} dicts
       Only created when load_forecaster_enabled=True.

Both price sensors carry the same "forecast" attribute list, which contains
interval_start, interval_end, import_price, export_price, calibrated_wholesale,
raw_rrp_per_mwh, and network_tou_rate for each slot.  Downstream tools (EMHASS,
Apex Charts, custom automations) can use the attribute directly.

TIMESTAMP CONVENTION (period-ending):
  Each slot carries BOTH timestamps.  `interval_end` is the PUBLISHED,
  settlement-aligned stamp (the price for [interval_start, interval_end) is
  labelled by its END) — this matches the NEM / Amber convention, so plot or
  overlay on `interval_end`.  `interval_start` (period-beginning) is retained
  for reference and is what the integration uses INTERNALLY to pick the current
  slot.

The "current" state is the slot whose half-open interval
[interval_start, interval_end) contains now — i.e., the slot we are currently
WITHIN.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import (
    ATTR_CALIBRATION_OBSERVATIONS,
    ATTR_CALIBRATION_STATUS,
    ATTR_FORECAST,
    ATTR_LOAD_FORECAST,
    ATTR_LOAD_IS_TRAINED,
    ATTR_LOAD_MODEL_NAME,
    ATTR_LOAD_TRAINING_OBSERVATIONS,
    ATTR_NEXT_UPDATE,
    ATTR_REGION,
    ATTR_RUN_DATETIME,
    CONF_LOAD_FORECASTER_ENABLED,
    CONF_REGION,
    DEFAULT_LOAD_FORECASTER_ENABLED,
    DOMAIN,
    SENSOR_EXPORT_PRICE,
    SENSOR_IMPORT_PRICE,
    SENSOR_LOAD_FORECAST,
    UNIT_DOLLARS_PER_KWH,
    UNIT_WATTS,
)
from .coordinator import CoordinatorData, ForecastSlot, NemPriceForecastCoordinator

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up sensor entities from a config entry."""
    coordinator: NemPriceForecastCoordinator = hass.data[DOMAIN][config_entry.entry_id]

    entities: list[SensorEntity] = [
        NemImportPriceSensor(coordinator, config_entry),
        NemExportPriceSensor(coordinator, config_entry),
    ]

    # Add the load forecast sensor only when enabled
    load_forecaster_enabled: bool = config_entry.data.get(
        CONF_LOAD_FORECASTER_ENABLED, DEFAULT_LOAD_FORECASTER_ENABLED
    )
    if load_forecaster_enabled:
        entities.append(NemLoadForecastSensor(coordinator, config_entry))

    async_add_entities(entities, update_before_add=True)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class _NemPriceBaseSensor(CoordinatorEntity[NemPriceForecastCoordinator], SensorEntity):
    """Base sensor with shared logic for import and export price sensors."""

    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_device_class = SensorDeviceClass.MONETARY
    _attr_native_unit_of_measurement = UNIT_DOLLARS_PER_KWH
    _attr_suggested_display_precision = 4
    _attr_has_entity_name = True
    # The full resampled price forecast is a multi-day list of dicts that
    # exceeds the recorder's 16384-byte per-state attribute cap, producing
    # "State attributes ... exceed maximum size of 16384 bytes" warnings and
    # bloating the database. Keep it live for templates/EMHASS but never
    # persist it.
    _unrecorded_attributes = frozenset({ATTR_FORECAST})

    def __init__(
        self,
        coordinator: NemPriceForecastCoordinator,
        config_entry: ConfigEntry,
        sensor_suffix: str,
        friendly_name: str,
    ) -> None:
        super().__init__(coordinator)
        region: str = config_entry.data[CONF_REGION]
        self._region = region
        self._sensor_suffix = sensor_suffix
        self._attr_name = friendly_name
        self._attr_unique_id = f"{DOMAIN}_{region}_{sensor_suffix}"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, region)},
            name=f"NEM Price Forecaster ({region})",
            manufacturer="AEMO NEMWeb",
            model="PD7DAY Predispatch + Sidecar ML Calibration",
            entry_type="service",
        )

    @property
    def coordinator_data(self) -> CoordinatorData | None:
        return self.coordinator.data

    def _current_slot(self) -> ForecastSlot | None:
        """Return the slot that covers the current moment.

        The "current" slot is the one whose half-open interval
        [interval_start, interval_end) contains now.  Selection uses the
        PERIOD-BEGINNING interval_start (kept internal/unchanged) so that the
        switch to period-ending PUBLISHED timestamps never shifts which slot is
        treated as "now".  A robust fallback (last slot with start <= now <
        start+30min) covers any malformed/gapped data.
        """
        if not self.coordinator_data:
            return None
        now_utc = datetime.now(timezone.utc)
        for forecast_slot in self.coordinator_data.forecast_slots:
            if forecast_slot.interval_start_utc <= now_utc < forecast_slot.interval_end_utc:
                return forecast_slot
        return None

    def _common_extra_attributes(self) -> dict[str, Any]:
        """Attributes shared by both import and export sensors."""
        data = self.coordinator_data
        if data is None:
            return {}
        return {
            ATTR_REGION: data.region,
            ATTR_RUN_DATETIME: data.run_datetime_utc.isoformat(),
            ATTR_CALIBRATION_STATUS: "active" if data.calibration_is_active else "warming_up",
            ATTR_CALIBRATION_OBSERVATIONS: data.calibration_observation_count,
            ATTR_NEXT_UPDATE: data.next_update.isoformat(),
            "forecast_horizon_hours": data.forecast_horizon_hours,
            "forecast_period_minutes": data.forecast_period_minutes,
            # Resampled forecast at the user-configured period and horizon
            ATTR_FORECAST: data.resampled_price_forecast,
        }


# ---------------------------------------------------------------------------
# Import price sensor
# ---------------------------------------------------------------------------

class NemImportPriceSensor(_NemPriceBaseSensor):
    """Current and forecast import price (wholesale + network + GST)."""

    def __init__(
        self,
        coordinator: NemPriceForecastCoordinator,
        config_entry: ConfigEntry,
    ) -> None:
        super().__init__(
            coordinator,
            config_entry,
            sensor_suffix=SENSOR_IMPORT_PRICE,
            friendly_name="Import Price",
        )
        self._attr_icon = "mdi:transmission-tower-import"

    @property
    def native_value(self) -> float | None:
        current_slot = self._current_slot()
        if current_slot is None:
            return None
        return round(current_slot.import_price_kwh, 6)

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        attrs = self._common_extra_attributes()
        current_slot = self._current_slot()
        if current_slot:
            attrs["current_interval_start"] = current_slot.interval_start_utc.isoformat()
            attrs["current_interval_end"] = current_slot.interval_end_utc.isoformat()
            attrs["current_network_tou_rate"] = round(
                current_slot.network_tou_rate_kwh, 6
            )
            attrs["current_calibrated_wholesale"] = round(
                current_slot.calibrated_wholesale_kwh, 6
            )
            attrs["current_raw_rrp_per_mwh"] = round(current_slot.raw_rrp_per_mwh, 4)
        return attrs


# ---------------------------------------------------------------------------
# Export price sensor
# ---------------------------------------------------------------------------

class NemExportPriceSensor(_NemPriceBaseSensor):
    """Current and forecast export (feed-in) price (wholesale, GST-excluded)."""

    def __init__(
        self,
        coordinator: NemPriceForecastCoordinator,
        config_entry: ConfigEntry,
    ) -> None:
        super().__init__(
            coordinator,
            config_entry,
            sensor_suffix=SENSOR_EXPORT_PRICE,
            friendly_name="Export Price",
        )
        self._attr_icon = "mdi:transmission-tower-export"

    @property
    def native_value(self) -> float | None:
        current_slot = self._current_slot()
        if current_slot is None:
            return None
        return round(current_slot.export_price_kwh, 6)

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        attrs = self._common_extra_attributes()
        current_slot = self._current_slot()
        if current_slot:
            attrs["current_interval_start"] = current_slot.interval_start_utc.isoformat()
            attrs["current_interval_end"] = current_slot.interval_end_utc.isoformat()
            attrs["current_raw_rrp_per_mwh"] = round(current_slot.raw_rrp_per_mwh, 4)
            attrs["current_calibrated_wholesale"] = round(
                current_slot.calibrated_wholesale_kwh, 6
            )
            attrs["gst_note"] = (
                "Export price is GST-excluded per ATO ruling on residential feed-in credits"
            )
        return attrs


# ---------------------------------------------------------------------------
# Load forecast sensor
# ---------------------------------------------------------------------------

class NemLoadForecastSensor(CoordinatorEntity[NemPriceForecastCoordinator], SensorEntity):
    """
    House-load forecast sensor.

    State: current 30-min load forecast in watts.
    Attribute 'forecast': list of {"datetime": ISO8601, "load_power": W} dicts
    covering the full planning horizon (default 48 h).

    EMHASS wiring:
        load_power_forecast: >
          {{ state_attr('sensor.nem_price_forecaster_nsw1_load_forecast', 'forecast')
             | tojson }}
    """

    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_native_unit_of_measurement = UNIT_WATTS
    _attr_suggested_display_precision = 0
    _attr_has_entity_name = True
    _attr_icon = "mdi:home-lightning-bolt"
    # The full resampled load forecast is a multi-day list of dicts that
    # exceeds the recorder's 16384-byte per-state attribute cap. Keep it
    # live for templates/EMHASS but never persist it.
    _unrecorded_attributes = frozenset({ATTR_LOAD_FORECAST})

    def __init__(
        self,
        coordinator: NemPriceForecastCoordinator,
        config_entry: ConfigEntry,
    ) -> None:
        super().__init__(coordinator)
        region: str = config_entry.data[CONF_REGION]
        self._region = region
        self._attr_name = "Load Forecast"
        self._attr_unique_id = f"{DOMAIN}_{region}_{SENSOR_LOAD_FORECAST}"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, region)},
            name=f"NEM Price Forecaster ({region})",
            manufacturer="AEMO NEMWeb",
            model="PD7DAY Predispatch + Sidecar ML Calibration",
            entry_type="service",
        )

    @property
    def coordinator_data(self) -> CoordinatorData | None:
        return self.coordinator.data

    def _current_load_slot(self):
        """Return the load slot that covers the current moment."""
        if not self.coordinator_data:
            return None
        now_utc = datetime.now(timezone.utc)
        current_slot = None
        for load_slot in self.coordinator_data.load_forecast_slots:
            if load_slot.interval_start_utc <= now_utc:
                current_slot = load_slot
            else:
                break
        return current_slot

    @property
    def native_value(self) -> float | None:
        current_slot = self._current_load_slot()
        if current_slot is None:
            return None
        return round(current_slot.load_watts, 1)

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        data = self.coordinator_data
        if data is None:
            return {}

        attrs: dict[str, Any] = {
            ATTR_REGION: data.region,
            ATTR_LOAD_MODEL_NAME: data.load_model_name,
            ATTR_LOAD_IS_TRAINED: data.load_is_trained,
            ATTR_LOAD_TRAINING_OBSERVATIONS: data.load_training_observations,
            "forecast_horizon_hours": data.forecast_horizon_hours,
            "forecast_period_minutes": data.forecast_period_minutes,
            # Resampled load forecast at the user-configured period and horizon
            ATTR_LOAD_FORECAST: data.resampled_load_forecast,
        }

        current_slot = self._current_load_slot()
        if current_slot:
            attrs["current_interval_start"] = current_slot.interval_start_utc.isoformat()

        return attrs
