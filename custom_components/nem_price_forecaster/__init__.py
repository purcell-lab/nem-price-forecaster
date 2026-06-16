"""
NEM Price Forecaster — Home Assistant custom integration (sidecar mode).

The integration is now a thin HTTP client for the NEM forecaster sidecar.
All ML compute (PD7DAY fetch, isotonic/Darts calibration, tariff calculation,
load forecasting) runs in the sidecar container.

Python 3.14 compatible: no darts, no sklearn, no scipy imported here.
Only numpy + homeassistant builtins + aiohttp (bundled with HA).

Sidecar:
  See sidecar/ directory for the Dockerfile and docker-compose.yml.
  Default URL: http://localhost:8765

Sensors published:
  - sensor.nem_price_forecaster_{region}_import_price  ($/kWh, GST-inclusive)
  - sensor.nem_price_forecaster_{region}_export_price  ($/kWh, GST-excluded)
  - sensor.nem_price_forecaster_{region}_load_forecast (W, optional)

See README.md for full setup instructions.
"""

from __future__ import annotations

import logging

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady

from .const import DOMAIN, PLATFORMS
from .coordinator import NemPriceForecastCoordinator

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry) -> bool:
    """Set up NEM Price Forecaster from a config entry."""
    hass.data.setdefault(DOMAIN, {})

    coordinator = NemPriceForecastCoordinator(hass, config_entry)

    # Perform an initial data fetch from the sidecar.
    # Raises ConfigEntryNotReady if the sidecar is unreachable so HA retries.
    await coordinator.async_config_entry_first_refresh()

    hass.data[DOMAIN][config_entry.entry_id] = coordinator

    await hass.config_entries.async_forward_entry_setups(config_entry, PLATFORMS)

    # Register update listener so options-flow changes trigger a reload
    config_entry.async_on_unload(
        config_entry.add_update_listener(_async_update_listener)
    )

    return True


async def async_unload_entry(hass: HomeAssistant, config_entry: ConfigEntry) -> bool:
    """Unload a config entry — close the HTTP session."""
    coordinator: NemPriceForecastCoordinator = hass.data[DOMAIN].get(
        config_entry.entry_id
    )
    if coordinator is not None:
        await coordinator.async_close()

    unload_ok = await hass.config_entries.async_unload_platforms(
        config_entry, PLATFORMS
    )
    if unload_ok:
        hass.data[DOMAIN].pop(config_entry.entry_id)
    return unload_ok


async def _async_update_listener(
    hass: HomeAssistant, config_entry: ConfigEntry
) -> None:
    """Handle options update: reload the entry so the new tariff takes effect."""
    await hass.config_entries.async_reload(config_entry.entry_id)
