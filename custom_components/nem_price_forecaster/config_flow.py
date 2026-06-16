"""
Config flow for NEM Price Forecaster.

Sets up the integration via the HA UI.  The flow collects:

  Step 1 — sidecar: sidecar URL + region + price model
  Step 2 — tariff: GST rate, fixed adder, feed-in toggle
  Step 3 — tou_bands: ToU network band definitions (may be skipped / empty)
  Step 4 — forecast: forecast horizon (hours) + forecast period (resolution)

The integration is now a thin HTTP client for the sidecar service.
No darts/sklearn/scipy is imported by the integration (Python 3.14 compatible).

Note: demand charges ($/kVA peak demand) cannot be expressed as per-kWh slot
forecasts and are therefore out of scope for this integration.
"""

from __future__ import annotations

import json
import logging
import voluptuous as vol

from homeassistant import config_entries
from homeassistant.core import callback
from homeassistant.data_entry_flow import FlowResult

from .const import (
    CONF_BAND_NAME,
    CONF_BAND_RATE_PER_KWH,
    CONF_BAND_WINDOWS,
    CONF_CALIBRATION_MIN_OBSERVATIONS,
    CONF_CALIBRATION_WINDOW_DAYS,
    CONF_FEED_IN_IS_WHOLESALE,
    CONF_FIXED_ADDER_PER_KWH,
    CONF_FORECAST_HORIZON_HOURS,
    CONF_FORECAST_PERIOD_MINUTES,
    CONF_GST_RATE,
    CONF_PLAUSIBILITY_CAP_DOLLARS_PER_KWH,
    CONF_REGION,
    CONF_SIDECAR_URL,
    CONF_PRICE_MODEL,
    CONF_TOU_BANDS,
    DEFAULT_CALIBRATION_MIN_OBSERVATIONS,
    DEFAULT_CALIBRATION_WINDOW_DAYS,
    DEFAULT_FEED_IN_IS_WHOLESALE,
    DEFAULT_FIXED_ADDER_PER_KWH,
    DEFAULT_FORECAST_HORIZON_HOURS,
    DEFAULT_FORECAST_PERIOD_MINUTES,
    DEFAULT_GST_RATE,
    DEFAULT_PLAUSIBILITY_CAP_DOLLARS_PER_KWH,
    DEFAULT_SIDECAR_URL,
    DEFAULT_PRICE_MODEL,
    DOMAIN,
    FORECAST_PERIOD_OPTIONS,
    NEM_REGIONS,
    PRICE_MODEL_ISOTONIC,
    PRICE_MODEL_DARTS,
)

_LOGGER = logging.getLogger(__name__)

# Example ToU bands JSON shown in the UI hint
_TOU_BANDS_EXAMPLE = json.dumps(
    [
        {
            "name": "peak",
            "rate_per_kwh": 0.12,
            "windows": [
                {"days": [0, 1, 2, 3, 4], "start": "07:00", "end": "21:00"}
            ],
        },
        {
            "name": "shoulder",
            "rate_per_kwh": 0.08,
            "windows": [
                {"days": [5, 6], "start": "07:00", "end": "21:00"}
            ],
        },
    ],
    indent=2,
)


class NemPriceForecastConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for NEM Price Forecaster."""

    VERSION = 2  # bumped for sidecar architecture

    def __init__(self) -> None:
        self._config_data: dict = {}

    # ------------------------------------------------------------------
    # Step 1: Sidecar URL + region + price model
    # ------------------------------------------------------------------

    async def async_step_user(
        self, user_input: dict | None = None
    ) -> FlowResult:
        """
        Collect sidecar URL, NEM region, and price model selection.

        Region auto-detection: HA's latitude/longitude is used to suggest the
        most likely NEM region.  Users can override.
        """
        errors: dict[str, str] = {}

        if user_input is not None:
            # Validate sidecar connectivity (best-effort — non-blocking if sidecar
            # isn't running yet; user can proceed and sensors will be unavailable)
            sidecar_url = user_input.get(CONF_SIDECAR_URL, DEFAULT_SIDECAR_URL)
            if isinstance(sidecar_url, str):
                sidecar_url = sidecar_url.rstrip("/")
            else:
                sidecar_url = DEFAULT_SIDECAR_URL
            connectivity_error = await self._async_test_sidecar_connectivity(sidecar_url)
            if connectivity_error:
                errors[CONF_SIDECAR_URL] = connectivity_error
            else:
                self._config_data.update(user_input)
                self._config_data[CONF_SIDECAR_URL] = sidecar_url
                return await self.async_step_tariff()

        # Suggest region from HA lat/long (hass attribute may be None in stubs/tests)
        suggested_region = _suggest_region_from_hass(getattr(self, "hass", None))

        schema = vol.Schema(
            {
                vol.Optional(CONF_SIDECAR_URL, default=DEFAULT_SIDECAR_URL): str,
                vol.Required(CONF_REGION, default=suggested_region): vol.In(NEM_REGIONS),
                vol.Optional(
                    CONF_PRICE_MODEL,
                    default=DEFAULT_PRICE_MODEL,
                ): vol.In([PRICE_MODEL_ISOTONIC, PRICE_MODEL_DARTS]),
                vol.Optional(
                    CONF_CALIBRATION_WINDOW_DAYS,
                    default=DEFAULT_CALIBRATION_WINDOW_DAYS,
                ): vol.All(vol.Coerce(int), vol.Range(min=7, max=365)),
                vol.Optional(
                    CONF_CALIBRATION_MIN_OBSERVATIONS,
                    default=DEFAULT_CALIBRATION_MIN_OBSERVATIONS,
                ): vol.All(vol.Coerce(int), vol.Range(min=2, max=100)),
                vol.Optional(
                    CONF_PLAUSIBILITY_CAP_DOLLARS_PER_KWH,
                    default=DEFAULT_PLAUSIBILITY_CAP_DOLLARS_PER_KWH,
                ): vol.All(vol.Coerce(float), vol.Range(min=0.5, max=50.0)),
            }
        )

        return self.async_show_form(
            step_id="user",
            data_schema=schema,
            errors=errors,
            description_placeholders={
                "regions": ", ".join(NEM_REGIONS),
                "default_sidecar_url": DEFAULT_SIDECAR_URL,
            },
        )

    async def _async_test_sidecar_connectivity(self, sidecar_url: str) -> str | None:
        """
        Try to reach the sidecar /health endpoint.

        Returns an error key string if unreachable, None if OK.

        Offline-first policy: if the URL points to localhost / 127.0.0.1, we
        return None immediately (sidecar may not be running yet during setup;
        the coordinator will mark sensors unavailable if needed). Only
        non-localhost URLs trigger a live connectivity check during setup.

        Silently passes on any import / unexpected error so setup is never
        permanently blocked.
        """
        # Localhost is always considered OK (offline-first; don't block initial setup)
        if any(
            host in sidecar_url
            for host in ("localhost", "127.0.0.1", "::1")
        ):
            return None

        try:
            from .sidecar_client import SidecarClient, SidecarUnavailable
            test_client = SidecarClient(sidecar_url)
            try:
                await test_client.async_get_health()
                return None
            except SidecarUnavailable:
                return "cannot_connect"
            finally:
                await test_client.async_close()
        except Exception:
            # Don't block setup on import / unexpected errors
            return None

    # ------------------------------------------------------------------
    # Step 2: Tariff settings
    # ------------------------------------------------------------------

    async def async_step_tariff(
        self, user_input: dict | None = None
    ) -> FlowResult:
        errors: dict[str, str] = {}

        if user_input is not None:
            self._config_data.update(user_input)
            return await self.async_step_tou_bands()

        schema = vol.Schema(
            {
                vol.Optional(CONF_GST_RATE, default=DEFAULT_GST_RATE): vol.All(
                    vol.Coerce(float), vol.Range(min=0.0, max=0.5)
                ),
                vol.Optional(
                    CONF_FIXED_ADDER_PER_KWH, default=DEFAULT_FIXED_ADDER_PER_KWH
                ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=2.0)),
                vol.Optional(
                    CONF_FEED_IN_IS_WHOLESALE, default=DEFAULT_FEED_IN_IS_WHOLESALE
                ): bool,
            }
        )

        return self.async_show_form(
            step_id="tariff",
            data_schema=schema,
            errors=errors,
        )

    # ------------------------------------------------------------------
    # Step 3: ToU bands (JSON text entry)
    # ------------------------------------------------------------------

    async def async_step_tou_bands(
        self, user_input: dict | None = None
    ) -> FlowResult:
        errors: dict[str, str] = {}

        if user_input is not None:
            raw_json: str = user_input.get("tou_bands_json", "[]").strip()
            if not raw_json:
                raw_json = "[]"
            try:
                parsed_bands = json.loads(raw_json)
                if not isinstance(parsed_bands, list):
                    raise ValueError("Expected a JSON array")
                self._config_data[CONF_TOU_BANDS] = parsed_bands
            except (json.JSONDecodeError, ValueError) as json_error:
                errors["tou_bands_json"] = "invalid_json"
                _LOGGER.debug("ToU bands JSON parse error: %s", json_error)
            else:
                return await self.async_step_forecast()

        schema = vol.Schema(
            {
                vol.Optional("tou_bands_json", default="[]"): str,
            }
        )

        return self.async_show_form(
            step_id="tou_bands",
            data_schema=schema,
            errors=errors,
            description_placeholders={
                "example": _TOU_BANDS_EXAMPLE,
            },
        )

    # ------------------------------------------------------------------
    # Step 4: Forecast horizon + period
    # ------------------------------------------------------------------

    async def async_step_forecast(
        self, user_input: dict | None = None
    ) -> FlowResult:
        errors: dict[str, str] = {}

        if user_input is not None:
            self._config_data[CONF_FORECAST_HORIZON_HOURS] = int(
                user_input.get(CONF_FORECAST_HORIZON_HOURS, DEFAULT_FORECAST_HORIZON_HOURS)
            )
            self._config_data[CONF_FORECAST_PERIOD_MINUTES] = int(
                user_input.get(CONF_FORECAST_PERIOD_MINUTES, DEFAULT_FORECAST_PERIOD_MINUTES)
            )
            region = self._config_data[CONF_REGION]
            await self.async_set_unique_id(f"{DOMAIN}_{region}")
            self._abort_if_unique_id_configured()
            return self.async_create_entry(
                title=f"NEM Price Forecaster ({region})",
                data=self._config_data,
            )

        schema = vol.Schema(
            {
                vol.Optional(
                    CONF_FORECAST_HORIZON_HOURS,
                    default=DEFAULT_FORECAST_HORIZON_HOURS,
                ): vol.All(vol.Coerce(int), vol.Range(min=1, max=336)),
                vol.Optional(
                    CONF_FORECAST_PERIOD_MINUTES,
                    default=DEFAULT_FORECAST_PERIOD_MINUTES,
                ): vol.In(FORECAST_PERIOD_OPTIONS),
            }
        )

        return self.async_show_form(
            step_id="forecast",
            data_schema=schema,
            errors=errors,
            description_placeholders={
                "period_options": ", ".join(str(minutes) for minutes in FORECAST_PERIOD_OPTIONS),
            },
        )

    # ------------------------------------------------------------------
    # Options flow (allow reconfiguring tariff after setup)
    # ------------------------------------------------------------------

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> "NemPriceForecastOptionsFlow":
        return NemPriceForecastOptionsFlow(config_entry)


class NemPriceForecastOptionsFlow(config_entries.OptionsFlow):
    """Handle options (reconfigure tariff + forecast settings without re-adding)."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        self._config_entry = config_entry

    async def async_step_init(
        self, user_input: dict | None = None
    ) -> FlowResult:
        errors: dict[str, str] = {}
        current_data = self._config_entry.data

        if user_input is not None:
            # Validate ToU bands JSON if provided
            raw_bands_json = user_input.pop("tou_bands_json", "")
            if raw_bands_json.strip():
                try:
                    parsed_bands = json.loads(raw_bands_json)
                    user_input[CONF_TOU_BANDS] = parsed_bands
                except json.JSONDecodeError:
                    errors["tou_bands_json"] = "invalid_json"
            else:
                user_input[CONF_TOU_BANDS] = current_data.get(CONF_TOU_BANDS, [])

            if not errors:
                return self.async_create_entry(title="", data=user_input)

        schema = vol.Schema(
            {
                vol.Optional(
                    CONF_GST_RATE,
                    default=current_data.get(CONF_GST_RATE, DEFAULT_GST_RATE),
                ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=0.5)),
                vol.Optional(
                    CONF_FIXED_ADDER_PER_KWH,
                    default=current_data.get(
                        CONF_FIXED_ADDER_PER_KWH, DEFAULT_FIXED_ADDER_PER_KWH
                    ),
                ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=2.0)),
                vol.Optional(
                    CONF_FEED_IN_IS_WHOLESALE,
                    default=current_data.get(
                        CONF_FEED_IN_IS_WHOLESALE, DEFAULT_FEED_IN_IS_WHOLESALE
                    ),
                ): bool,
                vol.Optional(
                    CONF_CALIBRATION_WINDOW_DAYS,
                    default=current_data.get(
                        CONF_CALIBRATION_WINDOW_DAYS, DEFAULT_CALIBRATION_WINDOW_DAYS
                    ),
                ): vol.All(vol.Coerce(int), vol.Range(min=7, max=365)),
                vol.Optional(
                    CONF_FORECAST_HORIZON_HOURS,
                    default=current_data.get(
                        CONF_FORECAST_HORIZON_HOURS, DEFAULT_FORECAST_HORIZON_HOURS
                    ),
                ): vol.All(vol.Coerce(int), vol.Range(min=1, max=336)),
                vol.Optional(
                    CONF_FORECAST_PERIOD_MINUTES,
                    default=current_data.get(
                        CONF_FORECAST_PERIOD_MINUTES, DEFAULT_FORECAST_PERIOD_MINUTES
                    ),
                ): vol.In(FORECAST_PERIOD_OPTIONS),
                vol.Optional("tou_bands_json", default=""): str,
            }
        )

        return self.async_show_form(
            step_id="init",
            data_schema=schema,
            errors=errors,
        )


# ---------------------------------------------------------------------------
# Region auto-detection from HA latitude/longitude
# ---------------------------------------------------------------------------

# Approximate bounding boxes for NEM regions (lat, lon ranges, Australia)
_REGION_BOUNDING_BOXES = [
    ("QLD1", (-29.0, -10.0, 138.0, 154.0)),   # Queensland
    ("NSW1", (-38.0, -28.0, 141.0, 154.0)),   # NSW + ACT
    ("VIC1", (-39.5, -34.0, 140.5, 150.0)),   # Victoria + ACT overlap
    ("SA1",  (-38.5, -26.0, 129.0, 141.5)),   # South Australia
    ("TAS1", (-43.7, -39.5, 144.0, 148.5)),   # Tasmania
]


def _suggest_region_from_hass(hass) -> str:
    """
    Return a suggested NEM region based on HA's configured latitude/longitude.

    Falls back to NSW1 if HA has no location configured or location is outside
    Australia.
    """
    try:
        if hass is None:
            return "NSW1"
        lat = hass.config.latitude
        lon = hass.config.longitude
        if lat is None or lon is None:
            return "NSW1"

        for region_code, (lat_min, lat_max, lon_min, lon_max) in _REGION_BOUNDING_BOXES:
            if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
                return region_code
    except Exception:
        pass

    return "NSW1"
