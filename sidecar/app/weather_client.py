"""
Open-Meteo weather client for the NEM Price Forecaster sidecar.

Fetches weather data from two Open-Meteo endpoints:
  - Forecast API  (api.open-meteo.com)   → future weather for prediction
  - Archive API   (archive-api.open-meteo.com) → past weather for training

Both are free, require no API key, and are offline-safe (errors fall back to
a no-weather path with a logged WARNING rather than crashing the forecast).

Variables fetched (price- and load-relevant):
  temperature_2m        °C — drives heating/cooling load + gas/peaker price spikes
  cloud_cover           %  — solar generation proxy → renewable penetration → price
  shortwave_radiation   W/m² — direct GHI solar input (more precise than cloud cover)
  wind_speed_10m        m/s — wind generation proxy (NEM wind penetration is significant)
  relative_humidity_2m  %  — comfort-driven load, correlates with temperature effect

Design decisions
----------------
- All timestamps are stored and returned as UTC-aware datetimes internally.
- Open-Meteo timestamps in the JSON response are "naive" UTC strings (no TZ suffix);
  we attach UTC timezone on parse.
- Forecast API returns hourly data from now through 16 days ahead.
  Archive API returns historical hourly data from 1940 through ~5 days ago (data lag).
- We request hourly resolution and forward-fill to 30-min slots on demand.
- Simple in-memory cache: fetched data is cached for _CACHE_TTL_MINUTES.
  The cache is intentionally short so the forecast window stays current.
- On any API error: log a WARNING, return empty lists, let callers use no-weather path.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta, timezone
from typing import Optional
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
import json

_LOGGER = logging.getLogger(__name__)

# Open-Meteo API base URLs
_FORECAST_BASE_URL = "https://api.open-meteo.com/v1/forecast"
_ARCHIVE_BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Weather variables to fetch — chosen for NEM price and load prediction
_WEATHER_VARIABLES = [
    "temperature_2m",
    "cloud_cover",
    "shortwave_radiation",
    "wind_speed_10m",
    "relative_humidity_2m",
]

# Cache TTL: re-use weather data for this many minutes before re-fetching
_CACHE_TTL_MINUTES = 30

# Request timeout and retry settings
_REQUEST_TIMEOUT_SECONDS = 20
_MAX_RETRIES = 2
_RETRY_BACKOFF_SECONDS = 2.0

# Maximum reasonable response size (10 MB)
_MAX_RESPONSE_BYTES = 10 * 1024 * 1024

# Default lat/lon per NEM dispatch region — representative capital city locations
# These provide sensible weather unless the user overrides with lat/lon config.
#   QLD1: Brisbane, NSW1: Sydney, VIC1: Melbourne, SA1: Adelaide, TAS1: Hobart
NEM_REGION_COORDS: dict[str, tuple[float, float]] = {
    "QLD1": (-27.47, 153.03),
    "NSW1": (-33.87, 151.21),
    "VIC1": (-37.81, 144.96),
    "SA1":  (-34.93, 138.60),
    "TAS1": (-42.88, 147.33),
}


@dataclass
class WeatherSlot:
    """One hourly weather observation or forecast."""
    time_utc: datetime               # UTC-aware
    temperature_celsius: float       # °C
    cloud_cover_percent: float       # 0–100
    shortwave_radiation_wm2: float   # W/m²
    wind_speed_ms: float             # m/s
    relative_humidity_percent: float # 0–100

    def as_dict(self) -> dict[str, float]:
        return {
            "temperature_2m": self.temperature_celsius,
            "cloud_cover": self.cloud_cover_percent,
            "shortwave_radiation": self.shortwave_radiation_wm2,
            "wind_speed_10m": self.wind_speed_ms,
            "relative_humidity_2m": self.relative_humidity_percent,
        }


@dataclass
class _WeatherCache:
    """Simple time-bounded cache for one weather fetch result."""
    slots: list[WeatherSlot] = field(default_factory=list)
    fetched_at: float = 0.0  # time.monotonic()


class OpenMeteoClient:
    """
    Fetches weather forecasts and historical data from the Open-Meteo public API.

    Usage
    -----
    client = OpenMeteoClient(latitude=-27.47, longitude=153.03)

    # Future weather for prediction (hourly, covers forecast horizon)
    forecast_slots = client.fetch_forecast(hours_ahead=168)

    # Historical weather for training (hourly, covers the training window)
    archive_slots  = client.fetch_archive(start_date=date(2024,1,1), end_date=date.today())

    Error handling
    --------------
    All public methods return an empty list on any API failure — callers must be
    prepared to receive [] and fall back to no-weather training/prediction.
    """

    def __init__(
        self,
        latitude: float,
        longitude: float,
        cache_ttl_minutes: int = _CACHE_TTL_MINUTES,
    ) -> None:
        self._latitude = latitude
        self._longitude = longitude
        self._cache_ttl_seconds = cache_ttl_minutes * 60

        self._forecast_cache = _WeatherCache()
        self._archive_cache: dict[tuple[date, date], _WeatherCache] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_forecast(self, hours_ahead: int = 168) -> list[WeatherSlot]:
        """
        Fetch future hourly weather from Open-Meteo forecast API.

        Returns hourly slots from now through now + hours_ahead.
        Returns [] on any network or parse error.

        Results are cached for cache_ttl_minutes; repeat calls within the TTL
        return the cached result without hitting the network.
        """
        elapsed_seconds = time.monotonic() - self._forecast_cache.fetched_at
        if (
            self._forecast_cache.slots
            and elapsed_seconds < self._cache_ttl_seconds
        ):
            _LOGGER.debug(
                "OpenMeteo forecast cache hit (age=%.0fs, ttl=%.0fs)",
                elapsed_seconds,
                self._cache_ttl_seconds,
            )
            return self._forecast_cache.slots

        # Clamp request to Open-Meteo's 16-day forecast limit
        forecast_days = min(16, max(1, (hours_ahead + 23) // 24))
        url = self._build_forecast_url(forecast_days)

        try:
            raw_bytes = self._fetch_url(url)
            slots = self._parse_response(raw_bytes)
        except Exception as fetch_error:
            _LOGGER.warning(
                "OpenMeteo forecast fetch failed (lat=%.4f, lon=%.4f): %s — "
                "proceeding without weather covariates for this cycle",
                self._latitude,
                self._longitude,
                fetch_error,
            )
            return []

        self._forecast_cache = _WeatherCache(slots=slots, fetched_at=time.monotonic())
        _LOGGER.info(
            "OpenMeteo forecast fetched: %d hourly slots (lat=%.4f, lon=%.4f)",
            len(slots),
            self._latitude,
            self._longitude,
        )
        return slots

    def fetch_archive(
        self,
        start_date: date,
        end_date: date,
    ) -> list[WeatherSlot]:
        """
        Fetch historical hourly weather from Open-Meteo archive API.

        start_date / end_date are inclusive; archive data is available from
        1940-01-01 through approximately 5 days ago (ECMWF ERA5 data lag).

        Returns [] on any network or parse error.

        Results are cached by (start_date, end_date) pair within this session
        (archive data does not change, so the TTL is infinite).
        """
        cache_key = (start_date, end_date)
        if cache_key in self._archive_cache and self._archive_cache[cache_key].slots:
            _LOGGER.debug(
                "OpenMeteo archive cache hit for %s to %s",
                start_date.isoformat(),
                end_date.isoformat(),
            )
            return self._archive_cache[cache_key].slots

        url = self._build_archive_url(start_date, end_date)
        try:
            raw_bytes = self._fetch_url(url)
            slots = self._parse_response(raw_bytes)
        except Exception as fetch_error:
            _LOGGER.warning(
                "OpenMeteo archive fetch failed (lat=%.4f, lon=%.4f, %s to %s): %s — "
                "training will proceed without weather covariates for this window",
                self._latitude,
                self._longitude,
                start_date.isoformat(),
                end_date.isoformat(),
                fetch_error,
            )
            return []

        self._archive_cache[cache_key] = _WeatherCache(
            slots=slots,
            fetched_at=time.monotonic(),
        )
        _LOGGER.info(
            "OpenMeteo archive fetched: %d hourly slots (%s to %s, lat=%.4f, lon=%.4f)",
            len(slots),
            start_date.isoformat(),
            end_date.isoformat(),
            self._latitude,
            self._longitude,
        )
        return slots

    def slots_to_30min_map(
        self,
        slots: list[WeatherSlot],
    ) -> dict[datetime, WeatherSlot]:
        """
        Convert hourly WeatherSlots to a UTC-datetime-keyed dict at 30-min resolution
        using forward-fill so every 30-min boundary has a value.

        Each hourly slot fills both the :00 and :30 boundaries within that hour.
        Returns an empty dict for empty input.
        """
        if not slots:
            return {}

        result: dict[datetime, WeatherSlot] = {}
        for hourly_slot in sorted(slots, key=lambda s: s.time_utc):
            # Forward-fill to :00 and :30 within this hour
            hour_base = hourly_slot.time_utc.replace(minute=0, second=0, microsecond=0)
            result[hour_base] = hourly_slot
            result[hour_base + timedelta(minutes=30)] = hourly_slot
        return result

    # ------------------------------------------------------------------
    # URL builders
    # ------------------------------------------------------------------

    def _build_forecast_url(self, forecast_days: int) -> str:
        variables_str = ",".join(_WEATHER_VARIABLES)
        return (
            f"{_FORECAST_BASE_URL}"
            f"?latitude={self._latitude:.6f}"
            f"&longitude={self._longitude:.6f}"
            f"&hourly={variables_str}"
            f"&forecast_days={forecast_days}"
            f"&timezone=UTC"
        )

    def _build_archive_url(self, start_date: date, end_date: date) -> str:
        variables_str = ",".join(_WEATHER_VARIABLES)
        return (
            f"{_ARCHIVE_BASE_URL}"
            f"?latitude={self._latitude:.6f}"
            f"&longitude={self._longitude:.6f}"
            f"&hourly={variables_str}"
            f"&start_date={start_date.isoformat()}"
            f"&end_date={end_date.isoformat()}"
            f"&timezone=UTC"
        )

    # ------------------------------------------------------------------
    # HTTP fetch with retry
    # ------------------------------------------------------------------

    def _fetch_url(self, url: str) -> bytes:
        """
        GET *url* with bounded retries and exponential back-off.

        Raises on final failure so callers can log a clean warning.
        """
        last_error: Optional[Exception] = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                request = Request(
                    url,
                    headers={"User-Agent": "ha-nem-price-forecaster/0.1"},
                )
                with urlopen(request, timeout=_REQUEST_TIMEOUT_SECONDS) as response:
                    raw_bytes = response.read(_MAX_RESPONSE_BYTES)
                return raw_bytes
            except (URLError, HTTPError) as network_error:
                last_error = network_error
                if attempt < _MAX_RETRIES:
                    backoff_seconds = _RETRY_BACKOFF_SECONDS * (2 ** attempt)
                    _LOGGER.debug(
                        "OpenMeteo fetch attempt %d/%d failed (%s); retry in %.0fs",
                        attempt + 1,
                        _MAX_RETRIES + 1,
                        network_error,
                        backoff_seconds,
                    )
                    time.sleep(backoff_seconds)
            except Exception as unexpected_error:
                last_error = unexpected_error
                break  # non-network errors are not retried

        raise last_error or RuntimeError("OpenMeteo fetch failed (unknown error)")

    # ------------------------------------------------------------------
    # Response parser
    # ------------------------------------------------------------------

    def _parse_response(self, raw_bytes: bytes) -> list[WeatherSlot]:
        """
        Parse the Open-Meteo hourly JSON response into a list of WeatherSlot.

        Open-Meteo returns:
          {
            "hourly": {
              "time": ["2026-06-08T00:00", ...],
              "temperature_2m": [18.3, ...],
              ...
            }
          }

        All time strings are naive UTC (no TZ suffix) — we attach UTC timezone.
        Missing values appear as null in the JSON; we forward-fill from the last
        known value (or 0 if no prior value).
        """
        try:
            data = json.loads(raw_bytes.decode("utf-8", errors="replace"))
        except json.JSONDecodeError as json_error:
            raise ValueError(f"OpenMeteo response is not valid JSON: {json_error}") from json_error

        if "hourly" not in data:
            # Check for API error message
            error_reason = data.get("reason", data.get("error", "unknown API error"))
            raise ValueError(f"OpenMeteo response missing 'hourly' block: {error_reason}")

        hourly_block = data["hourly"]
        time_strings = hourly_block.get("time", [])
        if not time_strings:
            return []

        temperature_values = hourly_block.get("temperature_2m", [None] * len(time_strings))
        cloud_cover_values = hourly_block.get("cloud_cover", [None] * len(time_strings))
        radiation_values   = hourly_block.get("shortwave_radiation", [None] * len(time_strings))
        wind_speed_values  = hourly_block.get("wind_speed_10m", [None] * len(time_strings))
        humidity_values    = hourly_block.get("relative_humidity_2m", [None] * len(time_strings))

        # Pad each array to the length of time_strings (guard against truncated responses)
        n_slots = len(time_strings)
        temperature_values = _pad_to_length(temperature_values, n_slots)
        cloud_cover_values = _pad_to_length(cloud_cover_values, n_slots)
        radiation_values   = _pad_to_length(radiation_values, n_slots)
        wind_speed_values  = _pad_to_length(wind_speed_values, n_slots)
        humidity_values    = _pad_to_length(humidity_values, n_slots)

        slots: list[WeatherSlot] = []
        last_temp     = 20.0
        last_cloud    = 50.0
        last_radiation = 0.0
        last_wind     = 0.0
        last_humidity = 50.0

        for slot_index, time_string in enumerate(time_strings):
            try:
                # Open-Meteo returns "2026-06-08T00:00" (no seconds, no TZ)
                naive_dt = datetime.strptime(time_string, "%Y-%m-%dT%H:%M")
                slot_utc = naive_dt.replace(tzinfo=timezone.utc)
            except ValueError:
                _LOGGER.debug("OpenMeteo: skipping unparseable time string: %r", time_string)
                continue

            # Forward-fill nulls
            last_temp     = _coerce_float(temperature_values[slot_index], last_temp)
            last_cloud    = _coerce_float(cloud_cover_values[slot_index], last_cloud)
            last_radiation = _coerce_float(radiation_values[slot_index], last_radiation)
            last_wind     = _coerce_float(wind_speed_values[slot_index], last_wind)
            last_humidity = _coerce_float(humidity_values[slot_index], last_humidity)

            slots.append(WeatherSlot(
                time_utc=slot_utc,
                temperature_celsius=last_temp,
                cloud_cover_percent=last_cloud,
                shortwave_radiation_wm2=last_radiation,
                wind_speed_ms=last_wind,
                relative_humidity_percent=last_humidity,
            ))

        return slots


# ---------------------------------------------------------------------------
# Module-level convenience constructor
# ---------------------------------------------------------------------------

def build_weather_client_for_region(
    region: str,
    latitude_override: Optional[float] = None,
    longitude_override: Optional[float] = None,
) -> OpenMeteoClient:
    """
    Build an OpenMeteoClient for *region*.

    If lat/lon overrides are given they take precedence over the NEM-region
    defaults.  Unknown regions fall back to the NSW1 (Sydney) default.
    """
    default_lat, default_lon = NEM_REGION_COORDS.get(region, NEM_REGION_COORDS["NSW1"])
    lat = latitude_override if latitude_override is not None else default_lat
    lon = longitude_override if longitude_override is not None else default_lon
    return OpenMeteoClient(latitude=lat, longitude=lon)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pad_to_length(values: list, target_length: int) -> list:
    """Extend *values* with None entries if shorter than *target_length*."""
    if len(values) >= target_length:
        return values
    return list(values) + [None] * (target_length - len(values))


def _coerce_float(raw_value, fallback: float) -> float:
    """Return float(*raw_value*) or *fallback* if raw_value is None/non-numeric."""
    if raw_value is None:
        return fallback
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return fallback
