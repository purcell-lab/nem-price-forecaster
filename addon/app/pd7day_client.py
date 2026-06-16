"""
NEMWeb PD7DAY client.

Fetches the latest 7-day predispatch (PD7DAY) file from the AEMO NEMWeb
CURRENT directory, extracts PRICESOLUTION rows for the requested NEM region,
and returns a list of (interval_datetime_utc, rrp_per_mwh) tuples.

All INTERVAL_DATETIME values in the CSV are in NEM time (AEST = UTC+10 always;
NEM does NOT observe daylight saving).  We convert to UTC for internal use so
downstream components never have to think about timezone handling.

Caching:  the latest filename is remembered; if it hasn't changed the cached
result is returned so we don't re-download 4–5 MB on every HA coordinator poll.
"""

from __future__ import annotations

import io
import logging
import re
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

NEMWEB_PD7DAY_INDEX_URL = "https://nemweb.com.au/Reports/CURRENT/PD7Day/"

_LOGGER = logging.getLogger(__name__)

# Maximum retries for transient network errors before giving up and returning cache
_MAX_RETRIES = 2
# Initial back-off in seconds between retries (doubles each attempt)
_RETRY_BACKOFF_SECONDS = 2.0
# Maximum response body size in bytes — guard against rogue/oversized responses
_MAX_RESPONSE_BYTES = 20 * 1024 * 1024  # 20 MB

# NEM time is always UTC+10 (no DST)
NEM_TIMEZONE = timezone(timedelta(hours=10))

# Column indices within PRICESOLUTION D-rows (0-based after stripping record-type prefix)
# Header: I,PD7DAY,PRICESOLUTION,1,RUN_DATETIME,INTERVENTION,INTERVAL_DATETIME,REGIONID,RRP,...
_COL_RUN_DATETIME = 4
_COL_INTERVAL_DATETIME = 6
_COL_REGIONID = 7
_COL_RRP = 8


@dataclass
class PriceSlot:
    """A single 30-minute dispatch interval with its forecast RRP."""
    interval_start_utc: datetime   # UTC, tz-aware
    interval_start_nem: datetime   # NEM time (UTC+10), tz-aware — kept for hour-of-day bucketing
    rrp_per_mwh: float             # $/MWh as published in PD7DAY


@dataclass
class Pd7DayForecast:
    """Full parsed PD7DAY forecast for one NEM region."""
    region: str
    run_datetime_utc: datetime
    slots: list[PriceSlot] = field(default_factory=list)


class Pd7DayClient:
    """
    Fetches and parses PD7DAY from NEMWeb.

    Thread-safety: not reentrant.  HA coordinators call this sequentially.
    """

    def __init__(self, timeout_seconds: int = 30) -> None:
        self._timeout_seconds = timeout_seconds
        self._cached_filename: Optional[str] = None
        self._cached_forecast: Optional[Pd7DayForecast] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_latest(self, region: str) -> Optional[Pd7DayForecast]:
        """
        Return the latest PD7DAY forecast for *region*.

        Implements bounded retries with exponential back-off for transient
        network errors (_MAX_RETRIES attempts before falling back to cache).

        Returns None only if both the network fetch AND the cache are
        unavailable — callers should treat None as a transient failure and
        retain whatever they had before.
        """
        # Sanitise region input
        region = str(region).strip().upper()

        # Discover latest filename with retry
        latest_filename: Optional[str] = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                latest_filename = self._discover_latest_filename()
                break
            except (URLError, HTTPError) as network_error:
                if attempt < _MAX_RETRIES:
                    backoff = _RETRY_BACKOFF_SECONDS * (2 ** attempt)
                    _LOGGER.warning(
                        "PD7DAY index fetch attempt %d/%d failed (%s); "
                        "retrying in %.0fs",
                        attempt + 1,
                        _MAX_RETRIES + 1,
                        network_error,
                        backoff,
                    )
                    time.sleep(backoff)
                else:
                    _LOGGER.warning(
                        "PD7DAY index fetch failed after %d attempts (%s); "
                        "returning cached forecast if available",
                        _MAX_RETRIES + 1,
                        network_error,
                    )
            except Exception as discovery_error:
                _LOGGER.warning(
                    "PD7DAY index fetch failed (%s); returning cached forecast if available",
                    discovery_error,
                )
                break

        if latest_filename is None:
            return self._cached_forecast_for_region(region)

        if latest_filename == self._cached_filename and self._cached_forecast is not None:
            if self._cached_forecast.region == region:
                _LOGGER.debug("PD7DAY cache hit for %s (%s)", region, latest_filename)
                return self._cached_forecast

        try:
            raw_bytes = self._download_zip(latest_filename)
            forecast = self._parse_zip(raw_bytes, region, latest_filename)
        except Exception as download_error:
            _LOGGER.warning(
                "PD7DAY download/parse failed for %s (%s); returning cached forecast",
                latest_filename,
                download_error,
            )
            return self._cached_forecast_for_region(region)

        self._cached_filename = latest_filename
        self._cached_forecast = forecast
        _LOGGER.info(
            "PD7DAY loaded: %s, region=%s, %d slots, run=%s",
            latest_filename,
            region,
            len(forecast.slots),
            forecast.run_datetime_utc.isoformat(),
        )
        return forecast

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _discover_latest_filename(self) -> str:
        """Scrape the NEMWeb index page and return the filename of the newest ZIP."""
        request = Request(
            NEMWEB_PD7DAY_INDEX_URL,
            headers={"User-Agent": "ha-nem-price-forecaster/0.1"},
        )
        with urlopen(request, timeout=self._timeout_seconds) as response:
            index_html = response.read().decode("utf-8", errors="replace")

        # Filenames look like: PUBLIC_PD7DAY_20260608124106_0000000521459051.zip
        all_filenames = re.findall(
            r'PUBLIC_PD7DAY_\d{14}_\d+\.zip',
            index_html,
            re.IGNORECASE,
        )
        if not all_filenames:
            raise ValueError("No PD7DAY ZIP files found on NEMWeb index page")

        # The filename timestamp is YYYYMMDDHHMMSS — sort lexicographically to get newest
        latest_filename = sorted(all_filenames)[-1]
        return latest_filename

    def _download_zip(self, filename: str) -> bytes:
        """
        Download a PD7DAY ZIP file and return its raw bytes.

        Bounded: response body is capped at _MAX_RESPONSE_BYTES to protect
        against malformed or hostile responses.
        """
        url = NEMWEB_PD7DAY_INDEX_URL + filename
        request = Request(url, headers={"User-Agent": "ha-nem-price-forecaster/0.1"})
        with urlopen(request, timeout=self._timeout_seconds) as response:
            raw_bytes = response.read(_MAX_RESPONSE_BYTES)
        if len(raw_bytes) == _MAX_RESPONSE_BYTES:
            _LOGGER.warning(
                "PD7DAY ZIP response for %s hit the %d-byte cap; file may be truncated",
                filename,
                _MAX_RESPONSE_BYTES,
            )
        return raw_bytes

    def _parse_zip(self, zip_bytes: bytes, region: str, filename: str) -> Pd7DayForecast:
        """
        Extract the single CSV from *zip_bytes* and parse PRICESOLUTION rows
        for *region*.

        Returns a Pd7DayForecast (may have zero slots if the region is absent,
        though that would be unexpected).
        """
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zip_file:
            csv_names = [name for name in zip_file.namelist() if name.upper().endswith(".CSV")]
            if not csv_names:
                raise ValueError(f"No CSV found inside {filename}")
            # There is always exactly one CSV inside a PD7DAY ZIP
            csv_text = zip_file.read(csv_names[0]).decode("utf-8", errors="replace")

        return self._parse_csv(csv_text, region)

    def _parse_csv(self, csv_text: str, region: str) -> Pd7DayForecast:
        """
        Parse the AEMO MMSDM-format CSV.

        The file uses a non-standard multi-table layout:
          C-rows  = file header
          I-rows  = column headers for the next D-row block
          D-rows  = data rows
          END     = end-of-file marker

        We only care about D-rows in the PRICESOLUTION table.
        """
        slots: list[PriceSlot] = []
        run_datetime_utc: Optional[datetime] = None
        in_price_solution = False

        for raw_line in csv_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("I,PD7DAY,PRICESOLUTION"):
                in_price_solution = True
                continue
            elif line.startswith("I,"):
                # Entering a different table
                in_price_solution = False
                continue

            if not (in_price_solution and line.startswith("D,")):
                continue

            # Split preserving quoted commas (AEMO datetimes are quoted)
            columns = _split_mmsdm_row(line)
            if len(columns) < _COL_RRP + 1:
                continue

            row_region = columns[_COL_REGIONID].strip().upper()
            if row_region != region.upper():
                continue

            try:
                interval_nem = _parse_nem_datetime(columns[_COL_INTERVAL_DATETIME])
                interval_utc = interval_nem.astimezone(timezone.utc)
                rrp = float(columns[_COL_RRP])
            except (ValueError, IndexError) as parse_error:
                _LOGGER.debug("Skipping malformed PRICESOLUTION row: %s", parse_error)
                continue

            if run_datetime_utc is None:
                try:
                    run_nem = _parse_nem_datetime(columns[_COL_RUN_DATETIME])
                    run_datetime_utc = run_nem.astimezone(timezone.utc)
                except (ValueError, IndexError):
                    run_datetime_utc = datetime.now(timezone.utc)

            slots.append(
                PriceSlot(
                    interval_start_utc=interval_utc,
                    interval_start_nem=interval_nem,
                    rrp_per_mwh=rrp,
                )
            )

        if run_datetime_utc is None:
            run_datetime_utc = datetime.now(timezone.utc)

        # Sort by interval time (should already be ordered, but be defensive)
        slots.sort(key=lambda price_slot: price_slot.interval_start_utc)

        return Pd7DayForecast(
            region=region,
            run_datetime_utc=run_datetime_utc,
            slots=slots,
        )

    def _cached_forecast_for_region(self, region: str) -> Optional[Pd7DayForecast]:
        """Return the in-memory cache if it matches the requested region."""
        if self._cached_forecast is not None and self._cached_forecast.region == region:
            return self._cached_forecast
        return None


# ---------------------------------------------------------------------------
# Module-level parsing helpers
# ---------------------------------------------------------------------------

def _split_mmsdm_row(line: str) -> list[str]:
    """
    Split a comma-delimited AEMO MMSDM row, respecting double-quoted fields.

    AEMO quotes datetime fields like "2026/06/08 13:00:00" which contain no
    commas, so a simple quote-aware split is sufficient (no escaped quotes).
    """
    columns: list[str] = []
    current_field = []
    inside_quotes = False

    for character in line:
        if character == '"':
            inside_quotes = not inside_quotes
        elif character == ',' and not inside_quotes:
            columns.append(''.join(current_field))
            current_field = []
        else:
            current_field.append(character)

    columns.append(''.join(current_field))
    return columns


def _parse_nem_datetime(raw_value: str) -> datetime:
    """
    Parse an AEMO datetime string of the form "YYYY/MM/DD HH:MM:SS"
    and return a timezone-aware datetime in NEM time (UTC+10).
    """
    cleaned = raw_value.strip().strip('"')
    naive_dt = datetime.strptime(cleaned, "%Y/%m/%d %H:%M:%S")
    return naive_dt.replace(tzinfo=NEM_TIMEZONE)
