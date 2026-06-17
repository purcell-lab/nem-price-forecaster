"""
AEMO NEMWeb historical price ingestion for the NEM Price Forecaster sidecar.

Purpose
-------
The Darts price model and load forecaster benefit enormously from years of
training history — but new installs only have a few weeks of observations
accumulated via the /calibration/import POST endpoint.

This module provides two capabilities:

1. Historical wholesale price download (AEMO NEMWeb ARCHIVE):
   Downloads and parses Public_Prices (5-min DISPATCHPRICE) CSVs from
   nemweb.com.au/Reports/ARCHIVE/DispatchIS_Reports/  and aggregates them
   to 30-min TRADING intervals to align with PD7DAY / Darts training.

2. Historical load proxy (not directly available from AEMO in this module):
   Load data is user-specific and must come from the HA recorder.  This
   module focuses on the price side.  The weather_client.py archive API
   provides the matching historical weather for training.

AEMO NEMWeb price archive layout
---------------------------------
URL pattern:
  https://nemweb.com.au/Reports/ARCHIVE/DispatchIS_Reports/PUBLIC_DISPATCHIS_<YYYYMMDD>.zip

Each ZIP contains one or more CSV files with table "DISPATCH,PRICE":
  I row header: I,DISPATCH,PRICE,4,SETTLEMENTDATE,RUNNO,REGIONID,DISPATCHTYPE,
                BANDAVAIL1,...,RRP,...
  D rows:       D,DISPATCH,PRICE,4,"2024/01/01 00:05:00",1,NSW1,...,RRP_value,...

SETTLEMENTDATE is NEM time (UTC+10, no DST), represents the END of the 5-min
dispatch interval.  We convert to interval START by subtracting 5 minutes, then
aggregate to 30-min averages.

Column positions (0-based) in D-rows:
  [0]  "D"
  [1]  "DISPATCH"
  [2]  "PRICE"
  [3]  "4"          (report sub-type)
  [4]  SETTLEMENTDATE
  [5]  RUNNO
  [6]  REGIONID
  [7]  DISPATCHTYPE
  [8..19] BANDAVAIL1..12  (cleared MW by bid band — not used here)
  ...
  We locate RRP by the column header (position varies by report version);
  default position is index 8 after stripping the leading 4 fixed fields.

Offline-safety
--------------
All downloads are cached on disk in <data_dir>/aemo_archive/.
Re-runs within the same process session and across restarts skip already-cached
files.  On network error the module falls back to what's cached and logs a
WARNING — it never crashes the sidecar.

Usage
-----
from aemo_historical_client import AemoHistoricalClient

client = AemoHistoricalClient(data_dir="/data", region="QLD1")
price_obs = client.fetch_price_history(days_back=365)
# price_obs is a list of PriceHistorySlot (30-min UTC intervals, $/MWh)
"""

from __future__ import annotations

import io
import json
import logging
import os
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta, timezone
from typing import Optional
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

_LOGGER = logging.getLogger(__name__)

# NEMWeb archive base URL for DISPATCHIS daily ZIP files
_DISPATCHIS_BASE_URL = (
    "https://nemweb.com.au/Reports/ARCHIVE/DispatchIS_Reports/"
)

# NEM time is always UTC+10 (no DST)
_NEM_TZ = timezone(timedelta(hours=10))

# HTTP settings
_REQUEST_TIMEOUT_SECONDS = 30
_MAX_RETRIES = 2
_RETRY_BACKOFF_SECONDS = 3.0
_MAX_RESPONSE_BYTES = 50 * 1024 * 1024  # 50 MB per daily ZIP

# Maximum days to request in one fetch call (guard against huge downloads)
_MAX_DAYS_PER_FETCH = 730  # 2 years


@dataclass
class PriceHistorySlot:
    """
    One 30-minute wholesale price observation from AEMO DISPATCHPRICE data.

    The 30-min average RRP is computed from the six 5-min dispatch intervals
    within each trading period.
    """
    interval_start_utc: datetime  # UTC-aware, start of 30-min trading interval
    rrp_per_mwh: float            # $/MWh, average over the 30-min period


class AemoHistoricalClient:
    """
    Downloads and caches AEMO NEMWeb historical dispatch price data.

    Each call to fetch_price_history() returns 30-min RRP observations for the
    requested NEM region covering up to *days_back* days of history.  Files are
    cached locally to avoid redundant downloads.

    Error handling: on any per-day download/parse failure the day is skipped
    with a WARNING.  Partial results are returned; callers should handle the
    case where fewer days than requested are available.
    """

    def __init__(
        self,
        data_dir: str,
        region: str,
    ) -> None:
        self._data_dir = data_dir
        self._region = region.strip().upper()
        self._cache_dir = os.path.join(data_dir, "aemo_archive")
        os.makedirs(self._cache_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_price_history(
        self,
        days_back: int = 365,
    ) -> list[PriceHistorySlot]:
        """
        Return 30-min wholesale RRP observations for the last *days_back* days.

        Downloads are cached in <data_dir>/aemo_archive/ per region+date so
        repeat calls are fast.  Days already cached are never re-downloaded.

        Returns a chronologically sorted list of PriceHistorySlot.
        The list may cover fewer days than requested if some files are missing
        (public holidays / system gaps are rare but possible).
        """
        days_back = min(max(1, days_back), _MAX_DAYS_PER_FETCH)

        # AEMO's archive lag is roughly 2 business days; don't request today or yesterday
        end_date = date.today() - timedelta(days=2)
        start_date = end_date - timedelta(days=days_back - 1)

        all_slots: list[PriceHistorySlot] = []

        current_date = start_date
        while current_date <= end_date:
            day_slots = self._get_day(current_date)
            all_slots.extend(day_slots)
            current_date += timedelta(days=1)

        # Sort chronologically and return
        all_slots.sort(key=lambda slot: slot.interval_start_utc)
        _LOGGER.info(
            "AemoHistoricalClient: %d 30-min price slots for region=%s "
            "covering %s to %s",
            len(all_slots),
            self._region,
            start_date.isoformat(),
            end_date.isoformat(),
        )
        return all_slots

    def fetch_price_history_for_date_range(
        self,
        start_date: date,
        end_date: date,
    ) -> list[PriceHistorySlot]:
        """
        Return 30-min wholesale RRP observations for an explicit date range.

        Useful when the training window is known (e.g. to align with Open-Meteo
        archive data).  Returns [] if the range is in the future or beyond the
        archive lag.
        """
        today = date.today()
        effective_end = min(end_date, today - timedelta(days=2))
        if start_date > effective_end:
            return []

        all_slots: list[PriceHistorySlot] = []
        current_date = start_date
        while current_date <= effective_end:
            day_slots = self._get_day(current_date)
            all_slots.extend(day_slots)
            current_date += timedelta(days=1)

        all_slots.sort(key=lambda slot: slot.interval_start_utc)
        return all_slots

    # ------------------------------------------------------------------
    # Per-day fetch + cache
    # ------------------------------------------------------------------

    def _get_day(self, target_date: date) -> list[PriceHistorySlot]:
        """
        Return 30-min price slots for *target_date*, using on-disk cache if available.
        """
        cache_file_path = self._cache_path(target_date)
        if os.path.exists(cache_file_path):
            return self._load_from_cache(cache_file_path)

        # Attempt to download
        zip_bytes = self._download_day(target_date)
        if zip_bytes is None:
            return []

        slots = self._parse_dispatchis_zip(zip_bytes, target_date)
        if slots:
            self._save_to_cache(slots, cache_file_path)

        return slots

    def _cache_path(self, target_date: date) -> str:
        """Return the on-disk cache file path for *target_date* + region."""
        filename = f"price_{self._region}_{target_date.strftime('%Y%m%d')}.json"
        return os.path.join(self._cache_dir, filename)

    def _load_from_cache(self, cache_file_path: str) -> list[PriceHistorySlot]:
        """Load pre-parsed slots from on-disk JSON cache."""
        try:
            with open(cache_file_path, encoding="utf-8") as cache_file:
                raw_list = json.load(cache_file)
            slots = []
            for entry in raw_list:
                slot_utc = datetime.fromisoformat(entry["interval_start_utc"])
                slots.append(PriceHistorySlot(
                    interval_start_utc=slot_utc,
                    rrp_per_mwh=float(entry["rrp_per_mwh"]),
                ))
            return slots
        except Exception as cache_error:
            _LOGGER.debug(
                "AemoHistoricalClient: cache load failed for %s: %s",
                cache_file_path,
                cache_error,
            )
            return []

    def _save_to_cache(
        self,
        slots: list[PriceHistorySlot],
        cache_file_path: str,
    ) -> None:
        """Persist parsed slots to on-disk JSON cache (atomic write via temp file)."""
        try:
            raw_list = [
                {
                    "interval_start_utc": slot.interval_start_utc.isoformat(),
                    "rrp_per_mwh": slot.rrp_per_mwh,
                }
                for slot in slots
            ]
            tmp_path = cache_file_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as tmp_file:
                json.dump(raw_list, tmp_file)
            os.replace(tmp_path, cache_file_path)
        except Exception as save_error:
            _LOGGER.debug(
                "AemoHistoricalClient: cache write failed for %s: %s",
                cache_file_path,
                save_error,
            )

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def _download_day(self, target_date: date) -> Optional[bytes]:
        """
        Download the DISPATCHIS daily ZIP for *target_date*.

        Returns raw bytes on success, None on failure (logged as WARNING).
        Implements bounded retries with exponential back-off.
        """
        url = _build_dispatchis_url(target_date)
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
            except HTTPError as http_error:
                if http_error.code == 404:
                    # 404 = this day genuinely missing (public holidays, gaps)
                    _LOGGER.debug(
                        "AemoHistoricalClient: %s not found on NEMWeb (404) — skipping",
                        target_date.isoformat(),
                    )
                    return None
                last_error = http_error
            except (URLError, Exception) as network_error:
                last_error = network_error

            if attempt < _MAX_RETRIES:
                backoff_seconds = _RETRY_BACKOFF_SECONDS * (2 ** attempt)
                _LOGGER.debug(
                    "AemoHistoricalClient: download attempt %d/%d failed for %s (%s); "
                    "retry in %.0fs",
                    attempt + 1,
                    _MAX_RETRIES + 1,
                    target_date.isoformat(),
                    last_error,
                    backoff_seconds,
                )
                time.sleep(backoff_seconds)

        _LOGGER.warning(
            "AemoHistoricalClient: failed to download %s after %d attempts (%s); "
            "skipping this day",
            target_date.isoformat(),
            _MAX_RETRIES + 1,
            last_error,
        )
        return None

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_dispatchis_zip(
        self,
        zip_bytes: bytes,
        target_date: date,
    ) -> list[PriceHistorySlot]:
        """
        Extract and parse the DISPATCH,PRICE table from an AEMO DISPATCHIS ZIP.

        Each D-row represents a 5-min dispatch interval.  We:
          1. Filter to our region
          2. Convert SETTLEMENTDATE (end of interval, NEM time) → interval start UTC
          3. Group by 30-min trading period (6 × 5-min intervals)
          4. Average the RRP within each 30-min group

        Returns a list of PriceHistorySlot sorted by interval_start_utc.
        Returns [] on parse failure (logged as WARNING).
        """
        try:
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zip_file:
                names = zip_file.namelist()
                csv_names = [name for name in names if name.upper().endswith(".CSV")]
                inner_zip_names = [name for name in names if name.upper().endswith(".ZIP")]

                # The NEMWeb DISPATCHIS archive ships a daily outer ZIP containing
                # 288 nested per-dispatch-interval ZIPs (one per 5-min run), each
                # of which contains exactly one MMSDM CSV. Earlier MMSDM archives
                # sometimes shipped a single flat CSV inside the daily ZIP. Handle
                # both layouts so the code works against the live and legacy
                # archive formats.
                if not csv_names and not inner_zip_names:
                    _LOGGER.warning(
                        "AemoHistoricalClient: no CSV or nested ZIP inside archive for %s",
                        target_date.isoformat(),
                    )
                    return []

                five_min_slots: dict[datetime, list[float]] = {}

                # Flat-CSV layout (legacy)
                for csv_name in csv_names:
                    csv_text = zip_file.read(csv_name).decode("utf-8", errors="replace")
                    self._parse_dispatch_csv(csv_text, five_min_slots)

                # Nested ZIP layout (current NEMWeb archive)
                for inner_zip_name in inner_zip_names:
                    try:
                        inner_bytes = zip_file.read(inner_zip_name)
                        with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner_zip:
                            inner_csv_names = [
                                inner_name for inner_name in inner_zip.namelist()
                                if inner_name.upper().endswith(".CSV")
                            ]
                            for inner_csv_name in inner_csv_names:
                                csv_text = inner_zip.read(inner_csv_name).decode(
                                    "utf-8", errors="replace"
                                )
                                self._parse_dispatch_csv(csv_text, five_min_slots)
                    except (zipfile.BadZipFile, KeyError) as inner_error:
                        _LOGGER.debug(
                            "AemoHistoricalClient: bad inner ZIP %s for %s: %s",
                            inner_zip_name,
                            target_date.isoformat(),
                            inner_error,
                        )
                        continue

        except zipfile.BadZipFile as zip_error:
            _LOGGER.warning(
                "AemoHistoricalClient: corrupt ZIP for %s: %s",
                target_date.isoformat(),
                zip_error,
            )
            return []
        except Exception as parse_error:
            _LOGGER.warning(
                "AemoHistoricalClient: parse error for %s: %s",
                target_date.isoformat(),
                parse_error,
            )
            return []

        if not five_min_slots:
            return []

        # Aggregate 5-min slots to 30-min trading intervals
        return _aggregate_to_30min(five_min_slots)

    def _parse_dispatch_csv(
        self,
        csv_text: str,
        accumulator: dict[datetime, list[float]],
    ) -> None:
        """
        Parse DISPATCH,PRICE D-rows from an AEMO MMSDM CSV and accumulate
        5-min RRP values into *accumulator* keyed by interval_start_utc.

        Modifies *accumulator* in place.  Silently skips malformed rows.
        """
        in_dispatch_price = False
        rrp_column_index: Optional[int] = None

        for raw_line in csv_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("I,DISPATCH,PRICE"):
                in_dispatch_price = True
                # Determine the RRP column index from the header row
                header_columns = line.split(",")
                try:
                    # Header columns use uppercase; locate RRP (not RAISE*RRP etc.)
                    rrp_column_index = _find_rrp_column(header_columns)
                except ValueError:
                    rrp_column_index = None
                    _LOGGER.debug(
                        "AemoHistoricalClient: could not locate RRP column in header: %s",
                        line[:120],
                    )
                continue
            elif line.startswith("I,"):
                in_dispatch_price = False
                rrp_column_index = None
                continue

            if not (in_dispatch_price and line.startswith("D,")):
                continue

            if rrp_column_index is None:
                continue

            columns = _split_mmsdm_row(line)
            if len(columns) <= max(4, 6, rrp_column_index):
                continue

            region_in_row = columns[6].strip().upper()
            if region_in_row != self._region:
                continue

            try:
                settlement_nem = _parse_nem_datetime(columns[4])
                # SETTLEMENTDATE = END of 5-min dispatch interval
                # interval start = settlement - 5 min
                interval_start_nem = settlement_nem - timedelta(minutes=5)
                interval_start_utc = interval_start_nem.astimezone(timezone.utc)
                rrp_value = float(columns[rrp_column_index])
            except (ValueError, IndexError):
                continue

            if interval_start_utc not in accumulator:
                accumulator[interval_start_utc] = []
            accumulator[interval_start_utc].append(rrp_value)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _build_dispatchis_url(target_date: date) -> str:
    """
    Build the NEMWeb ARCHIVE URL for a DISPATCHIS daily ZIP.

    URL format:
      https://nemweb.com.au/Reports/ARCHIVE/DispatchIS_Reports/
        PUBLIC_DISPATCHIS_<YYYYMMDD>.zip
    """
    date_string = target_date.strftime("%Y%m%d")
    return f"{_DISPATCHIS_BASE_URL}PUBLIC_DISPATCHIS_{date_string}.zip"


def _find_rrp_column(header_columns: list[str]) -> int:
    """
    Return the column index of "RRP" in an AEMO DISPATCH,PRICE header.

    We look for an exact "RRP" entry (case-insensitive) and avoid matching
    sub-strings like RAISEFASTRRP.  Raises ValueError if not found.
    """
    for column_index, header_value in enumerate(header_columns):
        if header_value.strip().upper() == "RRP":
            return column_index
    raise ValueError(f"RRP column not found in header: {header_columns}")


def _aggregate_to_30min(
    five_min_slots: dict[datetime, list[float]],
) -> list[PriceHistorySlot]:
    """
    Group 5-min interval RRP values into 30-min trading periods and average them.

    Each 30-min trading period consists of 6 consecutive 5-min dispatch intervals.
    The trading period start is defined by truncating to the nearest 30-min boundary.

    We use the SIMPLE average (not dispatch-quantity-weighted) because we don't
    have cleared MW data here.  This is consistent with how AEMO computes the
    TRADINGPRICE RRP (simple average of the 6 dispatch RRPs).
    """
    # Group 5-min slots by their parent 30-min boundary
    trading_buckets: dict[datetime, list[float]] = {}
    for five_min_start_utc, rrp_values in five_min_slots.items():
        # Round down to the 30-min boundary
        minutes_into_period = five_min_start_utc.minute % 30
        trading_start_utc = five_min_start_utc - timedelta(
            minutes=minutes_into_period,
            seconds=five_min_start_utc.second,
            microseconds=five_min_start_utc.microsecond,
        )
        if trading_start_utc not in trading_buckets:
            trading_buckets[trading_start_utc] = []
        trading_buckets[trading_start_utc].extend(rrp_values)

    result: list[PriceHistorySlot] = []
    for trading_start_utc in sorted(trading_buckets):
        bucket_values = trading_buckets[trading_start_utc]
        if not bucket_values:
            continue
        average_rrp = sum(bucket_values) / len(bucket_values)
        result.append(PriceHistorySlot(
            interval_start_utc=trading_start_utc,
            rrp_per_mwh=average_rrp,
        ))

    return result


def _split_mmsdm_row(line: str) -> list[str]:
    """
    Split a comma-delimited AEMO MMSDM row, respecting double-quoted fields.
    """
    columns: list[str] = []
    current_field: list[str] = []
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
    Parse an AEMO NEM datetime string "YYYY/MM/DD HH:MM:SS" → UTC+10 aware datetime.
    """
    cleaned = raw_value.strip().strip('"')
    naive_dt = datetime.strptime(cleaned, "%Y/%m/%d %H:%M:%S")
    return naive_dt.replace(tzinfo=_NEM_TZ)
