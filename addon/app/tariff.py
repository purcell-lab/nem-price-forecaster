"""
Tariff layer: convert calibrated wholesale RRP ($/kWh) to retail import and
export prices.

Import price
------------
    import_price = (wholesale_rrp + network_tou_rate + fixed_adder) × (1 + gst_rate)

where network_tou_rate is looked up from the configured ToU band windows for
the interval's local (NEM) datetime.

Export price (feed-in tariff)
------------------------------
When feed_in_is_wholesale is True (the default, matching Amber's model):

    export_price = wholesale_rrp   (GST-excluded, per ATO ruling)

The export price is intentionally NOT multiplied by GST.  Per the ATO:
electricity retailers who buy exported solar are not required to remit GST on
the feed-in credit, and residential generators are not registered for GST.
This means import > export at all times (arbitrage-free by construction).

Network ToU bands
-----------------
Bands are defined as a list of dicts, each with:
    {
        "name": "peak",
        "rate_per_kwh": 0.12,          # $/kWh added on top of wholesale
        "windows": [
            {"days": [0, 1, 2, 3, 4],  # Monday=0 … Sunday=6
             "start": "07:00",
             "end": "21:00"}
        ]
    }

Windows are evaluated in list order; the first matching band wins.  A fallback
"off-peak" rate of 0.0 is used if no band matches.

"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, time
from typing import Optional

_LOGGER = logging.getLogger(__name__)


@dataclass
class TouWindow:
    """A single time-of-use window within a band."""
    days: list[int]    # 0=Monday … 6=Sunday (Python weekday convention)
    start_time: time   # inclusive
    end_time: time     # exclusive


@dataclass
class TouBand:
    """One named ToU network band with its windows and per-kWh adder."""
    name: str
    rate_per_kwh: float
    windows: list[TouWindow]


class TariffCalculator:
    """
    Convert calibrated wholesale price ($/kWh) to import and export retail prices.

    All datetime inputs should be in NEM time (UTC+10) so that peak/shoulder
    windows match what the customer sees on their bill.
    """

    def __init__(
        self,
        tou_bands: list[TouBand],
        fixed_adder_per_kwh: float = 0.0,
        gst_rate: float = 0.10,
        feed_in_is_wholesale: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        tou_bands
            Ordered list of ToU bands (first match wins).  May be empty.
        fixed_adder_per_kwh
            Flat per-kWh network/service charge added to import price ($/kWh,
            pre-GST).  Does NOT apply to export.
        gst_rate
            GST multiplier (default 0.10 = 10%).  Applied to import only.
        feed_in_is_wholesale
            When True, export_price = calibrated_wholesale_rrp (GST-excluded).
            When False (rare — retailer-specific FiT), export = separate value
            the caller must supply via compute_export_price_override.
        """
        self._tou_bands = tou_bands
        self._fixed_adder_per_kwh = fixed_adder_per_kwh
        self._gst_rate = gst_rate
        self._feed_in_is_wholesale = feed_in_is_wholesale

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_import_price(
        self,
        calibrated_wholesale_kwh: float,
        interval_nem_datetime: datetime,
    ) -> float:
        """
        Return the retail import price ($/kWh, GST-inclusive) for an interval.
        """
        network_rate = self._lookup_tou_rate(interval_nem_datetime)
        pre_gst = calibrated_wholesale_kwh + network_rate + self._fixed_adder_per_kwh
        return pre_gst * (1.0 + self._gst_rate)

    def compute_export_price(
        self,
        calibrated_wholesale_kwh: float,
        interval_nem_datetime: Optional[datetime] = None,
    ) -> float:
        """
        Return the feed-in / export price ($/kWh, GST-excluded).

        When feed_in_is_wholesale=True, this is simply the calibrated wholesale
        value (which may be negative during solar oversupply — this is correct
        and should propagate to the battery optimiser).
        """
        if self._feed_in_is_wholesale:
            return calibrated_wholesale_kwh
        # Non-wholesale feed-in (flat retailer FiT) — caller must configure a
        # zero-rate ToU band or a separate mechanism.  Placeholder: return 0.
        _LOGGER.debug(
            "feed_in_is_wholesale=False; export price defaults to 0; configure explicitly"
        )
        return 0.0

    def network_rate_for_interval(self, interval_nem_datetime: datetime) -> float:
        """Public accessor for the ToU network rate at a given NEM datetime."""
        return self._lookup_tou_rate(interval_nem_datetime)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _lookup_tou_rate(self, nem_datetime: datetime) -> float:
        """Return the $/kWh network ToU rate for *nem_datetime* (NEM time)."""
        weekday = nem_datetime.weekday()  # 0=Monday … 6=Sunday
        query_time = nem_datetime.time()

        for band in self._tou_bands:
            for window in band.windows:
                if weekday in window.days and _time_in_window(
                    query_time, window.start_time, window.end_time
                ):
                    return band.rate_per_kwh

        # No band matched — off-peak (zero network adder over wholesale)
        return 0.0


# ---------------------------------------------------------------------------
# Config-dict → dataclass converters (used by config_flow + coordinator)
# ---------------------------------------------------------------------------

def parse_tou_bands_from_config(tou_bands_config: list[dict]) -> list[TouBand]:
    """
    Convert the list-of-dicts stored in the config entry into TouBand objects.

    Expected input format per band:
    {
        "name": "peak",
        "rate_per_kwh": 0.12,
        "windows": [
            {"days": [0, 1, 2, 3, 4], "start": "07:00", "end": "21:00"}
        ]
    }
    """
    bands: list[TouBand] = []
    for band_config in tou_bands_config:
        windows: list[TouWindow] = []
        for window_config in band_config.get("windows", []):
            windows.append(
                TouWindow(
                    days=list(window_config["days"]),
                    start_time=_parse_time_string(window_config["start"]),
                    end_time=_parse_time_string(window_config["end"]),
                )
            )
        bands.append(
            TouBand(
                name=band_config["name"],
                rate_per_kwh=float(band_config["rate_per_kwh"]),
                windows=windows,
            )
        )
    return bands


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _time_in_window(query_time: time, start_time: time, end_time: time) -> bool:
    """
    Return True if *query_time* falls within [start_time, end_time).

    Handles windows that do NOT wrap midnight.  For midnight-crossing windows
    (e.g., 22:00–06:00) split them into two bands in the config.
    """
    if start_time <= end_time:
        return start_time <= query_time < end_time
    # Wrap-around: start > end (e.g., 22:00 to 06:00)
    return query_time >= start_time or query_time < end_time


def _parse_time_string(time_string: str) -> time:
    """Parse "HH:MM" or "HH:MM:SS" into a datetime.time object."""
    parts = time_string.strip().split(":")
    hour = int(parts[0])
    minute = int(parts[1]) if len(parts) > 1 else 0
    second = int(parts[2]) if len(parts) > 2 else 0
    return time(hour, minute, second)
