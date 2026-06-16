"""
Forecast resampling — resample 30-min native PD7DAY/load-forecast slots to a
user-configured resolution over a user-configured horizon.

Strategy
--------
PD7DAY is natively 30-min.  The load forecaster also outputs 30-min slots.

* **Upsample** (target period < 30 min, e.g. 5-min or 15-min):
  Linear interpolation between consecutive 30-min slot prices/loads.

* **Same resolution** (target period = 30 min):
  Pass-through with horizon truncation only.

* **Downsample** (target period > 30 min, e.g. 60-min):
  Arithmetic mean of the 30-min slots falling within each output bucket.
  For prices this is reasonable ($/kWh averages linearly); for load (W) it
  is also correct because watts already represents a rate.

The horizon is applied **after** resampling: only slots whose
interval_start falls within [now, now + horizon_hours] are returned.

All timestamps are UTC, tz-aware throughout.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

_LOGGER = logging.getLogger(__name__)

# Native PD7DAY / load forecaster resolution in minutes
_NATIVE_SLOT_MINUTES = 30


def resample_price_slots(
    slots: list[dict[str, Any]],
    target_period_minutes: int,
    horizon_hours: int,
    now_utc: datetime | None = None,
) -> list[dict[str, Any]]:
    """
    Resample a list of price slot dicts to *target_period_minutes* over
    *horizon_hours* starting from *now_utc*.

    Each slot dict is expected to contain at least the keys produced by
    ForecastSlot.as_dict():
        interval_start, import_price, export_price, calibrated_wholesale,
        raw_rrp_per_mwh, network_tou_rate

    Returns a new list of dicts with the same keys.  Timestamps are ISO-8601
    strings (UTC, tz-aware) matching the output period boundaries.
    """
    if not slots:
        return []

    if now_utc is None:
        now_utc = datetime.now(timezone.utc)

    horizon_end_utc = now_utc + timedelta(hours=horizon_hours)

    # Filter to horizon
    horizon_slots = [
        slot for slot in slots
        if _parse_iso(slot["interval_start"]) < horizon_end_utc
    ]

    if not horizon_slots:
        return []

    if target_period_minutes == _NATIVE_SLOT_MINUTES:
        return horizon_slots

    if target_period_minutes < _NATIVE_SLOT_MINUTES:
        return _upsample_price_slots(horizon_slots, target_period_minutes, horizon_end_utc)

    # Downsample
    return _downsample_price_slots(horizon_slots, target_period_minutes, now_utc, horizon_end_utc)


def resample_load_slots(
    slots: list[dict[str, Any]],
    target_period_minutes: int,
    horizon_hours: int,
    now_utc: datetime | None = None,
) -> list[dict[str, Any]]:
    """
    Resample a list of load slot dicts to *target_period_minutes* over
    *horizon_hours*.

    Each slot dict is expected to contain the keys produced by
    LoadForecastSlot.as_dict():
        datetime, load_power

    Returns a new list of dicts with the same keys.
    """
    if not slots:
        return []

    if now_utc is None:
        now_utc = datetime.now(timezone.utc)

    horizon_end_utc = now_utc + timedelta(hours=horizon_hours)

    # Filter to horizon
    horizon_slots = [
        slot for slot in slots
        if _parse_iso(slot["datetime"]) < horizon_end_utc
    ]

    if not horizon_slots:
        return []

    if target_period_minutes == _NATIVE_SLOT_MINUTES:
        return horizon_slots

    if target_period_minutes < _NATIVE_SLOT_MINUTES:
        return _upsample_load_slots(horizon_slots, target_period_minutes, horizon_end_utc)

    return _downsample_load_slots(horizon_slots, target_period_minutes, now_utc, horizon_end_utc)


# ---------------------------------------------------------------------------
# Price slot resampling helpers
# ---------------------------------------------------------------------------

def _upsample_price_slots(
    slots: list[dict[str, Any]],
    target_period_minutes: int,
    horizon_end_utc: datetime,
) -> list[dict[str, Any]]:
    """
    Interpolate 30-min price slots to a finer period.

    Between each pair of consecutive 30-min slots, we linearly interpolate
    all numeric fields (import_price, export_price, calibrated_wholesale,
    raw_rrp_per_mwh, network_tou_rate) at the target period boundaries.
    """
    if len(slots) < 2:
        # Only one slot — replicate it at target period within its 30-min window
        return _replicate_single_slot_price(slots[0], target_period_minutes, horizon_end_utc)

    numeric_keys = [
        "import_price", "export_price", "calibrated_wholesale",
        "raw_rrp_per_mwh", "network_tou_rate",
    ]
    output_slots: list[dict[str, Any]] = []

    for segment_index in range(len(slots) - 1):
        slot_a = slots[segment_index]
        slot_b = slots[segment_index + 1]
        time_a = _parse_iso(slot_a["interval_start"])
        time_b = _parse_iso(slot_b["interval_start"])
        segment_duration_seconds = (time_b - time_a).total_seconds()

        if segment_duration_seconds <= 0:
            continue

        step_seconds = target_period_minutes * 60.0
        num_steps = max(1, round(segment_duration_seconds / step_seconds))

        for step_index in range(num_steps):
            fraction = (step_index * step_seconds) / segment_duration_seconds
            slot_start = time_a + timedelta(seconds=step_index * step_seconds)
            if slot_start >= horizon_end_utc:
                break

            interp_slot: dict[str, Any] = {
                "interval_start": slot_start.isoformat(),
                "interval_start_nem": slot_a.get("interval_start_nem", ""),
            }
            for key in numeric_keys:
                value_a = float(slot_a.get(key, 0.0))
                value_b = float(slot_b.get(key, 0.0))
                interp_slot[key] = round(value_a + fraction * (value_b - value_a), 6)

            output_slots.append(interp_slot)

    # Append the final slot
    last_slot_start = _parse_iso(slots[-1]["interval_start"])
    if last_slot_start < horizon_end_utc:
        output_slots.append(dict(slots[-1]))

    return output_slots


def _downsample_price_slots(
    slots: list[dict[str, Any]],
    target_period_minutes: int,
    now_utc: datetime,
    horizon_end_utc: datetime,
) -> list[dict[str, Any]]:
    """
    Aggregate 30-min price slots to a coarser period via arithmetic mean.

    Buckets are aligned to the start of the first slot, stepping by
    *target_period_minutes*.
    """
    numeric_keys = [
        "import_price", "export_price", "calibrated_wholesale",
        "raw_rrp_per_mwh", "network_tou_rate",
    ]

    first_slot_time = _parse_iso(slots[0]["interval_start"])
    step_delta = timedelta(minutes=target_period_minutes)

    # Build an index from slot_start → slot dict for fast lookup
    slot_by_start: dict[datetime, dict[str, Any]] = {}
    for slot in slots:
        slot_time = _parse_iso(slot["interval_start"])
        slot_by_start[slot_time] = slot

    output_slots: list[dict[str, Any]] = []
    bucket_start = first_slot_time

    while bucket_start < horizon_end_utc:
        bucket_end = bucket_start + step_delta

        # Collect all native 30-min slots that fall within this bucket
        bucket_member_slots: list[dict[str, Any]] = []
        candidate_time = bucket_start
        while candidate_time < bucket_end:
            if candidate_time in slot_by_start:
                bucket_member_slots.append(slot_by_start[candidate_time])
            candidate_time += timedelta(minutes=_NATIVE_SLOT_MINUTES)

        if not bucket_member_slots:
            bucket_start = bucket_end
            continue

        averaged_slot: dict[str, Any] = {
            "interval_start": bucket_start.isoformat(),
            "interval_start_nem": bucket_member_slots[0].get("interval_start_nem", ""),
        }
        for key in numeric_keys:
            values = [float(slot.get(key, 0.0)) for slot in bucket_member_slots]
            averaged_slot[key] = round(sum(values) / len(values), 6)

        output_slots.append(averaged_slot)
        bucket_start = bucket_end

    return output_slots


def _replicate_single_slot_price(
    slot: dict[str, Any],
    target_period_minutes: int,
    horizon_end_utc: datetime,
) -> list[dict[str, Any]]:
    """
    When only one 30-min slot is available, replicate its values at the
    target period within the available window (up to horizon_end_utc).
    """
    slot_start = _parse_iso(slot["interval_start"])
    window_end = min(slot_start + timedelta(minutes=_NATIVE_SLOT_MINUTES), horizon_end_utc)
    output_slots: list[dict[str, Any]] = []
    step_delta = timedelta(minutes=target_period_minutes)
    current_time = slot_start
    while current_time < window_end:
        output_slots.append(dict(slot) | {"interval_start": current_time.isoformat()})
        current_time += step_delta
    return output_slots


# ---------------------------------------------------------------------------
# Load slot resampling helpers
# ---------------------------------------------------------------------------

def _upsample_load_slots(
    slots: list[dict[str, Any]],
    target_period_minutes: int,
    horizon_end_utc: datetime,
) -> list[dict[str, Any]]:
    """Interpolate 30-min load slots to a finer period."""
    if len(slots) < 2:
        return _replicate_single_slot_load(slots[0], target_period_minutes, horizon_end_utc)

    output_slots: list[dict[str, Any]] = []

    for segment_index in range(len(slots) - 1):
        slot_a = slots[segment_index]
        slot_b = slots[segment_index + 1]
        time_a = _parse_iso(slot_a["datetime"])
        time_b = _parse_iso(slot_b["datetime"])
        segment_duration_seconds = (time_b - time_a).total_seconds()

        if segment_duration_seconds <= 0:
            continue

        step_seconds = target_period_minutes * 60.0
        num_steps = max(1, round(segment_duration_seconds / step_seconds))

        load_a = float(slot_a.get("load_power", 0.0))
        load_b = float(slot_b.get("load_power", 0.0))

        for step_index in range(num_steps):
            fraction = (step_index * step_seconds) / segment_duration_seconds
            slot_start = time_a + timedelta(seconds=step_index * step_seconds)
            if slot_start >= horizon_end_utc:
                break
            interp_load = load_a + fraction * (load_b - load_a)
            output_slots.append({
                "datetime": slot_start.isoformat(),
                "load_power": round(interp_load, 1),
            })

    last_slot_start = _parse_iso(slots[-1]["datetime"])
    if last_slot_start < horizon_end_utc:
        output_slots.append(dict(slots[-1]))

    return output_slots


def _downsample_load_slots(
    slots: list[dict[str, Any]],
    target_period_minutes: int,
    now_utc: datetime,
    horizon_end_utc: datetime,
) -> list[dict[str, Any]]:
    """Aggregate 30-min load slots to a coarser period via arithmetic mean."""
    first_slot_time = _parse_iso(slots[0]["datetime"])
    step_delta = timedelta(minutes=target_period_minutes)

    slot_by_start: dict[datetime, dict[str, Any]] = {}
    for slot in slots:
        slot_time = _parse_iso(slot["datetime"])
        slot_by_start[slot_time] = slot

    output_slots: list[dict[str, Any]] = []
    bucket_start = first_slot_time

    while bucket_start < horizon_end_utc:
        bucket_end = bucket_start + step_delta

        bucket_member_slots: list[dict[str, Any]] = []
        candidate_time = bucket_start
        while candidate_time < bucket_end:
            if candidate_time in slot_by_start:
                bucket_member_slots.append(slot_by_start[candidate_time])
            candidate_time += timedelta(minutes=_NATIVE_SLOT_MINUTES)

        if not bucket_member_slots:
            bucket_start = bucket_end
            continue

        load_values = [float(slot.get("load_power", 0.0)) for slot in bucket_member_slots]
        output_slots.append({
            "datetime": bucket_start.isoformat(),
            "load_power": round(sum(load_values) / len(load_values), 1),
        })
        bucket_start = bucket_end

    return output_slots


def _replicate_single_slot_load(
    slot: dict[str, Any],
    target_period_minutes: int,
    horizon_end_utc: datetime,
) -> list[dict[str, Any]]:
    """When only one 30-min slot is available, replicate it at the target period."""
    slot_start = _parse_iso(slot["datetime"])
    window_end = min(slot_start + timedelta(minutes=_NATIVE_SLOT_MINUTES), horizon_end_utc)
    output_slots: list[dict[str, Any]] = []
    step_delta = timedelta(minutes=target_period_minutes)
    current_time = slot_start
    while current_time < window_end:
        output_slots.append({"datetime": current_time.isoformat(), "load_power": slot["load_power"]})
        current_time += step_delta
    return output_slots


# ---------------------------------------------------------------------------
# Parsing helper
# ---------------------------------------------------------------------------

def _parse_iso(iso_string: str) -> datetime:
    """
    Parse an ISO-8601 datetime string to a tz-aware UTC datetime.

    Handles both '+00:00' suffix and 'Z' suffix.  Assumes UTC if no offset given.
    """
    cleaned = iso_string.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(cleaned)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
