"""
HTTP client for the NEM Price Forecaster sidecar.

Fetches /price_forecast and /load_forecast from the sidecar service.
Returns typed dataclasses so the coordinator can publish HA sensor states.

Design constraints:
  - Python 3.14 compatible (no darts, no sklearn, no scipy)
  - Only stdlib + homeassistant builtins (aiohttp is available in HA)
  - All HTTP calls are async (called via hass.async_add_executor_job is NOT
    needed — we use aiohttp which is natively async)
  - Raises SidecarUnavailable on HTTP errors so coordinator can mark unavailable

Retry strategy:
  - 2 retries on connection error or 5xx, with 1s / 2s back-off
  - 503 (not yet ready) is treated as a transient error
  - 404 (endpoint disabled on sidecar) is NOT retried
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

_LOGGER = logging.getLogger(__name__)

_RETRY_COUNT = 2
_RETRY_BACKOFF_SECONDS = 1.0
_REQUEST_TIMEOUT_SECONDS = 15


class SidecarUnavailable(Exception):
    """Raised when the sidecar cannot be reached or returns an error."""


class SidecarClient:
    """
    Async HTTP client for the NEM forecaster sidecar.

    Lifecycle: one instance per config entry, created in coordinator __init__.
    Call async_close() when the config entry is unloaded.
    """

    def __init__(self, base_url: str) -> None:
        """
        base_url: e.g. "http://localhost:8765" (no trailing slash)
        """
        self._base_url = base_url.rstrip("/")
        self._session: Any = None  # aiohttp.ClientSession, created lazily

    async def async_get_price_forecast(self) -> dict[str, Any]:
        """
        Fetch /price_forecast from the sidecar.

        Returns the full response dict on success.
        Raises SidecarUnavailable on any error.
        """
        return await self._async_get("/price_forecast")

    async def async_get_load_forecast(self) -> Optional[dict[str, Any]]:
        """
        Fetch /load_forecast from the sidecar.

        Returns None (not raises) if the endpoint returns 404 (load disabled).
        Raises SidecarUnavailable on connection / 5xx errors.
        """
        try:
            return await self._async_get("/load_forecast")
        except SidecarUnavailable as unavailable_error:
            if "404" in str(unavailable_error):
                _LOGGER.debug("Load forecaster is disabled on sidecar")
                return None
            raise

    async def async_get_health(self) -> dict[str, Any]:
        """Fetch /health from the sidecar (used by config_flow to test connectivity)."""
        return await self._async_get("/health")

    async def async_post_import_calibration(
        self,
        predicted_rrp_per_mwh: float,
        actual_rrp_per_mwh: float,
        hour_of_day: int,
        observed_at: Optional[datetime] = None,
    ) -> None:
        """Post an import calibration observation to the sidecar (best-effort)."""
        payload = {
            "predicted_rrp_per_mwh": predicted_rrp_per_mwh,
            "actual_rrp_per_mwh": actual_rrp_per_mwh,
            "hour_of_day": hour_of_day,
            "observed_at": (observed_at or datetime.now(timezone.utc)).isoformat(),
        }
        try:
            await self._async_post("/calibration/import", payload)
        except SidecarUnavailable as post_error:
            _LOGGER.debug("Import calibration post failed (non-fatal): %s", post_error)

    async def async_post_export_calibration(
        self,
        predicted_rrp_per_mwh: float,
        actual_rrp_per_mwh: float,
        hour_of_day: int,
        observed_at: Optional[datetime] = None,
    ) -> None:
        """Post an export calibration observation to the sidecar (best-effort)."""
        payload = {
            "predicted_rrp_per_mwh": predicted_rrp_per_mwh,
            "actual_rrp_per_mwh": actual_rrp_per_mwh,
            "hour_of_day": hour_of_day,
            "observed_at": (observed_at or datetime.now(timezone.utc)).isoformat(),
        }
        try:
            await self._async_post("/calibration/export", payload)
        except SidecarUnavailable as post_error:
            _LOGGER.debug("Export calibration post failed (non-fatal): %s", post_error)

    async def async_post_load_observation(
        self,
        interval_start_utc: datetime,
        load_watts: float,
    ) -> None:
        """Post a 30-min load measurement to the sidecar (best-effort)."""
        payload = {
            "interval_start_utc": interval_start_utc.isoformat(),
            "load_watts": load_watts,
        }
        try:
            await self._async_post("/load_observation", payload)
        except SidecarUnavailable as post_error:
            _LOGGER.debug("Load observation post failed (non-fatal): %s", post_error)

    async def async_close(self) -> None:
        """Close the aiohttp session. Call on config entry unload."""
        if self._session is not None:
            await self._session.close()
            self._session = None

    # ------------------------------------------------------------------
    # Internal HTTP helpers
    # ------------------------------------------------------------------

    async def _async_get(self, path: str) -> dict[str, Any]:
        session = await self._ensure_session()
        url = f"{self._base_url}{path}"
        last_error: Optional[Exception] = None

        for attempt_number in range(_RETRY_COUNT + 1):
            try:
                import asyncio
                async with session.get(url, timeout=_build_timeout()) as response:
                    if response.status == 200:
                        return await response.json()
                    body = await response.text()
                    raise SidecarUnavailable(
                        f"Sidecar GET {path} returned HTTP {response.status}: {body[:200]}"
                    )
            except SidecarUnavailable:
                raise
            except Exception as connection_error:
                last_error = connection_error
                if attempt_number < _RETRY_COUNT:
                    import asyncio
                    backoff = _RETRY_BACKOFF_SECONDS * (2 ** attempt_number)
                    _LOGGER.debug(
                        "Sidecar GET %s attempt %d failed (%s); retrying in %.0fs",
                        path,
                        attempt_number + 1,
                        connection_error,
                        backoff,
                    )
                    await asyncio.sleep(backoff)

        raise SidecarUnavailable(
            f"Sidecar GET {path} failed after {_RETRY_COUNT + 1} attempts: {last_error}"
        )

    async def _async_post(self, path: str, payload: dict[str, Any]) -> None:
        session = await self._ensure_session()
        url = f"{self._base_url}{path}"
        try:
            async with session.post(url, json=payload, timeout=_build_timeout()) as response:
                if response.status not in (200, 202):
                    body = await response.text()
                    raise SidecarUnavailable(
                        f"Sidecar POST {path} returned HTTP {response.status}: {body[:200]}"
                    )
        except SidecarUnavailable:
            raise
        except Exception as connection_error:
            raise SidecarUnavailable(
                f"Sidecar POST {path} failed: {connection_error}"
            ) from connection_error

    async def _ensure_session(self) -> Any:
        """Create aiohttp.ClientSession lazily."""
        if self._session is None:
            import aiohttp
            self._session = aiohttp.ClientSession()
        return self._session


def _build_timeout() -> Any:
    """Build an aiohttp.ClientTimeout for request + connect."""
    try:
        import aiohttp
        return aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT_SECONDS)
    except ImportError:
        return None
