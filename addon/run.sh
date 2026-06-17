#!/usr/bin/with-contenv bash
# NEM Price Forecaster — Add-on entry point
#
# Reads user-set options from /data/options.json (mounted by Supervisor) and
# exports them as SIDECAR_* environment variables before launching uvicorn.
#
# Why not bashio? bashio::config reads the Supervisor API endpoint
# /addons/self/options/config, which returns HTTP 403 for this add-on
# regardless of the hassio_api flag. Reading /data/options.json directly with
# jq is the documented escape hatch and is what most production add-ons do.

set -euo pipefail

OPTIONS=/data/options.json
log() { echo "[$(date -u +%H:%M:%S)] INFO: $*"; }

if [ ! -f "${OPTIONS}" ]; then
    echo "[$(date -u +%H:%M:%S)] ERROR: ${OPTIONS} not found — Supervisor did not mount add-on options" >&2
    exit 1
fi

log "Starting NEM Price Forecaster sidecar ..."

# Helper: read a scalar (string/number/bool) option as plain text.
get() {
    local key="$1"
    local default="${2-}"
    local v
    v=$(jq -r --arg k "${key}" '.[$k] // empty' "${OPTIONS}")
    if [ -z "${v}" ] && [ -n "${default}" ]; then
        echo "${default}"
    else
        echo "${v}"
    fi
}

# ---- Read add-on options ----
export SIDECAR_REGION="$(get region)"
export SIDECAR_PRICE_MODEL="$(get price_model)"
export SIDECAR_CALIBRATOR="$(get calibrator)"
export SIDECAR_NAIVE_BLEND_WEIGHT="$(get naive_blend_weight)"
export SIDECAR_FORECAST_HORIZON_DAYS="$(get forecast_horizon_days)"
export SIDECAR_FORECAST_HORIZON_HOURS="$(get forecast_horizon_hours)"
export SIDECAR_CHAIN_SEAM_DAYS="$(get chain_seam_days)"
export SIDECAR_CHAIN_BLEND_WINDOW_HOURS="$(get chain_blend_window_hours)"
export SIDECAR_GST_RATE="$(get gst_rate)"
export SIDECAR_FIXED_ADDER_PER_KWH="$(get fixed_adder_per_kwh)"
export SIDECAR_FEED_IN_IS_WHOLESALE="$(get feed_in_is_wholesale)"
export SIDECAR_LOAD_FORECASTER_ENABLED="$(get load_forecaster_enabled)"
export SIDECAR_WEATHER_ENABLED="$(get weather_enabled)"
export SIDECAR_AEMO_HISTORY_DAYS="$(get aemo_history_days)"
export SIDECAR_HYBRID_CROSSOVER_HOURS="$(get hybrid_crossover_hours)"
export SIDECAR_HYBRID_BLEND_WINDOW_HOURS="$(get hybrid_blend_window_hours)"
export SIDECAR_HYBRID_BLEND_ENABLED="$(get hybrid_blend_enabled)"
export SIDECAR_CALIBRATOR_ADJACENCY_ALPHA="$(get calibrator_adjacency_alpha)"

# Latitude / longitude override (0.0 means "use NEM region default")
LATITUDE="$(get latitude)"
LONGITUDE="$(get longitude)"
if [ "${LATITUDE}" != "0.0" ] || [ "${LONGITUDE}" != "0.0" ]; then
    export SIDECAR_LATITUDE="${LATITUDE}"
    export SIDECAR_LONGITUDE="${LONGITUDE}"
fi

# Data directory.
#
# The Supervisor maps the host /share directory onto /nem_forecaster_data
# inside the container (see config.yaml `map:` block).  /share is shared
# with other add-ons (EMHASS, etc.), so NPF writes into a dedicated
# subdirectory to keep its state isolated:
#   host:      /share/nem_forecaster_data/
#   container: /nem_forecaster_data/nem_forecaster_data/
# Legacy installs that wrote at the parent path are auto-migrated on first
# startup; see sidecar/app/main.py::_migrate_legacy_data_dir.
export SIDECAR_DATA_DIR=/nem_forecaster_data/nem_forecaster_data

log "Region: ${SIDECAR_REGION}, Price model: ${SIDECAR_PRICE_MODEL}, Calibrator: ${SIDECAR_CALIBRATOR}, Horizon: ${SIDECAR_FORECAST_HORIZON_HOURS}h"
log "Weather: ${SIDECAR_WEATHER_ENABLED}, Load forecaster: ${SIDECAR_LOAD_FORECASTER_ENABLED}"
log "Data dir: ${SIDECAR_DATA_DIR}"

exec uvicorn main:app \
    --host 0.0.0.0 \
    --port 8765 \
    --workers 1
