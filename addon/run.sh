#!/usr/bin/env bashio
# NEM Price Forecaster — Add-on entry point
#
# Reads the options set by the user in the HA Supervisor UI and translates
# them into SIDECAR_* environment variables before launching uvicorn.

set -e

bashio::log.info "Starting NEM Price Forecaster sidecar ..."

# ---- Read add-on options (bashio reads from /data/options.json) ----
export SIDECAR_REGION="$(bashio::config 'region')"
export SIDECAR_PRICE_MODEL="$(bashio::config 'price_model')"
export SIDECAR_CALIBRATOR="$(bashio::config 'calibrator')"
export SIDECAR_NAIVE_BLEND_WEIGHT="$(bashio::config 'naive_blend_weight')"
export SIDECAR_FORECAST_HORIZON_HOURS="$(bashio::config 'forecast_horizon_hours')"
export SIDECAR_GST_RATE="$(bashio::config 'gst_rate')"
export SIDECAR_FIXED_ADDER_PER_KWH="$(bashio::config 'fixed_adder_per_kwh')"
export SIDECAR_FEED_IN_IS_WHOLESALE="$(bashio::config 'feed_in_is_wholesale')"
export SIDECAR_LOAD_FORECASTER_ENABLED="$(bashio::config 'load_forecaster_enabled')"
export SIDECAR_WEATHER_ENABLED="$(bashio::config 'weather_enabled')"
export SIDECAR_AEMO_HISTORY_DAYS="$(bashio::config 'aemo_history_days')"

# Latitude / longitude override (0.0 means "use NEM region default")
LATITUDE="$(bashio::config 'latitude')"
LONGITUDE="$(bashio::config 'longitude')"
if [ "${LATITUDE}" != "0.0" ] || [ "${LONGITUDE}" != "0.0" ]; then
    export SIDECAR_LATITUDE="${LATITUDE}"
    export SIDECAR_LONGITUDE="${LONGITUDE}"
fi

# Data directory — mapped by Supervisor from share/nem_forecaster_data
export SIDECAR_DATA_DIR=/nem_forecaster_data

bashio::log.info "Region: ${SIDECAR_REGION}, Price model: ${SIDECAR_PRICE_MODEL}, Calibrator: ${SIDECAR_CALIBRATOR}, Horizon: ${SIDECAR_FORECAST_HORIZON_HOURS}h"
bashio::log.info "Weather: ${SIDECAR_WEATHER_ENABLED}, Load forecaster: ${SIDECAR_LOAD_FORECASTER_ENABLED}"
bashio::log.info "Data dir: ${SIDECAR_DATA_DIR}"

exec uvicorn main:app \
    --host 0.0.0.0 \
    --port 8765 \
    --workers 1
