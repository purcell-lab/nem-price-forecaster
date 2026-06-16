# NEM Price Forecaster — HA Add-on Dockerfile
#
# Uses the official Home Assistant base images for multi-arch add-on compatibility.
# The sidecar code is identical to the standalone Docker image; only the entry
# point and data directory are adapted for the Supervisor environment.
#
# Build args injected by the Supervisor builder:
#   BUILD_FROM  — e.g. homeassistant/amd64-base-python:3.12-alpine3.19

ARG BUILD_FROM
FROM ${BUILD_FROM}

# System deps for LightGBM (libgomp) — Alpine uses musl so we need libgomp-dev
RUN apk add --no-cache libgomp

# Install Python dependencies into the image's Python environment
# We install into the system Python, not a venv, so the CMD below works cleanly.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Copy sidecar application code
WORKDIR /app
COPY app/ /app/

# Copy the add-on run script — reads HA Supervisor options and starts uvicorn
COPY run.sh /run.sh
RUN chmod +x /run.sh

# Data directory: the Supervisor maps the "share/nem_forecaster_data" path
# declared in config.yaml to /nem_forecaster_data inside the container.
VOLUME ["/nem_forecaster_data"]

ENV PYTHONUNBUFFERED=1
ENV SIDECAR_DATA_DIR=/nem_forecaster_data

EXPOSE 8765

CMD ["/run.sh"]
