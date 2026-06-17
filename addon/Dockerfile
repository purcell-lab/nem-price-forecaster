# NEM Price Forecaster — HA Add-on Dockerfile
#
# Uses the official Home Assistant DEBIAN (glibc) base images for multi-arch
# add-on compatibility.  The sidecar code is identical to the standalone Docker
# image; only the entry point and data directory are adapted for the Supervisor
# environment.
#
# Why debian (base-debian) and not alpine (base-python)?
#   LightGBM and Darts publish manylinux (glibc) binary wheels.  On an alpine
#   (musl libc) base those wheels are incompatible, so pip tries to build them
#   from source — which routinely fails — and the Darts price model never loads.
#   The debian base ships glibc, so the prebuilt wheels install out of the box.
#
# Build args injected by the Supervisor builder:
#   BUILD_FROM  — e.g. homeassistant/amd64-base-debian:bookworm

ARG BUILD_FROM
FROM ${BUILD_FROM}

# System deps:
#   python3 / python3-pip / python3-dev — base-debian ships bashio + s6 but NO
#     Python interpreter, so we install it (bookworm provides Python 3.11).
#   libgomp1 — OpenMP runtime required by LightGBM at import/predict time.
#   build-essential / cmake — build deps for any dependency that has no prebuilt
#     wheel for the target arch (e.g. armv7) and must compile from source.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        libgomp1 \
        build-essential \
        cmake \
        jq \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies into the system interpreter.
# --break-system-packages: debian's pip is PEP-668 "externally managed"; this is
# a single-purpose container, so installing into the system environment is fine
# and keeps uvicorn on PATH for the run script below.
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir --break-system-packages -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# Copy sidecar application code
WORKDIR /app
COPY app/ /app/

# Copy the add-on run script — reads HA Supervisor options and starts uvicorn
COPY run.sh /run.sh
RUN chmod +x /run.sh

# Data directory: the Supervisor maps the "share/nem_forecaster_data" path
# declared in config.yaml to /nem_forecaster_data inside the container.
VOLUME ["/nem_forecaster_data"]

# Build-time identity baked into the image so /version can report which commit
# is actually running.  The HA Supervisor builder injects:
#   BUILD_REF      git SHA of the source tree being built
#   BUILD_DATE     ISO-8601 UTC timestamp of the build
#   BUILD_VERSION  add-on version from config.yaml
#   BUILD_ARCH     target architecture
# See https://developers.home-assistant.io/docs/add-ons/configuration/#docker-build-arguments
ARG BUILD_REF=unknown
ARG BUILD_DATE=unknown
ARG BUILD_VERSION=unknown
ARG BUILD_ARCH=unknown
ENV SIDECAR_GIT_SHA=${BUILD_REF}
ENV SIDECAR_BUILD_TIME=${BUILD_DATE}
ENV SIDECAR_ADDON_VERSION=${BUILD_VERSION}
ENV SIDECAR_BUILD_ARCH=${BUILD_ARCH}

ENV PYTHONUNBUFFERED=1
ENV SIDECAR_DATA_DIR=/nem_forecaster_data

EXPOSE 8765

CMD ["/run.sh"]
