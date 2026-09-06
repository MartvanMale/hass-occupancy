#!/usr/bin/with-contenv bashio
# Read the add-on options and Supervisor's MQTT service, then hand over.
#
# Everything below is passed as environment rather than written into a config
# file, because Supervisor owns these values and re-injects them on every start.

set -e

# The data source is infrastructure, so the ADD-ON OPTION is authoritative and
# overrides whatever is persisted in /data/config.json. Without this the option
# was accepted, shown in the UI, and quietly ignored -- the add-on kept using
# the store because that is what its saved settings said.
export OCCUPANCY_SOURCE="$(bashio::config 'source')"

# log.py maps bashio's seven level names onto Python's five.
export LOG_LEVEL="$(bashio::config 'log_level')"

if bashio::config.equals 'source' 'influx'; then
    export INFLUX_URL="$(bashio::config 'influx_url')"
    export INFLUX_ORG="$(bashio::config 'influx_org')"
    export INFLUX_BUCKET="$(bashio::config 'influx_bucket')"
    export INFLUX_TOKEN="$(bashio::config 'influx_token')"
fi

# Home Assistant user ids allowed to POST to the mutating endpoints. Empty --
# which is the default, and what every existing install has -- means everyone,
# exactly as before this option existed. See config.admin_users().
#
# `admin_users // []` rather than `admin_users`: on an install that predates the
# option the key is absent, and jq's `join` on null is a hard error, which under
# `set -e` is a container that will not start over a setting nobody has touched.
export OCCUPANCY_ADMIN_USERS="$(bashio::config 'admin_users // [] | join(",")')"

if bashio::services.available 'mqtt'; then
    export MQTT_HOST="$(bashio::services 'mqtt' 'host')"
    export MQTT_PORT="$(bashio::services 'mqtt' 'port')"
    export MQTT_USER="$(bashio::services 'mqtt' 'username')"
    export MQTT_PASSWORD="$(bashio::services 'mqtt' 'password')"
    # Whether the broker wants TLS. Ignored until now, so a TLS-only broker
    # got a plaintext CONNECT and the add-on reported it unavailable.
    export MQTT_SSL="$(bashio::services 'mqtt' 'ssl' 2>/dev/null || echo false)"
else
    # Not fatal -- see `mqtt:want` in config.yaml.
    bashio::log.warning "No MQTT service. Entities will not be published."
fi

# The add-on's own name, not a literal: this file is shared by the stable and
# edge builds, so a hard-coded "Occupancy Forecast" makes edge's log claim to be
# stable's. `|| true` because a log line is never worth failing a start over.
addon_name="$(bashio::addon.name 2>/dev/null || true)"
bashio::log.info "Starting ${addon_name:-Occupancy Forecast} (source: $(bashio::config 'source'))"

# Everything above this line needs root: bashio reads /data/options.json and
# Supervisor's service credentials, and neither is readable by the user we are
# about to become. Everything BELOW it is a forecaster that reads history and
# writes /data, and has no business being able to modify the image it is running
# from.
#
# `s6-setuidgid` rather than a Dockerfile `USER`: s6-overlay is PID 1 here
# (`init: false` in config.yaml hands it the entrypoint) and it has to start as
# root to set its own supervision tree up. Dropping at the last exec is the only
# place that leaves both halves working.
#
# /data arrives root-owned -- it did on every install that predates this, and
# Supervisor creates it as root on new ones -- so it is handed over once, and
# then not again, because the test is cheaper than a recursive chown over an
# archive that only grows.
if [ "$(stat -c %u /data)" != "$(id -u occupancy)" ]; then
    bashio::log.info "Handing /data to the unprivileged user"
    chown -R occupancy:occupancy /data
fi

# Guarded, because if a future base image drops s6-overlay's command directory
# the alternative is a container that exits with "s6-setuidgid: not found" and no
# forecaster at all. Running as root and saying so is the better failure -- and
# since there are two outcomes, each one logs which it took.
if command -v s6-setuidgid >/dev/null 2>&1; then
    bashio::log.info "Dropping to user occupancy (uid $(id -u occupancy))"
    exec s6-setuidgid occupancy python -m occupancy_forecast.server
fi
bashio::log.warning "s6-setuidgid not found -- running as root"
exec python -m occupancy_forecast.server
