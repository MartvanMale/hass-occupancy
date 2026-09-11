#!/usr/bin/with-contenv bashio
# Read the add-on options and Supervisor's MQTT service, then hand over.
#
# Everything below is passed as environment rather than written into a config
# file, because Supervisor owns these values and re-injects them on every start.

set -e

# The ADD-ON OPTION is authoritative and overrides /data/config.json: the option
# was otherwise accepted, shown in the UI and ignored.
export OCCUPANCY_SOURCE="$(bashio::config 'source')"

# log.py maps bashio's seven level names onto Python's five.
export LOG_LEVEL="$(bashio::config 'log_level')"

if bashio::config.equals 'source' 'influx'; then
    export INFLUX_URL="$(bashio::config 'influx_url')"
    export INFLUX_ORG="$(bashio::config 'influx_org')"
    export INFLUX_BUCKET="$(bashio::config 'influx_bucket')"
    export INFLUX_TOKEN="$(bashio::config 'influx_token')"
fi

# Empty means everyone, which is what every existing install has. `admin_users // []`
# because jq's `join` on an absent key is a hard error, and under `set -e` that
# is a container that will not start.
export OCCUPANCY_ADMIN_USERS="$(bashio::config 'admin_users // [] | join(",")')"

# An explicit broker wins over the Supervisor service: a broker outside Supervisor registers no service.
if bashio::config.has_value 'mqtt_host'; then
    export MQTT_HOST="$(bashio::config 'mqtt_host')"
    export MQTT_PORT="$(bashio::config 'mqtt_port')"
    export MQTT_USER="$(bashio::config 'mqtt_user')"
    export MQTT_PASSWORD="$(bashio::config 'mqtt_password')"
    export MQTT_SSL="$(bashio::config 'mqtt_ssl')"
    bashio::log.info "MQTT broker from the add-on options: ${MQTT_HOST}:${MQTT_PORT}"
elif bashio::services.available 'mqtt'; then
    export MQTT_HOST="$(bashio::services 'mqtt' 'host')"
    export MQTT_PORT="$(bashio::services 'mqtt' 'port')"
    export MQTT_USER="$(bashio::services 'mqtt' 'username')"
    export MQTT_PASSWORD="$(bashio::services 'mqtt' 'password')"
    # Whether the broker wants TLS. Ignored until now, so a TLS-only broker
    # got a plaintext CONNECT and the add-on reported it unavailable.
    export MQTT_SSL="$(bashio::services 'mqtt' 'ssl' 2>/dev/null || echo false)"
else
    # Not fatal -- see `mqtt:want` in config.yaml.
    bashio::log.warning "No MQTT broker: no mqtt_host option and no Supervisor mqtt service. Entities will not be published."
fi

# The add-on's own name, not a literal: this file is shared, so a hardcoded one
# makes edge's log claim to be stable's. `|| true`: a log line is not worth a failed start.
addon_name="$(bashio::addon.name 2>/dev/null || true)"
bashio::log.info "Starting ${addon_name:-Occupancy Forecast} (source: $(bashio::config 'source'))"

# Everything above needs root (bashio reads /data/options.json and Supervisor's
# credentials); everything below is a forecaster with no business writing the
# image. `s6-setuidgid` rather than a Dockerfile `USER`, because s6-overlay is
# PID 1 and must start as root -- the last exec is the only place both work.
# Guarded: under `set -e` a refused chown kills the container, which is worse than running as root.
if [ "$(stat -c %u /data)" != "$(id -u occupancy)" ]; then
    bashio::log.info "Handing /data to the unprivileged user"
    if ! chown -R occupancy:occupancy /data; then
        bashio::log.warning "Could not take ownership of /data -- running as root"
        exec python -m occupancy_forecast.server
    fi
fi

# Guarded: a base image without s6-setuidgid would otherwise exit with "not found"
# and no forecaster. Root and saying so is the better failure.
if command -v s6-setuidgid >/dev/null 2>&1; then
    bashio::log.info "Dropping to user occupancy (uid $(id -u occupancy))"
    exec s6-setuidgid occupancy python -m occupancy_forecast.server
fi
bashio::log.warning "s6-setuidgid not found -- running as root"
exec python -m occupancy_forecast.server
