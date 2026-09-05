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

# The log_level option has been in config.yaml and in the Configuration tab
# since the first release, and until now nothing read it: it was a knob that
# did nothing. occupancy_forecast/log.py maps bashio's seven level names onto
# Python's five.
export LOG_LEVEL="$(bashio::config 'log_level')"

if bashio::config.equals 'source' 'influx'; then
    export INFLUX_URL="$(bashio::config 'influx_url')"
    export INFLUX_ORG="$(bashio::config 'influx_org')"
    export INFLUX_BUCKET="$(bashio::config 'influx_bucket')"
    export INFLUX_TOKEN="$(bashio::config 'influx_token')"
fi

if bashio::services.available 'mqtt'; then
    export MQTT_HOST="$(bashio::services 'mqtt' 'host')"
    export MQTT_PORT="$(bashio::services 'mqtt' 'port')"
    export MQTT_USER="$(bashio::services 'mqtt' 'username')"
    export MQTT_PASSWORD="$(bashio::services 'mqtt' 'password')"
else
    # Not fatal. The add-on still collects history, still trains and still
    # answers its API -- it just cannot publish entities until a broker exists.
    bashio::log.warning "No MQTT service. Entities will not be published."
fi

# The add-on's own name, not a literal: this file is shared by the stable and
# edge builds, so a hard-coded "Occupancy Forecast" makes edge's log claim to be
# stable's. `|| true` because a log line is never worth failing a start over.
addon_name="$(bashio::addon.name 2>/dev/null || true)"
bashio::log.info "Starting ${addon_name:-Occupancy Forecast} (source: $(bashio::config 'source'))"
exec python -m occupancy_forecast.server
