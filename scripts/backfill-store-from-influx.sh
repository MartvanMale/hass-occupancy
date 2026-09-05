#!/usr/bin/env bash
# One-off: fill an add-on's local store from an InfluxDB archive.
#
#   ./scripts/backfill-store-from-influx.sh                    # edge, creds from stable
#   ./scripts/backfill-store-from-influx.sh local_occupancy_forecast # the other way round
#
# The store normally accumulates from the day the add-on is installed, so a
# fresh install has days of history where the Influx has months. This imports
# the whole Influx history in one pass, using the add-on's OWN InfluxSource and
# HistoryStore -- so what lands in the store is exactly what the influx source
# would have read, which is what makes the two comparable at all.
#
# Idempotent. Re-running inserts nothing. The add-on does not need to be
# stopped: the store is WAL and its primary key makes the write a no-op.
#
# Not part of the add-on, and deliberately so -- see the .py for why.
#
# CREDENTIALS. Taken from another add-on's Supervisor options by default, and
# piped container-to-container on the Home Assistant box: the token never
# reaches this machine, never appears in an argv and never lands in `ps`. To use
# an Influx no add-on is configured for, export INFLUX_URL / INFLUX_ORG /
# INFLUX_TOKEN (and optionally INFLUX_BUCKET) instead -- they are sent over the
# ssh connection's stdin, not on the command line, for the same reason.
#
# AFTERWARDS the store has history the models have never seen. Nothing retrains
# on its own until the next scheduled run, so trigger one -- the script prints
# the command.
set -euo pipefail
cd "$(dirname "$0")"

HOST=ha
TARGET=${1:-local_occupancy_forecast_edge}    # whose store gets filled
CREDS_FROM=${2:-local_occupancy_forecast}     # whose options.json holds the Influx

# Supervisor names an add-on's container `app_<slug>`. `ha addons info` would be
# the polite way to ask, but it does not report the container name, and this
# needs `docker exec` regardless.
target_container="app_${TARGET}"
creds_container="app_${CREDS_FROM}"

if [[ -n "${INFLUX_URL:-}" ]]; then
    echo "Influx from the environment: $INFLUX_URL"
    creds_cmd="cat"                            # we supply the JSON on stdin below
else
    echo "Influx from ${CREDS_FROM}'s add-on options"
    creds_cmd="docker exec $creds_container cat /data/options.json"
fi

echo "Filling ${TARGET}'s store. This is safe to re-run."
echo

# Two hops on purpose. The payload goes in first as a file, which leaves the
# container's stdin free to carry the credentials into the interpreter.
ssh "$HOST" "docker exec -i $target_container sh -c 'cat > /tmp/backfill.py'" \
    < backfill-store-from-influx.py

if [[ -n "${INFLUX_URL:-}" ]]; then
    printf '{"influx_url":"%s","influx_org":"%s","influx_bucket":"%s","influx_token":"%s"}' \
        "$INFLUX_URL" "${INFLUX_ORG:-}" "${INFLUX_BUCKET:-homeassistant}" "${INFLUX_TOKEN:-}" \
        | ssh "$HOST" "docker exec -i $target_container python /tmp/backfill.py"
else
    ssh "$HOST" "$creds_cmd | docker exec -i $target_container python /tmp/backfill.py"
fi

ssh "$HOST" "docker exec $target_container rm -f /tmp/backfill.py"

cat <<EOF

Done. The models have not seen any of this yet -- retrain with:

  ssh $HOST 'docker exec $target_container python -c "
import urllib.request
urllib.request.urlopen(urllib.request.Request(
    \"http://127.0.0.1:8099/train\", data=b\"\", method=\"POST\"), timeout=2700)"'

Takes a few minutes: 48 horizon models plus one ETA model per person.
EOF
