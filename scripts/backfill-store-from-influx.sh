#!/usr/bin/env bash
# One-off: fill an add-on's local store from an InfluxDB archive.
#   ./scripts/backfill-store-from-influx.sh [target_slug] [creds_from_slug]
# Credentials are piped container to container, so the token never reaches this
# machine, an argv or `ps`; INFLUX_URL/_ORG/_TOKEN/_BUCKET in the environment
# override them, over ssh stdin for the same reason. Nothing retrains afterwards.
set -euo pipefail
cd "$(dirname "$0")"

HOST=ha
TARGET=${1:-local_occupancy_forecast_edge}    # whose store gets filled
CREDS_FROM=${2:-local_occupancy_forecast}     # whose options.json holds the Influx

# Supervisor names an add-on's container `app_<slug>`; `ha addons info` does not
# report it, and this needs `docker exec` regardless.
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

# Two hops: the payload goes in as a file, leaving stdin free for the credentials.
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
