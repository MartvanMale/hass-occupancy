#!/usr/bin/env bash
# Deploy the edge add-on to the Home Assistant box and rebuild it. Edge is a
# LOCAL add-on in `ha:/addons/occupancy_forecast_edge/`, so deploying is a file
# copy plus a rebuild -- no version bump, no push, nothing published.
set -euo pipefail
cd "$(dirname "$0")/.."

HOST=ha
DIR=/addons/occupancy_forecast_edge
SLUG=local_occupancy_forecast_edge

# Stamp the deployed copy with its commit; the committed config.yaml keeps its
# plain `-dev`. Build the panel first, so a failed build costs nothing.
scripts/build-panel.sh occupancy-forecast-edge

sha=$(git rev-parse --short HEAD)

stamped=$(mktemp -d)
trap 'rm -rf "$stamped"' EXIT
# rsync, not `cp -a`: node_modules has no business being staged or stamped.
rsync -a --exclude 'node_modules' occupancy-forecast-edge/ "$stamped/"

# The dirty marker hashes the staged tree, not the bare word: the branch below
# keys off the version string, and an identical one means rebuild -- which does
# NOT reload apparmor.txt.
dirty=""
if [[ -n "$(git status --porcelain occupancy-forecast-edge/)" ]]; then
    # Relative paths and LC_ALL=C, or mktemp's name lands in every line and two
    # shells stamp one tree differently.
    dirty=".dirty$( (cd "$stamped" && find . -type f -print0 | LC_ALL=C sort -z \
                     | xargs -0 sha1sum) | sha1sum | cut -c1-7)"
    echo "note: deploying uncommitted changes; version will be marked ${dirty#.}"
fi

sed -i -E "s/^version: \"([^\"]+)\"/version: \"\1.${sha}${dirty}\"/" "$stamped/config.yaml"
grep '^version:' "$stamped/config.yaml"

# dist is NOT excluded: it is the artifact this deploy ships, and the box cannot
# build it.
rsync -a --delete \
  --exclude '__pycache__' --exclude '*.pyc' --exclude '.pytest_cache' \
  --exclude 'node_modules' \
  "$stamped/" "$HOST:$DIR/"

# `ha store reload` first: only it re-reads config.yaml, which shows the stamp.
# `update`, not `rebuild`: rebuild does NOT reload apparmor.txt, so it is only
# the fallback for redeploying one commit. The test reads `"installed": false`,
# not the exit status, which is 0 once the store has seen the directory.
ssh "$HOST" "
    set -e
    ha store reload
    if [ \"\$(ha addons info $SLUG --raw-json | grep -o '\"installed\": *false')\" ]; then
        ha addons install $SLUG
        echo 'installed but NOT started -- start it once /data is in place'
    elif [ \"\$(ha addons info $SLUG --raw-json | grep -o '\"update_available\": *true')\" ]; then
        ha addons update $SLUG
    else
        ha addons rebuild $SLUG
        echo 'note: rebuilt, not updated -- the AppArmor profile was NOT reloaded'
    fi
"

echo
echo "Deployed. Watch it come up with:"
echo "  ssh $HOST 'ha addons logs $SLUG'"
