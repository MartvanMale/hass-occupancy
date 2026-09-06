#!/usr/bin/env bash
# Deploy the edge add-on to the Home Assistant box and rebuild it.
#
# These are LOCAL add-ons -- they live in `ha:/addons/<slug>/` and Supervisor
# builds them from that directory. They are not installed from GitHub, so
# deploying is a file copy plus a rebuild: no version bump needed, no push
# needed, nothing published. `ha addons rebuild` rebuilds unconditionally,
# which is why the edge loop can be this short.
#
# (An add-on installed from a repository URL updates only when config.yaml's
# `version:` changes. If this repo is ever published and installed that way,
# that becomes the trigger instead and this script stops being the whole story.)
#
# Run it as often as you like. It only ever touches occupancy_forecast_edge; the
# stable add-on is deployed by deploy-stable.sh and by nothing else.
set -euo pipefail
cd "$(dirname "$0")/.."

HOST=ha
DIR=/addons/occupancy_forecast_edge
SLUG=local_occupancy_forecast_edge

# Stamp the deployed copy with the commit it came from, so the add-on page in
# Home Assistant tells you exactly which build is running. Without this every
# edge build shows the same version string and "is it running my latest change?"
# has no answer short of reading the logs.
#
# The stamp goes ONLY into the copy on the HA box. The committed config.yaml
# keeps its plain `0.2.0-dev`, because a version that changes on every commit
# would put a diff in every commit.
# The panel is compiled here, never on the HA box; the Dockerfile only COPYs
# dist/. Build before the version stamp so a failed build costs nothing.
scripts/build-panel.sh occupancy-forecast-edge

sha=$(git rev-parse --short HEAD)
dirty=""
if [[ -n "$(git status --porcelain occupancy-forecast-edge/)" ]]; then
    dirty=".dirty"
    echo "note: deploying uncommitted changes; version will be marked .dirty"
fi

stamped=$(mktemp -d)
trap 'rm -rf "$stamped"' EXIT
# rsync, not `cp -a`: node_modules is 60-odd MB of toolchain that has no
# business being staged, let alone stamped and diffed.
rsync -a --exclude 'node_modules' occupancy-forecast-edge/ "$stamped/"
sed -i -E "s/^version: \"([^\"]+)\"/version: \"\1.${sha}${dirty}\"/" "$stamped/config.yaml"
grep '^version:' "$stamped/config.yaml"

# node_modules is excluded and dist is NOT: the bundle is the artifact this
# deploy exists to ship, and the box has no way to produce it.
rsync -a --delete \
  --exclude '__pycache__' --exclude '*.pyc' --exclude '.pytest_cache' \
  --exclude 'node_modules' \
  "$stamped/" "$HOST:$DIR/"

# `ha store reload`, not `ha addons reload`. Supervisor caches each add-on's
# manifest, and only the store reload re-reads config.yaml off disk -- without it
# the new version stamp is invisible and Supervisor builds against the manifest
# it read last time.
#
# Then `update`, not `rebuild`. A rebuild rebuilds the image but leaves the
# INSTALLED version where it was, so the add-on page keeps showing the old stamp
# while running the new code -- which defeats the point of stamping it. An update
# does both. It is only offered when the version actually changed, so fall back
# to a rebuild for the case where nothing moved (a redeploy of the same commit).
#
# And `install` first, for a box that has never seen this add-on -- which is the
# case on a fresh machine and, less obviously, the first deploy after the slug
# changes: a renamed add-on is a NEW add-on to Supervisor, and `rebuild` on
# something uninstalled is an error that `set -e` turns into a failed deploy.
# `install` deliberately does NOT start it, which is what you want when the
# previous slug's /data still has to be copied across before the first run.
# Note the install test reads \`\"installed\": false\`, not the exit status of
# \`ha addons info\`. Once \`ha store reload\` has seen the directory the add-on
# EXISTS and info returns 0 for it, reporting \`\"version\": null\`, \`\"state\":
# \"unknown\"\` -- so an exit-status test says \"installed\" for something that has
# never been built.
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
    fi
"

echo
echo "Deployed. Watch it come up with:"
echo "  ssh $HOST 'ha addons logs $SLUG'"
