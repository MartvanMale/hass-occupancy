#!/usr/bin/env bash
# Deploy the edge add-on to the Home Assistant box and rebuild it.
# Edge is a LOCAL add-on in `ha:/addons/occupancy_forecast_edge/`, so deploying
# is a file copy plus a rebuild -- no version bump, no push, nothing published.
# Stable has no deploy script: it installs from this repo's URL, so a version
# bump and a push are its trigger. See DEVELOPMENT.md.
set -euo pipefail
cd "$(dirname "$0")/.."

HOST=ha
DIR=/addons/occupancy_forecast_edge
SLUG=local_occupancy_forecast_edge

# Stamp the deployed copy with its commit, so the add-on page says which build is
# running. The stamp goes ONLY to the box; the committed config.yaml keeps its
# plain `-dev`. Build the panel first, so a failed build costs nothing.
scripts/build-panel.sh occupancy-forecast-edge

sha=$(git rev-parse --short HEAD)

stamped=$(mktemp -d)
trap 'rm -rf "$stamped"' EXIT
# rsync, not `cp -a`: node_modules has no business being staged or stamped.
rsync -a --exclude 'node_modules' occupancy-forecast-edge/ "$stamped/"

# The dirty marker hashes the staged tree rather than being the bare word
# ".dirty", because the update/rebuild branch below keys off the version string:
# an identical one means rebuild, and rebuild does NOT reload apparmor.txt. Two
# different dirty trees at one commit would then enforce a stale profile
# silently. Hashed before the stamp is written, so it is not a hash of itself.
dirty=""
if [[ -n "$(git status --porcelain occupancy-forecast-edge/)" ]]; then
    # Relative paths, or mktemp's random directory name lands in every line and
    # the hash changes every run. LC_ALL=C because sort collates by locale, so
    # two shells would otherwise stamp one tree differently.
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

# `ha store reload` first: only it re-reads config.yaml, so without it the new
# version stamp is invisible. Then `update` rather than `rebuild`, for two
# reasons -- rebuild leaves the INSTALLED version behind, and it does NOT reload
# apparmor.txt (Supervisor calls install_apparmor() from install() and update()
# only). Rebuild survives as the fallback for redeploying the same commit, and
# says so. `install` first for a box that has never seen this add-on, including
# the first deploy after a slug change; it deliberately does not start it, so
# the previous slug's /data can be copied across first. The test reads
# `"installed": false` rather than the exit status, because once the store has
# seen the directory `ha addons info` returns 0 for a add-on never built.
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
