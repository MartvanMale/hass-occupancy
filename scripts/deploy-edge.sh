#!/usr/bin/env bash
# Deploy the edge add-on to the Home Assistant box and rebuild it.
#
# Edge is a LOCAL add-on -- it lives in `ha:/addons/occupancy_forecast_edge/` and
# Supervisor builds it from that directory. It is not installed from GitHub, so
# deploying is a file copy plus a rebuild: no version bump needed, no push
# needed, nothing published. That is why the edge loop can be this short.
#
# Run it as often as you like. It only ever touches occupancy_forecast_edge.
# Stable is not deployed by any script: it installs from this repository's URL,
# so a version bump and a push are its trigger. See DEVELOPMENT.md.
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

stamped=$(mktemp -d)
trap 'rm -rf "$stamped"' EXIT
# rsync, not `cp -a`: node_modules is 60-odd MB of toolchain that has no
# business being staged, let alone stamped and diffed.
rsync -a --exclude 'node_modules' occupancy-forecast-edge/ "$stamped/"

# The dirty marker carries a hash of the staged tree, not the bare word
# ".dirty". Two different uncommitted trees at the same commit used to stamp
# the SAME version string, and the update/rebuild branch below keys off exactly
# that string: identical version means `update_available: false` means
# `ha addons rebuild`. Rebuild is fine for code -- but Supervisor reloads the
# AppArmor profile from install_apparmor(), which App.install() and App.update()
# call and App.rebuild() does not. So an edit to apparmor.txt on a tree that was
# already dirty at this commit deployed the new file and kept enforcing the old
# profile, with nothing anywhere saying so. Hashing the tree makes every
# distinct deploy a distinct version, which puts it back on the update path.
#
# Hashed before the stamp is written, so the hash is of the source and not of
# itself.
dirty=""
if [[ -n "$(git status --porcelain occupancy-forecast-edge/)" ]]; then
    # Relative paths, from inside the staging directory: `find "$stamped"` puts
    # mktemp's random directory name into every line of sha1sum's output, so the
    # hash would change on every run and the point -- a stable identity for an
    # unchanged tree -- would be lost.
    dirty=".dirty$( (cd "$stamped" && find . -type f -print0 | sort -z \
                     | xargs -0 sha1sum) | sha1sum | cut -c1-7)"
    echo "note: deploying uncommitted changes; version will be marked ${dirty#.}"
fi

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
# The second reason, and the one with teeth: `update` reloads apparmor.txt and
# `rebuild` does not. Supervisor calls install_apparmor() from App.install() and
# App.update() only. The version stamp above now hashes a dirty tree, so any real
# edit lands on the update path -- but the rebuild branch survives for the
# redeploy-the-same-thing case, and it says out loud that the profile is
# whatever was loaded last.
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
        echo 'note: rebuilt, not updated -- the AppArmor profile was NOT reloaded'
    fi
"

echo
echo "Deployed. Watch it come up with:"
echo "  ssh $HOST 'ha addons logs $SLUG'"
