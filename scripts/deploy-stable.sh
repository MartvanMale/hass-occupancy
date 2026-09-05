#!/usr/bin/env bash
# Deploy the STABLE add-on to the Home Assistant box and rebuild it.
#
# This is the one that touches the add-on you trust -- the one whose forecast
# `home_state.py` reads to decide the house is "Arriving soon". It is deliberately
# fussier than deploy-edge.sh, and it is not something to run casually.
#
# It refuses unless the stable tree is committed AND identical to what
# promote.sh would produce from edge. Both checks exist because occupancy-forecast/ is
# generated: a hand-edit there survives only until the next promotion, so
# deploying one would put code on the HA box that exists nowhere else and
# vanishes without trace the moment somebody promotes.
set -euo pipefail
cd "$(dirname "$0")/.."

HOST=ha
DIR=/addons/occupancy_forecast
SLUG=local_occupancy_forecast

if [[ -n "$(git status --porcelain occupancy-forecast/)" ]]; then
    echo "error: occupancy-forecast/ has uncommitted changes. Commit the promotion first." >&2
    git status --short occupancy-forecast/ >&2
    exit 1
fi

# Dry-run the generator. If it would change anything, the stable tree is not a
# clean promotion of edge and we do not know what we would be shipping.
drift=$(rsync -an --delete --out-format='%n' \
  --exclude '__pycache__' --exclude '*.pyc' --exclude '.pytest_cache' \
  --exclude 'node_modules' --exclude 'dist' \
  --exclude 'config.yaml' --exclude 'DOCS.md' --exclude 'CHANGELOG.md' \
  occupancy-forecast-edge/ occupancy-forecast/ | grep -v '/$' || true)
if [[ -n "$drift" ]]; then
    echo "error: occupancy-forecast/ is not a clean promotion of occupancy-forecast-edge/." >&2
    echo "       These files differ -- run scripts/promote.sh:" >&2
    printf '  %s\n' $drift >&2
    exit 1
fi

version=$(grep -oP '^version: "\K[^"]+' occupancy-forecast/config.yaml)
echo "About to deploy Occupancy Forecast $version to $HOST:$DIR and rebuild it."
read -rp "Type the version to confirm: " confirm
[[ "$confirm" == "$version" ]] || { echo "aborted"; exit 1; }

# Check the committed bundle, do not rebuild it. promote.sh builds stable's
# panel from stable's own promoted source so that the bundle is in the
# promotion commit; building here instead would dirty the tree the clean check
# above just validated, and ship something that is in no commit anywhere.
scripts/check-panel.sh occupancy-forecast

rsync -a --delete \
  --exclude '__pycache__' --exclude '*.pyc' --exclude '.pytest_cache' \
  --exclude 'node_modules' \
  occupancy-forecast/ "$HOST:$DIR/"

ssh "$HOST" "ha addons reload && ha addons rebuild $SLUG"

echo
echo "Deployed $version. Watch it come up with:"
echo "  ssh $HOST 'ha addons logs $SLUG'"
echo "Tag it so the release point is recoverable:"
echo "  git tag v$version && git push --tags   # push needs Mart's go-ahead"
