#!/usr/bin/env bash
# Fail if <tree>/panel/dist was not built from <tree>/panel's current source.
#
# The bundle is committed and the Dockerfile only COPYs it, so a stale dist/ is
# not a build error -- it is an add-on that installs cleanly and serves last
# week's panel, with nothing anywhere saying so.
#
# scripts/test.sh runs this, so a green suite means the committed bundle matches
# its source. promote.sh does not need it -- it rebuilds both bundles itself, so
# the pair is fresh by construction at promotion.
set -euo pipefail
cd "$(dirname "$0")/.."

TREE="${1:-occupancy-forecast-edge}"
STAMP="$TREE/panel/dist/.source-hash"

if [[ ! -f "$TREE/panel/dist/index.html" ]]; then
    echo "error: $TREE/panel/dist is missing -- the panel was never built." >&2
    echo "       scripts/build-panel.sh $TREE" >&2
    exit 1
fi

if [[ ! -f "$STAMP" ]]; then
    echo "error: $STAMP is missing." >&2
    echo "       The bundle predates the freshness stamp; rebuild it." >&2
    echo "       scripts/build-panel.sh $TREE" >&2
    exit 1
fi

want=$(scripts/panel-source-hash.sh "$TREE")
have=$(cat "$STAMP")

if [[ "$want" != "$have" ]]; then
    echo "error: $TREE/panel/dist is stale -- built from different source." >&2
    echo "       source $want" >&2
    echo "       dist   $have" >&2
    echo "       scripts/build-panel.sh $TREE" >&2
    exit 1
fi
