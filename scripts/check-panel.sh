#!/usr/bin/env bash
# Fail if <tree>/panel/dist was not built from <tree>/panel's current source.
#
# The bundle is committed and the Dockerfile only COPYs it, so a stale dist/ is
# not a build error -- it is an add-on that installs cleanly and serves last
# week's panel, with nothing anywhere saying so. This is the check that makes
# that impossible: .githooks/pre-commit runs it before a commit that touches
# panel source, and deploy-stable.sh runs it before shipping.
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
