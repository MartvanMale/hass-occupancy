#!/usr/bin/env bash
# Build the Ingress panel into <tree>/panel/dist.
#
# The panel is a React app and Vite compiles it, but never on the Home Assistant
# box, and the result is committed. DEVELOPMENT.md ("The panel") is the one place
# that argument is written out. This repo's stand-in for CI is the deploy
# scripts, and this is the step they run first; the Dockerfile only COPYs it.
#
# The script finishes by stamping dist/ with a hash of its inputs, so a stale
# bundle can be detected later -- see scripts/panel-source-hash.sh.
#
# In a container rather than a local node, for the same reason scripts/test.sh
# runs pytest in one: the toolchain is pinned by package-lock.json and nothing
# has to be installed on the host.
#
# `--user` is not optional. Without it npm writes node_modules/ and dist/ back
# into the working tree owned by root, and the next non-root build cannot
# overwrite them.
set -euo pipefail
cd "$(dirname "$0")/.."

TREE="${1:-occupancy-forecast-edge}"
PANEL="$PWD/$TREE/panel"
[[ -d "$PANEL" ]] || { echo "no panel in $TREE" >&2; exit 1; }

# A named volume for npm's cache, so a rebuild is not a re-download. It is
# created root-owned, which the --user below cannot write to, so claim it once.
if ! docker volume inspect occupancy-panel-npm >/dev/null 2>&1; then
    docker volume create occupancy-panel-npm >/dev/null
    docker run --rm -v occupancy-panel-npm:/cache alpine \
        chown -R "$(id -u):$(id -g)" /cache
fi

docker run --rm \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e npm_config_cache=/tmp/.npm \
  -v "$PANEL":/w -w /w \
  -v occupancy-panel-npm:/tmp/.npm \
  node:22-alpine \
  sh -c '[ -d node_modules ] || npm ci --no-audit --no-fund; npm run build'

# After the build, never before: vite.config.ts sets `emptyOutDir: true`, so
# anything written into dist/ ahead of this is deleted by it. The file sits at
# dist/ root rather than in dist/assets, which is the directory web/__init__.py
# mounts -- so it is not reachable over Ingress.
scripts/panel-source-hash.sh "$TREE" > "$TREE/panel/dist/.source-hash"

echo "built $TREE/panel/dist  (source-hash $(cut -c1-12 < "$TREE/panel/dist/.source-hash"))"
