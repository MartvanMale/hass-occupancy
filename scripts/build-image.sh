#!/usr/bin/env bash
# Build <tree>'s real add-on image, the one Supervisor builds; then run
# scripts/smoke-image.sh. Native only, so arch defaults to this machine's.
#
#   scripts/build-image.sh [tree] [arch]     arch: amd64 | aarch64
set -euo pipefail
cd "$(dirname "$0")/.."
. scripts/container-guard.sh

TREE="${1:-occupancy-forecast-edge}"

case "${2:-$(uname -m)}" in
    amd64|x86_64)   ARCH=amd64 ;;
    aarch64|arm64)  ARCH=aarch64 ;;
    *) echo "error: unsupported arch '${2:-$(uname -m)}' -- amd64 or aarch64" >&2
       exit 2 ;;
esac

if [[ ! -f "$TREE/build.yaml" ]]; then
    echo "error: $TREE/build.yaml does not exist." >&2
    exit 2
fi

# From build.yaml, the file Supervisor reads, so the tags exist in one place.
BASE=$(sed -n "s/^  ${ARCH}: *//p" "$TREE/build.yaml")
if [[ -z "$BASE" ]]; then
    echo "error: no build_from for '$ARCH' in $TREE/build.yaml" >&2
    exit 2
fi

TAG="occupancy-forecast-image:$TREE-$ARCH"

echo "=== building $TREE for $ARCH ==="
echo "    BUILD_FROM $BASE"
guard_start
docker build --build-arg "BUILD_FROM=$BASE" -t "$TAG" "$TREE"

echo
echo "built $TAG"
