#!/usr/bin/env bash
# Exercise the image scripts/build-image.sh built: the pinned stack inside it,
# then a server boot read back through /health. --qemu-a72 adds the stack under
# qemu-user as a Pi 4's Cortex-A72 (needs qemu-aarch64-static; takes minutes).
#
#   scripts/smoke-image.sh [tree] [arch] [--qemu-a72]
set -euo pipefail
cd "$(dirname "$0")/.."
. scripts/container-guard.sh

TREE="occupancy-forecast-edge"
ARCH=""
QEMU=0
positional=0
while (( $# )); do
    case "$1" in
        --qemu-a72) QEMU=1; shift ;;
        -h|--help)
            echo "usage: $(basename "$0") [tree] [arch] [--qemu-a72]"; exit 0 ;;
        -*) echo "error: unknown argument '$1'" >&2
            echo "usage: $(basename "$0") [tree] [arch] [--qemu-a72]" >&2
            exit 2 ;;
        *) if (( positional == 0 )); then TREE="$1"; else ARCH="$1"; fi
           positional=$(( positional + 1 )); shift ;;
    esac
done

case "${ARCH:-$(uname -m)}" in
    amd64|x86_64)   ARCH=amd64 ;;
    aarch64|arm64)  ARCH=aarch64 ;;
    *) echo "error: unsupported arch '${ARCH:-$(uname -m)}'" >&2; exit 2 ;;
esac

TAG="occupancy-forecast-image:$TREE-$ARCH"
if ! docker image inspect "$TAG" >/dev/null 2>&1; then
    echo "error: $TAG does not exist -- scripts/build-image.sh $TREE $ARCH" >&2
    exit 2
fi

if (( QEMU )) && [[ "$ARCH" != aarch64 ]]; then
    echo "error: --qemu-a72 needs the aarch64 image, not $ARCH." >&2
    exit 2
fi

QEMU_BIN=/usr/bin/qemu-aarch64-static
if (( QEMU )) && [[ ! -x "$QEMU_BIN" ]]; then
    echo "error: $QEMU_BIN is missing." >&2
    echo "       apt-get install --no-install-recommends qemu-user-static" >&2
    exit 2
fi

# server.code_fingerprint() over the tree, for /health to be checked against.
FINGERPRINT=$(python3 - "$TREE" <<'PY'
import hashlib, sys
from pathlib import Path
digest = hashlib.sha256()
for path in sorted((Path(sys.argv[1]) / "occupancy_forecast").glob("*.py")):
    digest.update(path.name.encode())
    digest.update(path.read_bytes())
print(digest.hexdigest()[:12])
PY
)

# The base's ENTRYPOINT is s6's /init; `timeout` replaces it and keeps the guard's
# in-container timeout.
guard_start
status=0

echo "=== $TAG: the pinned stack, inside the image ==="
guard_run smoke-cpu \
    --entrypoint /usr/bin/timeout \
    -e HOME=/tmp \
    -v "$PWD/scripts/cpu_smoke.py":/cpu_smoke.py:ro \
    "$TAG" \
    -k 30 600 /opt/venv/bin/python3 /cpu_smoke.py || status=$?
if (( status )); then
    echo "note: exit 132 or 'Illegal instruction' means a wheel used an" >&2
    echo "      instruction this CPU lacks; the last '...' line names the step." >&2
    exit "$status"
fi
echo

echo "=== $TAG: booting the server ==="
guard_run smoke-boot \
    --entrypoint /usr/bin/timeout \
    -e HOME=/tmp -e EXPECT_FINGERPRINT="$FINGERPRINT" \
    -v "$PWD/scripts/boot_smoke.py":/boot_smoke.py:ro \
    "$TAG" \
    -k 30 180 /opt/venv/bin/python3 /boot_smoke.py || status=$?
if (( status )); then
    exit "$status"
fi

if (( QEMU )); then
    echo
    echo "=== $TAG: the pinned stack on an emulated Cortex-A72 ==="
    "$QEMU_BIN" -version | head -1
    # --require-no-lse catches a dropped QEMU_CPU, which would mean -cpu max.
    guard_run smoke-a72 \
        --entrypoint /usr/bin/timeout \
        -e HOME=/tmp -e QEMU_CPU=cortex-a72 -e OMP_NUM_THREADS=1 \
        -e OPENBLAS_NUM_THREADS=1 \
        -v "$QEMU_BIN":/qemu-aarch64-static:ro \
        -v "$PWD/scripts/cpu_smoke.py":/cpu_smoke.py:ro \
        "$TAG" \
        -k 60 2400 /qemu-aarch64-static \
        /opt/venv/bin/python3 /cpu_smoke.py --require-no-lse || status=$?
    if (( status )); then
        exit "$status"
    fi
fi

echo
echo "=== $TREE ($ARCH) ==="
echo "  the pinned stack ran inside the image"
echo "  the server booted and served /health, code $FINGERPRINT"
if (( QEMU )); then
    echo "  and it ran on an emulated ARMv8.0 Cortex-A72"
fi
