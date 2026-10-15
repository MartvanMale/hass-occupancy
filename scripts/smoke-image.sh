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

# /init stays PID 1 here, as for a user. It hands CMD a cleared environment;
# with-contenv restores it, as the shebang in standalone.sh does.
STANDALONE=0
# Runs the standalone image to its refusal; $1 is the message it must print.
expect_refusal() {
    local want="$1" out
    shift
    out=$(GUARD_INIT=0 guard_run smoke-refusal "$@" "$TAG-standalone" \
        /usr/bin/with-contenv /usr/bin/timeout -k 5 60 /standalone.sh 2>&1) \
        && { echo "FAILED: it started, expected: $want" >&2; exit 1; }
    if ! grep -qF "$want" <<<"$out"; then
        echo "FAILED: expected '$want', got:" >&2
        echo "$out" >&2
        exit 1
    fi
    echo "refuses: $want"
}

if docker image inspect "$TAG-standalone" >/dev/null 2>&1; then
    STANDALONE=1
    SETTINGS=(-e HA_URL=http://127.0.0.1:9 -e HA_TOKEN=smoke -e OCCUPANCY_PASSWORD=smoke-password)
    echo
    echo "=== $TAG-standalone: booting without Supervisor ==="
    expect_refusal "not starting: set HA_URL HA_TOKEN OCCUPANCY_PASSWORD"
    expect_refusal "not starting: set OCCUPANCY_SSL_KEY too, or neither" \
        "${SETTINGS[@]}" -e OCCUPANCY_SSL_CERT=/etc/hostname

    echo "--- with the certificate it makes itself"
    GUARD_INIT=0 guard_run smoke-standalone \
        "${SETTINGS[@]}" -e EXPECT_FINGERPRINT="$FINGERPRINT" \
        -v "$PWD/scripts/standalone_smoke.py":/standalone_smoke.py:ro \
        "$TAG-standalone" \
        /usr/bin/with-contenv /usr/bin/timeout -k 30 180 \
        /opt/venv/bin/python3 /standalone_smoke.py || status=$?
    if (( status )); then
        exit "$status"
    fi

    echo "--- with a certificate of the user's own"
    certs=$(mktemp -d)
    openssl req -x509 -newkey rsa:2048 -nodes -days 1 -subj /CN=smoke-supplied \
        -addext "subjectAltName=IP:127.0.0.1" \
        -keyout "$certs/key.pem" -out "$certs/cert.pem" 2>/dev/null
    chmod 600 "$certs"/*.pem
    GUARD_INIT=0 guard_run smoke-standalone \
        "${SETTINGS[@]}" -e EXPECT_FINGERPRINT="$FINGERPRINT" \
        -e OCCUPANCY_SSL_CERT=/certs/cert.pem -e OCCUPANCY_SSL_KEY=/certs/key.pem \
        -e EXPECT_CERT=/certs/cert.pem -v "$certs":/certs:ro \
        -v "$PWD/scripts/standalone_smoke.py":/standalone_smoke.py:ro \
        "$TAG-standalone" \
        /usr/bin/with-contenv /usr/bin/timeout -k 30 180 \
        /opt/venv/bin/python3 /standalone_smoke.py || status=$?
    rm -rf "$certs"
    if (( status )); then
        exit "$status"
    fi
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
if (( STANDALONE )); then
    echo "  the standalone image booted through /init, served HTTPS and guarded its port"
fi
if (( QEMU )); then
    echo "  and it ran on an emulated ARMv8.0 Cortex-A72"
fi
