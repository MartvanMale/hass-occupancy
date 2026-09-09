#!/usr/bin/env bash
# Run when a numerical pin in <tree>/requirements.txt changes (~75s).
# Checks both shipping architectures: amd64 by installing and RUNNING the wheels,
# aarch64 -- which cannot run here -- by disassembling them for ARMv8.1 LSE
# outside libgcc's dispatch, the pyarrow 21.0.0 bug that aborted on a Pi 4.
# Not in test.sh: no network there, and it runs the developer's native arch.
#
#   scripts/check-pins.sh [tree] [--only arm|amd64] [--update-baseline]
#
# --update-baseline is honest only once the pin has run on the actual Pi.
set -euo pipefail
cd "$(dirname "$0")/.."
. scripts/container-guard.sh

TREE="occupancy-forecast-edge"
ONLY="both"
UPDATE=""
while (( $# )); do
    case "$1" in
        --only) ONLY="${2:-}"; shift 2 || true ;;
        --only=*) ONLY="${1#*=}"; shift ;;
        --update-baseline) UPDATE="--update-baseline"; shift ;;
        -h|--help)
            echo "usage: $(basename "$0") [tree] [--only arm|amd64] [--update-baseline]"
            exit 0 ;;
        -*) echo "error: unknown argument '$1'" >&2
            echo "usage: $(basename "$0") [tree] [--only arm|amd64] [--update-baseline]" >&2
            exit 2 ;;
        *) TREE="$1"; shift ;;
    esac
done

case "$ONLY" in
    both|arm|amd64) ;;
    *) echo "error: --only takes 'arm' or 'amd64', not '$ONLY'" >&2; exit 2 ;;
esac

if [[ ! -f "$TREE/requirements.txt" ]]; then
    echo "error: $TREE/requirements.txt does not exist." >&2
    exit 2
fi

guard_start

amd64_status="skipped"
arm_status="skipped"

# Both halves run even if the first fails, so one invocation reports everything.
# Note the `|| status=$?` on each: inside an `if ! cmd` branch $? is the status
# of the negation and is always 0, which reports a failed run as a pass.
if [[ "$ONLY" != "arm" ]]; then
    echo "=== amd64: executing the pinned stack ==="
    status=0
    guard_run pins-amd64 --platform linux/amd64 \
        -e HOME=/tmp -e PIP_DISABLE_PIP_VERSION_CHECK=1 \
        -v "$PWD/$TREE/requirements.txt":/requirements.txt:ro \
        -v "$PWD/scripts/cpu_smoke.py":/cpu_smoke.py:ro \
        python:3.13-slim \
        timeout -k 30 600 \
        sh -c 'pip install --quiet --no-cache-dir --root-user-action=ignore \
                   -r /requirements.txt \
               && exec python /cpu_smoke.py' || status=$?
    if (( status )); then
        amd64_status="FAILED (exit $status)"
        echo "note: exit 132 or 'Illegal instruction' means a wheel used an" >&2
        echo "      instruction this CPU lacks; the last '...' line names the step." >&2
    else
        amd64_status="passed"
    fi
    echo
fi

# Root here, unlike above: apt-get needs it for the cross binutils. Hence the
# chown -- --update-baseline is the only write to the repo, and a root-owned
# file would outlive the run.
if [[ "$ONLY" != "amd64" ]]; then
    echo "=== aarch64: reading the instructions ==="
    status=0
    guard_run pins-arm \
        -e HOME=/tmp -e PIP_DISABLE_PIP_VERSION_CHECK=1 \
        -e OWNER="$(id -u):$(id -g)" \
        -v "$PWD":/w -w /w \
        python:3.13-slim \
        timeout -k 30 900 \
        sh -c '
            set -e
            apt-get update -qq >/dev/null 2>&1 || {
                echo "error: apt-get update failed (no network?)" >&2; exit 2; }
            apt-get install -y -qq --no-install-recommends \
                binutils-aarch64-linux-gnu >/dev/null 2>&1
            inner=0
            python scripts/arm_baseline.py "$1" scripts/arm-baseline.json $2 || inner=$?
            if [ -f scripts/arm-baseline.json ]; then
                chown "$OWNER" scripts/arm-baseline.json
            fi
            exit $inner
        ' sh "$TREE/requirements.txt" "$UPDATE" || status=$?
    if (( status )); then
        arm_status="FAILED (exit $status)"
    else
        arm_status="passed"
    fi
    echo
fi

echo "=== $TREE ==="
echo "  amd64   (executed):            $amd64_status"
echo "  aarch64 (static LSE scan):     $arm_status"

if [[ "$amd64_status" == FAILED* || "$arm_status" == FAILED* ]]; then
    exit 1
fi
