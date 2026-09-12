#!/usr/bin/env bash
# Run the suite against the edge tree, the source of truth; occupancy-forecast/
# is generated from it. In a container so the Python and the pinned deps are the
# ones the add-on ships. No network, no Home Assistant, no broker.
# Minutes, not seconds: a caller that gives up after two never sees the end.
set -euo pipefail
cd "$(dirname "$0")/.."
. scripts/container-guard.sh

echo "the suite takes about four minutes -- allow fifteen, not two" >&2
guard_start

# Panel first: the fast half. `tsc --noEmit` is the whole UI test budget, and it
# checks the same shapes test_api_contract.py does.
guard_run tsc \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e npm_config_cache=/tmp/.npm \
  -v "$PWD/occupancy-forecast-edge/panel":/w -w /w \
  -v occupancy-panel-npm:/tmp/.npm \
  node:22-alpine \
  timeout -k 30 600 \
  sh -c '[ -d node_modules ] || npm ci --no-audit --no-fund; npx tsc --noEmit'

# A stale bundle installs cleanly and serves last week's panel, so a green run
# has to mean it is fresh. If this fails: scripts/build-panel.sh, then re-test.
scripts/check-panel.sh occupancy-forecast-edge

# The repository is public and DEVELOPMENT.md's "Never" forbids a real name or
# entity id in the tree; until this ran, nothing enforced it.
scripts/check-privacy.sh

# Tagged with a hash of the pins it holds, so a moved pin cannot be served a
# stale image -- the same trick as panel/dist/.source-hash, without the compare.
tag="occupancy-forecast-test:$(cat occupancy-forecast-edge/requirements.txt \
                                   occupancy-forecast-edge/requirements-dev.txt \
                               | sha1sum | cut -c1-12)"
if ! docker image inspect "$tag" >/dev/null 2>&1; then
    echo "building $tag (one minute; the pins moved or this is a fresh clone)" >&2
    docker build -q -t "$tag" -f - occupancy-forecast-edge >/dev/null <<'EOF'
FROM python:3.13-slim
COPY requirements.txt requirements-dev.txt /
RUN pip install --no-cache-dir --root-user-action=ignore -r /requirements-dev.txt
EOF
fi

# pyarrow sizes its pool from hardware_concurrency(), which reads the Proxmox
# host's threads. Safe only because every train_all in the suite passes n_jobs=1.
threads=$(( $(nproc) > 3 ? $(nproc) - 2 : 1 ))

limit=${TEST_TIMEOUT:-900}
status=0
guard_run pytest \
  --cpus "$(( threads + 1 ))" --pids-limit 1024 \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e PYTHONPATH=/w -e PYTHONDONTWRITEBYTECODE=1 \
  -e OMP_NUM_THREADS="$threads" \
  -v "$PWD/occupancy-forecast-edge":/w -w /w \
  "$tag" \
  timeout -k 30 "$limit" \
  sh -c 'exec python -m pytest -q "$@"' sh "$@" || status=$?

if (( status == 124 || status == 137 )); then
    echo "the suite was killed at TEST_TIMEOUT=${limit}s" >&2
fi
exit "$status"
