#!/usr/bin/env bash
# Generate the stable add-on's code from edge's: edge -> stable, never back.
# config.yaml, DOCS.md and CHANGELOG.md are hand-written per add-on and excluded.
# Note the --delete: a file only in occupancy-forecast/ and not excluded below is
# removed. It promotes the WORKING tree, so edge need not be committed first.
set -euo pipefail
cd "$(dirname "$0")/.."

# Arguments first, so a typo'd flag says so rather than dying on the tree.
# `--no-test` is honest only when test.sh has just run against these exact files.
run_tests=1
for arg in "$@"; do
    case "$arg" in
        --no-test) run_tests=0 ;;
        -h|--help) echo "usage: $(basename "$0") [--no-test]"; exit 0 ;;
        *) echo "error: unknown argument '$arg'" >&2
           echo "usage: $(basename "$0") [--no-test]" >&2
           exit 2 ;;
    esac
done

if (( run_tests )); then
    echo "Running tests against the edge tree..."
    scripts/test.sh
else
    echo "SKIPPING TESTS (--no-test) -- promoting on the strength of an earlier run."
fi

# Rebuild EDGE's bundle first: a stale one is silent, and one commit carries both
# trees, so it is made fresh by construction rather than checked.
scripts/build-panel.sh occupancy-forecast-edge

# The panel's SOURCE is promoted, not its build output; dist/ is rebuilt below
# so stable's bundle comes from stable's own tree.
rsync -a --delete \
  --exclude '__pycache__' --exclude '*.pyc' --exclude '.pytest_cache' \
  --exclude 'node_modules' --exclude 'dist' \
  --exclude 'config.yaml' --exclude 'DOCS.md' --exclude 'CHANGELOG.md' \
  occupancy-forecast-edge/ occupancy-forecast/

# From stable's own promoted source, so each tree is independently reproducible.
scripts/build-panel.sh occupancy-forecast

echo
if [[ -z "$(git status --porcelain occupancy-forecast/)" ]]; then
    echo "occupancy-forecast/ is already identical to occupancy-forecast-edge/. Nothing to promote."
    exit 0
fi

echo "=== promoted into occupancy-forecast/ ==="
git diff --stat occupancy-forecast/
echo
echo "Not done yet. Still yours to do:"
echo "  1. Bump version: in occupancy-forecast/config.yaml (semver)."
echo "  2. Retitle the ## Unreleased block in occupancy-forecast-edge/CHANGELOG.md to"
echo "     \"## <version> - $(date +%F)\", keeping the ### Added/Changed/Fixed headings,"
echo "     and open a fresh empty ## Unreleased above it. COPY that block -- do not move"
echo "     it -- to the top of the release history in occupancy-forecast/CHANGELOG.md."
echo "     Edge keeps its copy: the two files must be identical from the first ## <version>"
echo "     heading down. Check it:"
echo "       diff <(sed -n '/^## [0-9]/,\$p' occupancy-forecast-edge/CHANGELOG.md) \\"
echo "            <(sed -n '/^## [0-9]/,\$p' occupancy-forecast/CHANGELOG.md)"
echo "  3. git add -A && git commit       (one commit, both trees)"
echo "     Use 'git add -A' rather than 'git commit -a' -- the rebuilt bundles"
echo "     can contain NEW files, and -a does not pick those up."
echo "  4. git push"
echo "     THAT is the deploy. Stable is installed from this repository's URL,"
echo "     so Supervisor offers the update once the pushed config.yaml version"
echo "     moves. There is no deploy script for stable -- see DEVELOPMENT.md."
