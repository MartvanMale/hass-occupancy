#!/usr/bin/env bash
# Promote the edge add-on's code into the stable add-on.
#
# The two add-ons run the SAME code at different release points. A Home
# Assistant add-on's Docker build context is its own directory, so it cannot
# reach a shared parent -- which leaves either two hand-maintained copies that
# drift, or one generated from the other. This is the generator, and it runs
# edge -> stable.
#
# The direction is the whole point. `occupancy-forecast-edge/` is where work happens;
# `occupancy-forecast/` changes ONLY here, and only when you have decided a change has
# earned its way into the add-on you trust. Generating the other way round --
# which is what this script used to do -- meant every edit landed in the stable
# add-on's directory first, so there was nowhere to put work in progress and
# "test it on edge before promoting" was not actually possible.
#
# config.yaml, DOCS.md and CHANGELOG.md are hand-written per add-on and are left
# alone here: the versions differ, the docs differ, and the changelogs say
# different things (edge's is the queue, stable's is the release history).
#
# Note the --delete: a file that exists ONLY in occupancy-forecast/ and is not excluded
# below gets removed on the next run. Anything the stable add-on needs in its own
# right belongs in occupancy-forecast-edge/ so that it mirrors, or in the exclude list.
#
# This script does not bump the version, does not move the changelog, does not
# commit and does not deploy. It stages the code and shows you what changed;
# every irreversible step after that is yours.
set -euo pipefail
cd "$(dirname "$0")/.."

# Promote what is committed, not what is in your editor. Without this you can
# promote a half-finished edit, and because the stable tree is generated there
# is then no record anywhere of what stable is actually running.
if [[ -n "$(git status --porcelain occupancy-forecast-edge/)" ]]; then
    echo "error: occupancy-forecast-edge/ has uncommitted changes." >&2
    echo "       Commit them first -- promote what you tested, not what is in the editor." >&2
    git status --short occupancy-forecast-edge/ >&2
    exit 1
fi

# The tests are cheap and the alternative is discovering it in production, where
# "production" is the forecast your heating reads.
echo "Running tests against the edge tree..."
scripts/test.sh

# The panel's SOURCE is promoted; its build output is not copied. dist/ is
# rebuilt below from the promoted source, so stable ships a bundle built from
# stable's own tree rather than a copy of whatever edge last compiled.
rsync -a --delete \
  --exclude '__pycache__' --exclude '*.pyc' --exclude '.pytest_cache' \
  --exclude 'node_modules' --exclude 'dist' \
  --exclude 'config.yaml' --exclude 'DOCS.md' --exclude 'CHANGELOG.md' \
  occupancy-forecast-edge/ occupancy-forecast/

# Build it here rather than at deploy time, because the bundle is committed and
# so has to be IN the promotion commit. deploy-stable.sh cannot build it: that
# script's first act is to require occupancy-forecast/ be clean, and a build
# would dirty the tree it just validated.
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
echo "  2. Move the ## Unreleased block from occupancy-forecast-edge/CHANGELOG.md into"
echo "     occupancy-forecast/CHANGELOG.md under \"## <version> - $(date +%F)\", keeping the"
echo "     ### Added/Changed/Fixed headings, and empty it."
echo "  3. PROMOTE=1 git commit -a      (the hook rejects occupancy-forecast/ without it)"
echo "     Stage occupancy-forecast/panel/dist/ with it -- the rebuilt bundle is"
echo "     part of the release, and \`git commit -a\` will not pick up new files."
echo "  4. scripts/deploy-stable.sh"
