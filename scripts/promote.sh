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
# alone here: the versions differ, the docs differ, and the changelogs are not
# the same file -- edge's carries a ## Unreleased queue on top of the same
# release history stable has.
#
# Note the --delete: a file that exists ONLY in occupancy-forecast/ and is not excluded
# below gets removed on the next run. Anything the stable add-on needs in its own
# right belongs in occupancy-forecast-edge/ so that it mirrors, or in the exclude list.
#
# This script does not bump the version, does not move the changelog, does not
# commit and does not deploy. It stages the code and shows you what changed;
# every irreversible step after that is yours.
#
# It does NOT require the edge tree to be committed first. The workflow is: work
# in edge, promote when happy, then ONE commit carrying both trees. That is why
# there is no pre-commit hook guarding occupancy-forecast/ any more -- the guard
# assumed edge and the promotion were separate commits, and they are not.
set -euo pipefail
cd "$(dirname "$0")/.."

# Arguments first, before any check that can fail -- otherwise `--help` and a
# typo'd flag both die on whatever the tree happens to look like instead of
# saying what the flag was.
#
# The tests are cheap and the alternative is discovering it in production, where
# "production" is the forecast your heating reads. So they run by default and
# skipping them has to be typed out.
#
# `--no-test` is for exactly one honest case: scripts/test.sh has just been run
# against the edge tree as it stands and nothing has been edited since. Since
# this script promotes the WORKING tree, "as it stands" means literally the
# files on disk -- so an edit made after that test run is an untested promotion.
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

# Rebuild EDGE's bundle before promoting. The pre-commit hook used to refuse a
# panel source edit whose committed dist/ did not match it -- that check is gone
# with the hook, and a stale bundle is silent: it installs cleanly and serves
# last week's panel. Since one commit now carries both trees, the honest fix is
# to make the bundle fresh by construction here rather than to check it. Cheap
# (sub-second) and a no-op when the source has not moved.
scripts/build-panel.sh occupancy-forecast-edge

# The panel's SOURCE is promoted; its build output is not copied. dist/ is
# rebuilt below from the promoted source, so stable ships a bundle built from
# stable's own tree rather than a copy of whatever edge last compiled.
rsync -a --delete \
  --exclude '__pycache__' --exclude '*.pyc' --exclude '.pytest_cache' \
  --exclude 'node_modules' --exclude 'dist' \
  --exclude 'config.yaml' --exclude 'DOCS.md' --exclude 'CHANGELOG.md' \
  occupancy-forecast-edge/ occupancy-forecast/

# Build stable's bundle from stable's OWN promoted source rather than copying
# edge's, so the two are independently reproducible from their own trees.
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
