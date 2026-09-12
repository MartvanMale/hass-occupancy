#!/usr/bin/env bash
# Fail if anything tracked names the maintainer's own household, or looks like a
# credential. DEVELOPMENT.md's "Never" has said so since the repo went public and
# nothing checked it, which is how a bake-off's per-day departure times reached
# the working tree. The name list lives OUTSIDE the repo so it cannot be
# committed; credentials are matched by SHAPE, so no secret is ever stored here.
set -euo pipefail
cd "$(dirname "$0")/.."

LIST="${PRIVACY_NAMES:-$HOME/.config/hass-occupancy/privacy-names}"

# Credential SHAPES, not values -- deliberately not in the name list. A real
# token in a denylist would be echoed by the report below and exposed in the
# process list, so this file may never learn one. Kept to patterns that do not
# false-positive on a key NAME like `mqtt_password:` with a templated value.
SECRET_SHAPES='eyJ[A-Za-z0-9_-]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{20,}|xox[baprs]-[A-Za-z0-9-]{10,}'

work=$(mktemp -d); trap 'rm -rf "$work"' EXIT
patterns="$work/patterns"
: > "$patterns"; chmod 600 "$patterns"

if [[ -f "$LIST" ]]; then
    # One extended-regex fragment per line; '#' comments and blanks ignored.
    grep -vE '^[[:space:]]*(#|$)' "$LIST" >> "$patterns" || true
else
    # Not an error: a fresh clone has no household to leak and neither does CI.
    echo "check-privacy: no name list at $LIST -- names not checked." >&2
fi
printf '%s\n' "$SECRET_SHAPES" >> "$patterns"

# `-f`, not `-e`: a pattern passed as an argument is visible to anyone who can
# read the process list. Tracked files only -- an ignored notebook or a scratch
# file is not what gets published. The repository URL carries the maintainer's
# name legitimately, so it is dropped by value.
hits=$(git grep -nEI -i -f "$patterns" -- . \
       | grep -vE 'github\.com/[A-Za-z0-9-]+/hass-occupancy' \
       || true)

# occupancy-forecast/ is GENERATED from the edge tree, and editing it is a
# promotion decision rather than a fix. A hit there is reported and does not
# fail: the next promote.sh overwrites it from an edge tree this check cleared.
generated=$(printf '%s\n' "$hits" | grep -E '^occupancy-forecast/' || true)
source_hits=$(printf '%s\n' "$hits" | grep -vE '^occupancy-forecast/' || true)

if [[ -n "$generated" ]]; then
    echo "check-privacy: WARNING -- the generated stable tree still carries" >&2
    echo "  these. A promotion clears them; they cannot be fixed in place." >&2
    printf '  %s\n' $(printf '%s\n' "$generated" | cut -d: -f1 | sort -u) >&2
fi

if [[ -n "$source_hits" ]]; then
    echo "error: tracked files name the maintainer's household, or carry" >&2
    echo "       something shaped like a credential. DEVELOPMENT.md: never" >&2
    echo "       write a real entity id, person name, token, IP or hostname" >&2
    echo "       into code, tests or docs." >&2
    echo >&2
    printf '%s\n' "$source_hits" >&2
    exit 1
fi

# It catches names, identifiers and credential shapes -- not de-identified
# statistics. "16 of 19 Saturdays" is a real measurement no pattern separates
# from a fixture, so the diff still has to be read.
