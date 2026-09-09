# Sourced by every script here that runs a container. A container's life belongs
# to the daemon, not to the client that asked for it, so a script killed mid-run
# leaves one burning the box -- three of them deadlocked it in 2026-09. The lock
# stops a second run piling on, the label lets the next run sweep a corpse, the
# name lets the trap kill this one, and callers wrap the container command in
# `timeout` for the case where neither the trap nor the sweep ever runs.
GUARD_LABEL=hass-occupancy.script
GUARD_LOCK=/tmp/hass-occupancy-scripts.lock
GUARD_NAME=""

# Never `exit` from here and never let a failure escape: this runs on the way out
# of a successful run too, and would overwrite the status promote.sh gates on.
guard_cleanup() {
    if [[ -n "$GUARD_NAME" ]]; then
        docker rm -f "$GUARD_NAME" >/dev/null 2>&1 || true
    fi
    return 0
}
trap guard_cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Refuse rather than queue, and sweep only behind the lock: holding it proves no
# sibling run is alive, so anything still wearing the label is a corpse.
guard_start() {
    exec 9>"$GUARD_LOCK"
    if ! flock -n 9; then
        echo "another scripts/ container run holds $GUARD_LOCK -- not starting" >&2
        exit 1
    fi
    local stale
    stale=$(docker ps -aq --filter "label=$GUARD_LABEL")
    [[ -n "$stale" ]] || return 0
    echo "sweeping $(wc -l <<<"$stale") container(s) left by an earlier run" >&2
    docker rm -f $stale >/dev/null
}

# Backgrounded on purpose: bash defers a trap until the foreground command
# returns, so a six-minute `docker run` would swallow SIGTERM for six minutes.
guard_run() {
    GUARD_NAME="hass-occupancy-$1-$$"
    shift
    local status=0
    docker run --rm --init --name "$GUARD_NAME" --label "$GUARD_LABEL=1" "$@" &
    wait $! || status=$?
    GUARD_NAME=""
    return "$status"
}

# A named volume is created root-owned, so claim it once for --user to write.
guard_volume() {
    if docker volume inspect "$1" >/dev/null 2>&1; then
        return 0
    fi
    docker volume create "$1" >/dev/null
    guard_run volume -v "$1":/v alpine chown -R "$(id -u):$(id -g)" /v
}
