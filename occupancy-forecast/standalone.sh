#!/usr/bin/with-contenv bash
# The entrypoint without Supervisor: the `standalone` image target runs this in
# place of run.sh. Configuration is plain environment (docs/standalone.md);
# there is no bashio and no options.json.

set -e

missing=()
for name in HA_URL HA_TOKEN OCCUPANCY_PASSWORD; do
    [ -n "${!name}" ] || missing+=("$name")
done
if [ "${#missing[@]}" -gt 0 ]; then
    echo "[standalone] not starting: set ${missing[*]}" >&2
    exit 1
fi

# Without a volume there is no /data at all; Supervisor always mounts one.
mkdir -p /data

# HTTPS: the user's certificate if both variables are set, otherwise one made here.
if [ -n "$OCCUPANCY_SSL_CERT" ] || [ -n "$OCCUPANCY_SSL_KEY" ]; then
    for name in OCCUPANCY_SSL_CERT OCCUPANCY_SSL_KEY; do
        if [ -z "${!name}" ]; then
            echo "[standalone] not starting: set $name too, or neither" >&2
            exit 1
        fi
        if [ ! -r "${!name}" ]; then
            echo "[standalone] not starting: ${!name} ($name) cannot be read. Check the path and that its folder is mounted in compose.yaml." >&2
            exit 1
        fi
    done
    # Copied so the drop to `occupancy` can read them, whatever their mode on the host.
    install -d -o occupancy -g occupancy -m 700 /run/occupancy-tls
    install -o occupancy -g occupancy -m 600 "$OCCUPANCY_SSL_CERT" /run/occupancy-tls/cert.pem
    install -o occupancy -g occupancy -m 600 "$OCCUPANCY_SSL_KEY" /run/occupancy-tls/key.pem
    export OCCUPANCY_SSL_CERT=/run/occupancy-tls/cert.pem OCCUPANCY_SSL_KEY=/run/occupancy-tls/key.pem
else
    if [ ! -f /data/tls/cert.pem ] || [ ! -f /data/tls/key.pem ]; then
        san="DNS:localhost,IP:127.0.0.1"
        addr=${LAN_ADDRESS#[}; addr=${addr%]}     # compose wants IPv6 in brackets
        if [ -n "$addr" ]; then
            if [[ "$addr" =~ ^[0-9.]+$ || "$addr" == *:* ]]; then
                san="$san,IP:$addr"
            else
                san="$san,DNS:$addr"
            fi
        fi
        echo "[standalone] making a self-signed certificate for $san"
        mkdir -p /data/tls
        openssl req -x509 -newkey rsa:2048 -nodes -days 3650 -subj /CN=occupancy-forecast \
            -addext "subjectAltName=$san" \
            -keyout /data/tls/key.pem -out /data/tls/cert.pem 2>/tmp/openssl.log \
            || { cat /tmp/openssl.log >&2; echo "[standalone] could not make a certificate" >&2; exit 1; }
    fi
    chown -R occupancy:occupancy /data/tls || true
    export OCCUPANCY_SSL_CERT=/data/tls/cert.pem OCCUPANCY_SSL_KEY=/data/tls/key.pem
fi

# The same drop as run.sh, with the same fallbacks.
if [ "$(stat -c %u /data)" != "$(id -u occupancy)" ]; then
    echo "[standalone] handing /data to the unprivileged user"
    if ! chown -R occupancy:occupancy /data; then
        echo "[standalone] could not take ownership of /data, so running as root" >&2
        exec python -m occupancy_forecast.server
    fi
fi
if command -v s6-setuidgid >/dev/null 2>&1; then
    exec s6-setuidgid occupancy python -m occupancy_forecast.server
fi
echo "[standalone] s6-setuidgid not found, so running as root" >&2
exec python -m occupancy_forecast.server
