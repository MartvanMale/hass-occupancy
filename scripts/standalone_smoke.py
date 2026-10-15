#!/usr/bin/env python3
"""Boot the standalone image through /standalone.sh; check HTTPS and the password gate.

Run via scripts/smoke-image.sh as the container's CMD, so s6's /init is PID 1 as
it is for a user. HA_URL points nowhere: the evidence is the gate, not a forecast.
EXPECT_CERT names the certificate the server must present (default: the generated one).
"""

from __future__ import annotations

import base64
import http.client
import json
import os
import re
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.request

BASE = "https://127.0.0.1:8099"
_tls: ssl.SSLContext | None = None
DEADLINE = 90.0


def get(path: str, password: str | None = None) -> tuple[int, bytes]:
    request = urllib.request.Request(BASE + path)
    if password is not None:
        token = base64.b64encode(f"smoke:{password}".encode()).decode()
        request.add_header("Authorization", f"Basic {token}")
    try:
        with urllib.request.urlopen(request, timeout=5, context=_tls) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as err:
        return err.code, err.read()


def uid_of(pid: int) -> int:
    with open(f"/proc/{pid}/status") as status:
        for line in status:
            if line.startswith("Uid:"):
                return int(line.split()[1])
    raise RuntimeError(f"no Uid line for pid {pid}")


def main() -> int:
    failures = []
    password = os.environ["OCCUPANCY_PASSWORD"]
    expected = os.environ.get("EXPECT_FINGERPRINT")

    global _tls
    server = subprocess.Popen(["/standalone.sh"])
    started = time.monotonic()
    while True:
        if server.poll() is not None:
            print(f"FAILED: the server exited with {server.returncode}.", file=sys.stderr)
            return 1
        try:
            # Verified against the expected certificate, so a 200 proves which one is served.
            if _tls is None:
                _tls = ssl.create_default_context(
                    cafile=os.environ.get("EXPECT_CERT", "/data/tls/cert.pem"))
            status, raw = get("/health")
            break
        except (urllib.error.URLError, OSError):
            if time.monotonic() - started > DEADLINE:
                server.kill()
                print(f"FAILED: no /health within {DEADLINE:.0f}s.", file=sys.stderr)
                return 1
            time.sleep(1.0)
    print(f"/health answered {status} after {time.monotonic() - started:.1f}s")

    body = json.loads(raw)
    if set(body) != {"status", "code"}:
        failures.append(f"/health behind a password shows {sorted(body)}, "
                        "expected only status and code")
    found = (body.get("code") or {}).get("fingerprint")
    if expected and found != expected:
        failures.append(f"code fingerprint {found!r}, expected {expected!r}")

    status, index = get("/", password)
    asset = re.search(rb'(?:\./|/)?(assets/[^"\']+\.js)', index)
    paths = ["/", "/api/status", "/openapi.json"]
    if asset:
        paths.append("/" + asset.group(1).decode())
    else:
        failures.append("no assets/*.js referenced from the panel's index")
    for path in paths:
        without, _ = get(path)
        wrong, _ = get(path, "not-" + password)
        right, _ = get(path, password)
        print(f"{path}: {without} without, {wrong} wrong, {right} with the password")
        if (without, wrong, right) != (401, 401, 200):
            failures.append(f"{path} answered {without}/{wrong}/{right}, "
                            "expected 401/401/200")

    try:
        urllib.request.urlopen("http://127.0.0.1:8099/health", timeout=5)
        failures.append("plain HTTP is still answered")
    except (urllib.error.URLError, OSError, http.client.HTTPException):
        print("plain HTTP is refused")

    uid = uid_of(server.pid)
    print(f"the server runs as uid {uid}")
    if uid == 0:
        failures.append("the server runs as root")

    server.terminate()
    try:
        server.wait(timeout=15)
    except subprocess.TimeoutExpired:
        server.kill()

    if failures:
        for failure in failures:
            print(f"FAILED: {failure}", file=sys.stderr)
        return 1
    print("\nPASSED: the standalone image serves HTTPS and guards everything but /health.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
