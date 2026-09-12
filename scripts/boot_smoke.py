#!/usr/bin/env python3
"""Start the add-on's server inside the built image and read /health back.

Run via scripts/smoke-image.sh, inside the container (the image has no curl).
Skips run.sh, which needs a Supervisor. With no Home Assistant 503 is expected,
so the evidence is the body: EXPECT_FINGERPRINT proves which code is inside.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

URL = "http://127.0.0.1:8099/health"
DEADLINE = 90.0


def read_health() -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(URL, timeout=5) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as err:  # 503 is a real answer, not a failure
        return err.code, json.loads(err.read())


def main() -> int:
    expected = os.environ.get("EXPECT_FINGERPRINT")
    server = subprocess.Popen([sys.executable, "-m", "occupancy_forecast.server"])

    status, body = 0, {}
    started = time.monotonic()
    while time.monotonic() - started < DEADLINE:
        if server.poll() is not None:
            print(f"FAILED: the server exited with {server.returncode} before "
                  "it answered.", file=sys.stderr)
            return 1
        try:
            status, body = read_health()
            break
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            time.sleep(1.0)
    else:
        server.terminate()
        print(f"FAILED: no answer from {URL} within {DEADLINE:.0f}s.",
              file=sys.stderr)
        return 1

    took = time.monotonic() - started
    print(f"/health answered {status} after {took:.1f}s")
    print(json.dumps(body, indent=2, sort_keys=True)[:2000])

    failures = []
    # Not `== 503`, or a legitimate change to /health would read as a break.
    if status not in (200, 503):
        failures.append(f"status {status}, expected 200 or 503")
    if body.get("status") != "collecting":
        failures.append(f"status field {body.get('status')!r}, expected 'collecting'")
    if body.get("people") != []:
        failures.append(f"people {body.get('people')!r}, expected [] with no HA")

    found = (body.get("code") or {}).get("fingerprint")
    if expected and found != expected:
        failures.append(f"code fingerprint {found!r}, but this tree hashes to "
                        f"{expected!r} -- the image is not built from it")
    elif expected:
        print(f"code fingerprint {found} matches the tree.")

    server.terminate()
    try:
        server.wait(timeout=15)
    except subprocess.TimeoutExpired:
        server.kill()

    if failures:
        for failure in failures:
            print(f"FAILED: {failure}", file=sys.stderr)
        return 1

    print("\nPASSED: the image boots and serves /health.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
