#!/usr/bin/env python3
"""Screenshot the panel against the demo instance, light and dark.

Deterministic and repeatable, which matters more than it sounds: a release
post's screenshots get retaken every time the UI moves, and doing it by hand is
how one of them ends up showing a real name six months later.

The panel's three tabs carry stable ids (`#tab-overview`, `#tab-config`,
`#tab-data`) because they are real ARIA tabs, so there is nothing to select on
by appearance.

Needs the browsers, so it wants `mcr.microsoft.com/playwright/python` -- the
PYTHON image. The plain `mcr.microsoft.com/playwright` tag is the Node one and
carries neither the `playwright` module nor a pip to install it with. The
`capture` service in `compose.yaml` is the invocation that gets this right:

    docker compose run --rm capture

Directly, against a `demo-serve.py` already running on the host:

    scripts/demo-capture.py --url http://127.0.0.1:8099 --out ~/occupancy-demo/shots
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

TABS = [("overview", "Overview"), ("config", "Setup"), ("data", "Data")]

# Wide enough for the panel's two-column layout, and a phone, because Ingress is
# opened from the Home Assistant app at least as often as from a desktop.
VIEWPORTS = {"desktop": (1280, 960), "mobile": (414, 896)}


def capture(url: str, out: Path, settle_ms: int, no_sandbox: bool = False) -> int:
    out.mkdir(parents=True, exist_ok=True)
    written = 0
    args = ["--force-color-profile=srgb"]
    if no_sandbox:
        # Chromium's sandbox needs user namespaces, which a container often does
        # not have -- and inside one the sandbox is buying very little anyway,
        # since the container is the boundary. Opt-in rather than always-on so a
        # run on the host keeps it.
        args.append("--no-sandbox")
    with sync_playwright() as p:
        browser = p.chromium.launch(args=args)
        for scheme in ("light", "dark"):
            for shape, (width, height) in VIEWPORTS.items():
                context = browser.new_context(
                    viewport={"width": width, "height": height},
                    device_scale_factor=2,          # legible when embedded in a post
                    color_scheme=scheme,
                    reduced_motion="reduce",        # no half-played transitions
                )
                page = context.new_page()
                page.goto(url, wait_until="networkidle", timeout=60_000)
                for tab, label in TABS:
                    page.click(f"#tab-{tab}")
                    page.wait_for_timeout(settle_ms)   # charts animate in
                    name = f"{tab}-{shape}-{scheme}.png"
                    page.screenshot(path=out / name, full_page=True)
                    print(f"  {name}")
                    written += 1
                context.close()
        browser.close()
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:8099")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--settle-ms", type=int, default=2500)
    parser.add_argument("--no-sandbox", action="store_true",
                        help="launch Chromium without its sandbox, for containers")
    args = parser.parse_args()
    count = capture(args.url, args.out, args.settle_ms, args.no_sandbox)
    print(f"{count} screenshots -> {args.out}")
    if count == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
