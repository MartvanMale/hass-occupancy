#!/usr/bin/env python3
"""Screenshot the panel against the demo instance, light and dark.

Runs inside `mcr.microsoft.com/playwright`, pointed at whatever
`demo-serve.py` is serving. Deterministic and repeatable, which matters more
than it sounds: a release post's screenshots get retaken every time the UI
moves, and doing it by hand is how one of them ends up showing a real name six
months later.

The panel's three tabs carry stable ids (`#tab-overview`, `#tab-config`,
`#tab-data`) because they are real ARIA tabs, so there is nothing to select on
by appearance.

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


def capture(url: str, out: Path, settle_ms: int) -> int:
    out.mkdir(parents=True, exist_ok=True)
    written = 0
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--force-color-profile=srgb"])
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
    args = parser.parse_args()
    count = capture(args.url, args.out, args.settle_ms)
    print(f"{count} screenshots -> {args.out}")
    if count == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
