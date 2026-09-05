"""Serving the Ingress panel.

The panel used to be built here: ~480 lines of Python concatenating HTML, on the
argument that a form and a list needed no framework and no build step. That held
right up until the page had to be responsive. Home Assistant's own options form
still has no entity picker, so the panel still has to exist and still has to
offer *only what this installation actually has* -- but it is now a React app
under `panel/`, and this module only hands it to the browser.

**Two halves of the old argument survive and are load-bearing.** Nothing is
fetched from anywhere but this add-on at runtime: no CDN, no font service, no
icon service, because an Ingress page on an installation with no internet must
still render. The panel does use a web font, IBM Plex -- but the woff2 files are
vendored into `panel/src/fonts/`, emitted into `dist/assets/` by the build, and
served by the mount below, which is the same rule and not an exception to it. And
nothing is built on
the Home Assistant box: `scripts/build-panel.sh` runs Vite here, on the
development machine, and the deploy scripts rsync the finished `dist/` alongside
the Python. That is what every other add-on on the box does -- Zigbee2MQTT and
Music Assistant ship frontends bundled in CI, not compiled on the user's
hardware -- and it is why the Dockerfile has no node in it.

The look, the Mushroom geometry and the icon set moved to `panel/src/style.css`
and `panel/src/components/Icon.tsx` unchanged; the reasoning for the numbers went
with them.
"""

from __future__ import annotations

import mimetypes
import re
from pathlib import Path

from .. import config

# `StaticFiles` types a response off `mimetypes`, and the standard library's
# table has no `.woff2` -- nor does `python:3.13-slim`, which ships no
# `/etc/mime.types` for it to read one from. The panel's fonts would go out as
# `text/plain`. No browser MIME-checks an `@font-face` source, so this is not
# what breaks anything; it is one line that stops it looking broken to whoever
# next opens the network tab wondering why the panel is in Roboto.
mimetypes.add_type("font/woff2", ".woff2")

# Where the built panel lives, in the two places it can be.
#
# In the image the Dockerfile copies it next to this file, so the package is
# self-contained and `dist` ships with the code that serves it. On a development
# machine the build lands in the source tree instead, which is what lets
# `uvicorn occupancy_forecast.server:app` serve a freshly built panel with nothing
# installed and nothing copied.
_CANDIDATE_DIRS = (
    Path(__file__).resolve().parent / "dist",
    Path(__file__).resolve().parents[2] / "panel" / "dist",
)

# What the browser gets when there is no build. Every local pytest run is in this
# state, and so is anyone who checks the repo out and starts the server without
# reading how to build the panel -- so it says how, rather than 404ing.
_NOT_BUILT = """<!doctype html>
<html><head><meta charset="utf-8"><title>{name}</title></head>
<body style="font: 14px/1.5 system-ui, sans-serif; margin: 3rem auto; max-width: 34rem">
<h1 style="font-weight: 400">{name}</h1>
<p>The panel has not been built. Run <code>scripts/build-panel.sh</code> and
deploy again; the API below is unaffected and the add-on is running normally.</p>
<p><a href="api/status">api/status</a></p>
</body></html>"""


def dist_dir() -> Path | None:
    """The built panel, or None if it was never built."""
    for path in _CANDIDATE_DIRS:
        if (path / "index.html").is_file():
            return path
    return None


def _template() -> str | None:
    """Read rather than cache: the file is 600 bytes, it is read once per panel
    load, and a cache would go stale the moment a deploy replaced the build
    underneath a running process."""
    directory = dist_dir()
    return (directory / "index.html").read_text() if directory else None


def index_html() -> str:
    """`dist/index.html` with the add-on's own name in the title.

    The substitution is not cosmetic. Two add-ons run side by side, their Ingress
    panels are identical, and the browser tab is one of the few things that can
    tell them apart -- so the title has to derive from `config.display_name()`
    like every other name this package emits. A literal baked into `index.html`
    at build time would be the same class of mistake as a hardcoded notification
    id: both add-ons build from one tree, so it would be stable's name on edge's
    panel.
    """
    name = config.display_name()
    template = _template()
    if template is None:
        return _NOT_BUILT.format(name=name)
    # A regex over the tag rather than a replace of the literal that is in
    # index.html today: a substitution that silently matches nothing is exactly
    # the failure this is here to prevent.
    return re.sub(r"<title>.*?</title>", f"<title>{name}</title>", template, count=1)


def mount(app) -> None:
    """Serve `dist/assets` if there is a build; do nothing if there is not.

    Mounting a missing directory is a startup crash, and a missing panel is not
    worth taking the forecaster down for.
    """
    directory = dist_dir()
    if directory is None:
        return
    from fastapi.staticfiles import StaticFiles

    app.mount("/assets", StaticFiles(directory=directory / "assets"), name="assets")
