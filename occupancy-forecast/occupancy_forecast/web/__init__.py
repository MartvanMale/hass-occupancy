"""Serving the Ingress panel: a React app under `panel/`, handed to the browser.

Nothing is fetched from anywhere but this add-on at runtime, because an Ingress
page on an offline installation must still render; and nothing is built on the
Home Assistant box, so the committed `dist/` travels with the Python.
"""

from __future__ import annotations

import mimetypes
import re
from pathlib import Path

from .. import config, log

_log = log.get(__name__)

# `mimetypes` has no `.woff2` and the slim image ships no `/etc/mime.types`, so
# `StaticFiles` would send the panel's fonts as `text/plain`.
mimetypes.add_type("font/woff2", ".woff2")

# Next to this file in the image; in the source tree on a development machine.
_CANDIDATE_DIRS = (
    Path(__file__).resolve().parent / "dist",
    Path(__file__).resolve().parents[2] / "panel" / "dist",
)

# The no-build page: every local pytest run gets it, so it says how to build.
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
    """Read rather than cache: a cache would go stale the moment a deploy
    replaced the build underneath a running process."""
    directory = dist_dir()
    return (directory / "index.html").read_text() if directory else None


def index_html() -> str:
    """`dist/index.html` with the add-on's own name in the title.

    From `config.display_name()`: both add-ons build from one tree, so a literal
    baked in at build time would put stable's name on edge's panel.
    """
    name = config.display_name()
    template = _template()
    if template is None:
        return _NOT_BUILT.format(name=name)
    # A regex over the tag rather than a replace of today's literal: a
    # substitution that silently matches nothing is the failure this prevents.
    return re.sub(r"<title>.*?</title>", f"<title>{name}</title>", template, count=1)


def mount(app) -> None:
    """Serve `dist/assets` if there is a build; do nothing if there is not.

    Mounting a missing directory is a startup crash, and no panel is worth that.
    """
    directory = dist_dir()
    if directory is None:
        return
    if not (directory / "assets").is_dir():
        # `StaticFiles` checks its directory at construction, so this is the
        # same start-up crash.
        _log.warning("%s has no assets/ directory; the panel's index will be "
                     "served but its bundle will not", directory)
        return
    from fastapi.staticfiles import StaticFiles

    app.mount("/assets", StaticFiles(directory=directory / "assets"), name="assets")
