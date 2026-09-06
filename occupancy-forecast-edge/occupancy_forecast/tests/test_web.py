"""Serving the Ingress panel.

The panel is a React app now, so the tests that used to live here -- that the
generated markup contained the ids the save script queried, that a person's
friendly name could not inject a `<script>` -- no longer describe anything real.
React escapes interpolated text by construction, and the save handler is a
function reference rather than a selector that can silently miss.

What is left is the part that is still Python, and both halves of it have bitten
before: the panel has to say WHICH add-on served it, and it has to degrade rather
than crash when there is no build. The typed half of the UI is covered by
`tsc --noEmit` in scripts/test.sh, and the shape of the API it consumes by
test_api_contract.py.
"""

import pytest

from occupancy_forecast import config, web


def test_the_title_says_which_build_served_the_panel(monkeypatch):
    """Both add-ons serve an identical-looking panel; only the name separates them.

    The bundle is built once from one tree and shipped to both, so the name
    cannot come from `index.html` -- it is substituted here, per request.
    """
    if web.dist_dir() is None:
        pytest.skip("panel not built; run scripts/build-panel.sh")

    monkeypatch.setattr(config, "_topic_prefix", "occupancy_forecast")
    assert "<title>Occupancy Forecast</title>" in web.index_html()

    monkeypatch.setattr(config, "_topic_prefix", "occupancy_forecast_edge")
    edge = web.index_html()
    assert "<title>Occupancy Forecast Edge</title>" in edge
    assert "<title>Occupancy Forecast</title>" not in edge


def test_an_unbuilt_panel_is_a_page_and_not_a_crash(monkeypatch):
    """The state of every checkout that has not run scripts/build-panel.sh.

    A 500 here would read as a broken add-on when the forecaster is running
    perfectly well, so the fallback says what to do and points at the API.
    """
    monkeypatch.setattr(web, "dist_dir", lambda: None)
    html = web.index_html()
    assert html.startswith("<!doctype html>")
    assert "build-panel.sh" in html
    assert config.display_name() in html


def test_the_assets_mount_is_skipped_when_there_is_no_build(monkeypatch):
    """Mounting a directory that does not exist is a startup crash, and a missing
    panel is not worth taking the forecaster down for."""
    monkeypatch.setattr(web, "dist_dir", lambda: None)

    class Recorder:
        mounted = False

        def mount(self, *args, **kwargs):
            self.mounted = True

    app = Recorder()
    web.mount(app)
    assert not app.mounted
