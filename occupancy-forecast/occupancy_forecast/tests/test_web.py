"""Serving the Ingress panel.

The two halves that are still Python, and have both bitten before: the panel
has to say WHICH add-on served it, and it has to degrade rather than crash when
there is no build.
"""

import pytest

from occupancy_forecast import config, web


def test_the_title_says_which_build_served_the_panel(monkeypatch):
    """Both add-ons serve an identical-looking panel; only the name separates them.
    The bundle is built once and shipped to both, so the name is substituted per
    request."""
    if web.dist_dir() is None:
        pytest.skip("panel not built; run scripts/build-panel.sh")

    monkeypatch.setattr(config, "_topic_prefix", "occupancy_forecast")
    assert "<title>Occupancy Forecast</title>" in web.index_html()

    monkeypatch.setattr(config, "_topic_prefix", "occupancy_forecast_edge")
    edge = web.index_html()
    assert "<title>Occupancy Forecast Edge</title>" in edge
    assert "<title>Occupancy Forecast</title>" not in edge


def test_an_unbuilt_panel_is_a_page_and_not_a_crash(monkeypatch):
    """The state of every checkout that has not run scripts/build-panel.sh. A 500
    here would read as a broken add-on when the forecaster is running fine."""
    monkeypatch.setattr(web, "dist_dir", lambda: None)
    html = web.index_html()
    assert html.startswith("<!doctype html>")
    assert "build-panel.sh" in html
    assert config.display_name() in html


def test_a_build_with_an_index_but_no_assets_does_not_crash_the_start(monkeypatch, tmp_path):
    """`StaticFiles` checks its directory at construction, so this was exactly
    the start-up crash the skip above exists to avoid, one level down."""
    (tmp_path / "index.html").write_text("<title>x</title>")
    monkeypatch.setattr(web, "dist_dir", lambda: tmp_path)

    class Recorder:
        mounted = False

        def mount(self, *args, **kwargs):
            self.mounted = True

    app = Recorder()
    web.mount(app)
    assert not app.mounted


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
