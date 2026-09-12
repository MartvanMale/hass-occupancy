"""The InfluxDB source, pinned against real response shapes.

Every CSV below was copied from a live InfluxDB 2 rather than invented, because
all three of the things that bite here are invisible in the Flux: an aggregate
returns no `_time` column, a `union` repeats its header mid-body, and HA writes
BOTH a `state` and a `value` field for the same entity.
"""

import urllib.request

import pytest

from occupancy_forecast.sources.influx import InfluxSource

# `count()` drops `_time` entirely -- the reason `_query` takes `timed`.
COUNTS = """,result,table,_start,_stop,_value,domain,entity_id,_field
,_result,0,1970-01-01T00:00:00Z,2026-09-12T14:38:38Z,44819,person,alice,state
,_result,1,1970-01-01T00:00:00Z,2026-09-12T14:38:38Z,43413,person,alice,value
,_result,2,1970-01-01T00:00:00Z,2026-09-12T14:38:38Z,1200,sensor,home_alice_distance,value
"""

# Two schemas in one response, so the header appears twice.
BOUNDS = """,result,table,_time,_value,_field,domain,entity_id
,_result,0,2026-03-12T21:22:29Z,home,state,person,alice
,_result,0,2026-09-12T14:35:55Z,not_home,state,person,alice
,result,table,_time,_value,_field,domain,entity_id
,_result,1,2026-03-12T21:22:29Z,1,value,person,alice
,_result,1,2026-09-12T14:35:55Z,0,value,person,alice
,_result,2,2026-04-01T00:00:00Z,1500,value,sensor,home_alice_distance
,_result,2,2026-09-12T14:00:00Z,250,value,sensor,home_alice_distance
"""


class _Response:
    def __init__(self, body: str):
        self._body = body.encode()

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


@pytest.fixture
def influx(monkeypatch):
    """Answers by which query was asked, and records the Flux for inspection."""
    sent: list[str] = []

    def fake_urlopen(request, timeout=None):
        flux = request.data.decode()
        sent.append(flux)
        return _Response(COUNTS if "count()" in flux else BOUNDS)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    source = InfluxSource(url="http://influx:8086", token="t", org="o",
                          bucket="homeassistant",
                          units={"sensor.home_alice_distance": "m"})
    source.sent = sent
    return source


ASKED = ["person.alice", "person.bob", "sensor.home_alice_distance"]


def test_an_entity_is_counted_once_and_read_as_its_string_state(influx):
    """HA writes a person as BOTH a string state and its numeric twin. Summing
    the pair doubles the archive, and reading the twin makes presence look
    numeric -- so `state` wins and `value` is only the fallback."""
    archive = influx.archive(ASKED)
    alice = next(e for e in archive["entities"] if e["entity_id"] == "person.alice")

    assert alice["rows"] == 44819, "the `value` twin must not be added in"
    assert alice["field"] == "state"
    assert set(alice["sample"]) == {"home", "not_home"}, "the header row is not data"
    assert alice["first"] == "2026-03-12T21:22:29Z"
    assert alice["last"] == "2026-09-12T14:35:55Z"


def test_a_number_only_entity_reports_its_value_field(influx):
    """A sensor with a unit has no string state at all. Falling back to `value`
    is what keeps it from reading as empty."""
    archive = influx.archive(ASKED)
    distance = next(e for e in archive["entities"]
                    if e["entity_id"] == "sensor.home_alice_distance")

    assert distance["field"] == "value"
    assert distance["rows"] == 1200
    assert distance["sample"] == ["1500", "250"]


def test_a_configured_entity_with_no_rows_says_so_rather_than_vanishing(influx):
    """The single most useful thing the card reports: configured, never seen."""
    archive = influx.archive(ASKED)
    bob = next(e for e in archive["entities"] if e["entity_id"] == "person.bob")

    assert bob["rows"] == 0
    assert bob["field"] is None
    assert bob["first"] is None


def test_the_span_adds_up_and_claims_no_disk(influx):
    span = influx.archive(ASKED)["span"]

    assert span["rows"] == 44819 + 1200
    assert span["bytes"] is None, "the bucket is shared; no figure is this add-on's"
    assert span["first"] == "2026-03-12T21:22:29Z"
    assert span["days"] == pytest.approx(183.7, abs=0.2)


def test_entities_are_selected_by_tag_not_by_measurement(influx):
    """A sensor WITH a unit is stored under the UNIT as measurement, so a
    `_measurement` filter silently drops it. Both tags are on every point."""
    influx.archive(ASKED)

    for flux in influx.sent:
        assert 'r.domain == "sensor" and r.entity_id == "home_alice_distance"' in flux
        assert '_measurement == "sensor.home_alice_distance"' not in flux


def test_nothing_asked_for_is_nothing_queried(influx):
    assert influx.archive([])["entities"] == []
    assert influx.sent == []
