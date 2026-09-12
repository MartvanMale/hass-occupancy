"""The InfluxDB source, pinned against real response shapes.

Every CSV below was copied from a live server rather than invented, because all
four of the things that bite here are invisible in the Flux: an aggregate
returns no `_time` column, a `union` repeats its header mid-body, HA writes BOTH
a `state` and a `value` field for the same entity, and 1.8 prefixes every
response with annotation lines that 2.x omits.
"""

import urllib.error
import urllib.request

import pytest

from occupancy_forecast.sources.influx import InfluxSource, check_connection

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


# --- InfluxDB 1.8 --------------------------------------------------------
#
# Captured from a live 1.8.10 in compatibility mode, CRLF and all. 1.8 ALWAYS
# prefixes the annotation lines; 2.x sends them only when the dialect asks, and
# we never ask -- so the filter that drops them is unconditional.

STATES_18 = (
    "#datatype,string,long,dateTime:RFC3339,string\r\n"
    "#group,false,false,false,false\r\n"
    "#default,_result,,,\r\n"
    ",result,table,_time,_value\r\n"
    ",,0,2026-09-11T15:51:51.747468032Z,home\r\n"
    ",,0,2026-09-11T16:21:51.747468032Z,not_home\r\n"
)

# The same union as BOUNDS above, but 1.8 restarts the annotations per table --
# so the header repeats too, and a blank line sits between the blocks.
BOUNDS_18 = (
    "#datatype,string,long,dateTime:RFC3339,string,string,string,string\r\n"
    "#group,false,false,false,false,true,true,true\r\n"
    "#default,_result,,,,,,\r\n"
    ",result,table,_time,_value,_field,domain,entity_id\r\n"
    ",,0,2026-09-11T15:51:51Z,home,state,person,alice\r\n"
    ",,0,2026-09-12T15:21:51Z,not_home,state,person,alice\r\n"
    "\r\n"
    "#datatype,string,long,dateTime:RFC3339,long,string,string,string\r\n"
    "#group,false,false,false,false,true,true,true\r\n"
    "#default,_result,,,,,,\r\n"
    ",result,table,_time,_value,_field,domain,entity_id\r\n"
    ",,1,2026-09-11T15:51:51Z,1,value,person,alice\r\n"
    ",,1,2026-09-12T15:21:51Z,0,value,person,alice\r\n"
)

COUNTS_18 = (
    "#datatype,string,long,dateTime:RFC3339,dateTime:RFC3339,string,string,string,long\r\n"
    "#group,false,false,true,true,true,true,true,false\r\n"
    "#default,_result,,,,,,,\r\n"
    ",result,table,_start,_stop,domain,entity_id,_field,_value\r\n"
    ",,0,1970-01-01T00:00:00Z,2026-09-12T15:22:19Z,person,alice,state,48\r\n"
)


def _answering(monkeypatch, body: str) -> InfluxSource:
    def fake_urlopen(request, timeout=None):
        return _Response(body)
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return InfluxSource(url="http://influx:8086", token="t", org="o",
                        bucket="homeassistant/autogen")


def test_a_1_8_response_is_read_rather_than_answering_zero_rows(monkeypatch):
    """Unfiltered, `#datatype` becomes the header, no row carries `_time`, and
    every query answers nothing against a database that is full."""
    source = _answering(monkeypatch, STATES_18)

    assert source.states("person.alice", "-24h") == [
        ("2026-09-11T15:51:51.747468032Z", "home"),
        ("2026-09-11T16:21:51.747468032Z", "not_home"),
    ]


def test_a_1_8_union_restarts_its_annotations_and_is_still_read(monkeypatch):
    """1.8 repeats the whole annotation block per table, not just the header."""
    rows = _answering(monkeypatch, BOUNDS_18)._query("union(...)")

    assert len(rows) == 4
    assert {r["_field"] for r in rows} == {"state", "value"}
    assert all(r["entity_id"] == "alice" for r in rows)


def test_the_annotation_lines_are_not_counted_as_rows_by_an_aggregate(monkeypatch):
    """`timed=False` skips the `_time` filter that would otherwise have hidden
    them, so an unfiltered count answers three junk rows plus the real one."""
    rows = _answering(monkeypatch, COUNTS_18)._query("count()", timed=False)

    assert len(rows) == 1
    assert rows[0]["_value"] == "48"


# --- the connection check ------------------------------------------------

class _Ping:
    """`/ping` answers 204 with the version in a header, on both 1.x and 2.x."""

    def __init__(self, version: str | None):
        self.headers = {"X-Influxdb-Version": version} if version else {}

    def read(self):
        return b""

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def _checking(monkeypatch, version, bodies):
    """`/ping` then one body per Flux query, in order."""
    remaining = list(bodies)

    def fake_urlopen(request, timeout=None):
        if request.full_url.endswith("/ping"):
            if isinstance(version, Exception):
                raise version
            return _Ping(version)
        answer = remaining.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return _Response(answer)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)


BUCKETS_18 = (
    "#datatype,string,long,string,string,string,string,long\r\n"
    "#group,false,false,false,false,true,false,false\r\n"
    "#default,_result,,,,,,\r\n"
    ",result,table,name,id,organizationID,retentionPolicy,retentionPeriod\r\n"
    ",,0,homeassistant/autogen,,,autogen,0\r\n"
)


def _stages(result) -> dict:
    return {s["name"]: s["ok"] for s in result["stages"]}


def test_the_check_reports_a_1_x_server_and_what_its_token_and_bucket_mean(monkeypatch):
    """The version is what makes the 1.x-only advice sayable at all."""
    _checking(monkeypatch, "1.8.10", [BUCKETS_18, COUNTS_18])

    result = check_connection("http://influx:8086", "u:p", "o",
                              "homeassistant/autogen", ["person.alice"])

    assert result["ok"] and result["version"] == "1.8.10"
    assert "compatibility mode" in result["stages"][0]["detail"]
    assert "48 rows" in result["stages"][-1]["detail"]
    assert any("username:password" in h for h in result["hints"])


def test_a_2_x_server_is_given_none_of_the_1_x_advice(monkeypatch):
    _checking(monkeypatch, "v2.7.12", [BUCKETS_18.replace(
        "homeassistant/autogen", "homeassistant"), COUNTS_18])

    result = check_connection("http://influx:8086", "t", "o", "homeassistant",
                              ["person.alice"])

    assert result["hints"] == []
    assert "compatibility" not in result["stages"][0]["detail"]


def test_something_that_is_not_influxdb_is_not_a_credentials_problem(monkeypatch):
    """A reverse proxy or a typo'd port must not read as a bad token."""
    _checking(monkeypatch, None, [])

    result = check_connection("http://nas:8086", "t", "o", "b", ["person.alice"])

    assert _stages(result) == {"reachable": False}
    assert "did not identify itself" in result["stages"][0]["detail"]


def test_a_refused_token_stops_before_the_bucket_stage(monkeypatch):
    _checking(monkeypatch, "1.8.10", [urllib.error.HTTPError(
        "u", 401, "Unauthorized", {}, None)])

    result = check_connection("http://influx:8086", "bad", "o", "b",
                              ["person.alice"])

    assert _stages(result) == {"reachable": True, "credentials": False}


def test_a_missing_bucket_lists_the_ones_the_token_can_see(monkeypatch):
    """Which is what untangles 1.x's `database/retention-policy` form."""
    _checking(monkeypatch, "1.8.10", [BUCKETS_18])

    result = check_connection("http://influx:8086", "u:p", "o",
                              "homeassistant", ["person.alice"])

    assert _stages(result) == {"reachable": True, "credentials": True,
                               "bucket": False}
    assert "homeassistant/autogen" in result["stages"][-1]["detail"]


def test_the_check_never_echoes_the_token_back(monkeypatch):
    """It answers over Ingress, and a token in a response is a token in a log."""
    _checking(monkeypatch, "1.8.10", [BUCKETS_18, COUNTS_18])

    result = check_connection("http://influx:8086", "supersecrettoken", "o",
                              "homeassistant/autogen", ["person.alice"])

    assert "supersecrettoken" not in str(result)
