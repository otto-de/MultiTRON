import os
from filecmp import cmp

from src.preprocessing import filter_carts, increment_aids, sort_events

os.makedirs("test/resources/out", exist_ok=True)


def test_increment_aids():
    events = [{
        "aid": 0,
        "ts": 1,
        "type": "clicks"
    }, {
        "aid": 1,
        "ts": 1,
        "type": "clicks"
    }, {
        "aid": 2,
        "ts": 1,
        "type": "clicks"
    }, {
        "aid": 1,
        "ts": 1,
        "type": "clicks"
    }]
    expected_events = [{
        "aid": 1,
        "ts": 1,
        "type": "clicks"
    }, {
        "aid": 2,
        "ts": 1,
        "type": "clicks"
    }, {
        "aid": 3,
        "ts": 1,
        "type": "clicks"
    }, {
        "aid": 2,
        "ts": 1,
        "type": "clicks"
    }]
    assert expected_events == increment_aids(events)


def test_filter_carts():
    num_sessions, num_click_events, num_order_events, num_items = filter_carts("test/resources/unfiltered_sessions.jsonl",
                                                                               "test/resources/out/filtered_sessions.jsonl")
    assert 5 == num_sessions
    assert 88 == num_click_events
    assert 12 == num_order_events
    assert 66 == num_items
    assert cmp("test/resources/expected_filtered_sessions.jsonl", "test/resources/out/filtered_sessions.jsonl")
    os.remove("test/resources/out/filtered_sessions.jsonl")


def test_sort_events():
    events = [{
        "aid": 1,
        "ts": 5,
        "type": "clicks"
    }, {
        "aid": 2,
        "ts": 1,
        "type": "clicks"
    }, {
        "aid": 3,
        "ts": 3,
        "type": "clicks"
    }, {
        "aid": 2,
        "ts": 0,
        "type": "clicks"
    }]
    expected_events = [{
        "aid": 2,
        "ts": 0,
        "type": "clicks"
    }, {
        "aid": 2,
        "ts": 1,
        "type": "clicks"
    }, {
        "aid": 3,
        "ts": 3,
        "type": "clicks"
    }, {
        "aid": 1,
        "ts": 5,
        "type": "clicks"
    }]
    assert expected_events == sort_events(events)
