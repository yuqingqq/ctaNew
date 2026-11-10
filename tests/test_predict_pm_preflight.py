"""Controls for the cross-venue engineering instrument, not strategy tests."""
import json
from collections import Counter

import pytest

from data_collectors.predict_pm_preflight import (
    NS, append_state, available, metadata, pm_updates, select_windows, touch_status,
)


def test_availability_positive_stale_and_future_controls():
    states = []
    append_state(states, 10 * NS, 9900, .4, .6)
    times = [10 * NS]
    assert available(states, times, 10 * NS, NS) == 'AVAILABLE'
    assert available(states, times, 10 * NS - 1, NS) == 'NO_BOOK_YET'
    assert available(states, times, 11 * NS + 1, NS) == 'RECEIPT_OLDER_THAN_LIMIT'


def test_bad_latest_state_does_not_forward_fill_previous_good_book():
    states = []
    append_state(states, 10 * NS, 9900, .4, .6)
    append_state(states, 10 * NS + 100, 9900, .5, .5)
    assert available(states, [x[0] for x in states], 10 * NS + 200, NS) == 'LOCKED'
    append_state(states, 10 * NS + 300, 9900, None, .6)
    assert available(states, [x[0] for x in states], 10 * NS + 400, NS) == 'ONE_SIDED_OR_EMPTY'


def test_clock_failures_are_named_statuses():
    assert available([(10 * NS, 11 * NS, 'TWO_SIDED')], [10 * NS], 10 * NS, NS) == 'SOURCE_CLOCK_AHEAD'
    assert available([(10 * NS, 4 * NS, 'TWO_SIDED')], [10 * NS], 10 * NS, NS) == 'SOURCE_OLDER_THAN_5S'
    assert available([(10 * NS, None, 'TWO_SIDED')], [10 * NS], 10 * NS, NS) == 'NO_SOURCE_TIMESTAMP'


def test_book_and_receipt_falsifiers():
    assert touch_status(.7, .6) == 'CROSSED'
    with pytest.raises(ValueError, match='INVALID_PRICE'):
        touch_status(float('nan'), .6)
    with pytest.raises(ValueError, match='RECEIPT_REGRESSION'):
        append_state([(10, None, 'TWO_SIDED')], 9, 0, .4, .6)


def test_up_identity_full_book_and_explicit_bbo():
    book = {'asset_id': 'UP', 'timestamp': '123',
            'bids': [{'price': '.4', 'size': '1'}, {'price': '.5', 'size': '0'}],
            'asks': [{'price': '.8', 'size': '1'}, {'price': '.6', 'size': '1'}]}
    assert pm_updates([book], 'UP') == [('123', .4, .6)]
    assert pm_updates([book], 'DOWN') == []
    change = {'event_type': 'price_change', 'timestamp': '124', 'price_changes': [
        {'asset_id': 'DOWN', 'best_bid': '.1', 'best_ask': '.2'},
        {'asset_id': 'UP', 'best_bid': '.8', 'best_ask': '.9'}]}
    assert pm_updates(change, 'UP') == [('124', .8, .9)]
    del change['price_changes'][1]['best_bid']
    with pytest.raises(KeyError):
        pm_updates(change, 'UP')


def test_pm_empty_side_sentinels_are_not_tradable_depth():
    event = {'event_type': 'price_change', 'timestamp': '124', 'price_changes': [
        {'asset_id': 'UP', 'best_bid': '0', 'best_ask': '1'}]}
    assert pm_updates(event, 'UP') == [('124', None, None)]


def test_declared_selection_accounts_for_stress_without_duplicates():
    audit = {'complete_window_coverage': [{'start_s': i * 300, 'market_id': i} for i in range(4)],
             'book_gap_alerts': [{'market_id': 1}, {'market_id': 2}]}
    result = select_windows(audit, 2)
    assert [(x['market_id'], x['selection']) for x in result] == [
        (0, 'REGULAR'), (1, 'KNOWN_GAP_STRESS'), (2, 'REGULAR_AND_STRESS')]
    audit['complete_window_coverage'].append(audit['complete_window_coverage'][0])
    with pytest.raises(ValueError, match='EMPTY_OR_DUPLICATED_COHORT'):
        select_windows(audit, 2)


def test_metadata_knowledge_cutoff_and_mapping_refusal(tmp_path):
    path = tmp_path / 'markets.jsonl'
    row = {'slug': 'test', 'recv_ns': 5, 'outcomes': '["Up", "Down"]', 'clobTokenIds': ['UP', 'DOWN']}
    path.write_text(json.dumps(row) + '\n')
    assert metadata(path, {'test'}, 5)['test']['_up_token'] == 'UP'
    assert metadata(path, {'test'}, 4) == {}
    row['outcomes'] = '["Down", "Up"]'
    path.write_text(json.dumps(row) + '\n')
    with pytest.raises(ValueError, match='PM_OUTCOME_IDENTITY'):
        metadata(path, {'test'}, 5)


def test_unidentifiable_ledger_records_are_counted_not_assigned_to_a_market(tmp_path):
    path = tmp_path / 'markets.jsonl'
    path.write_text('{}\n')
    issues = Counter()
    assert metadata(path, {'test'}, 5, issues) == {}
    assert issues == {'UNIDENTIFIABLE_METADATA_RECORD': 1}
