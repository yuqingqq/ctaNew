"""Bounded review of saved ledger artifacts; no tape replay or training.

Usage via PF_SLOT=codex run_light.sh: python ABSOLUTE_SCRIPT PINNED_CHECKOUT
"""
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

TARGET = '656f322fab161354f26032a43a2b917138c33a0e'
CHECKOUT = Path(sys.argv[1]).resolve()
ROOT = Path('/home/yuqing/ctaNew')
OUT = Path(__file__).resolve().parent
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=CHECKOUT, text=True).strip() == TARGET
sys.path.insert(0, str(CHECKOUT / 'predictfun/research'))
import pf_queue_place_ledger as QL  # noqa: E402


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    copies = CHECKOUT / 'predictfun/reviews/queue_place_ledger_20261117_v2'
    data = ROOT / 'data/predict_fun/research/queue_place_20261117_v2'
    original = ROOT / 'data/predict_fun/research/queue_place_20261117'
    claimed = json.loads((copies / 'queue_place_ledger.json').read_text())
    evidence = dict(target_commit=TARGET, review_asof=QL.C.stamp(),
                    result_asof=claimed['asof'], scope='Existing derived tables on consumed Oct23-27 only; no raw tape, replay, training or forward validation.',
                    review_script_sha256=sha(__file__), committed_artifacts={}, per_day={})

    # Check all newly committed result artifacts, including logs, against their
    # live output copies. Receipt paths are flattened only in the committed copy.
    for saved in sorted(copies.rglob('*')):
        if not saved.is_file():
            continue
        relative = saved.relative_to(copies)
        if saved.name.startswith('receipt_'):
            folder = saved.stem.removeprefix('receipt_')
            actual = data / folder / 'receipt.json'
        else:
            actual = data / relative
        assert sha(saved) == sha(actual), str(relative)
        evidence['committed_artifacts'][str(relative)] = sha(saved)
        if saved.suffix == '.log' and saved.name != 'driver.log':
            assert ' rc=0 ' in saved.read_text().splitlines()[-1], str(relative)
    assert '29 / 29 planted code errors fail it' in (copies / 'logs/qpl_mutants.log').read_text()

    accumulated = {k: [] for k in ('state_arms', 'grid', 'event_arms')}
    old_rows = []
    positive_controls = {}
    for day in QL.DAYS:
        receipt = json.loads((copies / f'receipt_{day}.json').read_text())
        old_receipt = json.loads((original / day / 'receipt.json').read_text())
        assert receipt['limit'] is None and receipt['day'] == day
        assert receipt['provenance']['uncommitted'] == ''
        for name, expected in receipt['provenance']['source'].items():
            assert sha(CHECKOUT / 'predictfun/research' / name) == expected
        assert sha(original / day / 'rows.parquet') == receipt['v1']['rows'] == old_receipt['outputs']['rows.parquet']
        frames = {}
        for name, expected in receipt['outputs'].items():
            assert sha(data / day / name) == expected, (day, name)
            frames[Path(name).stem] = pd.read_parquet(data / day / name)
        old = pd.read_parquet(original / day / 'rows.parquet')
        assert not QL.conserve(frames, receipt['stat'])
        mismatches, summary = QL.reconcile(old, old_receipt['stat'], frames, receipt['stat'])
        assert not mismatches and summary == receipt['reconciliation']

        # Independent identities on saved action keys, not just receipt counts.
        grid = frames['grid']
        counted = grid[grid.cand_status == 'counted']
        arms = frames['state_arms']
        keys = ['day', 'm', 'side', 'k']
        assert not arms.duplicated(keys + ['arm']).any()
        assert (arms.groupby(keys).size() == 3).all()
        assert len(arms) == 3 * len(counted)
        assert (arms.groupby('arm').size() == len(counted)).all()
        lost = counted[counted.path != 'eligible']
        old_states = old[old.kind == 'state'].drop_duplicates(['m', 'side', 't'])
        assert len(counted) - len(old_states) == len(lost)
        missing2 = arms[(arms.status == 'eligible') & (arms.status_2s != 'eligible')]
        assert len(missing2) == int(old[old.kind == 'state'].v_2s.isna().sum())
        assert missing2.status_2s.eq('missing_endpoint').all()
        for table in (arms, frames['event_arms']):
            for tag in QL.TAGS:
                eligible = table['status' + tag].eq('eligible')
                for col in QL.VALS[tag]:
                    assert np.array_equal(table[col].notna(), eligible)
                    assert np.isfinite(table.loc[eligible, col]).all()
                if 'rej' + tag in table:
                    refused = table['rej' + tag] > 0
                    assert (table.loc[refused, 'q' + tag] == 0).all()
                    assert (table.loc[refused, 'v' + tag] == 0).all()
        windows = np.sort(frames['markets'].ws.values)
        start = pd.Timestamp(day, tz='UTC').timestamp() * 1000
        complete_day = np.array_equal(windows, start + 300000 * np.arange(288))
        assert complete_day
        stat = receipt['stat']
        assert stat['parity_bad'] == 0 and stat['parity_checked'] > 0
        assert stat['mut_checked'] == stat['mut_differs'] > 0
        assert all(receipt['planted_errors_flagged'].values())
        for name in ('state_arms', 'grid', 'event_arms'):
            accumulated[name].append(frames[name])
        old_rows.append(old)
        if day == QL.DAYS[0]:
            positive_controls = QL.falsifiers(old, old_receipt['stat'], frames, stat, True)
            assert len(positive_controls) == 10 and all(positive_controls.values())
            assert not QL._l5_cases()
        evidence['per_day'][day] = dict(asof=receipt['asof'], counted=len(counted), saved_states=len(old_states),
            gap=len(lost), gap_by_status=lost.path.value_counts().to_dict(), missing_2s_arm_rows=len(missing2),
            complete_288_market_UTC_day=bool(complete_day), conservation_violations=[], reconciliation_mismatches=[],
            output_hashes=receipt['outputs'])
        print(f'{day}: artifact hashes, conservation and original-row reconciliation pass', flush=True)
        del frames
        gc.collect()

    ledger = {k: pd.concat(v, ignore_index=True) for k, v in accumulated.items()}
    A = pd.concat(old_rows, ignore_index=True)
    readings_old = QL.v1_readings(A)
    readings_new = QL.v1_readings(QL.to_v1(ledger))
    flags = {key: value[0] for key, value in readings_new.items()}
    assert len(flags) == 18
    assert flags == claimed['v1_readings_from_ledger'] == json.loads((original / 'queue_place.json').read_text())['readings']
    for key, (flag, mean) in readings_old.items():
        assert flag == readings_new[key][0] and np.isclose(mean, readings_new[key][1], rtol=0, atol=1e-9, equal_nan=True)
    states = ledger['state_arms']
    counted = ledger['grid'][ledger['grid'].cand_status == 'counted']
    lost = counted[counted.path != 'eligible']
    assert len(lost) == claimed['gap'] == 2054
    assert lost.path.value_counts().to_dict() == claimed['gap_by_status']
    missing2 = states[(states.status == 'eligible') & (states.status_2s != 'eligible')]
    assert missing2.status_2s.value_counts().to_dict() == claimed['arm_rows_without_2s']
    values = {}
    for arm in QL.ARMS_S:
        subset = states[states.arm == arm]
        included = subset[subset.status == 'eligible']
        excluded = subset[(subset.arm_status == 'eligible') & (subset.status != 'eligible')]
        old2 = included[included.status_2s == 'eligible']
        all2 = subset[subset.status_2s == 'eligible']
        ex2 = excluded[excluded.status_2s == 'eligible']
        shift = float(all2.v_2s.mean() - old2.v_2s.mean())
        assert np.isclose(shift, claimed['move_2s'][arm], rtol=0, atol=1e-12)
        # Independent payoff-bound formula: q in [0,30], terminal mark in [0,100].
        lower = -30 * np.where(excluded.side == 0, excluded.p, 100 - excluded.p)
        upper = 30 * np.where(excluded.side == 0, 100 - excluded.p, excluded.p)
        saved_sum = float(included.v.sum())
        low, high = saved_sum + lower.sum(), saved_sum + upper.sum()
        robust = bool(high < 0 if saved_sum < 0 else low > 0 if saved_sum > 0 else False)
        assert robust == claimed['L5b_robust_at_payoff_bounds'][arm]
        assert QL.tip(included.v, len(excluded))[2] == claimed['L5a_tip_outside_observed_range'][arm]
        values[arm] = dict(eligible_5s=len(included), mean_5s_c=float(included.v.mean()), excluded_feasible_5s=len(excluded),
            eligible_2s=len(all2), mean_2s_c=float(all2.v_2s.mean()), coverage_shift_2s_c=shift,
            excluded_but_2s_eligible=len(ex2), excluded_mean_2s_c=float(ex2.v_2s.mean()),
            worst_case_total_value_c=[float(low), float(high)], sign_robust_at_bounds=robust)
    for name, arm in (('R1 REST1 - JOIN', 'REST1'), ('R2 IMPROVE - JOIN', 'IMPROVE')):
        keys = ['day', 'm', 'side', 'k']
        left = states[states.arm == arm].set_index(keys).sort_index()
        right = states[states.arm == 'JOIN'].set_index(keys).sort_index()
        assert left.index.equals(right.index)
        feasible = left.arm_status.eq('eligible') & right.arm_status.eq('eligible')
        included = feasible & left.status.eq('eligible') & right.status.eq('eligible')
        excluded = feasible & ~included
        differences = left.loc[included, 'v'] - right.loc[included, 'v']
        lo, hi = QL.pair_bounds(left.index.get_level_values('side')[excluded], left.loc[excluded, 'p'], right.loc[excluded, 'p'])
        robust = QL.worst(float(differences.sum()), lo, hi)
        assert robust == claimed['L5b_robust_at_payoff_bounds'][name]
        assert QL.tip(differences, int(excluded.sum()))[2] == claimed['L5a_tip_outside_observed_range'][name]
    smoke = json.loads((copies / 'receipt_smoke_20261023.json').read_text())
    for filename, expected in smoke['outputs'].items():
        assert sha(data / 'smoke_20261023' / filename) == expected
    assert claimed['L1'] and claimed['L2'] and claimed['L3']
    assert not claimed['L4_coverage_matters_at_2s']
    assert all(abs(v['coverage_shift_2s_c']) < .25 for v in values.values())
    evidence.update(counted_states=len(counted), gap=len(lost), gap_by_status=lost.path.value_counts().to_dict(),
        gap_by_contract_minute={int(k): int(v) for k, v in lost.groupby('phase').size().items()},
        missing_2s_arm_rows=len(missing2), original_readings_reproduced=18, arms=values,
        controls=dict(clean_artifacts_pass=True, ten_planted_corruptions_refused=positive_controls, legal_payoff_controls_pass=True),
        verdict='No discrepancy found in saved-table reconciliation. Accounting gaps are explained; profitability and fill-model validity are not established.')
    (OUT / 'receipt_v1.json').write_text(json.dumps(QL.C.clean(evidence), indent=2) + '\n')
    print(json.dumps(QL.C.clean(dict(gap=evidence['gap'], gap_by_status=evidence['gap_by_status'], missing_2s_arm_rows=len(missing2), arms=values)), indent=2))


if __name__ == '__main__':
    main()
