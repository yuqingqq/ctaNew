"""Reconcile submitted bank ablation artifacts; no fitting or policy selection."""
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
import pf_bank_two_sided as B

C = B.C
ROOT = Path(__file__).resolve().parents[2]
REVIEW = ROOT / 'predictfun/reviews/bank_two_sided_review_20261117'


def same(a, b):
    C.require(np.allclose(a, b, atol=1e-9, rtol=0, equal_nan=True), 'numeric mismatch')


def hash_check(path, expected):
    C.require(C.sha(path) == expected, 'artifact hash mismatch')


def refused(fn):
    try:
        fn()
    except (ValueError, AssertionError):
        return True
    raise ValueError('known-bad input accepted')


def controls():
    same([1., np.nan], [1., np.nan])
    hash_check(__file__, C.sha(__file__))
    # Duplicating paths adds no information but increases this nominal z score.
    def dispersion_z(x):
        return float(x.mean() / (x.std() / np.sqrt(len(x))))
    x = np.array([-1., 3.]); copied = np.repeat(x, 4)
    same(x.mean(), copied.mean()); same(x.std(), copied.std())
    same(dispersion_z(copied), 2 * dispersion_z(x))
    return dict(equal_positive=True, corrupt_number_refused=refused(lambda: same([1.], [2.])),
                valid_hash_positive=True, corrupt_hash_refused=refused(lambda: hash_check(__file__, '0'*64)),
                duplicate_paths_same_distribution=True, duplicate_paths_change_nominal_z=True)


def run():
    C.require(not (REVIEW/'review_v1.json').exists(), 'review exists')
    checks = controls(); B.selftest(); B.V.verify(); inputs = {}
    source = ROOT/'predictfun/research/pf_bank_two_sided.py'
    committed = subprocess.check_output(['git', 'show', '693960d8:predictfun/research/pf_bank_two_sided.py'], cwd=ROOT)
    hash_check(source, hashlib.sha256(committed).hexdigest())
    def read(path):
        path = Path(path); inputs[str(path)] = C.sha(path)
        return json.loads(path.read_text())
    def checked(directory):
        receipt = read(directory/'receipt.json')
        for name, digest in receipt['outputs'].items():
            hash_check(directory/name, digest); inputs[str(directory/name)] = digest
        return receipt
    published = ROOT/'predictfun/reviews/bank_two_sided_20261117'
    original = read(published/'two_sided_bank.json')
    hash_check(B.OUT/'two_sided_bank.json', C.sha(published/'two_sided_bank.json'))
    panels = []; lineage = {}
    for day in B.EVAL_DAYS:
        dest = B.OUT/('actions_'+day); receipt = checked(dest)
        hash_check(published/('actions_'+day+'_receipt.json'), C.sha(dest/'receipt.json'))
        archive = B.V.OUT/('actions_'+day); checked(archive)
        hash_check(archive/'receipt.json', receipt['archive_receipt_sha256'])
        frame = pd.read_parquet(dest/'values.parquet')
        C.require(not frame.duplicated(['uid','bank']).any(), 'duplicate actions')
        old = pd.read_parquet(archive/'values.parquet'); old = old[old.case == B.CASE].set_index('uid')
        for bank in B.BANKS:
            x = frame[frame.bank == bank].set_index('uid')
            C.require(set(x.index) == set(old.index), 'action population mismatch')
            old = old.loc[x.index]
            same(x.actual, old.actual); same(x.actual_quantity, old.actual_quantity)
            if bank == 'OLD':
                C.require((x.evaluation_status == old.evaluation_status).all(), 'status mismatch')
                for col in B.QCOLS: same(x[col], old[col])
            path = B.F.OUT/'bank/paths.pkl' if bank == 'OLD' else B.OUT/'banks'/bank/'paths.pkl'
            hash_check(path, receipt['banks'][bank]); inputs[str(path)] = C.sha(path)
        lineage[day] = dict(n=len(old), statuses=frame.groupby('bank').evaluation_status.value_counts().to_dict())
        panels.append(frame)
    values = pd.concat(panels, ignore_index=True)
    eligible = values.groupby('uid').evaluation_status.apply(lambda x: (x == 'eligible').all())
    common = set(eligible[eligible].index); v = values[values.uid.isin(common)].copy()
    v['z'] = v.predicted/(v.scenario_advantage_sd/np.sqrt(v.effective_paths.clip(lower=1))).replace(0, np.nan)
    metrics = {}
    for (bank, day, side, kind), g in v.groupby(['bank','day','side','kind']):
        key = f'{bank}:{day}:{side}:{kind}'
        rmse = float(np.sqrt(np.mean((g.predicted-g.actual)**2)))
        same(rmse, original['rmse'][key])
        zero = float(np.sqrt(np.mean(g.actual**2)))
        metrics[key] = dict(n=len(g), rmse_cents=rmse, zero_rmse_cents=zero,
                            relative_to_zero=rmse/zero-1)
    def a1(bank):
        ask = sum(metrics[f'{bank}:{d}:1:{k}']['rmse_cents'] < metrics[f'OLD:{d}:1:{k}']['rmse_cents']
                  for d in B.EVAL_DAYS for k in ('post','cancel'))
        bid = sum(metrics[f'{bank}:{d}:0:{k}']['rmse_cents'] <= 1.05*metrics[f'OLD:{d}:0:{k}']['rmse_cents']
                  for d in B.EVAL_DAYS for k in ('post','cancel'))
        return dict(ask_improves=ask, bid_within_5pct=bid, passes=ask>=3 and bid>=3)
    nominal_posts = v[(v.bank == 'TWO')&(v.kind == 'post')&(v.z >= 2)]
    readings = dict(A1='ASK-CALIBRATION-IMPROVES' if a1('TWO')['passes'] else 'NOT-SHOWN',
                    A2=('CONFIDENT-POSITIVE-POSTS' if nominal_posts.actual.mean()>0 else 'MISCALIBRATED-POSITIVE-POSTS')
                    if len(nominal_posts)>=5 else 'NONE')
    pilot = B.OUT/'pilot'; receipt = checked(pilot)
    hash_check(published/'pilot_receipt.json', C.sha(pilot/'receipt.json'))
    archive = B.V.OUT/'pilot'; checked(archive)
    hash_check(archive/'receipt.json', receipt['archive_receipt_sha256'])
    fills = pd.read_parquet(pilot/'fills.parquet')
    old = pd.read_parquet(archive/'fills.parquet'); old = old[old.case == B.CASE]
    key = ['m','side','rung','fill_seq','fill_ms']
    pd.testing.assert_frame_equal(fills[fills.bank=='OLD'][old.columns].sort_values(key).reset_index(drop=True),
                                  old.sort_values(key).reset_index(drop=True), check_exact=True)
    stats = {}; marked = {}
    for bank in ('OLD','TWO'):
        for side in ('all',0,1):
            g = fills[(fills.bank==bank)&(fills.minute<5)]
            if side != 'all': g = g[g.side==side]
            marks = [B._cs(g, f'markout_{h}_usd') for h in (1000,5000)]
            if side != 'all':
                same(marks, original['markouts'][f'{bank}:{side}:1-4'])
                marked[bank,side] = marks
            stats[f'{bank}:{side}'] = dict(fills=len(g), shares=float(g.shares.sum()),
                markouts_cents_per_share=marks,
                mark_statuses={str(h): g[f'markout_{h}_status'].value_counts().to_dict() for h in (1000,5000)})
    r1 = marked['TWO',1][0] > marked['OLD',1][0] and marked['TWO',1][1] > marked['OLD',1][1] and marked['TWO',0][0] >= marked['OLD',0][0]-.5
    readings['R1'] = 'ASK-MARKOUT-IMPROVES' if r1 else 'NOT-SHOWN'
    C.require(readings == original['readings'], 'published readings mismatch')
    result = dict(asof=C.stamp(), reviewed_source_commit='693960d8', reviewed_results_commit='f1d53700',
        original_asof=original['asof'], source_sha256=C.sha(__file__), controls=checks, inputs=inputs,
        lineage=lineage, common_actions=len(common), excluded_actions=int((~eligible).sum()),
        readings_reproduced=readings, ask_calibration={b:a1(b) for b in ('TWO','BID48')}, metrics=metrics,
        nominal_z_ge2_posts=len(nominal_posts), nominal_z_is_calibrated_confidence=False,
        all_old_fill_fields_match=True, pilot_markets=len(receipt['selected_markets']), minutes_1_to_4=stats,
        more_than_5pct_worse_than_zero=[k for k,x in metrics.items() if x['relative_to_zero']>.05],
        limits=['Consumed Oct26-27; descriptive artifact review, no new strategy or model.',
                'Scenario dispersion and effective paths are not calibrated forecast confidence.',
                'A1/R1 retain their original descriptive definitions; no day-cluster intervals.',
                'Bank correction failure does not identify which new estimator will succeed.'])
    C.dump(REVIEW/'review_v1.json', result)
    print(json.dumps(dict(asof=result['asof'], readings=readings, controls=checks,
                         actions=len(common), all_old_fill_fields_match=True, output=str(REVIEW/'review_v1.json'))), flush=True)


if __name__ == '__main__':
    run()
