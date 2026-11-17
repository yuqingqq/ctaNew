"""Bounded synthetic review of the pinned unmatched-acceptance fix.

Run via PF_SLOT=codex run_light.sh, with the pinned checkout as argv[1].
No market tape is loaded. The full report is exercised with constructed frames.
"""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

TARGET = '323ddd7d224b7b155b0f8873271b0fdc8346cddb'
CHECKOUT = Path(sys.argv[1]).resolve()
OUT = Path(__file__).resolve().parent
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=CHECKOUT, text=True).strip() == TARGET
sys.path.insert(0, str(CHECKOUT / 'predictfun/research'))
import pf_keep_pull as KP  # noqa: E402


def build():
    day = np.repeat(np.arange(5), 240)
    market = np.tile(np.repeat(np.arange(20), 12), 5)
    short = np.tile(np.r_[np.ones(6, bool), np.zeros(6, bool)], 100)
    n = len(day)
    E = pd.DataFrame(dict(
        day=np.array(KP.DAYS)[day], oid=np.arange(n), m=market, phase=0,
        side=0, mark=49., p=50., status='eligible', fair_missing=0,
        cause=np.where(short, 'behind', ''),
        g_struct=np.where(short, 150., np.nan),
        land_struct=np.where(short, 187., np.nan)))
    for tag in KP.CTAG.values():
        E['v' + tag] = -1.
        E['q' + tag] = 1.
        E['t_full' + tag] = np.nan
    P = np.where(short, 2, 3)
    F = np.where(np.arange(KP.NSTEP + 1)[None, :] >= P[:, None], 49., 51.)
    E = pd.concat([E, pd.DataFrame(F, columns=[f'f{j:02d}' for j in range(KP.NSTEP + 1)])], axis=1)
    Fl = pd.concat([pd.DataFrame(dict(day=E.day, oid=E.oid, clock=c,
                                    t=np.where(short, 170., 300.), q=1., src=0))
                    for c in KP.CLOCKS], ignore_index=True)
    return E, Fl, P, day, day * 20 + market


def main():
    E, Fl, P, day, cell = build()
    assert np.array_equal(KP.score_steps(E[[f'f{j:02d}' for j in range(99)]].values,
                                       E.side.values, E.p.values)['A+0'], P)
    V = KP.variant(E, Fl, 50., False, 37.)
    res = KP.outcome(V, P, day, 5)
    pk = KP.null_picks(V['endj'], P, cell, KP.SEED, 500)
    sk = KP.state_picks(V['endj'], P, cell, KP.SEED + 100, 500)
    null = [KP.outcome(V, pn, day, 5) for pn, _ in pk]
    states = [KP.outcome_states(V, oi, oj, day, 5) for oi, oj in sk]
    unmatched = np.array([um for _, um in pk])
    duplicate_states = np.array([len(oi) - len(np.unique(oi)) for oi, _ in sk])
    assert (unmatched > 0).all()
    assert KP.k2(res, null)[0] and not KP.k3(res, null)[0]  # old failure still reproduced
    assert not KP.k2(res, states)[0] and not KP.k3(res, states)[0]
    assert all(np.array_equal(x['n'], res['n']) for x in states)

    # Exercise actual report dispatch and emitted JSON, not just hierarchy().
    report_dir = OUT / 'synthetic_report'
    report_dir.mkdir(exist_ok=False)
    original_load = KP.load
    KP.load = lambda *args, **kwargs: (E, Fl)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            KP.report(report_dir)
    finally:
        KP.load = original_load
    artifact = json.loads((report_dir / 'keep_pull.json').read_text())
    primary = artifact['D']['M50_L37']['A+0']
    assert primary['traj']['evaluable'] is False
    assert primary['traj']['k2'] is None and primary['traj']['k3'] is None
    assert primary['estimand'] == 'decision states'
    assert primary['k2'] is False and primary['k3'] is False
    assert artifact['readings']['K2 SCORE-EARNS'] == 'NOT-SHOWN'
    for variants in artifact['D'].values():
        for result in variants.values():
            if not result['traj']['evaluable']:
                assert result['traj']['k2'] is None and result['traj']['k3'] is None
            chosen = result['traj'] if result['traj']['evaluable'] else result['states']
            assert result['k2'] == chosen['k2'] and result['k3'] == chosen['k3']

    # Equal-lifetime control: all trajectory pulls match, no economic ranking.
    E2 = E.assign(g_struct=np.nan, land_struct=np.nan)
    Fl2 = Fl.assign(t=300.)
    V2 = KP.variant(E2, Fl2, 50., False, 37.)
    p2 = KP.null_picks(V2['endj'], P, cell, KP.SEED, 200)
    n2 = [KP.outcome(V2, pn, day, 5) for pn, _ in p2]
    r2 = KP.outcome(V2, P, day, 5)
    assert all(um == 0 for _, um in p2)
    assert KP.hierarchy(True) == 'traj'
    assert not KP.k2(r2, n2)[0] and not KP.k3(r2, n2)[0]

    # Positive discrimination: the score pulls only the losing half.
    win = (np.arange(len(E)) % 12) < 6
    E3 = E2.copy()
    E3['mark'] = np.where(win, 51., 49.)
    for tag in KP.CTAG.values():
        E3['v' + tag] = np.where(win, 1., -1.)
    V3 = KP.variant(E3, Fl2, 50., False, 37.)
    p3 = np.where(win, 99, 2)
    r3 = KP.outcome(V3, p3, day, 5)
    n3 = [KP.outcome(V3, pn, day, 5) for pn, _ in KP.null_picks(V3['endj'], p3, cell, KP.SEED, 200)]
    s3 = [KP.outcome_states(V3, oi, oj, day, 5)
          for oi, oj in KP.state_picks(V3['endj'], p3, cell, KP.SEED + 100, 200)]
    assert KP.k2(r3, n3)[0] and KP.k3(r3, n3)[0]
    assert KP.k2(r3, s3)[0] and KP.k3(r3, s3)[0]

    # Known-bad decision: order zero is no longer eligible at step 3.
    try:
        KP.outcome_states(V, np.array([0]), np.array([3]), day, 5)
    except Exception as error:
        assert 'not a live decision' in str(error)
    else:
        raise AssertionError('ineligible state was accepted')

    # Mirrored ask losses must have the same values as the bid case.
    Va = KP.variant(E.assign(side=1, mark=51.), Fl, 50., False, 37.)
    assert all(np.array_equal(V[k], Va[k]) for k in ('Vk', 'Qk', 'Mv', 'Mq', 'endj'))
    # Old repaired controls remain reachable and pass.
    assert not KP.QL._l5_cases()
    assert KP.outcome(V, np.ones(len(E), int), day, 5)['n'].sum() == 0
    assert KP.landing(2, 0.) == 100. and KP.landing(2, 37.) == 137.

    receipt = dict(asof=KP.C.stamp(), target_commit=TARGET,
        scope='Synthetic only. Date strings are fixture labels, not observations from those days. No tape read or performance estimate.',
        source_hashes={name: KP.C.sha(CHECKOUT / 'predictfun/research' / name)
                       for name in ('pf_keep_pull.py', 'pf_queue_place_ledger.py', 'pf_keep_pull_mutants.py')},
        review_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        population=dict(synthetic_orders=len(E), synthetic_day_cells=5, strata=100, null_draws=500),
        exhaustion=dict(score_pulls=int(res['n'].sum()), trajectory_mean_pulls=float(np.mean([x['n'].sum() for x in null])),
            unmatched_mean=float(unmatched.mean()), draws_with_unmatched=int((unmatched > 0).sum()),
            raw_trajectory_k2=True, gated_trajectory_k2=None, gated_trajectory_k3=None,
            state_pulls=1200, state_k2=False, state_k3=False,
            mean_repeated_order_states=float(duplicate_states.mean()),
            interpretation='STATES is a local state-selection diagnostic. Repeated-order suffix values cannot be summed as executable portfolio P&L.'),
        controls=dict(equal_lifetime_draws=200, equal_lifetime_all_matched=True, equal_lifetime_k2=False, equal_lifetime_k3=False,
                      perfect_ranking_draws=200, perfect_ranking_k2_k3_both_nulls=True,
                      ineligible_state_refused=True, ask_bid_mirror_equal=True, legal_payoff_bounds=True,
                      pending_excluded_from_live=True, live_latency_landing=True,
                      report_gates_every_unmatched_variant=True),
        report_sha256=KP.C.sha(report_dir / 'keep_pull.json'))
    (OUT / 'receipt_v1.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
