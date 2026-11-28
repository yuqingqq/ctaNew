"""Latency sensitivity of the frozen chase forward candidate at the Tokyo-measured order path (USER 2026-11-28: "Host you mean this server?
We will run live in Tokyo"; "We can rerun the test again"). P-2026-005, public data, no orders. Declared before the run. A sensitivity of the
frozen candidate on the forward days it was tested on (11-20 .. 11-26, read by this line on 11-27), not a new test: nothing is chosen and
there is no PASS rule.
Why these latencies: the replays put our order at Predict 22 ms after sending (half the Tokyo probe's 44 ms create round trip,
predictfun/research/pf_clock_rl.py). The exec port's correction from Predict's wallet-event stamps (2026-11-03) has an order live 58 ms
after sending (median; at least 46 given the clock offset) and a cancel in effect 37 ms after sending (at least 25). + 36 ms = the median
order path, + 50 ms = a margin for slower orders; + 25 ms (about the lower bound, in the forward result) is re-run as the parity check.
L is added to every message between us and Predict, each way, by chase_forward.exits_day unchanged (EXIT_FIXED's convention): the entry
IOC's arrival (the 150 ms hold follows), our fill notice, post-only orders, cancels, refusals and the exit IOC's arrival. The Binance and
Predict feeds are unchanged (both already on the Tokyo clock). A uniform L overstates the measured cancel delay (+ 15 ms at the median)
and understates the wallet fill notice (~100 ms after the match, latency test L2c); taking at the 1 s calm sends no cancel, and the day
stage counts the primary's filled entries whose calm would move if the notice came 100 ms after the match.
Readings (declared; computed in read): the primary cell (RECEIPT_v5's, taking at the 1 s calm) at + 36 / + 50 ms: total > 0, days > 0 of
7, day-equivalent Sharpe, max drawdown, filled; every cell and exit as point estimates; the control (random move onsets) at each latency.
No null at these latencies: the selection's skill against random entries was the PASS rule's test at + 0 ms.
Parity: the + 25 ms columns and the identity columns equal the forward run's stored exits exactly on every day (the day stage refuses to
write otherwise), and the read's + 25 ms cells equal chase_forward_result.json's.
Stages: selftest | exits <day> | read
Outputs (write-once): OUT/chase_forward/exits_lat/<day>.parquet + <day>.json, chase_forward_latency.json (CFL_OUT=<dir>: a rehearsal
writes there instead, and its read skips the cell parity when it has fewer than the seven days).
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/chase_forward_latency.py <stage> ...
"""
import datetime as dt
import json
import os
import re
import resource
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import chase_forward as CF  # noqa: E402

E, EF, M2, M3, SX = CF.E, CF.EF, CF.M2, CF.M3, CF.SX
C, TR, TM, CT, MM = CF.C, CF.TR, CF.TM, CF.CT, CF.MM
LATS2 = (25., 36., 50.)
L_PARITY = 25.
NOTICE_MS = 100.        # the wallet fill notice after the match (latency test L2c, 2026-11-03)
REHEARSAL = bool(os.environ.get('CFL_OUT'))
OUT_LAT = Path(os.environ['CFL_OUT']) if REHEARSAL else CF.FOUT / 'exits_lat'
RESULT = OUT_LAT / 'read.json' if REHEARSAL else H / 'chase_forward_latency.json'
LCOL = re.compile(r'L\d+_')


def primary(rc):
    """The receipt's primary cell -> ('SEL|AB_sN', SEL, N)."""
    p = re.search(r'\*\*([A-Za-z0-9_]+\|AB_s\d+)\*\*', rc).group(1)
    sel, pre_s = p.split('|')
    return p, sel, float(pre_s.split('_s')[1])


def same_col(x, y):
    if x.dtype.kind in 'fiu' and y.dtype.kind in 'fiu':
        return x.shape == y.shape and np.array_equal(x.to_numpy(float), y.to_numpy(float), equal_nan=True)
    return x.reset_index(drop=True).equals(y.reset_index(drop=True))


def parity(new, old, L=L_PARITY):
    """new's identity columns and L{L}_* columns must equal old's exactly (NaN = NaN), with the same column sets and rows."""
    pre = f'L{int(L)}_'
    idn = sorted(c for c in new.columns if not LCOL.match(c)); ido = sorted(c for c in old.columns if not LCOL.match(c))
    ln = sorted(c for c in new.columns if c.startswith(pre)); lo = sorted(c for c in old.columns if c.startswith(pre))
    bad = []
    if idn != ido:
        bad.append('identity columns differ')
    if ln != lo:
        bad.append(f'{pre} columns differ')
    if len(new) != len(old):
        bad.append('rows differ')
    cols = [c for c in idn + ln if c in old.columns]
    if not bad:
        bad += [c for c in cols if not same_col(new[c], old[c])]
    return dict(columns=len(cols), rows=int(len(new)), mismatched=bad)


def notice_check(X, kn, sel, s_p, lats=LATS2):
    """Filled EARLY entries of the selection whose 1 s calm moves if our fill notice comes NOTICE_MS after the match instead of the
    replay's 22 ms + L."""
    m = X[sel].to_numpy(bool) & (X.cut_minus_act_ms.to_numpy(float) >= EF.EARLY_MS)
    out = {}
    for L in lats:
        f = m & (np.nan_to_num(X[f'L{int(L)}_AB_s{int(s_p)}_fill'].to_numpy(float)) > 0)
        mt = X.t.to_numpy(float)[f] + TM.FEED + TM.DECIDE + E.NET + L + E.HOLD
        c_rep = M2.calm_times(kn, mt + M2.LEARN + L, EF.GAP)
        c_slow = M2.calm_times(kn, mt + NOTICE_MS, EF.GAP)
        out[f'L{int(L)}'] = dict(filled=int(f.sum()), calm_moves=int((c_slow != c_rep).sum()))
    return out


def stage_exits(day):
    C.require(day in CF.FWD, f'{day} is not a declared forward day')
    fp, fj = OUT_LAT / f'{day}.parquet', OUT_LAT / f'{day}.json'
    C.require(not fp.exists() and not fj.exists(), f'{fp.name} exists (write-once)')
    resource.setrlimit(resource.RLIMIT_CPU, (2400, 2401))
    rc = CF.frozen()
    _, sel, s_p = primary(rc)
    CF.forward_mode((day,))
    S = pd.read_parquet(CF.FOUT / 'forward_selections.parquet')
    X, st = CF.exits_day(day, S, OUT_LAT, lats=LATS2)
    old = pd.read_parquet(CF.FOUT / 'exits' / f'{day}.parquet')
    par = parity(X, old)
    del old
    C.require(par['columns'] > 0 and not par['mismatched'], f'+ {int(L_PARITY)} ms parity failed on {day}: {par["mismatched"][:8]}')
    BN = M3.Binance(*CT.mid_series(day))
    st.update(parity_L25=par, notice_100ms=notice_check(X, BN.kn, sel, s_p), grid_drops=dict(CF.GRID_DROPS), latencies_ms=list(LATS2),
              rehearsal=REHEARSAL)
    X.to_parquet(fp, index=False)
    TR.dump(fj, st)
    print(day, json.dumps(TR.clean(st)), flush=True)


def cell_parity(cells, ref, L=L_PARITY):
    """The L cells (through the writer's serialization) against the stored result's: (count, mismatched keys)."""
    tag = f'|L{int(L)}_'
    a = json.loads(json.dumps(TR.clean({k: v for k, v in cells.items() if tag in k}), allow_nan=False))
    b = {k: v for k, v in ref.items() if tag in k}
    bad = sorted(set(a) ^ set(b)) + sorted(k for k in set(a) & set(b) if a[k] != b[k])
    return len(a), bad


def stage_read():
    C.require(not RESULT.exists(), f'{RESULT.name} exists (write-once)')
    resource.setrlimit(resource.RLIMIT_CPU, (1800, 1801))
    rc = CF.frozen()
    p, sel, s_p = primary(rc)
    days = tuple(d for d in CF.FWD if (OUT_LAT / f'{d}.parquet').exists())
    C.require(days == CF.FWD or (REHEARSAL and days), f'exits_lat has {len(days)} of {len(CF.FWD)} days')
    X = pd.concat([pd.read_parquet(OUT_LAT / f'{d}.parquet') for d in days], ignore_index=True)
    keys, dayk = [], []
    for d in days:
        reg = MM.registry(d).sort_values('ws')
        keys += [f'{d}:{int(m)}' for m in reg.mid]; dayk += [d] * len(reg)
    keys, dayk = np.array(keys), np.array(dayk)
    SX.DAYS = days
    cells = CF.cells_of(X, keys, dayk, len(keys) / len(days), days, EF.EARLY_MS, LATS2)
    R0 = json.loads((H / 'chase_forward_result.json').read_text())
    C.require(R0['primary'] == p, 'the result and the receipt name different primaries')
    n25, bad25 = cell_parity(cells, R0['cells'])
    if days == CF.FWD:
        C.require(n25 > 0 and not bad25, f'+ {int(L_PARITY)} ms cells differ from chase_forward_result.json: {bad25[:8]}')
    key = lambda L: f'{sel}|L{int(L)}_AB_s{int(s_p)}|TAKE_C1000'  # noqa: E731
    by_lat = {f'L{int(L)}': R0['cells'][key(L)] for L in CF.LATS if L not in LATS2}
    by_lat.update({f'L{int(L)}': TR.clean(cells[key(L)]) for L in LATS2})
    readings = {f'L{int(L)}': dict(total_positive=bool(cells[key(L)]['total_usd'] > 0), days_positive=cells[key(L)]['days_positive'],
                                   days=len(days), sharpe_day_equiv=cells[key(L)]['sharpe_day_equiv'],
                                   max_drawdown_usd=cells[key(L)]['max_drawdown_usd'], actions=cells[key(L)]['actions'],
                                   filled=cells[key(L)]['filled']) for L in LATS2 if L != L_PARITY}
    st = {d: json.loads((OUT_LAT / f'{d}.json').read_text()) for d in days}
    notice = {f'L{int(L)}': {k: sum(v['notice_100ms'][f'L{int(L)}'][k] for v in st.values()) for k in ('filled', 'calm_moves')} for L in LATS2}
    out = dict(asof=dt.datetime.now(dt.timezone.utc).isoformat(), receipt=CF.receipt_path().name, primary=p, days=list(days),
               rehearsal=REHEARSAL, latencies_ms=list(LATS2), parity_latency_ms=L_PARITY,
               replay_base=dict(order_ms=E.NET, hold_ms=E.HOLD, fill_notice_ms=M2.LEARN, binance_feed_ms=TM.FEED, decide_ms=TM.DECIDE),
               tokyo_measured=dict(order_live_p50_ms=58., order_live_lower_ms=46., cancel_p50_ms=37., cancel_lower_ms=25.,
                                   fill_notice_ms=NOTICE_MS, source='predictfun/research/pf_clock_rl.py (exec-port Tokyo probe 2026-11-02, corrected 2026-11-03)'),
               readings=readings, primary_by_latency=by_lat, parity_L25=dict(cells=n25, mismatched=bad25, checked=days == CF.FWD),
               notice_100ms=notice,
               statuses={d: dict(actions=v['actions'], unloaded=v['unloaded'], capped_loops=v['capped_loops'], grid_drops=v['grid_drops'])
                         for d, v in st.items()},
               cells={k: v for k, v in cells.items() if f'|L{int(L_PARITY)}_' not in k})
    TR.dump(RESULT, out)
    print('primary', p, json.dumps(TR.clean(readings)), 'parity L25', n25, len(bad25), flush=True)


def selftest():
    C.require(L_PARITY in LATS2 and L_PARITY in CF.LATS and all(L not in CF.LATS for L in LATS2 if L != L_PARITY), 'latency sets')
    rc = CF.receipt_path().read_text()
    p, sel, s_p = primary(rc)
    C.require(p == json.loads((H / 'chase_forward_result.json').read_text())['primary'] and (sel, s_p) == ('MOVE_top10', 3.), f'primary {p}')
    # parity: a positive control (equal, with NaN pairs) and three known-bad inputs it must flag
    new = pd.DataFrame(dict(day=['a', 'b'], t=[1., 2.], MOVE_top10=[True, False], cut_minus_act_ms=[1., np.nan],
                            L25_AB_s3_fill=[1., np.nan], L36_AB_s3_fill=[2., 3.]))
    old = new.drop(columns=['L36_AB_s3_fill']).assign(L0_AB_s3_fill=[5., 6.])
    assert parity(new, old)['mismatched'] == [] and parity(new, old)['columns'] == 5, parity(new, old)
    o2 = old.copy(); o2.loc[1, 'L25_AB_s3_fill'] = 0.
    assert parity(new, o2)['mismatched'] == ['L25_AB_s3_fill'], parity(new, o2)
    o3 = old.copy(); o3.loc[0, 't'] = 1.5
    assert parity(new, o3)['mismatched'] == ['t'], parity(new, o3)
    assert 'L25_ columns differ' in parity(new.drop(columns=['L25_AB_s3_fill']), old)['mismatched']
    assert 'identity columns differ' in parity(new.drop(columns=['MOVE_top10']), old)['mismatched']
    # notice: Binance changes known at 0 and 5,000 ms. Entry A (t 100): its calm (1,000) lies past both notices -> unchanged.
    # Entry B (t 2,000): Binance already calm at the notice -> the calm moves with it. Entry C: not filled -> not counted.
    kn = np.array([0., 5000.])
    X = pd.DataFrame(dict(t=[100., 2000., 2000.], MOVE_top10=[True, True, True], cut_minus_act_ms=[1e5, 1e5, 1e5]))
    for L in LATS2:
        X[f'L{int(L)}_AB_s3_fill'] = [10., 10., 0.]
    nc = notice_check(X, kn, 'MOVE_top10', 3.)
    assert all(v == dict(filled=2, calm_moves=1) for v in nc.values()), nc
    X2 = X.assign(cut_minus_act_ms=[1e5, 1., 1e5])          # B outside EARLY -> nothing moves
    assert all(v == dict(filled=1, calm_moves=0) for v in notice_check(X2, kn, 'MOVE_top10', 3.).values())
    # cell parity through the writer's serialization: equal, then one changed value and one missing key
    ref = {'A|L25_AB_s3|TAKE_C1000': dict(total_usd=1.5, sharpe_day_equiv=None, daily_usd=[1.5]), 'A|L0_AB_s3|TAKE_C1000': dict(total_usd=9.)}
    cel = {'A|L25_AB_s3|TAKE_C1000': dict(total_usd=np.float64(1.5), sharpe_day_equiv=float('nan'), daily_usd=np.array([1.5]))}
    assert cell_parity(cel, ref) == (1, []), cell_parity(cel, ref)
    assert cell_parity({k: dict(v, total_usd=1.25) for k, v in cel.items()}, ref)[1] == ['A|L25_AB_s3|TAKE_C1000']
    assert cell_parity(dict(cel, **{'B|L25_AB_s3|HOLD': dict(total_usd=0.)}), ref)[1] == ['B|L25_AB_s3|HOLD']
    # refusals, with their reasons
    try:
        stage_exits('20261101')
        raise AssertionError('stage_exits accepted a consumed day')
    except ValueError as e_:
        assert 'not a declared forward day' in str(e_), e_
    global RESULT
    keep = RESULT
    with tempfile.NamedTemporaryFile(suffix='.json') as tf:
        RESULT = Path(tf.name)
        try:
            stage_read()
            raise AssertionError('stage_read overwrote its result')
        except ValueError as e_:
            assert 'write-once' in str(e_), e_
        finally:
            RESULT = keep
    print('chase_forward_latency selftest PASS')


if __name__ == '__main__':
    cmd, a_ = sys.argv[1], sys.argv[2:]
    dict(selftest=selftest, read=stage_read, exits=lambda: stage_exits(a_[0]))[cmd]()
