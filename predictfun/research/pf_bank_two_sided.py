"""TWO-SIDED SCENARIO BANK (2026-11-17, session aea5bfb9; USER: "Yes can you do the next and validate", after the audit: "Next priority: a
controlled bank-representation correction with the model and policy held fixed ... Judge improvement by economic calibration and fill
markout - not fewer cancellations alone"). Declared before any run; exploratory, consumed days, not profitability validation.

CHANGE (the only one): SOFT_JOINT's training scenario bank. OLD = the other session's frozen bank (flow_control_20261116_v3/bank): paths
sampled at neutral BID rung-0 at-best probe times only (pf_flow_control_paths.build: side == 0), the first 24 per market by identity hash.
TWO = the same rule applied to the ASK rung-0 at-best probes as well (24 per market per side; a time drawn by both sides kept once).
BID48 = the bid rule with 48 per market: a control for TWO's larger candidate pool. Everything else is the frozen SOFT_JOINT, unchanged and
imported: pf_flow_control_paths.extract (paired both-side tape and book), cells (phase, spread, bid class, ask class) with 3 hash-chosen
templates weighted by the cell count, pf_value_bridge.pool / labels (one-tick spread prior, current-quote relabel), 24 IPF iterations to the
frozen Oct27 classifier (trained Oct23-25, temperature Oct26), the greedy controller, cap 180 / lambda 1/360 / 30 sh x 6 rungs, clocks
58 / 37 / 100 / 50 ms, M50, the 1 s horizon. Bank training days Oct23-25 only.
VALIDATION:
 (A) the audited 861 neutral actions (Oct26 418, Oct27 443): pf_value_bridge_actions' own selection, client states and realized advantages
     (read from its archive by action id); SOFT_JOINT's predicted advantage under OLD (must reproduce the archive), TWO and BID48.
 (B) the 12 preselected Oct27 markets, full five-minute contracts: SOFT_JOINT replayed with OLD (must reproduce the archived pilot) and TWO.
READINGS (computed by report(); actions eligible under every bank):
 A1 ASK-CALIBRATION: TWO's RMSE (predicted - realized) below OLD's in >= 3 of the 4 ask cells (day x post / cancel) AND TWO's RMSE <= 1.05 x
    OLD's in >= 3 of the 4 bid cells -> ASK-CALIBRATION-IMPROVES, else NOT-SHOWN. The same rule is evaluated for BID48: if BID48 passes too,
    the improvement is not attributed to two-sidedness (reported MORE-DATA).
 A2 CONFIDENT-POSTS (TWO, both days): posts predicted positive by more than 2 scenario standard errors (SD / sqrt(effective paths)): n >= 5
    and mean realized > 0 -> CONFIDENT-POSITIVE-POSTS; n >= 5 and mean realized <= 0 -> MISCALIBRATED-POSITIVE-POSTS; n < 5 -> NONE.
 R1 ASK-MARKOUT (replay, fills in minutes 1-4): TWO's ask +1 s AND +5 s markouts (c/sh) above OLD's, AND TWO's bid +1 s >= OLD's - 0.5
    c/sh -> ASK-MARKOUT-IMPROVES, else NOT-SHOWN.
 Printed, not decided on: RMSE of the zero predictor, Spearman, positive-band realized means, |z| shares, by bank x day x side x kind;
 replay per-side shares and 1 s / 5 s markouts for minutes 1-4 and minute 5, orders, pulls, first-review cancels, terminal and settlement;
 bank composition (paths, cells, templates drawn at ask times).
CONTROLS (any failure -> no output): the OLD rule rebuilt here equals the other session's bank (path ids, order, base weights, arrays);
 OLD predictions reproduce the archived SOFT_JOINT predictions (four quantities, |diff| <= 1e-9) on the same action ids and statuses; the
 OLD replay reproduces the archived SOFT_JOINT pilot per market (shares, terminal, settlement, orders, pulls, fills); the TWO ask draw comes
 from ask probes; bank identity by hash. Planted errors (selftest): an ask draw taken from bid rows and a bank swapped under another name
 must each be refused; the per-side hash rule is checked on a toy frame.
LIMITS: consumed days (Oct23-27); the replay is 12 markets of one day; simulated (rule M, queue M, unknown depth 0); the realized action
 values are the archive's canonical branches. Usage (run_light, PF_SLOT=claude): pf_bank_two_sided.py selftest | build | actions <day> |
 pilot | report; driver run_bank_two_sided.sh
"""
import argparse
from collections import Counter, defaultdict
import gc
import json
import pickle
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_value_bridge as V                               # noqa: E402
import pf_value_bridge_actions as A                       # noqa: E402
import pf_flow_control_run as RUN                         # noqa: E402
import pf_stop_ev_minutes_metrics as MET                  # noqa: E402
import pf_stop_ev_minutes as AUDIT                        # noqa: E402
import pf_flow_price_model as FM                          # noqa: E402

F, C, I, P, K, PATH, pd = V.F, V.C, V.I, V.P, V.K, V.PATH, V.pd
OUT = Path(I.K.IO) / 'bank_two_sided_20261117'
BANKS = ('OLD', 'TWO', 'BID48')
RULES = {'OLD': ((0,), 24), 'TWO': ((0, 1), 24), 'BID48': ((0,), 48)}
EVAL_DAYS = ('20261026', '20261027')
CASE = 'SOFT_JOINT'
QCOLS = ('predicted', 'predicted_wealth', 'predicted_quantity', 'scenario_advantage_sd')
MUT_ASK_FROM_BID = MUT_BANK_SWAP = 0                      # planted errors (selftest only)


def candidate_times(frame, day, sides, per):
    """pf_flow_control_paths.build's rule per side: rung-0 at-best probes, one row per (m, time), ranked by the identity hash, the first `per`
    per market; a time drawn by both sides is kept once (the bid draw)."""
    out = []
    for s in sides:
        src = 0 if (MUT_ASK_FROM_BID and s == 1) else s
        f = frame.loc[(frame.side == src) & (frame.rung == 0) & (frame.x_depth == 0)].drop_duplicates(['m', 'decision_ms']).copy()
        f['hash'] = [PATH.keyhash(day, m, t) for m, t in f[['m', 'decision_ms']].itertuples(index=False, name=None)]
        f = f.sort_values('hash').groupby('m', sort=False).head(per)
        out.append(pd.DataFrame(dict(m=f.m.astype(int).values, decision_ms=f.decision_ms.astype(float).values, event_side=s)))
    return pd.concat(out, ignore_index=True).drop_duplicates(['m', 'decision_ms'], keep='first').reset_index(drop=True)


def ask_draw_ok(frame, z):
    """The TWO rule's ask draw is non-empty and every time is an ask rung-0 at-best probe."""
    ask = z[z.event_side == 1]
    side1 = set(zip(frame.loc[(frame.side == 1) & (frame.rung == 0) & (frame.x_depth == 0), 'm'].astype(int),
                    frame.loc[(frame.side == 1) & (frame.rung == 0) & (frame.x_depth == 0), 'decision_ms'].astype(float)))
    return len(ask) > 0 and all((int(m), float(t)) in side1 for m, t in zip(ask.m, ask.decision_ms))


def bank_paths(name, root=None):
    """A bank by name, its file hash checked against its receipt and against OLD's."""
    if name == 'OLD':
        return PATH.load()
    root = root or (OUT / 'banks')
    p = root / ('OLD' if MUT_BANK_SWAP else name) / 'paths.pkl'; r = json.loads((root / name / 'receipt.json').read_text())
    C.require(C.sha(p) == r['path_sha256'], f'bank {name} identity')
    C.require(r['path_sha256'] != r.get('old_sha256'), f'bank {name} equals OLD')
    return pickle.loads(p.read_bytes())


def selftest():
    bad = []
    fr = pd.DataFrame(dict(m=[1, 1, 1, 1, 2, 2, 1, 1], side=[0, 0, 1, 1, 0, 1, 0, 1], rung=[0, 0, 0, 0, 0, 0, 1, 0],
                           x_depth=[0, 0, 0, 0, 0, 0, 0, 1], decision_ms=[10., 20., 20., 30., 10., 40., 50., 60.]))
    z = candidate_times(fr, 'toy', (0, 1), 24)
    want = {(1, 10., 0), (1, 20., 0), (1, 30., 1), (2, 10., 0), (2, 40., 1)}
    if set(zip(z.m, z.decision_ms, z.event_side)) != want:
        bad.append(f'per-side rule {sorted(zip(z.m, z.decision_ms, z.event_side))}')
    one = candidate_times(fr, 'toy', (0,), 1)
    h = {(m, t): PATH.keyhash('toy', m, t) for m, t in ((1, 10.), (1, 20.), (2, 10.))}
    exp1 = {1: min((h[(1, 10.)], 10.), (h[(1, 20.)], 20.))[1], 2: 10.}
    if dict(zip(one.m, one.decision_ms)) != exp1:
        bad.append('hash cap per market')
    if not ask_draw_ok(fr, z):
        bad.append('ask draw refused on valid input')
    globals()['MUT_ASK_FROM_BID'] = 1
    try:
        caught_ask = not ask_draw_ok(fr, candidate_times(fr, 'toy', (0, 1), 24))
    finally:
        globals()['MUT_ASK_FROM_BID'] = 0
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for nm, payload in (('OLD', [dict(uid='a')]), ('TWO', [dict(uid='b')])):
            (root / nm).mkdir(); (root / nm / 'paths.pkl').write_bytes(pickle.dumps(payload))
        sha_old = C.sha(root / 'OLD' / 'paths.pkl')
        C.dump(root / 'TWO' / 'receipt.json', dict(path_sha256=C.sha(root / 'TWO' / 'paths.pkl'), old_sha256=sha_old))
        try:
            ok = bank_paths('TWO', root) == [dict(uid='b')]
        except ValueError:
            ok = False
        globals()['MUT_BANK_SWAP'] = 1
        try:
            bank_paths('TWO', root); caught_swap = False
        except ValueError:
            caught_swap = True
        finally:
            globals()['MUT_BANK_SWAP'] = 0
    if not ok:
        bad.append('bank identity refused a valid bank')
    caught = dict(MUT_ASK_FROM_BID=caught_ask, MUT_BANK_SWAP=caught_swap)
    if bad or not all(caught.values()):
        sys.exit(f'TWO-SIDED BANK SELFTEST FAILED -> no output: {bad}, planted errors caught {caught}')
    print(f'two-sided bank selftests passed; planted errors caught: {caught}', flush=True)
    return caught


def build():
    selftest(); V.verify(); out = OUT / 'banks'; C.require(not out.exists(), 'banks exist'); out.mkdir(parents=True)
    start = time.time(); cells = {b: defaultdict(list) for b in BANKS}; counts = {b: Counter() for b in BANKS}
    statuses = {b: Counter() for b in BANKS}; drawn = {b: Counter() for b in BANKS}
    for day in F.TRAIN:
        frame = FM.read(day)
        rules = {b: candidate_times(frame, day, *RULES[b]) for b in BANKS}
        if not ask_draw_ok(frame, rules['TWO']):
            sys.exit(f'TWO-SIDED BANK CONTROL FAILED on {day}: the ask draw is not from ask probes -> no output')
        need = defaultdict(dict)
        for b, z in rules.items():
            drawn[b].update(Counter(f'side{int(e)}' for e in z.event_side))
            for m, t, e in zip(z.m, z.decision_ms, z.event_side):
                need[int(m)].setdefault(float(t), {})[b] = int(e)
        offset = json.loads((Path(C.V.E.OUTD) / f'day_{day}_imp.json').read_text())['untimed']['imp_off_ms']
        markets = sorted(need)
        for i in range(0, len(markets), 24):
            batch = markets[i:i + 24]; data = I.load_day(day, set(batch), offset); seen = set()
            for W in I.pairs(data):
                m = int(W[0]['m']); seen.add(m)
                for t, owners in need[m].items():
                    p, status = PATH.extract(W, float(t), day)
                    for b, e in owners.items():
                        statuses[b][status] += 1
                        if p is None:
                            continue
                        q = dict(p); q['event_side'] = e
                        cell = (p['phase'], p['spread'], *p['classes']); counts[b][cell] += 1
                        lst = cells[b][cell]; lst.append(q); lst.sort(key=lambda z_: z_['uid']); del lst[F.CELL_TEMPLATES:]
            for m in set(batch) - seen:
                for t, owners in need[m].items():
                    for b in owners:
                        statuses[b]['missing_market_probes'] += 1
            del data; C.guard('two-sided bank batch')
            print(json.dumps(dict(stage='bank', day=day, markets=min(i + 24, len(markets)), seconds=round(time.time() - start))), flush=True)
    theirs = PATH.load(); old_sha = C.sha(F.OUT / 'bank' / 'paths.pkl'); assembled = {}
    for b in BANKS:
        paths = []
        for cell, items in sorted(cells[b].items()):
            for p in items:
                p['base_weight'] = counts[b][cell] / len(items); paths.append(p)
        assembled[b] = paths
    mine = assembled['OLD']
    same = len(mine) == len(theirs) and all(
        a['uid'] == b_['uid'] and a['base_weight'] == b_['base_weight'] and np.array_equal(a['frames'], b_['frames'])
        and np.array_equal(a['trades'], b_['trades']) and a['marks'] == b_['marks'] and a['classes'] == b_['classes'] for a, b_ in zip(mine, theirs))
    if not same:
        sys.exit(f'TWO-SIDED BANK CONTROL FAILED: the OLD rule rebuilt here ({len(mine)} paths) differs from the frozen bank ({len(theirs)}) -> no output')
    for b in ('TWO', 'BID48'):
        d = out / b; d.mkdir(); pp = d / 'paths.pkl'; pp.write_bytes(pickle.dumps(assembled[b], protocol=5))
        comp = Counter(f'side{p["event_side"]}' for p in assembled[b])
        C.dump(d / 'receipt.json', dict(asof=C.stamp(), bank=b, rule=dict(sides=list(RULES[b][0]), per_market=RULES[b][1]), training_days=list(F.TRAIN),
                                        drawn_times=dict(drawn[b]), statuses=dict(statuses[b]), paths=len(assembled[b]), cells=len(cells[b]),
                                        templates_by_event_side=dict(comp), path_sha256=C.sha(pp), old_sha256=old_sha,
                                        old_rule_rebuilt_equal=True, seconds=time.time() - start))
    C.dump(out / 'receipt.json', dict(asof=C.stamp(), old=dict(paths=len(mine), cells=len(cells['OLD']), drawn_times=dict(drawn['OLD']),
                                                               statuses=dict(statuses['OLD']), sha256=old_sha, rebuilt_equal=True),
                                      seconds=time.time() - start))
    print(json.dumps(dict(stage='bank done', paths={b: len(v) for b, v in assembled.items()}, cells={b: len(cells[b]) for b in BANKS},
                          ask_templates={b: sum(p['event_side'] == 1 for p in assembled[b]) for b in BANKS}, seconds=round(time.time() - start))), flush=True)


def actions(day):
    C.require(day in EVAL_DAYS, 'undeclared action day'); V.verify(); start = time.time()
    out = OUT / f'actions_{day}'; C.require(not out.exists(), 'action output exists'); out.mkdir(parents=True)
    banks = {b: bank_paths(b) for b in BANKS}
    adir = V.OUT / f'actions_{day}'; ar = json.loads((adir / 'receipt.json').read_text())
    for name, h in ar['outputs'].items():
        C.require(C.sha(adir / name) == h, 'archived action output changed')
    arch = pd.read_parquet(adir / 'values.parquet'); arch = arch[arch.case == CASE].set_index('uid')
    selected, data, fair, depth, raw = A.load(day); rows_out = []
    for W in I.pairs(data):
        m = int(W[0]['m']); ref, rows = A.reference_candidates(W, data, day); bytime = {}
        for row in rows:
            if row['status'] in ('eligible', 'eligible_estimated_timing'):
                bytime.setdefault(row['decision_ms'], []).append(row)
        X, _ = I.market_inputs(W, fair[m], depth); prob = P.forecasts(W, X)
        ctl = {b: V.Controller(W, paths, prob, CASE) for b, paths in banks.items()}; saved = {}

        def observe(k, t, ST, kpos):                      # pf_value_bridge_actions.run's observer, unchanged
            if t not in bytime:
                return
            state = P.snapshot(W, k, t, ST, kpos)
            for row in bytime[t]:
                matches = [i for i, info in enumerate(state['info']) if info['kind'] == row['kind'] and info['side'] == row['side'] and info['rung'] == row['rung']
                           and (row['kind'] == 'post' or abs(info['activation'] - (t - row['x_age_ms'])) < .001)]
                if len(matches) != 1:
                    row['status'] = 'not_available_in_client_state'; continue
                i = matches[0]
                if row['kind'] == 'post':
                    a = state['baseline'].copy(); a[i] = 1.
                    if not P.feasible(state, a):
                        row['status'] = 'inventory_reservation_limit'; continue
                C.require(abs(kpos) <= F.CAP + 1e-7, 'neutral state beyond cap')
                saved[row['kind'], row['side'], row['rung'], row['generation']] = (k, state, i)
        replay = RUN.run_lane(W, cap=F.CAP, observer=observe)
        for field in ('fl', 'stat', 'orders', 'cancels'):
            np.testing.assert_equal(replay[field], ref[field])
        frame = pd.DataFrame(rows); eligible = frame[frame.status.isin(['eligible', 'eligible_estimated_timing'])]
        chosen = eligible.sort_values('hash').groupby(['kind', 'side', 'phase'], sort=False).head(2)
        for row in chosen.to_dict('records'):
            k, state, index = saved[row['kind'], row['side'], row['rung'], row['generation']]
            uid = f"{day}:{m}:{row['kind']}:{row['side']}:{row['rung']}:{row['generation']}"
            C.require(uid in arch.index, 'action id missing from the archive ' + uid); a = arch.loc[uid]
            for b, c in ctl.items():
                ctx, status = c.context(k, row['decision_ms']); est = dict(evaluation_status=status)
                if ctx is not None:
                    est.update(A.prediction(state, index, row['kind'], ctx))
                rows_out.append(dict(uid=uid, day=day, m=m, side=int(row['side']), rung=int(row['rung']), kind=row['kind'], phase=int(row['phase']), bank=b,
                                     actual=float(a['actual']), actual_quantity=float(a['actual_quantity']), **est))
        C.guard('two-sided actions'); gc.collect()
        print(json.dumps(dict(stage='actions', day=day, m=m, selected=len(chosen), seconds=round(time.time() - start))), flush=True)
    df = pd.DataFrame(rows_out); old = df[df.bank == 'OLD'].set_index('uid')
    C.require(set(old.index) == set(arch.index) and len(old) == len(arch), 'action ids differ from the archive')
    C.require((old.evaluation_status == arch.loc[old.index, 'evaluation_status']).all(), 'evaluation statuses differ from the archive')
    ok = arch.index[arch.evaluation_status == 'eligible']
    for c in QCOLS:
        diff = float((old.loc[ok, c] - arch.loc[ok, c]).abs().max())
        C.require(diff <= 1e-9, f'OLD does not reproduce the archived {c} ({diff})')
    df.to_parquet(out / 'values.parquet', index=False)
    C.dump(out / 'receipt.json', dict(asof=C.stamp(), day=day, actions=int(df.uid.nunique()), archive_receipt_sha256=C.sha(adir / 'receipt.json'),
                                      old_reproduced=True, banks={b: (C.sha(OUT / 'banks' / b / 'paths.pkl') if b != 'OLD' else C.sha(F.OUT / 'bank' / 'paths.pkl')) for b in BANKS},
                                      outputs={'values.parquet': C.sha(out / 'values.parquet')}, seconds=time.time() - start))


def pilot():
    V.verify(); start = time.time(); out = OUT / 'pilot'; C.require(not out.exists(), 'pilot output exists'); out.mkdir(parents=True)
    selected, data, fair, depth, raw = A.load(F.TEST)
    adir = V.OUT / 'pilot'; ar = json.loads((adir / 'receipt.json').read_text())
    for name, h in ar['outputs'].items():
        C.require(C.sha(adir / name) == h, 'archived pilot changed')
    am = pd.read_parquet(adir / 'markets.parquet'); am = am[am.case == CASE].set_index('m')
    af = pd.read_parquet(adir / 'fills.parquet'); af = af[af.case == CASE]
    C.require(sorted(am.index.astype(int)) == sorted(selected), 'pilot population differs from the archive')
    probs = {}
    for W in I.pairs(data):
        m = int(W[0]['m']); X, _ = I.market_inputs(W, fair[m], depth); probs[m] = P.forecasts(W, X)
    frames = {k_: [] for k_ in ('markets', 'fills', 'inventory', 'decisions')}
    for b in ('OLD', 'TWO'):
        paths = bank_paths(b)
        for W in I.pairs(data):
            m = int(W[0]['m']); pol = V.Controller(W, paths, probs[m], CASE); r = RUN.run_lane(W, pol)
            f = MET.engine_fills(W, r, data, F.TEST); f['case'] = CASE; f = MET.value_fills(f, W); AUDIT.raw_audit(f, W); AUDIT.audit_controls(f, W)
            inv = MET.minute_inventory(W, r); summary = MET.summarize_fills(f)
            row = dict(m=m, bank=b, **summary, orders=sum(len(z) for z in r['order_states'].values()), pulls=sum(r['stat']['pulls']),
                       end_position=r['endpos'], both_quote_ms=sum(x['quote_both_ms'] for x in inv), any_quote_ms=sum(x['quote_any_ms'] for x in inv))
            if b == 'OLD':
                a = am.loc[m]
                for c in ('shares', 'terminal_pnl_usd', 'settlement_pnl_usd', 'orders', 'pulls', 'fills'):
                    C.require(abs(float(row[c]) - float(a[c])) <= 1e-6 * max(1., abs(float(a[c]))), f'OLD replay differs from the archive: market {m} {c}')
                g = af[af.m == m]
                C.require(len(g) == len(f) and np.allclose(np.sort(g.terminal_pnl_usd.values), np.sort(f.terminal_pnl_usd.values), atol=1e-9),
                          f'OLD replay fills differ from the archive: market {m}')
            f['bank'] = b; frames['markets'].append(pd.DataFrame([row])); frames['fills'].append(f)
            frames['inventory'].append(pd.DataFrame([dict(m=m, bank=b, **x) for x in inv]))
            frames['decisions'].append(pd.DataFrame([dict(bank=b, **x) for x in pol.trace]))
            print(json.dumps(dict(stage='pilot', bank=b, m=m, seconds=round(time.time() - start), rss_mib=round(I.K.rss_mib()))), flush=True)
            del pol, r; gc.collect(); C.guard('two-sided pilot')
    for name, fr in frames.items():
        pd.concat(fr, ignore_index=True).to_parquet(out / f'{name}.parquet', index=False)
    C.dump(out / 'receipt.json', dict(asof=C.stamp(), selected_markets=selected, old_reproduced=True, archive_receipt_sha256=C.sha(adir / 'receipt.json'),
                                      banks={'OLD': C.sha(F.OUT / 'bank' / 'paths.pkl'), 'TWO': C.sha(OUT / 'banks' / 'TWO' / 'paths.pkl')},
                                      outputs={p.name: C.sha(p) for p in out.glob('*.parquet')}, seconds=time.time() - start))


def _cs(y, col):
    y = y.dropna(subset=[col]); q = float(y.shares.sum())
    return 100.0 * float(y[col].sum()) / q if q > 0 else float('nan')


def report():
    selftest()
    V_ = pd.concat([pd.read_parquet(OUT / f'actions_{d}' / 'values.parquet') for d in EVAL_DAYS], ignore_index=True)
    ok = V_.groupby('uid').evaluation_status.apply(lambda s: (s == 'eligible').all()); common = set(ok[ok].index)
    V_ = V_[V_.uid.isin(common)].copy()
    V_['z'] = V_.predicted / (V_.scenario_advantage_sd / np.sqrt(V_.effective_paths.clip(lower=1))).replace(0, np.nan)
    asof = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    L = [f'TWO-SIDED SCENARIO BANK (controlled: only the bank changes; SOFT_JOINT, classifier, controller, clocks fixed); as of {asof}',
         'controls: OLD rule rebuilt = the frozen bank; OLD reproduces the archived action predictions and the archived 12-market pilot; ask draw from ask probes; '
         'bank identities by hash; planted errors refused', '']
    br = json.loads((OUT / 'banks' / 'receipt.json').read_text())
    L.append(f"BANKS: OLD {br['old']['paths']} paths / {br['old']['cells']} cells (bid times {br['old']['drawn_times']})")
    for b in ('TWO', 'BID48'):
        r = json.loads((OUT / 'banks' / b / 'receipt.json').read_text())
        L.append(f"  {b}: {r['paths']} paths / {r['cells']} cells; drawn times {r['drawn_times']}; templates by event side {r['templates_by_event_side']}")
    L += ['', f'PART A: the audited neutral actions eligible under every bank ({len(common)} of {V_.uid.nunique() if len(V_) else 0}): RMSE (predicted - realized) | '
          'zero-predictor RMSE | Spearman | positive predictions n, mean predicted / realized | |z| < 1 share | z >= 2 positives n, mean realized']
    cells_ = {}
    for (day, side, kind), g0 in V_.groupby(['day', 'side', 'kind']):
        parts = []
        for b in BANKS:
            g = g0[g0.bank == b]; e = g.predicted - g.actual; rm = float(np.sqrt(np.mean(e ** 2))); cells_[(b, day, side, kind)] = rm
            pos = g[g.predicted > 0]; zz = g[(g.z >= 2)]
            parts.append(f'{b} {rm:6.2f} | sp {g.predicted.corr(g.actual, method="spearman"):+.3f} | pos {len(pos):3d} {pos.predicted.mean() if len(pos) else float("nan"):+6.2f}/'
                         f'{pos.actual.mean() if len(pos) else float("nan"):+6.2f} | |z|<1 {np.mean(g.z.abs() < 1):4.0%} | z>=2 {len(zz):3d} {zz.actual.mean() if len(zz) else float("nan"):+6.2f}')
        zero = float(np.sqrt(np.mean(g0[g0.bank == 'OLD'].actual ** 2)))
        L.append(f'  {day} {"bid" if side == 0 else "ask"} {kind:<6} n {len(g0[g0.bank == "OLD"]):3d} zero {zero:6.2f} || ' + ' || '.join(parts))

    def a1(b):
        ask = [cells_[(b, d, 1, k)] < cells_[('OLD', d, 1, k)] for d in EVAL_DAYS for k in ('post', 'cancel')]
        bid = [cells_[(b, d, 0, k)] <= 1.05 * cells_[('OLD', d, 0, k)] for d in EVAL_DAYS for k in ('post', 'cancel')]
        return sum(ask) >= 3 and sum(bid) >= 3, sum(ask), sum(bid)
    two, bid48 = a1('TWO'), a1('BID48')
    r_a1 = ('ASK-CALIBRATION-IMPROVES' + (' (MORE-DATA: BID48 passes too)' if bid48[0] else '')) if two[0] else 'NOT-SHOWN'
    tp = V_[(V_.bank == 'TWO') & (V_.kind == 'post') & (V_.z >= 2)]
    r_a2 = ('CONFIDENT-POSITIVE-POSTS' if tp.actual.mean() > 0 else 'MISCALIBRATED-POSITIVE-POSTS') if len(tp) >= 5 else 'NONE'
    L += ['', f'A1 ASK-CALIBRATION: TWO ask cells better {two[1]} / 4, bid cells within 5% {two[2]} / 4; BID48 {bid48[1]} / 4, {bid48[2]} / 4 -> {r_a1}',
          f'A2 CONFIDENT-POSTS (TWO): posts with z >= 2: n {len(tp)}, mean realized {tp.actual.mean() if len(tp) else float("nan"):+.2f} c -> {r_a2}']
    pm = pd.read_parquet(OUT / 'pilot' / 'markets.parquet'); pf = pd.read_parquet(OUT / 'pilot' / 'fills.parquet')
    pdx = pd.read_parquet(OUT / 'pilot' / 'decisions.parquet')
    L += ['', 'PART B: 12 Oct27 markets, full contracts - per side, fills in minutes 1-4 and in minute 5: shares | +1 s | +5 s c/sh']
    mk = {}
    for b in ('OLD', 'TWO'):
        f = pf[pf.bank == b]; parts = []
        for sd in (0, 1):
            for nm, y in (('1-4', f[(f.side == sd) & (f.minute < 5)]), ('5', f[(f.side == sd) & (f.minute == 5)])):
                mk[(b, sd, nm)] = (_cs(y, 'markout_1000_usd'), _cs(y, 'markout_5000_usd'))
                parts.append(f'{"bid" if sd == 0 else "ask"} m{nm} {y.shares.sum():6,.0f} {mk[(b, sd, nm)][0]:+6.2f} {mk[(b, sd, nm)][1]:+6.2f}')
        m_ = pm[pm.bank == b]; c_ = pdx[(pdx.bank == b) & (pdx.kind == 'cancel')]
        L.append(f'  {b:<4} ' + ' | '.join(parts))
        L.append(f'       orders {int(m_.orders.sum()):,} pulls {int(m_.pulls.sum()):,} first-review cancels {int(((c_.time - c_.activation) <= 50. + 1e-7).sum()):,} of {len(c_):,} | '
                 f'terminal ${m_.terminal_pnl_usd.sum():+.1f} settlement ${m_.settlement_pnl_usd.sum():+.1f} | both quote {m_.both_quote_ms.sum() / (len(m_) * 300000.):.1%}')
    r1 = (mk[('TWO', 1, '1-4')][0] > mk[('OLD', 1, '1-4')][0] and mk[('TWO', 1, '1-4')][1] > mk[('OLD', 1, '1-4')][1]
          and mk[('TWO', 0, '1-4')][0] >= mk[('OLD', 0, '1-4')][0] - 0.5)
    L += [f'R1 ASK-MARKOUT (minutes 1-4): ask +1 s {mk[("OLD", 1, "1-4")][0]:+.2f} -> {mk[("TWO", 1, "1-4")][0]:+.2f}, +5 s {mk[("OLD", 1, "1-4")][1]:+.2f} -> '
          f'{mk[("TWO", 1, "1-4")][1]:+.2f}; bid +1 s {mk[("OLD", 0, "1-4")][0]:+.2f} -> {mk[("TWO", 0, "1-4")][0]:+.2f} -> {"ASK-MARKOUT-IMPROVES" if r1 else "NOT-SHOWN"}',
          '', 'LIMITS: consumed days; 12 markets of one day for the replay; simulated (rule M, queue M, unknown depth 0); exploratory, not profitability validation']
    (OUT / 'two_sided_bank.txt').write_text('\n'.join(L) + '\n')
    C.dump(OUT / 'two_sided_bank.json', dict(asof=asof, readings=dict(A1=r_a1, A2=r_a2, R1='ASK-MARKOUT-IMPROVES' if r1 else 'NOT-SHOWN'),
                                              rmse={f'{b}:{d}:{s}:{k}': v for (b, d, s, k), v in cells_.items()},
                                              markouts={f'{b}:{s}:{n}': v for (b, s, n), v in mk.items()}))
    print('\n'.join(L), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['selftest', 'build', 'actions', 'pilot', 'report']); ap.add_argument('day', nargs='?')
    a_ = ap.parse_args()
    if a_.mode == 'selftest':
        selftest()
    elif a_.mode == 'build':
        build()
    elif a_.mode == 'actions':
        actions(a_.day)
    elif a_.mode == 'pilot':
        pilot()
    else:
        report()
