"""BOOK-CLOCK BOUNDS AND CANCEL-ANSWER CENSUS ON EVERY KNOWN LEG (2026-11-18, session aea5bfb9; USER "Go" to the two next steps of the timing
case audit, reviews/timing_case_audit_20261118/report.md). Declared before any data is read.

WHY  The timing case audit removed K6's three S3 false avoidances with the venue book clock bounded from the bot's own frame receipts, and it
found that the venue's Canceled answer does not order a cancel against a fill. The known ledger's gate of record moves to one offset bound per
clock domain (pf_keep_pull_exec_windows.py known v6). This module supplies the book-clock bound for EVERY known leg and counts the cancel race
on every known order.
POPULATION  the 99 legs / 86 orders of known v5 (keep_pull_exec_windows_known_v5_20261118, outputs hash-checked against its receipt). Consumed
            Nov 8; every reader stops at Nov 8 23:02Z (pf_timing_case_audit's readers, cutoff and identity rule, unchanged).
PER LEG (pf_timing_case_audit.lifecycle / book_pairs): A0 the case table reproduced from the kernel records, and one bot clock (the order's
            Accepted carries the same instant in both bot journals, keyed by market and engine id); A1 the first cancel confirmation's class
            (MATCHED-OR-GONE / CANCELED / NO-EFFECT / NO-ANSWER), the post-cancel terminal snapshots and pf-exec's alerts naming the order; A4 the
            book-clock lower bound B (the ceil(n/2)-th largest u - receipt over identity pairs within +-W_LOCAL of the leg's cancel confirmation,
            or of its first fill notice when it has none), its pairs, statuses and the bot clock's step.
BOOK BOUND per leg: IDENTIFIED iff A0's one_bot_clock holds, the bot clock did not step (<= CLOCK_STEP_MS) and n >= N_MIN; then B is the lowest
            admissible book-clock offset (venue book - bot) near the leg. Otherwise UNIDENTIFIED, with every failed condition named; the known
            ledger fails closed on it.
CENSUS per unique order (its legs' records): the first cancel confirmation's class; the filled size (core) against the order size; post-cancel
            terminal snapshots (size_matched below the final fill; received after the first fill notice); pf-exec alerts naming the order.
READINGS (declared; evaluated in code):
  N0 REPRODUCES-THE-CASE-AUDIT: on the timing case audit's three legs, B, n and the clock step equal its v2 receipt exactly. YES / NO.
  N1 RECORDS-REPRODUCE: legs whose every A0 field holds, of 99 (an instrument check; each failing leg is named with its fields).
  N2 BOOK-BOUND: legs with a cancel confirmation whose bound is IDENTIFIED, of those legs; B's distribution (min / p50 / max); every
     UNIDENTIFIED leg with its causes.
  N3 CANCEL-ANSWER-ORDERS-FILLS: NO iff some order whose first cancel confirmation is CANCELED ended filled IN FULL - the venue acknowledged
     a cancel of an order that then filled completely; else NOT-CONTRADICTED. Counts: class x (full / partial fill).
  N4 TERMINAL-SNAPSHOT-UNDERSTATES-FILLS: orders with a post-cancel terminal snapshot whose size_matched is below the order's final filled
     size; of them, those whose snapshot was received after the first fill notice.
  N5 PF-EXEC-ALERTS: orders named in a late-fill or quantity-conflict alert.
Exclusions are statuses: a leg without kernel records, an order without any cancel confirmation (class NO-ANSWER), a leg without bot frames.
SELFTEST (synthetic, no data): the bound status rule (identified; too few pairs; a stepped or unmeasured clock; one bot clock failing); the
 order census on planted orders (CANCELED full / partial, MATCHED-OR-GONE, NO-ANSWER; snapshots before / after the notice, understating or
 not; alerts); the reading predicates on planted counts; the per-leg path on pf_timing_case_audit's planted leg. Mutations:
 pf_known_clock_census_mutants.py.
LIMITS: one consumed partial day; filled own orders only (outcome-selected: N3 / N4 count races among orders that filled, not a race rate over
 all cancels); the book-clock premise of the audit (a frame is stamped before the bot receives it; one stamp per update across subscribers);
 the bound is one-sided.
V2 (2026-11-18, after v1's output was read; v1's receipt known_clock_census_20261118 stays as provenance). v1's N1 read 76 of 99 on three
 defects of the lifecycle code it reuses from pf_timing_case_audit (fixed there, with falsifiers and mutants): a leg's fill notices were matched
 by trade id alone, so a settlement that filled two own orders lent the sibling's notices (12 legs); the case table's confirmed cancel was
 compared with the FIRST answer, a refusal on 10 legs whose cancel was confirmed on a retry; the one-bot-clock tolerance was 1 us, while the two
 journals' stamps are separate reads of the host clock (4.4 us apart on 1 leg; now 0.1 ms). Every declared reading and threshold is unchanged;
 added, descriptive only: the orders with any venue acknowledgement of the cancel (a retry's included) by fill.
V3 (after v2's output was read): v2 compared the case table's confirmed cancel with the first answer showing the order no longer live, a
 "can't be found" answer included; the fill-clock audit defines it as the order's first CancelPendingAudit only (orders.parquet
 t_CancelPendingAudit), so v2 failed A0 on 10 legs without one. The comparison now mirrors that definition; nothing else changes.
Usage (run_light, PF_SLOT=claude): pf_known_clock_census.py selftest | run
"""
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1]))
import pf_keep_pull as KP                                 # noqa: E402
import pf_keep_pull_exec_windows as X                     # noqa: E402  the Nov 8 export paths
import pf_timing_case_audit as TCA                        # noqa: E402  the readers, lifecycle, book_pairs, the identity rule

C, pd, I = KP.C, KP.pd, KP.I
OUT = Path(I.K.IO) / 'known_clock_census_v3_20261118'   # v3: the lifecycle fixes in pf_timing_case_audit (v1 / v2 stay as provenance)
KNOWN = Path(I.K.IO) / 'keep_pull_exec_windows_known_v5_20261118'   # the case table (K6 is not read here)
AUDIT = Path(I.K.IO) / 'timing_case_audit_v2_20261118' / 'audit.json'
ALERT_KEYS = ('late_fill', 'cancel_quantity_conflict')
LEG_COLS = ['run', 'cid', 'mid', 'trade_id', 'order_hash', 'ack', 'center', 'B', 'n', 'B_max', 'B_min', 'statuses', 'clock_step_ms', 'clock_samples',
            'one_bot_clock', 'A0', 'A0_failed', 'A1', 'A1_at', 'A1_source', 'acknowledged', 'snapshots', 'alerts', 'first_notice', 'size', 'filled', 'bound_status',
            'bound_missing']


def bound_status(one_bot_clock, step, b):
    """IDENTIFIED / UNIDENTIFIED and the failed conditions of one leg's book-clock bound."""
    miss = []
    if not one_bot_clock:
        miss.append('one bot clock (the Accepted in both journals)')
    if step is None or not np.isfinite(step) or step > TCA.CLOCK_STEP_MS:
        miss.append('a bot wall clock without a step')
    if b['n'] < TCA.N_MIN:
        miss.append(f'>= {TCA.N_MIN} identity pairs')
    return ('IDENTIFIED' if not miss else 'UNIDENTIFIED'), miss


def leg_row(r, rec, br, Fm, bf, ck, size, filled):
    """One leg's census row (r: its known case row; its kernel records, input-journal records, market frames, bot frames, clock samples)."""
    L_ = TCA.lifecycle(r, rec, br); center = float(r.ack) if np.isfinite(float(r.ack)) else float(r.upper)
    _, st, b, step = TCA.book_pairs(center, Fm, bf, ck); bs, miss = bound_status(L_['a0f']['one_bot_clock'], step, b)
    return dict(run=r.run, cid=int(r.cid), mid=int(r.mid), trade_id=r.trade_id, order_hash=r.order_hash, ack=float(r.ack), center=center,
                B=b['B'], n=b['n'], B_max=b['B_max'], B_min=b['B_min'], statuses=json.dumps(st, sort_keys=True), clock_step_ms=step, clock_samples=int(len(ck)),
                one_bot_clock=bool(L_['a0f']['one_bot_clock']), A0=bool(L_['a0']), A0_failed=json.dumps(sorted(k for k, v in L_['a0f'].items() if not v)),
                A1=L_['a1'], A1_at=float(L_['a1_at']), A1_source=L_['a1_src'], acknowledged=bool(L_['pa']),
                snapshots=json.dumps([dict(at=t, size_matched=int(x['size_matched'])) for t, x in L_['cq']]),
                alerts=json.dumps(sorted({a for t, k, line in L_['rec'] if k == 'outbox' and r.order_hash.lower() in line for a in json.loads(line)['frame']['alerts']})),
                first_notice=float(min(L_['pos'])) if L_['pos'] else np.nan, size=float(size), filled=float(filled), bound_status=bs, bound_missing=json.dumps(miss))


def order_census(L):
    """Per unique order (run, cid) from the per-leg rows L -> DataFrame: its first cancel confirmation's class, full / partial fill, the
    post-cancel terminal snapshots that understate the final fill (and those received after the first fill notice), pf-exec's alerts."""
    rows = []
    for (run, cid), g in L.groupby(['run', 'cid'], sort=True):
        x = g.iloc[0]; snaps = json.loads(x['snapshots']); alerts = json.loads(x['alerts'])
        fn = g.first_notice.values.astype(float); fn = float(np.nanmin(fn)) if np.isfinite(fn).any() else np.nan
        under = [s for s in snaps if s['size_matched'] < round(float(x['filled']) * 1e6)]
        rows.append(dict(run=run, cid=int(cid), legs=int(len(g)), cls=x['A1'], acknowledged=bool(g.acknowledged.any()), size=float(x['size']), filled=float(x['filled']),
                         full=bool(float(x['filled']) >= float(x['size']) - 1e-9), snapshots=len(snaps), understating=len(under),
                         understating_after_notice=sum(1 for s in under if np.isfinite(fn) and s['at'] >= fn),
                         alerts=json.dumps(sorted({a.split(':')[0] for a in alerts})), alerted=any(any(k in a for k in ALERT_KEYS) for a in alerts)))
    return pd.DataFrame(rows)


def readings(L, O, audit_legs):
    """N0-N5 on the per-leg rows L, the order census O and the audit's v2 legs (B / n / step per (run, cid, trade_id))."""
    read, R = {}, {}
    keys = [(x['run'], x['cid'], x['trade_id']) for x in audit_legs]; got = L.set_index(['run', 'cid', 'trade_id'])
    same = all(k in got.index and got.loc[k, 'B'] == x['A4']['B'] and int(got.loc[k, 'n']) == x['A4']['n'] and got.loc[k, 'clock_step_ms'] == x['A4']['clock_step_ms']
               for k, x in zip(keys, audit_legs)) and len(keys) > 0
    read['N0 REPRODUCES-THE-CASE-AUDIT'] = 'YES' if same else 'NO'
    bad0 = L[~L.A0.values.astype(bool)]
    read['N1 RECORDS-REPRODUCE'] = f"{int(L.A0.sum())} of {len(L)} legs"
    R['N1_failed'] = [dict(run=r.run, cid=int(r.cid), trade_id=r.trade_id, fields=json.loads(r.A0_failed)) for r in bad0.itertuples()]
    can = np.isfinite(L.ack.values.astype(float)); idf = can & (L.bound_status.values == 'IDENTIFIED'); Bv = L.B.values[idf].astype(float)
    read['N2 BOOK-BOUND'] = f"IDENTIFIED on {int(idf.sum())} of {int(can.sum())} legs with a cancel confirmation"
    R['N2_B_ms'] = dict(min=float(Bv.min()), p50=float(np.median(Bv)), max=float(Bv.max())) if len(Bv) else {}
    R['N2_unidentified'] = [dict(run=r.run, cid=int(r.cid), trade_id=r.trade_id, n=int(r.n), missing=json.loads(r.bound_missing)) for r in L[can & ~idf].itertuples()]
    xt = {f'{c}|{"full" if f else "partial"}': int(((O.cls == c) & (O.full == f)).sum()) for c in TCA.ANSWER for f in (True, False)}
    race = int(((O.cls == 'CANCELED') & O.full).sum())
    read['N3 CANCEL-ANSWER-ORDERS-FILLS'] = (f'NO: {race} orders acknowledged Canceled then filled in full' if race else 'NOT-CONTRADICTED'); R['N3_class_x_fill'] = xt
    u = int((O.understating > 0).sum()); ua = int((O.understating_after_notice > 0).sum())
    read['N4 TERMINAL-SNAPSHOT-UNDERSTATES-FILLS'] = f'{u} orders ({ua} with the snapshot received after the first fill notice)'
    read['N5 PF-EXEC-ALERTS'] = f"{int(O.alerted.sum())} orders named in a late-fill or quantity-conflict alert"
    # v2, descriptive (added after v1's output was read; no reading uses it): any venue acknowledgement of the cancel, a retry's included
    R['N3b_any_acknowledgement_x_fill'] = {f'{"acknowledged" if a else "never acknowledged"}|{"full" if f else "partial"}': int(((O.acknowledged == a) & (O.full == f)).sum())
                                           for a in (True, False) for f in (True, False)}
    R['N5_alerts'] = dict(collections.Counter(a for s in O.alerts for a in json.loads(s)))
    return read, R


def run(out=OUT):
    """The census on consumed Nov 8 (every known leg)."""
    t0 = time.time(); out = Path(out); C.require(not (out / 'census.json').exists(), f'{out} exists: a new version needs a new directory'); out.mkdir(parents=True, exist_ok=True)
    kj = json.loads((KNOWN / 'known.json').read_text()); inputs = {str(KNOWN / 'known.json'): C.sha(KNOWN / 'known.json')}
    C.require(C.sha(KNOWN / 'known_cases.parquet') == kj['outputs']['known_cases.parquet'], 'known_cases differs from its known v5 receipt')
    K = pd.read_parquet(KNOWN / 'known_cases.parquet').reset_index(drop=True); aj = json.loads(AUDIT.read_text()); inputs[str(AUDIT)] = C.sha(AUDIT)
    core = {r: pd.read_parquet(X.LIVE / f'core_orders_{r}.parquet', columns=['cid', 'id', 'size', 'filled']) for r in ('v6', 'basis')}
    for r in core:
        inputs[str(X.LIVE / f'core_orders_{r}.parquet')] = C.sha(X.LIVE / f'core_orders_{r}.parquet')
    eng = {}
    for i, r in K.iterrows():
        c_ = core[r.run]; j = np.flatnonzero(c_.cid.values == r.cid); C.require(len(j) == 1, f'cid {r.cid} not once in core_orders_{r.run}'); j = int(j[0])
        eng[i] = (int(c_.id.values[j]), float(c_['size'].values[j]), float(c_.filled.values[j]))
    center = {i: (float(r.ack) if np.isfinite(float(r.ack)) else float(r.upper)) for i, r in K.iterrows()}
    K_rec, K_clk, BF, BR, st = {}, {}, {}, {}, {}
    for run_ in sorted(set(K.run)):
        ix = [i for i in K.index if K.run[i] == run_]
        legs = {i: dict(cid=int(K.cid[i]), oid_json='"order_id":[' + ','.join(str(b) for b in bytes.fromhex(K.order_hash[i][2:])) + ']', trade_id=K.trade_id[i],
                        hex=K.order_hash[i].lower()) for i in ix}
        p = TCA.KX / f'{run_}.kernel.jsonl'; inputs[str(p)] = C.sha(p)
        a, b, c = TCA.kernel_records(p, legs, {i: (center[i] - TCA.W_LOCAL, center[i] + TCA.W_LOCAL) for i in ix}); K_rec.update(a); K_clk.update(b); st[f'kernel_{run_}'] = c
        wins = collections.defaultdict(list)
        for i in ix:
            wins[int(K.mid[i])].append((center[i] - TCA.W_LOCAL - 2 * TCA.W_ID, center[i] + TCA.W_LOCAL + 2 * TCA.W_ID))
        p = TCA.INPUTS[run_]; inputs[str(p)] = C.sha(p)
        a, b, c = TCA.input_frames(p, dict(wins), sorted({(int(K.mid[i]), eng[i][0]) for i in ix})); BF[run_] = a; BR[run_] = b; st[f'inputs_{run_}'] = c
    wins, hrs = collections.defaultdict(list), set()
    for i in K.index:
        lo_, hi_ = center[i] - TCA.W_LOCAL - 3 * TCA.W_ID, center[i] + TCA.W_LOCAL + 3 * TCA.W_ID; wins[int(K.mid[i])].append((lo_, hi_))
        hrs |= set(TCA.hours_between(lo_ - 60_000., hi_ + 60_000.))
    RF, _, h1 = TCA.research_frames(sorted(hrs), dict(wins)); inputs.update(h1)
    empty = pd.DataFrame(columns=['u', 'recv_ns', 'key', 'pend'])
    rows = [leg_row(K.loc[i], K_rec.get(i, []), BR[K.run[i]].get((int(K.mid[i]), eng[i][0]), []), RF.get(int(K.mid[i]), empty), BF[K.run[i]].get(int(K.mid[i]), []),
                    np.array(K_clk.get(i, []), float), eng[i][1], eng[i][2]) for i in K.index]
    L = pd.DataFrame(rows, columns=LEG_COLS); O = order_census(L); read, R = readings(L, O, aj['legs'])
    L.to_parquet(out / 'legs.parquet', index=False); O.to_parquet(out / 'orders.parquet', index=False)
    files = {f: C.sha(out / f) for f in ('legs.parquet', 'orders.parquet')}
    J = dict(asof=C.stamp(), readings=read, R=R, readers=st, cutoff_ms=TCA.CUTOFF_MS, params=dict(W_LOCAL=TCA.W_LOCAL, W_ID=TCA.W_ID, N_MIN=TCA.N_MIN, DP=TCA.DP,
             CLOCK_STEP_MS=TCA.CLOCK_STEP_MS), outputs=files, inputs=inputs, source_sha256=C.sha(Path(__file__)), audit_source_sha256=C.sha(Path(TCA.__file__)),
             seconds=time.time() - t0)
    C.dump(out / 'census.json', J); txt = report(J, L, O); (out / 'census.txt').write_text(txt); print(txt)
    return J


def report(J, L, O):
    o = ['BOOK-CLOCK BOUNDS AND CANCEL-ANSWER CENSUS (pf_known_clock_census.py run; the 99 known legs of consumed Nov 8, read to 23:02Z)']
    for r in L.itertuples():
        o.append(f"  {r.run} cid {r.cid} trade {r.trade_id}: bound {r.bound_status} B {r.B} (n {r.n}, max {r.B_max}, min {r.B_min}, step {r.clock_step_ms}) "
                 f"{'' if r.bound_status == 'IDENTIFIED' else json.loads(r.bound_missing)}; A0 {r.A0} {'' if r.A0 else json.loads(r.A0_failed)}; A1 {r.A1}")
    o.append('  orders: ' + '; '.join(f"{r.run} {r.cid} {r.cls} {r.filled:g}/{r.size:g} snaps {r.snapshots} under {r.understating} (after notice {r.understating_after_notice}) "
                                    f"{json.loads(r.alerts)}" for r in O.itertuples()))
    o += ['READINGS (declared before the data was read)'] + [f'  {k}: {v}' for k, v in J['readings'].items()]
    o += [f"  N1 failing legs: {J['R']['N1_failed']}", f"  N2 B (ms): {J['R']['N2_B_ms']}; unidentified: {J['R']['N2_unidentified']}",
          f"  N3 class x fill: {J['R']['N3_class_x_fill']}; descriptive (v2): any acknowledgement x fill {J['R']['N3b_any_acknowledgement_x_fill']}",
          f"  N5 alerts: {J['R']['N5_alerts']}",
          'LIMITS: one consumed partial day; filled own orders only (outcome-selected); the book-clock premise (stamp before receipt; one stamp per update across '
          'subscribers); a one-sided bound; development evidence - Nov 8 is not untouched validation.']
    return '\n'.join(o) + '\n'


# ------------------------------------------------------------------ selftest (synthetic only)
def selftest():
    bad = []
    good = dict(n=40, B=-5., B_max=1., B_min=-80.)
    cases = [((True, 0.1, good), 'IDENTIFIED', None), ((True, 0.1, dict(good, n=19)), 'UNIDENTIFIED', 'pairs'), ((True, 5., good), 'UNIDENTIFIED', 'step'),
             ((True, None, good), 'UNIDENTIFIED', 'step'), ((False, 0.1, good), 'UNIDENTIFIED', 'clock')]
    for a, want, why in cases:
        s_, m_ = bound_status(*a)
        if s_ != want or (why and not any(why in x or (why == 'step' and 'step' in x) or (why == 'clock' and 'journals' in x) for x in m_)):
            bad.append(f'bound status {a[:2]} n {a[2]["n"]}: {s_} {m_}')
    # the per-leg path on the audit's planted leg (lifecycle + book_pairs through leg_row)
    try:
        v, rec, br, Fm, bf, MR, ck, K7 = TCA._planted_leg(); r = v['row']
        g = leg_row(r, rec, br, Fm, bf, ck, 5., 5.)
        g2 = leg_row(r, rec, br, Fm, [(t - 300., k) for t, k in bf], ck, 5., 5.)
        g3 = leg_row(r, rec, ['{"Accepted":{"market":7,"known_ms":%.4f,"id":146}}' % (r.accepted + 0.5)], Fm, bf, ck, 5., 5.)
        g4 = leg_row(r.copy().rename(None).pipe(lambda x: x.where(x.index != 'upper', r.ack + 30_000.)), rec + [
            (r.place, 'outbox', '{"frame":{"alerts":["late_fill_beyond_cancel_bound:0xdeadbeef: another order"],"cancels":[],"control":[],"places":[],"requests":[],'
                                '"seq":99,"timers":[]},"i":99,"type":"outbox","version":1}')], br, Fm, bf, ck, 5., 5.)
    except Exception as e:
        g = g2 = g3 = g4 = dict(bound_status=f'raised {e}', B=None, n=-1, A0=False, A1='', one_bot_clock=False, alerts='["raised"]')
    if not (g['bound_status'] == 'IDENTIFIED' and g['B'] == -3. and g['n'] == 31 and g['A0'] and g['A1'] == 'CANCELED' and json.loads(g['alerts']) == []
            and json.loads(g['snapshots']) == [dict(at=json.loads(g['snapshots'])[0]['at'], size_matched=5000000)]):
        bad.append(f'leg_row on the planted leg: {g}')
    if not (g2['bound_status'] == 'IDENTIFIED' and abs(g2['B'] - 297.) < 1e-6):
        bad.append(f'leg_row: B must follow the pairs (x = u - receipt): {g2.get("B")}')
    if not (g3['bound_status'] == 'UNIDENTIFIED' and not g3['one_bot_clock']):
        bad.append(f'leg_row: a second bot clock must leave the bound unidentified: {g3.get("bound_status")}')
    if not (g4['bound_status'] == 'IDENTIFIED' and g4['n'] == 31 and json.loads(g4['alerts']) == []):
        bad.append(f'leg_row: the window centres on the cancel confirmation, and only alerts naming the order count: {g4.get("n")} {g4.get("alerts")}')
    # the order census on planted orders
    snap = lambda *x: json.dumps([dict(at=a, size_matched=m) for a, m in x])
    L = pd.DataFrame([dict(run='v6', cid=1, A1='CANCELED', acknowledged=True, size=5., filled=5., snapshots=snap((45., 0)), alerts=json.dumps(['late_fill_beyond_cancel_bound:0x1: x']), first_notice=50.),
                      dict(run='v6', cid=1, A1='CANCELED', acknowledged=True, size=5., filled=5., snapshots=snap((45., 0)), alerts=json.dumps(['late_fill_beyond_cancel_bound:0x1: x']), first_notice=40.),
                      dict(run='v6', cid=2, A1='CANCELED', acknowledged=True, size=5., filled=2., snapshots=snap((100., 2000000)), alerts='[]', first_notice=150.),
                      dict(run='v6', cid=3, A1='MATCHED-OR-GONE', acknowledged=False, size=5., filled=5., snapshots='[]', alerts=json.dumps(['order_reopened:0x3: z']), first_notice=10.),
                      dict(run='basis', cid=4, A1='NO-ANSWER', acknowledged=True, size=5., filled=5., snapshots=snap((200., 1000000)), alerts=json.dumps(['cancel_quantity_conflict:0x4: y']), first_notice=300.)])
    O = order_census(L).set_index(['run', 'cid'])
    want = {('v6', 1): (True, 1, 1, True), ('v6', 2): (False, 0, 0, False), ('v6', 3): (True, 0, 0, False), ('basis', 4): (True, 1, 0, True)}
    if [bool(O.loc[k].acknowledged) for k in want] != [True, True, False, True]:
        bad.append('order census: the acknowledgement flag')
    for k, (full, under, after, alerted) in want.items():
        x = O.loc[k]
        if (bool(x.full), int(x.understating), int(x.understating_after_notice), bool(x.alerted)) != (full, under, after, alerted) or (k == ('v6', 1) and int(x.legs) != 2):
            bad.append(f'order census {k}: {x.to_dict()}')
    # the reading predicates
    Lr = pd.DataFrame([dict(run='v6', cid=1, trade_id='t1', A0=True, A0_failed='[]', ack=10., bound_status='IDENTIFIED', B=-5., n=30, clock_step_ms=0.1, bound_missing='[]'),
                       dict(run='v6', cid=2, trade_id='t2', A0=False, A0_failed='["one_bot_clock"]', ack=20., bound_status='UNIDENTIFIED', B=-9., n=30, clock_step_ms=0.1,
                            bound_missing='["one bot clock"]'),
                       dict(run='v6', cid=3, trade_id='t3', A0=True, A0_failed='[]', ack=np.nan, bound_status='UNIDENTIFIED', B=None, n=0, clock_step_ms=0.1, bound_missing='[]')])
    Or = pd.DataFrame([dict(cls='CANCELED', acknowledged=True, full=True, understating=1, understating_after_notice=1, alerted=True, alerts='["late_fill_beyond_cancel_bound"]'),
                       dict(cls='MATCHED-OR-GONE', acknowledged=False, full=True, understating=0, understating_after_notice=0, alerted=False, alerts='[]')])
    rd, Rr = readings(Lr, Or, [dict(run='v6', cid=1, trade_id='t1', A4=dict(B=-5., n=30, clock_step_ms=0.1))])
    if not (rd['N0 REPRODUCES-THE-CASE-AUDIT'] == 'YES' and rd['N1 RECORDS-REPRODUCE'] == '2 of 3 legs' and rd['N2 BOOK-BOUND'] == 'IDENTIFIED on 1 of 2 legs with a cancel confirmation'
            and rd['N3 CANCEL-ANSWER-ORDERS-FILLS'].startswith('NO: 1 ') and rd['N4 TERMINAL-SNAPSHOT-UNDERSTATES-FILLS'].startswith('1 orders (1 ') and Rr['N2_B_ms'] == dict(min=-5., p50=-5., max=-5.)
            and len(Rr['N2_unidentified']) == 1 and Rr['N3_class_x_fill']['CANCELED|full'] == 1
            and Rr['N3b_any_acknowledgement_x_fill'] == {'acknowledged|full': 1, 'acknowledged|partial': 0, 'never acknowledged|full': 1, 'never acknowledged|partial': 0}):
        bad.append(f'readings on planted counts: {rd} {Rr}')
    rd2, _ = readings(Lr, Or.assign(cls=['MATCHED-OR-GONE', 'CANCELED'], full=[True, False]), [dict(run='v6', cid=1, trade_id='t1', A4=dict(B=-4., n=30, clock_step_ms=0.1))])
    if rd2['N3 CANCEL-ANSWER-ORDERS-FILLS'] != 'NOT-CONTRADICTED' or rd2['N0 REPRODUCES-THE-CASE-AUDIT'] != 'NO':
        bad.append(f'readings: a partial fill after Canceled is not a race / a different B must not reproduce: {rd2}')
    print('SELFTEST', 'PASS' if not bad else 'FAIL')
    for b_ in bad:
        print('  -', b_)
    return not bad


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'selftest'
    if cmd == 'selftest':
        sys.exit(0 if selftest() else 1)
    if cmd == 'run':
        C.require(selftest() and TCA.selftest(), 'selftest failed: no run'); run()
    else:
        sys.exit(f'unknown command {cmd}')
