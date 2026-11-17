"""QUEUE PLACE TEST: POST FIRST, POST LATE, REST EARLY, JOIN, REST ONE BEHIND, IMPROVE (2026-11-17, session aea5bfb9; USER "Ok go" on: "go
back to the order setup ... compare the current post with resting at the next price level before it becomes the best and posting first
the moment a new best forms" - the USER's own idea: be at the front of the queue). Declared before any run; descriptive with declared
predicates; no model, no controller.

ENGINE: the other session's validated C++ branch kernel (pf_flow_control_kernel.evaluate) on the ACTUAL future tape and book of each window
(not scenarios): queue M (displayed size at activation, min-updated by later frames, unknown depth 0), trades shifted back 50 ms (M50;
M250 printed), placement 58 ms, structural drift / cross cancels (hold 3 ticks + rung), no other cancels. The path builder mirrors
pf_flow_control_paths.extract with a longer end (must reproduce it exactly at 2 s). VALUE of an order (30 sh): cash + inventory x the
valid mid at the common END, c per order; its fills in the order's first 100 ms (the kernel re-run with the order removed 100 ms after
activation) are split out - young replay fills are mostly phantom on live orders (B1a).
EVENT STUDY (births of a new best price, per window and side, frames in venue time; up to 40 per window-side by identity hash): b = the
venue time the new best p appears, seen at its receive time; END = b + H.
 FIRST  post at p at the first 50 ms decision step after the birth is received (p must still be the known best then).
 LATE   post at p 300 ms after FIRST (our ladder arrived a median 359 ms after a level became the best; QUEUE-PLACE).
 EARLY  (depletion births: the old best emptied and p, behind it, became the best) an order posted at p, 1-3 ticks behind the best, at the
        last decision step >= 2 s before b - EX POST: it is conditioned on the birth that follows, a measure of queue value at the birth, not
        a strategy.
CAUSAL STATES (every 100th decision step - 5 s - per window and side, valid book; END = t + H):
 JOIN   post at the best (rung 0).   REST1  post one tick behind the best (rung 1).   IMPROVE  post one tick inside, creating a new best
 (only when the spread is >= 2 ticks; it fills on any trade at or through its price - an optimistic counterfactual).
HORIZONS: H = 5 s primary, 2 s printed. Days Oct23-27 (consumed), every selected market, full five-minute contracts, ends >= 2 s before expiry.
READINGS (H = 5 s, M50, mean c per order, day rule >= 4 of 5):
 F1 FIRST-BEATS-LATE: mean(FIRST - LATE) > 0 pooled and on >= 4 of 5 days.  F2 FIRST-PROFITABLE: mean(FIRST) > 0 pooled and >= 4 of 5.
 E1 EARLY-BEATS-FIRST (ex post): mean(EARLY - FIRST) > 0 pooled and >= 4 of 5.  E2 EARLY-PROFITABLE (ex post): mean(EARLY) > 0, >= 4 of 5.
 R1 REST1-BEATS-JOIN, R2 IMPROVE-BEATS-JOIN (states with spread >= 2): the paired mean difference > 0 pooled and >= 4 of 5.
 R3 PROFITABLE (each of JOIN / REST1 / IMPROVE): mean > 0 pooled and >= 4 of 5.
 Every reading is printed with and without the orders' first-100-ms fills.
 Printed: fill share, value per filled share, by birth type, side and contract minute; H = 2 s; M250.
CONTROLS (any failure -> no output): the path builder reproduces pf_flow_control_paths.extract (frames, trades, 2 s mark) on sampled
 states (planted error: trades without the 50 ms shift must differ); kernel cash / position reconciliation (inside evaluate); young + old
 = total by construction, young fills <= total fills; EARLY's decision precedes b - 58 ms (planted error: a decision after the birth
 refused); FIRST's p is the known best at its decision step.
FIRST RUN 06:30Z stopped on Oct25 (kernel refused a horizon <= 0: a birth received > 1.7 s late leaves the 2 s LATE arm no time before
 END); such arms now carry status no_time_before_end; the partial run is kept as provenance (failed_run_1) and every day re-run.
LIMITS: consumed days; replay fill model (queue M unvalidated on live, counterfactual fills for orders that change the book - IMPROVE
 especially); no strategy cancels (orders held to END); a single order at a time (no ladder, no inventory limit); births are ex post for
 EARLY. Usage (run_light, PF_SLOT=claude): pf_queue_place_test.py selftest | day <day> | report
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_flow_control_common as F                        # noqa: E402
import pf_flow_control_paths as PATH                      # noqa: E402
import pf_flow_control_kernel as K                        # noqa: E402
import pf_joint_moment_data as JD                         # noqa: E402

C, pd, I = F.C, F.pd, F.I
AC = C.AC
OUT = Path(I.K.IO) / 'queue_place_20261117'
DAYS = ('20261023', '20261024', '20261025', '20261026', '20261027')
H_MAIN, H_SHORT = 5000., 2000.
LATE_MS, EARLY_MS, YOUNG_MS = 300., 2000., 100.
PER_SIDE, STATE_EVERY = 40, 100
MUT_NOSHIFT = 0                                           # planted error (the path-parity control)


def build_path(W, t, end, shift=50.):
    """pf_flow_control_paths.extract with END and the trade shift as parameters."""
    w = W[0]; j = int(np.searchsorted(w['kf'], t, side='right') - 1)
    if j < 0:
        return None, 'no_known_book'
    bid, ask = float(W[0]['P5'][j, 0]), float(W[1]['P5'][j, 0])
    if not 0 <= bid < ask <= 100:
        return None, 'invalid_known_book'
    if end > w['ws'] + 298000.:
        return None, 'expiry_censored'
    e = int(np.searchsorted(w['tf'], end, side='right'))
    if e <= j or w['tf'][-1] < end:
        return None, 'missing_future_book'
    if np.max(np.diff(np.r_[w['tf'][j:e], end])) > 1000.:
        return None, 'stale_future_book'
    frames = np.column_stack([np.asarray(w['tf'])[j:e] - t, np.asarray(w['kf'])[j:e] - t, W[0]['P5'][j:e], W[0]['S5'][j:e], W[1]['P5'][j:e], W[1]['S5'][j:e]]).astype(float)
    tape = []; sh = 0. if MUT_NOSHIFT else shift
    for side in (0, 1):
        z = W[side]; tt = np.asarray(z['anc']) - sh - t; ix = np.flatnonzero((tt > 0) & (tt <= end - t))
        tape.extend((float(tt[i]), side, float(z['ptl'][i]), float(z['shl'][i]), int(i)) for i in ix)
    tape.sort(key=lambda x: (x[0], x[1], x[4]))
    trades = np.asarray([x[:4] for x in tape], float).reshape(-1, 4)
    mt, mm = AC.valid_marks(w, 0); ix = np.searchsorted(mt, end, side='right') - 1
    if ix < 0 or end - mt[ix] > 1000.:
        return None, 'missing_endpoint'
    return dict(frames=frames, trades=trades, mark=float(mm[ix]), bid=bid, ask=ask, j=j, mid=(bid + ask) / 2.), 'eligible'


def evaluate(p, orders, horizon, young=False):
    """Each order alone (one branch per order): value at END (c) and filled shares; young=True removes each order 100 ms after activation."""
    fo = np.array([0, len(p['frames'])], np.int64); to = np.array([0, len(p['trades'])], np.int64)
    packed = (fo, to, np.ascontiguousarray(p['frames']), np.ascontiguousarray(p['trades']), np.zeros(1), np.array([p['mark']]))
    o = np.asarray(orders, float).reshape(-1, 9).copy()
    if young:
        o[:, 6] = o[:, 5] + YOUNG_MS
    acts = np.eye(len(o))
    out, qty = K.evaluate(packed, o, acts, 0., p['mid'], horizon)
    return out[0, :, 0].copy(), np.array([qty[0, b, b] for b in range(len(o))])


def order(side, rung, price, act=58.):
    return [side, rung, price, 30., 0., act, 1e30, 0., 1.]


def keyhash(*a):
    return hashlib.sha256(':'.join(map(str, a)).encode()).hexdigest()


def births(W, s):
    """New best prices on side s: (frame index, venue time, price, kind) with kind improve / deplete."""
    best = np.asarray(W[s]['P5'], float)[:, 0]; opp = np.asarray(W[1 - s]['P5'], float)[:, 0]; tf = np.asarray(W[s]['tf'], float)
    ok = (best >= 0) & (opp >= 0) & ((best < opp) if s == 0 else (best > opp)); out = []
    for j in range(1, len(best)):
        if ok[j] and ok[j - 1] and best[j] != best[j - 1]:
            imp = (best[j] > best[j - 1]) if s == 0 else (best[j] < best[j - 1])
            out.append((j, float(tf[j]), float(best[j]), 'improve' if imp else 'deplete'))
    return out


def selftest():
    bad = []
    if order(0, 1, 50.)[:3] != [0, 1, 50.]:
        bad.append('order layout')
    if keyhash('a', 1) == keyhash('a', 2):
        bad.append('hash')
    try:
        early_ok(1000., 990.); caught = False                  # planted error: a decision 10 ms before the birth (after b - 58 ms)
    except ValueError:
        caught = True
    if bad or not caught:
        sys.exit(f'QUEUE-PLACE SELFTEST FAILED -> no output: {bad}, early-after-birth refused {caught}')
    print('queue-place selftests passed; planted early-after-birth decision refused', flush=True)


def early_ok(b, t_e):
    C.require(t_e < b - 58., 'EARLY decides after the birth')
    return True


def parity(W, t, day):
    """Control: build_path(t, t + 2 s) == pf_flow_control_paths.extract(W, t, day) (frames, trades, 2 s mark)."""
    a, sa = build_path(W, t, t + 2000.); b, sb = PATH.extract(W, t, day)
    if a is None or b is None:
        return None
    return bool(np.array_equal(a['frames'], b['frames']) and np.array_equal(a['trades'], b['trades']) and a['mark'] == b['marks'][1]), len(b['trades'])


def day_run(day, limit=None):
    C.require(day in DAYS, 'undeclared day'); selftest(); start = time.time()
    out = OUT / (f'smoke_{day}' if limit else day); C.require(not out.exists(), 'day output exists'); out.mkdir(parents=True)
    raw, reg = JD.metadata(day, True); markets = reg.mid.astype(int).tolist()[:limit] if limit else reg.mid.astype(int).tolist()
    offset = json.loads((Path(C.V.E.OUTD) / f'day_{day}_imp.json').read_text())['untimed']['imp_off_ms']
    rows = []; st = dict(parity_checked=0, parity_bad=0, mut_checked=0, mut_differs=0, births=0, states=0)
    for begin in range(0, len(markets), 24):
        batch = markets[begin:begin + 24]; data = I.load_day(day, set(batch), offset)
        for W in I.pairs(data):
            m = int(W[0]['m']); tg = np.asarray(W[0]['tg'], float); kn = np.asarray(W[0]['jk'])
            # control: path parity on two states per window, and the planted no-shift error
            for t in (float(tg[len(tg) // 3]), float(tg[2 * len(tg) // 3])):
                r = parity(W, t, day)
                if r is None:
                    continue
                st['parity_checked'] += 1; st['parity_bad'] += int(not r[0])
                if r[1] > 0:                                   # the planted no-shift error is visible only where trades exist
                    globals()['MUT_NOSHIFT'] = 1
                    try:
                        st['mut_checked'] += 1; st['mut_differs'] += int(parity(W, t, day)[0] is False)
                    finally:
                        globals()['MUT_NOSHIFT'] = 0
            for s in (0, 1):
                sgn = 1 if s == 0 else -1; P5 = np.asarray(W[s]['P5'], float)
                # --- event study
                bl = [x for x in births(W, s) if x[1] >= W[0]['ws'] + 2500. and x[1] + H_MAIN <= W[0]['ws'] + 298000.]
                bl = sorted(bl, key=lambda x: keyhash(day, m, s, x[1]))[:PER_SIDE]
                for (j, b, p, kind) in bl:
                    kf_b = float(W[s]['kf'][j]); kf_ = np.searchsorted(tg, kf_b, side='left')
                    if kf_ >= len(tg):
                        continue
                    t_f = float(tg[kf_]); jj = int(kn[kf_])
                    if jj < 0 or P5[jj, 0] != p:
                        rows.append(dict(day=day, m=m, side=s, kind=kind, arm='FIRST', status='gone_before_first')); continue
                    arms = [('FIRST', t_f, 0), ('LATE', t_f + LATE_MS, None)]
                    if kind == 'deplete':
                        ke = int(np.searchsorted(tg, b - EARLY_MS, side='right') - 1)
                        if ke >= 0:
                            t_e = float(tg[ke]); early_ok(b, t_e); je = int(kn[ke])
                            if je >= 0 and P5[je, 0] >= 0:
                                dist = int(round(sgn * (P5[je, 0] - p)))
                                if 1 <= dist <= 3:
                                    arms.append(('EARLY', t_e, dist))
                    st['births'] += 1
                    for arm, t_a, rung in arms:
                        rec = dict(day=day, m=m, side=s, kind=kind, arm=arm, b=b, t=t_a, p=p, phase=min(5, int((b - W[0]['ws']) // 60000) + 1))
                        if arm == 'LATE':
                            ka = int(np.searchsorted(tg, t_a, side='left'))
                            ja = int(kn[min(ka, len(kn) - 1)]) if ka < len(kn) else -1
                            rung = int(round(sgn * (P5[ja, 0] - p))) if ja >= 0 and P5[ja, 0] >= 0 else 0
                            rung = max(rung, 0)
                        for H, tag in ((H_MAIN, ''), (H_SHORT, '_2s')):
                            if b + H - t_a < 50.:                       # seen too late: no decision time before END (a status, not a crash)
                                rec['status' + tag] = 'no_time_before_end'; continue
                            pth, status = build_path(W, t_a, b + H)
                            if pth is None:
                                rec['status' + tag] = status; continue
                            o = [order(s, rung, p)]
                            v, q = evaluate(pth, o, b + H - t_a); vy, qy = evaluate(pth, o, b + H - t_a, young=True)
                            C.require(qy[0] <= q[0] + 1e-9, 'young fills exceed total fills')
                            rec.update({'status' + tag: 'eligible', 'v' + tag: float(v[0]), 'q' + tag: float(q[0]), 'vy' + tag: float(vy[0]), 'qy' + tag: float(qy[0])})
                            if H == H_MAIN:
                                p250, s250 = build_path(W, t_a, b + H, shift=250.)
                                if p250 is not None:
                                    v2, q2 = evaluate(p250, o, b + H - t_a); rec.update(v_m250=float(v2[0]), q_m250=float(q2[0]))
                        rows.append(rec)
                # --- causal states
                for k in range(0, len(tg), STATE_EVERY):
                    t = float(tg[k])
                    if t < W[0]['ws'] + 2000. or t + H_MAIN > W[0]['ws'] + 298000.:
                        continue
                    jj = int(kn[k])
                    if jj < 0:
                        continue
                    best = float(P5[jj, 0]); oppb = float(np.asarray(W[1 - s]['P5'], float)[jj, 0])
                    if best < 0 or oppb < 0:
                        continue
                    spread = (oppb - best) if s == 0 else (best - oppb)
                    if spread < 1:
                        continue
                    st['states'] += 1
                    cands = [('JOIN', 0, best), ('REST1', 1, best - sgn)]
                    if spread >= 2:
                        cands.append(('IMPROVE', 0, best + sgn))
                    cands = [c for c in cands if 1 <= c[2] <= 99]
                    rec0 = dict(day=day, m=m, side=s, t=t, spread=spread, phase=min(5, int((t - W[0]['ws']) // 60000) + 1))
                    o = [order(s, r, pr) for _, r, pr in cands]; vals = {}
                    for H, tag in ((H_MAIN, ''), (H_SHORT, '_2s')):
                        pth, status = build_path(W, t, t + H)
                        if pth is None:
                            vals[tag] = None; continue
                        v, q = evaluate(pth, o, H); vy, qy = evaluate(pth, o, H, young=True); vals[tag] = (v, q, vy, qy)
                        if H == H_MAIN:
                            p250, _ = build_path(W, t, t + H, shift=250.)
                            vals['m250'] = evaluate(p250, o, H)[0] if p250 is not None else np.full(len(o), np.nan)
                    if vals.get('') is None:
                        continue
                    for n_, (arm, r, pr) in enumerate(cands):
                        rec = dict(rec0, arm=arm, p=pr, kind='state', status='eligible')
                        for tag in ('', '_2s'):
                            if vals.get(tag) is not None:
                                v, q, vy, qy = vals[tag]; rec.update({'v' + tag: float(v[n_]), 'q' + tag: float(q[n_]), 'vy' + tag: float(vy[n_]), 'qy' + tag: float(qy[n_])})
                        rec['v_m250'] = float(vals['m250'][n_]); rows.append(rec)
            print(json.dumps(dict(stage='day', day=day, m=m, rows=len(rows), seconds=round(time.time() - start), rss=round(I.K.rss_mib()))), flush=True)
        del data; C.guard('queue-place day')
    if st['parity_bad'] or not st['parity_checked'] or not st['mut_checked'] or st['mut_differs'] != st['mut_checked']:
        sys.exit(f'QUEUE-PLACE CONTROL FAILED on {day} -> no output: {st}')
    pd.DataFrame(rows).to_parquet(out / 'rows.parquet', index=False)
    C.dump(out / 'receipt.json', dict(asof=C.stamp(), day=day, markets=len(markets), stat=st, outputs={'rows.parquet': C.sha(out / 'rows.parquet')}, seconds=time.time() - start))
    print(json.dumps(dict(day=day, stat=st, seconds=round(time.time() - start))), flush=True)


def report():
    selftest(); R = []
    for d in DAYS:
        r = json.loads((OUT / d / 'receipt.json').read_text()); C.require(C.sha(OUT / d / 'rows.parquet') == r['outputs']['rows.parquet'], 'rows changed'); R.append(pd.read_parquet(OUT / d / 'rows.parquet'))
    A = pd.concat(R, ignore_index=True); asof = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    L = [f'QUEUE PLACE TEST (actual tape and book through the validated kernel; one 30 sh order at a time, held to END); as of {asof}', '']
    ev = A[(A.kind != 'state')]; stt = A[A.kind == 'state']

    def summ(g, tag=''):
        g = g[g['status' + tag] == 'eligible'] if 'status' + tag in g else g
        v = g['v' + tag]; vo = g['v' + tag] - g['vy' + tag]; q = g['q' + tag]
        return dict(n=len(g), v=float(v.mean()), v_old=float(vo.mean()), fill=float((q > 1e-9).mean()), sh=float(q.sum()),
                    per_sh=float(v.sum() / max(q.sum(), 1e-9)))
    L.append('PART 1: event study - births of a new best (H = 5 s, M50): arm | n | value c/order (all fills | w/o first-100-ms fills) | filled | c per filled share || H 2 s | M250')
    for kind in ('improve', 'deplete'):
        for arm in ('FIRST', 'LATE', 'EARLY'):
            g = ev[(ev.kind == kind) & (ev.arm == arm)]
            if len(g) == 0 or 'v' not in g:
                continue
            a = summ(g); b2 = summ(g, '_2s') if 'v_2s' in g else None
            L.append(f'  {kind:<8} {arm:<6} n {a["n"]:6,} | {a["v"]:+7.2f} | {a["v_old"]:+7.2f} | {a["fill"]:5.1%} | {a["per_sh"]:+6.2f} || '
                     f'{b2["v"] if b2 else float("nan"):+7.2f} | {g.v_m250.mean():+7.2f}')
    L.append(f'  gone before FIRST (the new best had changed by the first decision step): {int((ev.status == "gone_before_first").sum()):,}')
    L += ['', 'PART 2: causal states (H = 5 s, M50): arm | n | value c/order (all | w/o first-100-ms fills) | filled | c per filled share || H 2 s | M250']
    for arm in ('JOIN', 'REST1', 'IMPROVE'):
        g = stt[stt.arm == arm]; a = summ(g); b2 = summ(g, '_2s')
        L.append(f'  {arm:<8} n {a["n"]:6,} | {a["v"]:+7.2f} | {a["v_old"]:+7.2f} | {a["fill"]:5.1%} | {a["per_sh"]:+6.2f} || {b2["v"]:+7.2f} | {g.v_m250.mean():+7.2f}')
    # paired differences and readings
    def paired(df, a, b, keys, col='v'):
        x = df[df.arm == a].set_index(keys)[col]; y = df[df.arm == b].set_index(keys)[col]; z = (x - y).dropna()
        return z

    def rule(series_by_day):
        pooled = float(np.mean(np.concatenate([s.values for s in series_by_day.values()]))) if series_by_day else float('nan')
        days = sum(float(s.mean()) > 0 for s in series_by_day.values())
        return pooled > 0 and days >= 4, pooled, days
    kev = ['day', 'm', 'side', 'b']; kst = ['day', 'm', 'side', 't']
    out = {}
    for name, df, a, b, keys in (('F1 FIRST-BEATS-LATE', ev, 'FIRST', 'LATE', kev), ('E1 EARLY-BEATS-FIRST (ex post)', ev[ev.kind == 'deplete'], 'EARLY', 'FIRST', kev),
                                 ('R1 REST1-BEATS-JOIN', stt, 'REST1', 'JOIN', kst), ('R2 IMPROVE-BEATS-JOIN', stt, 'IMPROVE', 'JOIN', kst)):
        for col, lab in (('v', 'all fills'), ('v_old', 'w/o first-100-ms fills')):
            d2 = df.copy(); d2['v_old'] = d2['v'] - d2['vy']
            by = {d: paired(d2[d2.day == d], a, b, keys, col) for d in DAYS}
            ok, pooled, days = rule(by); out[(name, col)] = ok
            L.append(f'{name} [{lab}]: {"YES" if ok else "NOT-SHOWN"} (pooled {pooled:+.2f} c/order; positive on {days} / 5 days)')
    for name, df, arm in (('F2 FIRST-PROFITABLE', ev, 'FIRST'), ('E2 EARLY-PROFITABLE (ex post)', ev[ev.kind == 'deplete'], 'EARLY'),
                          ('R3 JOIN-PROFITABLE', stt, 'JOIN'), ('R3 REST1-PROFITABLE', stt, 'REST1'), ('R3 IMPROVE-PROFITABLE', stt, 'IMPROVE')):
        for col, lab in (('v', 'all fills'), ('v_old', 'w/o first-100-ms fills')):
            g = df[(df.arm == arm) & (df.status == 'eligible')].copy(); g['v_old'] = g['v'] - g['vy']
            by = {d: g[g.day == d][col].dropna() for d in DAYS}
            ok, pooled, days = rule(by); out[(name, col)] = ok
            L.append(f'{name} [{lab}]: {"YES" if ok else "NOT-SHOWN"} (pooled {pooled:+.2f} c/order; positive on {days} / 5 days)')
    L += ['', 'By day (H 5 s, M50, all fills) - event FIRST / LATE / EARLY | states JOIN / REST1 / IMPROVE:']
    for d in DAYS:
        e = ev[(ev.day == d) & (ev.status == 'eligible')]; s_ = stt[stt.day == d]
        L.append(f'  {d}: ' + ' / '.join(f'{e[e.arm == a_].v.mean():+6.2f}' for a_ in ('FIRST', 'LATE', 'EARLY')) + ' | ' +
                 ' / '.join(f'{s_[s_.arm == a_].v.mean():+6.2f}' for a_ in ('JOIN', 'REST1', 'IMPROVE')))
    L += ['', 'LIMITS: consumed days; replay fill model (queue M, counterfactual fills - IMPROVE optimistic); orders held to END (no strategy cancels); one order at a time; EARLY is ex post']
    (OUT / 'queue_place.txt').write_text('\n'.join(L) + '\n')
    C.dump(OUT / 'queue_place.json', dict(asof=asof, readings={f'{k[0]} [{k[1]}]': v for k, v in out.items()}))
    print('\n'.join(L), flush=True)


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'selftest':
        selftest()
    elif cmd == 'day':
        day_run(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else None)
    elif cmd == 'report':
        report()
    else:
        sys.exit('usage: pf_queue_place_test.py selftest | day <day> | report')
