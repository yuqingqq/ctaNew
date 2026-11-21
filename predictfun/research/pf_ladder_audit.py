"""RECEIPT-LEAD-TIME-AUDIT-v1 WITH THE CORRECTED LADDER DIAGNOSTICS (P-2026-005; 2026-11-21, session aea5bfb9). Review 0065191f (target 869fa85f)
accepted the LADDER_REST_INVENTORY_v1 ledger of the 19 complete markets and named ONE next priority: on the same 19 markets / 5,288 submitted
generations (all 96 statuses kept), can information received before an effective cancellation distinguish bad future fills from useful
fills? It bundled the diagnostic corrections (findings 1-4, 7, 8). No fit, guard, threshold search, profit or oracle avoided-loss claim, no
new model, no baseline replay; consumed Oct 24-27 only. The frozen run (47d2eec1; pf_ladder_rest, pf_branch_ladder unchanged) is read
through pf_ladder_rest.load_outputs; every day record's sources must equal the 47d2eec1 blobs.
WINDOW RULE (engine-consistent, pf_branch_engine.Engine.request): a request at d on order o lands at eff(d, o) = max(d + 37, submit + 95) when
 d < activation, else d + 37; the engine fills an order up to its removal EXCLUSIVE, so a fill at t >= eff is preventable by that request and
 one at t < eff is not. A fill after a theoretical effect is NOT automatically avoidable: an eligible observed signal is still required.
CORRECTED DIAGNOSTICS:
 F1 per generation three named boundaries: pending_first = eff(the first grid tick after the submission tick) (the client structural rule
   may request a post-pending order there; submit + 95), live_first = (the first grid tick at or after activation) + 37, actual = the logged
   removal effect; quantity, END value and +1 s markout split at each; the fills in [pending_first, live_first) separately;
 F2 public-trade sweep events: (side, trade index) rows that fill >= 2 own generations, with their tranches' END and +1 s values; the
   former same-timestamp grouping and the ever-exposed-generation view are reported under their own names;
 F3 stage labels on ONE window per snapshot: entry -> every fill of its generation; first / live at d -> the fills at t >= eff(d, o). On that
   window: quantity, any-fill, END value, the +1 s markout (PRIMARY: sum of s q (mid(t + 1 s) - p)), +50 / +100 ms markouts, severe (END
   < -30 c; END / submitted < -1 c); a window fill without its +1 s mid censors the +1 s label (censored_1s, never 0); lifetime_* named
   apart; the entry / first / live cohorts are never pooled;
 F4 policy-only first-review pulls; accepted generations requested no later than their first eligible live tick, by reason; restriction
   episodes recovered (exit to UNRESTRICTED) / reversal (direct exit to the opposite restriction) / censored at END;
 F7 the net market loss concentration and the +1 s generation tails beside the END tails; a closed round trip (bid 50 / ask 51, mark 80)
   shows END legs of +3,000 / -2,900 c for a +100 c joint value (the selftest);
 F8 exact durations: per side the union of venue-active intervals [activation, min(removal, full fill, END)); bid / ask / either / both
   shares of [0, END] beside the 50 ms grid estimates.
AUDIT (states.csv; per snapshot): kind, side, decision d, the observation ages pf_age / bn_age (ms) and receipt times d - age, the command's
 effect (entry: activation d + 58), the window labels (F3), the fills before the effect (unpreventable by a command at d), the generation's
 actual cancel-pending window fills (req <= t < removal), and the already acquired client-known inventory I(d) with its venue-mid move to
 d + 1 s (I x (mid(d + 1 s) - mid(d)); missing beyond END). Per kind x side x dictionary input x the frozen Oct 24-25 bins (the frozen
 measurement's audit.json edges, pf_ladder_rest.bin_of, unchanged on every day) x day: opportunities, censored, window quantity, any-fill,
 +1 s sum / positive / negative / mean, END sum / positive / negative.
READINGS (computed; a lead for the next experiment, never a claim): a (kind, input) is a LEAD when q5 - q1 of the mean +1 s window outcome
 has the same nonzero sign in all eight (day, side) cells, each bin with >= MIN_CELL uncensored opportunities; the ONE candidate hypothesis =
 the lead with the largest pooled |q5 - q1| (ties by kind then dictionary order), else NO_LEAD.
CAUSAL AUDIT (causal <day>): per day the AUDIT_N lowest sha256(PREFIX|audit|{day, m, kind, side, gid, t}) ok snapshots; per snapshot the
 71 market inputs by pf_pipeline_export.causal_check (future-only perturbation -> unchanged; delivered same-time step change -> changed),
 and known_steps_q (frames received after d and fills notified after d perturbed -> unchanged; the frame received at d with the order's
 price size set to 0 -> the queue estimate changed, counted where it was positive).
Usage (run_light): pf_ladder_audit.py selftest | saved | causal <day>"""
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_ladder_rest as M                                # noqa: E402  frozen: the run's loader, helpers, dictionary and bins

C, pd, LB, BE, X, LS, A = M.C, M.pd, M.LB, M.BE, M.X, M.LS, M.A
VERSION = 'receipt_lead_time_audit_v1'
PREFIX = 'receipt_lead_time_audit_v1|'
RUN_FREEZE = '47d2eec1'
STEP, CANCEL, PENDING, PLACE, NOTICE = M.STEP, M.CANCEL, float(BE.PENDING), M.PLACE, M.NOTICE
MIN_CELL, AUDIT_N = 10, 16
KINDS = ('entry', 'first', 'live')
H1 = 6                                                     # fill_mids index of mid(t + 1,000 ms) (pf_ladder_rest.mids_of)
REPO = HERE.parents[1]
RECEIPT = 'predictfun/reviews/ladder_audit_20261121'
OUT = A.DATA / 'ladder_audit_v1_20261121'
SOURCES = ('pf_ladder_audit.py', 'pf_ladder_audit_selftest.py', 'pf_ladder_rest.py', 'pf_branch_ladder.py', 'pf_branch_engine.py', 'pf_pipeline_export.py')


def sources():
    return {n: C.sha(HERE / n) for n in SOURCES}


def blob_sha(commit, rel):
    r = subprocess.run(['git', '-C', str(REPO), 'show', f'{commit}:{rel}'], capture_output=True)
    return hashlib.sha256(r.stdout).hexdigest() if r.returncode == 0 else None


def frozen():
    return bool(all(M.committed_equal(f'predictfun/research/{n}', HERE / n) for n in ('pf_ladder_audit.py', 'pf_ladder_audit_selftest.py'))
                and all(M.committed_equal(f'{RECEIPT}/{n}', REPO / RECEIPT / n) for n in ('declaration.md', 'launch_audit.sh')))


def sgn(s):
    return 1. if int(s) == 0 else -1.


def eff(d, o):
    """The engine's landing of a removal requested at d (pf_branch_engine.Engine.request)."""
    return max(d + CANCEL, float(o['submit']) + PENDING) if d < float(o['act']) else d + CANCEL


def boundaries(o):
    """F1: pending_first, live_first and the actual effect (None without a request)."""
    return dict(pending_first=eff(float(o['submit']) + STEP, o), live_first=math.ceil(float(o['act']) / STEP) * STEP + CANCEL,
                actual=float(o['remove']) if o['req'] is not None else None)


def fill_view(L_):
    """Per gid its fills [(t, idx, side, q, p, mid+1s)] in (t, trade index) order."""
    out = {}
    for f, mm in zip(L_['fills'], L_['fill_mids']):
        out.setdefault(int(f[1]), []).append((float(f[0]), int(f[5]), int(f[2]), float(f[3]), float(f[4]), mm))
    return {g: sorted(v, key=lambda x: (x[0], x[1])) for g, v in out.items()}


def vals(fl, mark, idx=H1):
    """(quantity, END value, the markout at mids index idx or None when any fill lacks it) of a fill list."""
    q = sum(x[3] for x in fl); v = sum(sgn(x[2]) * x[3] * (mark - x[4]) for x in fl)
    mo = None if any(x[5][idx] is None for x in fl) else sum(sgn(x[2]) * x[3] * (x[5][idx] - x[4]) for x in fl)
    return q, v, mo


def gen_v2(r):
    """F1 / F2 / F4 rows of one complete market: per generation the boundaries, splits, actual cancel-pending fills and sweep membership."""
    L_ = r['ledger']; mark, end = float(L_['mark']), float(L_['end']); fv = fill_view(L_); ev = {}
    for f in L_['fills']:
        ev.setdefault((int(f[2]), int(f[5])), set()).add(int(f[1]))
    rows = []
    for o in L_['orders']:
        gid = int(o['gid']); fl = fv.get(gid, []); b = boundaries(o); x = dict(day=r['day'], m=int(r['m']), gid=gid, side=int(o['side']), rung=int(round(float(o['rung']))),
                                                                          size=float(o['size']), submit=float(o['submit']), act=float(o['act']), refused=bool(o['refused']), **b)
        for nm in ('pending_first', 'live_first'):
            bb = b[nm]; pre = [y for y in fl if y[0] < bb]; post = [y for y in fl if y[0] >= bb]
            (q0, v0, m0), (q1, v1, m1) = vals(pre, mark), vals(post, mark)
            x.update({f'q_before_{nm}': q0, f'v_before_{nm}': v0, f'mo1s_before_{nm}': m0, f'q_from_{nm}': q1, f'v_from_{nm}': v1, f'mo1s_from_{nm}': m1})
        gap = [y for y in fl if b['pending_first'] <= y[0] < b['live_first']]; qg, vg, mg = vals(gap, mark)
        x.update(n_gap=len(gap), q_gap=qg, v_gap=vg, mo1s_gap=mg)
        if o['req'] is not None:
            cp = [y for y in fl if float(o['req']) <= y[0] < float(o['remove'])]; br = [y for y in fl if y[0] < float(o['req'])]
        else:
            cp, br = [], list(fl)
        (qc, vc, mc), (qb, vb, mb) = vals(cp, mark), vals(br, mark)
        x.update(n_cancel_pending=len(cp), q_cancel_pending=qc, v_cancel_pending=vc, mo1s_cancel_pending=mc, n_before_request=len(br), q_before_request=qb,
                 v_before_request=vb, mo1s_before_request=mb, request_reason=None if o['req'] is None else ('policy' if o['why'] == 'policy' else str(o['why'])),
                 req=None if o['req'] is None else float(o['req']))
        sw = [y for y in fl if len(ev[(y[2], y[1])]) >= 2]; qs, vs, ms = vals(sw, mark)
        lq, lv, lm = vals(fl, mark)
        x.update(n_sweep_event_fills=len(sw), q_sweep=qs, v_sweep=vs, mo1s_sweep=ms, ever_swept=bool(sw), lifetime_q=lq, lifetime_value=lv, lifetime_mo1s=lm,
                 lifetime_any=bool(lq > LB.TOL), n_fills=len(fl))
        rows.append(x)
    return rows


def sweep_rows(r):
    """F2: one row per fill of a public-trade event (side, trade index) filling >= 2 own generations; and the former timestamp grouping."""
    L_ = r['ledger']; mark = float(L_['mark']); ev, ts = {}, {}
    for f in L_['fills']:
        ev.setdefault((int(f[2]), int(f[5])), set()).add(int(f[1])); ts.setdefault((int(f[2]), float(f[0])), set()).add(int(f[1]))
    out = []
    for f, mm in zip(L_['fills'], L_['fill_mids']):
        s, i, t = int(f[2]), int(f[5]), float(f[0]); tr = len(ev[(s, i)]) >= 2; tg = len(ts[(s, t)]) >= 2
        if tr or tg:
            out.append(dict(day=r['day'], m=int(r['m']), side=s, trade=i, t=t, gid=int(f[1]), q=float(f[3]), p=float(f[4]), trade_event=tr, time_group=tg,
                            end_value=sgn(s) * float(f[3]) * (mark - float(f[4])), mo1s=None if mm[H1] is None else sgn(s) * float(f[3]) * (mm[H1] - float(f[4]))))
    return out


def churn(r, arrs):
    """F4: policy-only first-review pulls; accepted generations requested no later than their first eligible live tick, by reason."""
    L_ = r['ledger']; cm = M.arr_of(arrs, r, 'commands'); first = {}
    for c in cm:
        if int(c[2]) == M.CMD_KINDS.index('live') and int(c[3]) >= 0 and int(c[3]) not in first:
            first[int(c[3])] = M.CMD_ACTIONS[int(c[7])]
    early = {}
    for o in L_['orders']:
        if o['refused'] or o['req'] is None:
            continue
        if float(o['req']) <= math.ceil(float(o['act']) / STEP) * STEP:
            k = 'policy' if o['why'] == 'policy' else str(o['why']); early[k] = early.get(k, 0) + 1
    return dict(reviewed=len(first), policy_first_review_pulls=sum(1 for v in first.values() if v == 'PULL'), requested_by_first_live=sum(early.values()), by_reason=early)


def recovery(r):
    """F4: restriction episodes from the mode log: recovered / reversal / censored."""
    L_ = r['ledger']; end = float(L_['end']); cur = None; t_in = None; out = []
    for g, I, old, new in L_['mode_log']:
        if cur is not None:
            out.append(dict(day=r['day'], m=int(r['m']), mode=cur, start=t_in, end=float(g), outcome='recovered' if new == 'UNRESTRICTED' else 'reversal', ms=float(g) - t_in))
            cur = None
        if new != 'UNRESTRICTED':
            cur, t_in = new, float(g)
    if cur is not None:
        out.append(dict(day=r['day'], m=int(r['m']), mode=cur, start=t_in, end=end, outcome='censored', ms=end - t_in))
    return out


def union_len(iv):
    """The total length and the merged list of [a, b) intervals."""
    out = []
    for a, b in sorted(iv):
        if b <= a:
            continue
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return sum(b - a for a, b in out), out


def inter_len(u, v):
    i = j = 0; tot = 0.
    while i < len(u) and j < len(v):
        a, b = max(u[i][0], v[j][0]), min(u[i][1], v[j][1]); tot += max(0., b - a)
        if u[i][1] < v[j][1]:
            i += 1
        else:
            j += 1
    return tot


def durations(r, arrs):
    """F8: exact venue-active shares from the order intervals, beside the 50 ms grid estimates."""
    L_ = r['ledger']; end = float(L_['end']); fv = fill_view(L_); iv = {0: [], 1: []}
    for o in L_['orders']:
        if o['refused']:
            continue
        cum = 0.; full = None
        for y in fv.get(int(o['gid']), []):
            cum += y[3]
            if full is None and cum >= float(o['size']) - LB.TOL:
                full = y[0]
        iv[int(o['side'])].append((float(o['act']), min(float(o['remove']), full if full is not None else np.inf, end)))
    (lb, ub), (la, ua) = union_len(iv[0]), union_len(iv[1]); both = inter_len(ub, ua); nva = M.arr_of(arrs, r, 'nva') > 0
    return dict(day=r['day'], m=int(r['m']), end=end, bid_exact=lb / end, ask_exact=la / end, both_exact=both / end, either_exact=(lb + la - both) / end,
                bid_grid=float(nva[:, 0].mean()), ask_grid=float(nva[:, 1].mean()), both_grid=float((nva[:, 0] & nva[:, 1]).mean()), both_ms=both, either_ms=lb + la - both)


def state_rows(F, recs, arrs):
    """The audit rows (header): every snapshot with its window labels, timeline, pre-effect fills, cancel-pending fills and inventory move."""
    byk = {(r['day'], int(r['m'])): r for r in recs if r['status'] == 'complete'}; out = []; cache = {}
    for x in F.to_dict('records'):
        r = byk.get((x['day'], int(x['m'])))
        if r is None or int(x['gid']) < 0:
            continue
        L_ = r['ledger']; mark = float(L_['mark']); end = float(L_['end']); key = (x['day'], int(x['m']))
        if key not in cache:
            cache[key] = ({int(y['gid']): y for y in L_['orders']}, fill_view(L_))
        o = cache[key][0][int(x['gid'])]; fl = cache[key][1].get(int(x['gid']), [])
        d = float(x['t']); kind = x['kind']
        e_ = float(o['act']) if kind == 'entry' else eff(d, o); win = fl if kind == 'entry' else [y for y in fl if y[0] >= e_]
        pre = [] if kind == 'entry' else [y for y in fl if d <= y[0] < e_]
        cp = [y for y in win if o['req'] is not None and float(o['req']) <= y[0] < float(o['remove'])]
        q, v, m1 = vals(win, mark); _, _, m50 = vals(win, mark, 2); _, _, m100 = vals(win, mark, 4); qp, vp, mp = vals(pre, mark); qc, vc, mc = vals(cp, mark)
        mid = M.arr_of(arrs, r, 'mid'); k = int(round(d / STEP)); k2 = k + int(1000. / STEP); I_ = float(x.get('ctx_inventory_shares', np.nan))
        mv = I_ * (mid[k2] - mid[k]) if (k2 < len(mid) and d + 1000. <= end and np.isfinite(mid[k]) and np.isfinite(mid[k2]) and np.isfinite(I_)) else None
        row = dict(day=x['day'], m=int(x['m']), kind=kind, side=int(x['side']), gid=int(x['gid']), rung=int(round(float(x['rung']))), t=d, size=float(o['size']),
                   status=x['status'], pf_age=x.get('pf_age'), bn_age=x.get('bn_age'), effect=e_, latency_ms=e_ - d, n_eligible=int(x['n_eligible']),
                   win_n=len(win), win_q=q, win_any=bool(q > LB.TOL), win_end=v, win_mo1s=m1, win_mo50=m50, win_mo100=m100, censored_1s=m1 is None and len(win) > 0,
                   win_severe_abs=bool(v < -M.SEVERE_ABS), win_severe_per_share=bool(v / float(o['size']) < -M.SEVERE_PER_SHARE), pre_n=len(pre), pre_q=qp, pre_end=vp,
                   pre_mo1s=mp, cp_n=len(cp), cp_q=qc, cp_mo1s=mc, inv_known=I_, inv_move_1s=mv, lifetime_q=sum(y[3] for y in fl), lifetime_any=bool(fl))
        for c in M.DICT_COLS:
            val = x.get('ctx_inventory_shares') if c == 'inv_signed' else x.get(c)
            row[c] = None if val is None or not np.isfinite(val) else (sgn(x['side']) * float(val) if c == 'inv_signed' else float(val))
        out.append(row)
    return pd.DataFrame(out)


def bin_table(S_, edges):
    """Per kind x side x input x frozen bin x day the support and the window outcomes (header)."""
    rows = []
    for (kind, s), g in S_.groupby(['kind', 'side']):
        for c in M.DICT_COLS:
            e = edges.get(f'{kind}|{s}|{c}', []); b = g[c].astype(float).map(lambda v: M.bin_of(v, e))
            for (day, bn), h in g.assign(_b=b).groupby(['day', '_b']):
                u = h[~h.censored_1s & h.win_mo1s.notna() | (h.win_n == 0)]; y = u.win_mo1s.fillna(0.).astype(float); en = h.win_end.astype(float)
                rows.append(dict(kind=kind, side=int(s), feature=c, day=day, bin=bn, n=int(len(h)), censored=int(h.censored_1s.sum()), uncensored=int(len(u)),
                                 win_q=float(h.win_q.sum()), win_any=int(h.win_any.sum()), mo1s_sum=float(y.sum()), mo1s_pos=float(y[y > 0].sum()), mo1s_neg=float(y[y < 0].sum()),
                                 mo1s_mean=float(y.mean()) if len(u) else None, end_sum=float(en.sum()), end_pos=float(en[en > 0].sum()), end_neg=float(en[en < 0].sum())))
    return pd.DataFrame(rows)


def leads(BT, S_, edges):
    """The declared lead rule (header) -> (leads table, the ONE candidate)."""
    out = []
    for kind in KINDS:
        for c in M.DICT_COLS:
            sub = BT[(BT.kind == kind) & (BT.feature == c)]; cells, ok = [], True
            for day in M.DAYS:
                for s in (0, 1):
                    q1 = sub[(sub.day == day) & (sub.side == s) & (sub.bin == 'q1')]; q5 = sub[(sub.day == day) & (sub.side == s) & (sub.bin == 'q5')]
                    if not (len(q1) and len(q5) and int(q1.uncensored.iloc[0]) >= MIN_CELL and int(q5.uncensored.iloc[0]) >= MIN_CELL):
                        ok = False; cells.append(None); continue
                    cells.append(float(q5.mo1s_mean.iloc[0]) - float(q1.mo1s_mean.iloc[0]))
            signs = {int(np.sign(x)) for x in cells if x is not None}
            g = S_[S_.kind == kind]; pooled = None
            if len(g):
                bins = pd.Series([M.bin_of(v, edges.get(f'{kind}|{s}|{c}', [])) for v, s in zip(g[c].astype(float), g.side)], index=g.index)
                un = g[(~g.censored_1s & g.win_mo1s.notna()) | (g.win_n == 0)].copy(); un['_b'] = bins[un.index]; y = un.win_mo1s.fillna(0.)
                if (un._b == 'q5').any() and (un._b == 'q1').any():
                    pooled = float(y[un._b == 'q5'].mean() - y[un._b == 'q1'].mean())
            lead = bool(ok and len(signs) == 1 and 0 not in signs and pooled is not None)
            out.append(dict(kind=kind, feature=c, cells=json.dumps(cells), lead=lead, sign=(next(iter(signs)) if lead else 0), pooled_q5_minus_q1=pooled))
    L_ = pd.DataFrame(out); ld = L_[L_.lead]
    if not len(ld):
        return L_, dict(answer='NO_LEAD')
    order = {(k, c): (KINDS.index(k), M.DICT_COLS.index(c)) for k in KINDS for c in M.DICT_COLS}
    best = sorted(ld.to_dict('records'), key=lambda x: (-abs(x['pooled_q5_minus_q1']), order[(x['kind'], x['feature'])]))[0]
    return L_, dict(answer='LEAD', kind=best['kind'], feature=best['feature'], sign=int(best['sign']), pooled_q5_minus_q1=best['pooled_q5_minus_q1'], n_leads=int(len(ld)),
                    note='a lead for the next experiment, not a guard, threshold, profit or avoided-loss estimate')


def common_markouts(recs, idx=(0, 2, H1)):
    """The equal-population edge / +50 ms / +1 s comparison on the fills with every mids index in idx valid (idx (0, 2, 4, 6, 8, 10): the frozen
    summary's all-horizon common subset)."""
    n = 0; q = e = m50 = m1 = 0.
    for r in recs:
        if r['status'] != 'complete':
            continue
        for f, mm in zip(r['ledger']['fills'], r['ledger']['fill_mids']):
            if any(mm[i] is None for i in idx):
                continue
            s, qq, p = int(f[2]), float(f[3]), float(f[4]); n += 1; q += qq; e += sgn(s) * qq * (mm[0] - p); m50 += sgn(s) * qq * (mm[2] - p); m1 += sgn(s) * qq * (mm[H1] - p)
    return dict(fills=n, q=q, edge=e, mo50=m50, mo1s=m1)


def run_saved(out=OUT):
    """The saved-evidence stage (header) -> out/*.csv + audit.json; write-once; nothing simulated."""
    out = Path(out); C.require(frozen(), 'the audit module / selftest / declaration / launcher are not the committed ones'); C.require(not (out / 'audit.json').exists(), 'the audit exists (write-once)')
    out.mkdir(parents=True, exist_ok=True)
    man, recs, arrs, F = M.load_outputs(False); comp = [r for r in recs if r['status'] == 'complete']
    for day in man['days']:
        rec = json.loads(M.day_paths(day, False)['rec'].read_text())
        C.require(all(rec['sources'][n] == blob_sha(RUN_FREEZE, f'predictfun/research/{n}') for n in rec['sources']), f'{day}: the run sources are not the {RUN_FREEZE} blobs')
    fm = json.loads((M.OUT / 'measure' / 'audit.json').read_text()); edges = fm['edges']
    G_ = pd.DataFrame([x for r in comp for x in gen_v2(r)]); SW = pd.DataFrame([x for r in comp for x in sweep_rows(r)])
    RC = pd.DataFrame([x for r in comp for x in recovery(r)]); DU = pd.DataFrame([durations(r, arrs) for r in comp]); CH = [dict(churn(r, arrs), day=r['day'], m=int(r['m'])) for r in comp]
    S_ = state_rows(F, recs, arrs); BT = bin_table(S_, edges); LT, cand = leads(BT, S_, edges)
    statuses = {}
    for r in recs:
        statuses[r['status']] = statuses.get(r['status'], 0) + 1
    gap = G_[G_.n_gap > 0]; live = S_[S_.kind == 'live']; first = S_[S_.kind == 'first']; ent = S_[S_.kind == 'entry']
    byreason = {}
    for c in CH:
        for k, v in c['by_reason'].items():
            byreason[k] = byreason.get(k, 0) + v
    mv = pd.DataFrame([dict(day=r['day'], m=int(r['m']), value=float(r['ledger']['value'])) for r in comp]); neg = -mv.value[mv.value < 0]
    tl = {}
    for kind, g in S_.groupby('kind'):
        un = g[(~g.censored_1s) & g.win_mo1s.notna()]
        tl[kind] = dict(n=int(len(g)), censored_1s=int(g.censored_1s.sum()), win_any=int(g.win_any.sum()), lifetime_any=int(g.lifetime_any.sum()),
                        win_mo1s=float(un.win_mo1s.sum()), pre_mo1s=float(g.pre_mo1s.dropna().sum()), cp_mo1s=float(g.cp_mo1s.dropna().sum()),
                        inv_move_1s=float(g.inv_move_1s.dropna().sum()), inv_move_missing=int(g.inv_move_1s.isna().sum()), latency_median_ms=float(g.latency_ms.median()),
                        pf_age_median_ms=float(g.pf_age.dropna().median()) if g.pf_age.notna().any() else None, bn_age_median_ms=float(g.bn_age.dropna().median()) if g.bn_age.notna().any() else None)
    cross = dict(locked_window=dict(fills=int(gap.n_gap.sum()), generations=int(len(gap)), shares=float(gap.q_gap.sum()), end_value=float(gap.v_gap.sum())),
                 sweeps=dict(trade_grouped_fill_rows=int(SW.trade_event.sum()) if len(SW) else 0, time_grouped_fill_rows=int(SW.time_group.sum()) if len(SW) else 0,
                             time_grouped_but_not_same_trade_rows=int((SW.time_group & ~SW.trade_event).sum()) if len(SW) else 0,
                             trade_event_end_value=float(SW[SW.trade_event].end_value.sum()) if len(SW) else 0.,
                             trade_event_mo1s=float(SW[SW.trade_event].mo1s.dropna().sum()) if len(SW) else 0.),
                 live_fill_denominators=dict(first=dict(n=int(len(first)), any_generation_fill=int(first.lifetime_any.sum()), future_fill=int(first.win_any.sum())),
                                             live=dict(n=int(len(live)), any_generation_fill=int(live.lifetime_any.sum()), future_fill=int(live.win_any.sum())),
                                             entry=dict(n=int(len(ent)), any_generation_fill=int(ent.lifetime_any.sum()), future_fill=int(ent.win_any.sum()))),
                 first_review=dict(reviewed=sum(c['reviewed'] for c in CH), policy_first_review_pulls=sum(c['policy_first_review_pulls'] for c in CH),
                                   requested_by_first_live=sum(c['requested_by_first_live'] for c in CH), by_reason=byreason),
                 inventory_modes=dict(recovered=int((RC.outcome == 'recovered').sum()) if len(RC) else 0, reversal=int((RC.outcome == 'reversal').sum()) if len(RC) else 0,
                                      censored=int((RC.outcome == 'censored').sum()) if len(RC) else 0,
                                      recovered_median_ms=float(RC[RC.outcome == 'recovered'].ms.median()) if len(RC) and (RC.outcome == 'recovered').any() else None),
                 fill_timing=dict(before_request=dict(fills=int(G_.n_before_request.sum()), q=float(G_.q_before_request.sum()), mo1s=float(G_.mo1s_before_request.dropna().sum())),
                                  cancel_pending=dict(fills=int(G_.n_cancel_pending.sum()), q=float(G_.q_cancel_pending.sum()), mo1s=float(G_.mo1s_cancel_pending.dropna().sum()))),
                 durations=dict(both_exact=float(DU.both_ms.sum() / DU.end.sum()), both_grid_mean=float(DU.both_grid.mean()), either_exact=float(DU.either_ms.sum() / DU.end.sum())),
                 net_concentration=dict(losing_markets=int((mv.value < 0).sum()), top5_share=float(np.sort(neg.values)[::-1][:5].sum() / neg.sum()) if len(neg) else None),
                 common_markouts=common_markouts(recs), common_markouts_all_horizons=common_markouts(recs, (0, 2, 4, 6, 8, 10)), generations=int(len(G_)), complete=len(comp),
                 selected=len(recs))
    gm = G_.lifetime_mo1s.dropna()
    cross['generation_tails'] = dict(end_worst=float(G_.lifetime_value.min()), end_severe_sum=float(G_.lifetime_value[G_.lifetime_value < -M.SEVERE_ABS].sum()),
                                     mo1s_worst=float(gm.min()) if len(gm) else None, mo1s_below_minus30_sum=float(gm[gm < -M.SEVERE_ABS].sum()) if len(gm) else None,
                                     mo1s_censored=int(G_.lifetime_mo1s.isna().sum() - (G_.n_fills == 0).sum()))
    for nm, D in (('generations_v2', G_), ('sweep_rows', SW), ('recovery', RC), ('durations', DU), ('churn', pd.DataFrame(CH)), ('states', S_), ('bins', BT), ('leads', LT)):
        D.to_csv(out / f'{nm}.csv', index=False)
    C.dump(out / 'audit.json', dict(version=VERSION, asof=C.stamp(), statuses=statuses, cross=cross, timeline=tl, candidate=cand, edges_source=str(M.OUT / 'measure' / 'audit.json'),
                                    edges_sha256=C.sha(M.OUT / 'measure' / 'audit.json'), outputs={f'{nm}.csv': C.sha(out / f'{nm}.csv') for nm in
                                                                                                  ('generations_v2', 'sweep_rows', 'recovery', 'durations', 'churn', 'states', 'bins', 'leads')},
                                    sources=sources(), run_freeze=RUN_FREEZE))
    print(json.dumps(dict(statuses=statuses, candidate=cand, cross=cross, timeline=tl), indent=1, default=str)[:5000])


def order_causal(w, s, act, price, ft, fq, d, terminal, size, end, rung, rng):
    """known_steps_q at d: frames received after d and fills notified after d perturbed -> (unchanged?); the frame received at d with the
    price's size set to 0 -> (eligible: the price shown there with a positive estimate, changed?)."""
    base = M.known_steps_q(w, s, act, price, (ft, fq), np.array([d]), terminal, size, end, rung=rung)
    kf = np.asarray(w['kf'], float); jt = int(np.searchsorted(kf, d, side='right')) - 1
    P5 = np.array(w['P5'], float, copy=True); S5 = np.array(w['S5'], float, copy=True); S5[jt + 1:] = S5[jt + 1:] * rng.uniform(1.5, 3., S5[jt + 1:].shape) + 1.
    P5[jt + 1:] = np.where(P5[jt + 1:] >= 0, np.clip(P5[jt + 1:] + rng.integers(-2, 3, P5[jt + 1:].shape), 1, 99), P5[jt + 1:])
    fq2 = np.where(np.asarray(ft, float) + NOTICE > d, np.asarray(fq, float) * .5 + 1., fq)
    pert = M.known_steps_q(dict(w, P5=P5, S5=S5), s, act, price, (ft, fq2), np.array([d]), terminal, size, end, rung=rung)
    same = all(np.array_equal(np.asarray(base[k_], float), np.asarray(pert[k_], float), equal_nan=True) for k_ in base)
    S6 = np.array(w['S5'], float, copy=True); P6 = np.asarray(w['P5'], float)
    shown = bool(jt >= 0 and (P6[jt] == price).any() and np.isfinite(base['queue_ahead_known'][0]) and base['queue_ahead_known'][0] > 0)
    if jt >= 0:
        S6[jt][P6[jt] == price] = 0.
    dl = M.known_steps_q(dict(w, S5=S6), s, act, price, (ft, fq), np.array([d]), terminal, size, end, rung=rung)
    fired = bool(shown and not np.array_equal(np.asarray(base['queue_ahead_known'], float), np.asarray(dl['queue_ahead_known'], float), equal_nan=True))
    return same, shown, fired


def sample_snapshots(F, day, n=AUDIT_N):
    g = F[(F.day == day) & (F.status == 'ok')].copy()
    g['_r'] = [hashlib.sha256((PREFIX + 'audit|' + LS.ckey(dict(day=day, m=int(m), kind=k, side=int(s), gid=int(gd), t=int(round(t))))).encode()).hexdigest()
               for m, k, s, gd, t in zip(g.m, g.kind, g.side, g.gid, g.t)]
    return g.sort_values(['_r', 'm', 'kind', 'side', 'gid', 't']).head(n)


def causal_day(day, out=OUT, loader=None):
    """The bounded causal audit of one day (header) -> out/causal_<day>.json; write-once."""
    out = Path(out); p = out / f'causal_{day}.json'; C.require(frozen(), 'not the committed audit'); C.require(not p.exists(), 'exists (write-once)'); out.mkdir(parents=True, exist_ok=True)
    man, recs, arrs, F = M.load_outputs(False); comp = {int(r['m']): r for r in recs if r['day'] == day and r['status'] == 'complete'}
    members = [x for x in man['selected'] if x['day'] == day and int(x['m']) in comp]; smp = sample_snapshots(F[F.m.isin(list(comp))], day)
    prs, fair, depth, feat_fn, allw, opens = (loader or (lambda: M.load_day(day, members, man)))(); rows = []
    for x in smp.to_dict('records'):
        m, s = int(x['m']), int(x['side']); r = comp[m]; W_ = prs[m]; t0 = float(r['path_info']['t_start']); d = t0 + float(x['t'])
        tg = np.asarray(W_[s]['tg'], float); kk = min(int(np.searchsorted(tg, d)), len(tg) - 1); rng = np.random.default_rng([20261121, m, int(round(x['t'])), s])
        same, fired = X.causal_check(W_, fair.get(m), depth, feat_fn, kk, s, rng, X.real_perturb, X.real_fire)
        o = {int(y['gid']): y for y in r['ledger']['orders']}[int(x['gid'])]; end = float(r['ledger']['end'])
        fl = [(float(f[0]), float(f[3])) for f in r['ledger']['fills'] if int(f[1]) == int(x['gid'])]
        ft = np.array([t0 + t for t, _ in fl], float); fq = np.array([q for _, q in fl], float); act = t0 + (float(o['act']))
        osame, shown, ofired = order_causal(W_[s], s, act, float(o['price']), ft, fq, d, t0 + end, float(o['size']), end, float(o['rung']), rng)
        rows.append(dict(m=m, kind=x['kind'], side=s, gid=int(x['gid']), t=float(x['t']), market_same=bool(same), market_fired=bool(fired), order_same=bool(osame),
                         order_fire_eligible=bool(shown), order_fired=ofired))
    C.dump(p, dict(day=day, n=len(rows), market_same=sum(r['market_same'] for r in rows), market_fired=sum(r['market_fired'] for r in rows),
                   order_same=sum(r['order_same'] for r in rows), order_fire_eligible=sum(r['order_fire_eligible'] for r in rows), order_fired=sum(r['order_fired'] for r in rows),
                   rows=rows, sources=sources(), asof=C.stamp()))
    print(json.dumps({k: v for k, v in json.loads(p.read_text()).items() if k != 'rows'}, default=str))


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'selftest':
        import pf_ladder_audit_selftest as ST
        ST.main(sys.modules[__name__])
    elif cmd == 'saved':
        run_saved()
    elif cmd == 'causal':
        causal_day(sys.argv[2])
    else:
        sys.exit('usage: pf_ladder_audit.py selftest | saved | causal <day>')
