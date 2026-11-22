"""INDEPENDENT CHECKS of ARRIVAL-QUEUE-DECOMPOSITION-v1 (declaration.md; session aea5bfb9, 2026-11-22). Recomputed from the stored evidence (the committed
audit states, the pressure receipt's disabled-arm ledgers and venue queue records, the frozen edges, the USER's committed condition table) with
this checker's own code; the module's outputs are read as data, the module is not imported.
 A identity: the frozen files committed; the states file sha256; the parent day records (the manifest's = the data files = the committed
   copies); the USER's table committed; the output hashes;
 B membership: states.csv = the audit's 4,840 live snapshots once each; intervals.csv = I0..I3 once per state;
 F rows: per state the effect, the generation end and cause, the pre-effect / before-decision counts, the queue / Binance / Predict bands and
   the nine view labels, the first-review / entry / recent statuses and dQ, the known remaining, the venue join and its notice gap; per
   interval the follow-up status and at-risk ms, fills, shares, filled-out, END and every horizon value; the union = the audit window;
 C cells: every cells.csv row recomputed from this checker's rows with separate groupbys per day / side level (and the same key set);
 R the USER's table: the committed 20261122_horizon_conditions/conditions.csv reproduced from this checker's union rows (rows, common,
   censored, no-fill, shares, net, state positive / negative, per opportunity, per share; the same key set);
 S summary: audit.json's status counts = this checker's;
 I resources: this round <= its cap; the ladder line (its eight receipts, the ladder reviews and the USER's 2026-11-22 diagnostics) <= 2.0.
falsify(): one corruption per family must fail while the clean evidence passes. Usage: aq_checks.py [outputs_dir manifest result_dir]"""
import collections
import glob
import gzip
import hashlib
import io
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path('/home/yuqing/ctaNew'); R = REPO / 'predictfun/reviews/ladder_arrival_queue_20261122'; D = REPO / 'data/predict_fun/research/ladder_arrival_queue_v1_20261122'
PD_ = REPO / 'data/predict_fun/research/ladder_pressure_v1_20261121'; STATES = REPO / 'predictfun/reviews/ladder_audit_20261121/outputs/states.csv.gz'
USER_T = 'predictfun/reviews/automatic/20261122_horizon_conditions/conditions.csv'
COMMITTED = ['predictfun/research/pf_ladder_arrival_queue.py', 'predictfun/research/pf_ladder_arrival_queue_selftest.py', 'predictfun/research/pf_ladder_arrival_queue_mutants.py',
             'predictfun/reviews/ladder_arrival_queue_20261122/declaration.md', 'predictfun/reviews/ladder_arrival_queue_20261122/manifest.json',
             'predictfun/reviews/ladder_arrival_queue_20261122/launch_arrival_queue.sh', 'predictfun/reviews/ladder_arrival_queue_20261122/aq_checks.py']
LINE = ['ladder_rest_20261121', 'ladder_audit_20261121', 'ladder_guard_20261121', 'ladder_pressure_20261121', 'ladder_pressure_lag_20261121', 'ladder_pressure_ablation_20261121',
        'ladder_pressure_residual_20261121', 'ladder_pressure_positive_20261121']
REVIEWS = ['20261121_ladder_rest_review', '20261121_ladder_audit_review', '20261121_ladder_guard_review', '20261121_ladder_pressure_review', '20261121_ladder_pressure_lag_review',
           '20261121_ladder_pressure_ablation_review', '20261121_ladder_pressure_residual_review', '20261122_horizon_conditions', '20261122_queue_conditions']
RL = re.compile(r'^(\S+) tag=(\S+) rc=(\d+) peak_rss_mib=(\d+) wall_s=([\d.]+) cpu_user_s=([\d.]+)')
H = (0, 50, 100, 1000, 5000, 30000); IV = (('I0', 0., 250.), ('I1', 250., 1000.), ('I2', 1000., 5000.), ('I3', 5000., float('inf'))); SG = {0: 1., 1: -1.}
VIEWS = ('pressure_flow', 'depletion', 'depth', 'age', 'inventory', 'queue', 'queue_pressure', 'queue_pressure_flow', 'progress'); FIVE = VIEWS[:5]
STAT = dict(removal='removed_in_interval', full_fill='filled_out_in_interval', end='end_censored'); LINE_CAP = 2.


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def blob(rel):
    r = subprocess.run(['git', '-C', str(REPO), 'show', f'HEAD:{rel}'], capture_output=True); return r.stdout if r.returncode == 0 else None


def gz(p):
    return [json.loads(x) for x in gzip.decompress(Path(p).read_bytes()).decode().splitlines() if x]


def num(a):
    return float('nan') if a is None or (isinstance(a, str) and a in ('', 'nan')) else float(a)


def close(a, b, tol=1e-6):
    a, b = num(a), num(b)
    return (np.isnan(a) and np.isnan(b)) or (not np.isnan(a) and not np.isnan(b) and abs(a - b) <= tol * max(1., abs(a), abs(b)))


def load(out=None, man=None):
    out = Path(out or D); gm = json.loads(Path(man or R / 'manifest.json').read_text()); E = dict(gm=gm, out=out, led={})
    for day in gm['days']:
        E['led'].update({(day, int(r['m'])): r['disabled'] for r in gz(PD_ / 'days' / f'{day}.jsonl.gz')})
    E['S'] = pd.read_csv(io.BytesIO(gzip.decompress(STATES.read_bytes())), dtype={'day': str}, float_precision='round_trip')
    for n in ('states', 'intervals', 'cells'):
        E[n] = pd.read_csv(out / f'{n}.csv', dtype={'day': str, 'side': str, 'band': str} if n == 'cells' else {'day': str}, float_precision='round_trip')
    E['audit'] = json.loads((out / 'audit.json').read_text()); E['last'] = {int(s): float(v[-1]) for s, v in gm['edges'].items()}
    E['user'] = pd.read_csv(REPO / USER_T, dtype={'day': str, 'side': str, 'band': str}, float_precision='round_trip'); return E


# ---------------------------------------------------------------------------------------------------------------------- this checker's rows
def my_rows(E):
    """This checker's own per-state and per-interval decomposition."""
    S = E['S']; live = S[S.kind == 'live']; other = collections.defaultdict(dict)
    for x in S[S.kind != 'live'].to_dict('records'):
        k = (x['day'], int(x['m']), int(x['gid'])); assert x['kind'] not in other[k], 'two snapshots of one kind'; other[k][x['kind']] = x
    cache = {}; st, iv = [], []
    for z in live.to_dict('records'):
        key = (z['day'], int(z['m'])); L_ = E['led'][key]
        if key not in cache:
            fl = collections.defaultdict(list)
            for f, mm in zip(L_['fills'], L_['fill_mids']):
                fl[int(f[1])].append((f, mm))
            cache[key] = ({int(o['gid']): o for o in L_['orders']}, {g: sorted(v, key=lambda x: (float(x[0][0]), int(x[0][5]))) for g, v in fl.items()})
        g = int(z['gid']); o = cache[key][0][g]; fl = cache[key][1].get(g, []); end, mark = float(L_['end']), float(L_['mark']); d = float(z['t']); act = float(o['act'])
        e = max(d + 37., float(o['submit']) + 95.) if d < act else d + 37.; cum, full = 0., float('inf')
        for f, _ in fl:
            cum += float(f[3])
            if cum >= float(o['size']) - 1e-9:
                full = float(f[0]); break
        G = min(float(o['remove']), full, end); cause = 'full_fill' if full == G else ('end' if end == G else 'removal')
        Q, Rr = num(z['fp_queue']), num(z['fp_remaining'])
        qb = 'invalid_R' if (np.isnan(Rr) or Rr <= 0) else ('unknown' if (np.isnan(Q) or Q < 0) else ('front' if Q == 0 else ('within_R' if Q <= Rr else 'beyond_R')))
        x = num(z['adv_bn_r250']); x32 = float(np.float32(x)) if not np.isnan(x) else x
        bb = 'missing' if np.isnan(x) else ('non_adverse' if x32 <= 0 else ('q5' if x32 >= E['last'][int(z['side'])] else 'positive_below_q5'))
        fpv = num(z['adv_ofi250']); pb = 'missing' if np.isnan(fpv) else ('adverse' if fpv > 0 else 'non_adverse')
        depl, dep, ag, inv = (num(z[c]) for c in ('own_d250', 'o_depth', 'ctx_generation_age_ms', 'inv_signed'))
        lab = dict(pressure_flow=f'{bb}|PF_{pb}', depletion='missing' if np.isnan(depl) else ('depleting' if depl < 0 else 'not_depleting'), depth='missing' if np.isnan(dep) else str(dep),
                   age='missing' if np.isnan(ag) else ('<0.5s' if ag < 500 else ('0.5-2s' if ag < 2000 else ('2-10s' if ag < 10000 else '>=10s'))),
                   inventory='missing' if np.isnan(inv) else ('adding' if inv > 0 else ('reducing' if inv < 0 else 'flat')), queue=qb, queue_pressure=f'{qb}|{bb}', queue_pressure_flow=f'{qb}|{bb}|PF_{pb}')
        same = lambda r_: r_ is not None and int(r_['side']) == int(o['side']) and int(r_['rung']) == round(float(o['rung'])) and float(r_['size']) == float(o['size'])
        pri = {}
        for kind, nm in (('first', 'first_review'), ('entry', 'entry')):
            r_ = other[(z['day'], int(z['m']), g)].get(kind)
            if d < act:
                pri[kind] = ('not_yet_born', float('nan'))
            elif r_ is None:
                pri[kind] = (f'no_{nm}_saved', float('nan'))
            elif not same(r_):
                pri[kind] = ('order_mismatch', float('nan'))
            elif float(r_['t']) > d:
                pri[kind] = ('future_state_refused', float('nan'))
            elif float(r_['t']) == d:
                pri[kind] = (f'{nm}_is_decision', float('nan'))
            elif np.isnan(num(r_['fp_queue'])) or np.isnan(Q) or num(r_['fp_queue']) < 0 or Q < 0:
                pri[kind] = ('unknown_queue', float('nan'))
            else:
                pri[kind] = ('ok', Q - float(r_['fp_queue']))
        lab['progress'] = pri['first'][0] if pri['first'][0] != 'ok' else ('advanced' if pri['first'][1] < 0 else ('unchanged' if pri['first'][1] == 0 else 'increased'))
        fr = other[(z['day'], int(z['m']), g)].get('first')
        rs = 'not_yet_born_at_t_minus_250' if d - 250. < act else ('available' if (fr is not None and float(fr['t']) == d - 250. and same(fr) and not np.isnan(num(fr['fp_queue'])) and not np.isnan(Q))
                                                                    else 'unavailable_no_saved_state')
        vf = L_['queue_venue']['first'].get(str(g))
        if fr is None:
            vs, gap_ok = 'no_first_review_saved', None
        elif vf is None:
            vs, gap_ok = 'no_venue_entry', None
        elif float(vf[0]) != float(fr['t']):
            vs, gap_ok = 'time_mismatch', None
        elif not same(fr):
            vs, gap_ok = 'order_mismatch', None
        else:
            un = sum(float(f[3]) for f, _ in fl if float(fr['t']) - 100. < float(f[0]) <= float(fr['t'])); vs = 'available'
            gap_ok = bool(abs((float(fr['fp_remaining']) - float(vf[2])) - un) <= 1e-6 * max(1., abs(float(fr['fp_remaining']))))
        kr = max(0., float(o['size']) - sum(float(f[3]) for f, _ in fl if float(f[0]) + 100. <= d))
        pre = sum(1 for f, _ in fl if d <= float(f[0]) < e); before = sum(1 for f, _ in fl if float(f[0]) < d)
        st.append(dict(day=z['day'], m=int(z['m']), gid=g, side=int(z['side']), effect=e, gen_end=G, gen_end_cause=cause, queue_band=qb, bn_band=bb, pf_band=pb, **lab,
                       first_status=pri['first'][0], dQ_first=pri['first'][1], entry_status=pri['entry'][0], dQ_from_entry=pri['entry'][1], recent_status=rs, venue_status=vs,
                       notice_gap_agrees=gap_ok, R_recomputed=kr, R_agrees=bool(not np.isnan(Rr) and abs(kr - Rr) <= 1e-6 * max(1., Rr)), pre_effect_fills=pre, before_decision_fills=before))
        for name, a, b in IV:
            lo, hi = e + a, e + b
            na = (G < lo) if cause == 'full_fill' else (G <= lo)
            fu, ms_ = ('not_at_risk', 0.) if na else (('at_risk_full', hi - lo) if G >= hi else (STAT[cause], G - lo))
            mine = [(f, mm) for f, mm in fl if lo <= float(f[0]) < hi]
            r = dict(day=z['day'], m=int(z['m']), gid=g, interval=name, cause=cause, follow_up=fu, at_risk_ms=ms_, fills=len(mine), shares=sum(float(f[3]) for f, _ in mine),
                     filled_out=any(cause == 'full_fill' and float(f[0]) == G for f, _ in mine), vEND=sum(SG[int(f[2])] * float(f[3]) * (mark - float(f[4])) for f, _ in mine))
            for j, h in enumerate(H):
                vals = [None if (mm[2 * j] is None or float(f[0]) + h > end) else SG[int(f[2])] * float(f[3]) * (float(mm[2 * j]) - float(f[4])) for f, mm in mine]
                r[f'ok{h}'] = all(v is not None for v in vals); r[f'v{h}'] = float(sum(vals)) if r[f'ok{h}'] else float('nan')
                r[f'fpos{h}'] = float(sum(max(v, 0.) for v in vals if v is not None)); r[f'fneg{h}'] = float(sum(min(v, 0.) for v in vals if v is not None))
            r['common'] = all(r[f'ok{h}'] for h in H); iv.append(r)
    return pd.DataFrame(st), pd.DataFrame(iv)


def my_union(mi):
    """Per state the union of its four interval rows (this checker's code)."""
    out = []
    for k, g in mi.groupby(['day', 'm', 'gid'], sort=False):
        g = g.set_index('interval').loc[['I0', 'I1', 'I2', 'I3']]; ar = any(s != 'not_at_risk' for s in g.follow_up)
        r = dict(day=k[0], m=k[1], gid=k[2], interval='union', follow_up=STAT[g.cause.iloc[0]] if ar else 'not_at_risk', fills=int(g.fills.sum()), shares=float(g.shares.sum()), filled_out=bool(g.filled_out.any()),
                 vEND=float(g.vEND.sum()), common=bool(g.common.all()))
        for h in H:
            r[f'ok{h}'] = bool(g[f'ok{h}'].all()); r[f'v{h}'] = float(g[f'v{h}'].sum()) if r[f'ok{h}'] else float('nan')
            r[f'fpos{h}'] = float(g[f'fpos{h}'].sum()); r[f'fneg{h}'] = float(g[f'fneg{h}'].sum())
        out.append(r)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------------------------------------------------- families
def fam_A(E):
    gm = E['gm']; c = dict(A_committed=all(blob(p) == (REPO / p).read_bytes() for p in COMMITTED))
    c['A_inputs'] = bool(sha(STATES) == gm['states']['sha256'] and all(sha(PD_ / 'days' / f'{d}.json') == gm['parent_day_records'][d]
                                                                       and blob(f'predictfun/reviews/ladder_pressure_20261121/outputs/days/{d}.json') == (PD_ / 'days' / f'{d}.json').read_bytes()
                                                                       for d in gm['days']) and blob(USER_T) == (REPO / USER_T).read_bytes())
    c['A_outputs'] = all(E['audit']['outputs'][n] == sha(E['out'] / n) for n in ('states.csv', 'intervals.csv', 'cells.csv')); return c, {}


def fam_B(E):
    live = E['S'][E['S'].kind == 'live']; want = sorted(zip(live.day, live.m.astype(int), live.gid.astype(int))); got = sorted(zip(E['states'].day, E['states'].m.astype(int), E['states'].gid.astype(int)))
    I_ = E['intervals']; per = I_.groupby(['day', 'm', 'gid']).interval.apply(lambda x: tuple(sorted(x)))
    return dict(B_membership=bool(got == want and len(got) == len(set(got)) and len(I_) == 4 * len(want) and len(per) == len(want) and (per == ('I0', 'I1', 'I2', 'I3')).all())), dict(B=dict(states=len(want)))


def fam_F(E):
    ms, mi = E['mine']; S_, I_ = E['states'], E['intervals']; c = {}
    J = ms.merge(S_, on=['day', 'm', 'gid'], suffixes=('', '_s'), how='left')
    lab = ['gen_end_cause', 'queue_band', 'bn_band', 'pf_band', 'first_status', 'entry_status', 'recent_status', 'venue_status'] + list(VIEWS)
    c['F_state_labels'] = bool(len(J) == len(ms) and all((J[k].astype(str) == J[f'{k}_s'].astype(str)).all() for k in lab))
    c['F_state_values'] = bool(all(all(close(a, b) for a, b in zip(J[k], J[f'{k}_s'])) for k in ('effect', 'gen_end', 'dQ_first', 'dQ_from_entry', 'R_recomputed'))
                               and (J.pre_effect_fills == J.pre_effect_fills_s).all() and (J.before_decision_fills == J.before_decision_fills_s).all()
                               and (J.R_agrees.astype(bool) == J.R_agrees_s.astype(bool)).all()
                               and (J.notice_gap_agrees.astype(str) == J.notice_gap_agrees_s.map(lambda v: 'None' if pd.isna(v) else str(bool(v)))).all())
    K = mi.merge(I_, on=['day', 'm', 'gid', 'interval'], suffixes=('', '_s'), how='left')
    c['F_intervals'] = bool(len(K) == len(mi) and (K.follow_up == K.follow_up_s).all() and (K.cause == K.cause_s).all() and (K.fills == K.fills_s).all() and (K.filled_out.astype(bool) == K.filled_out_s.astype(bool)).all()
                            and (K.common.astype(bool) == K.common_s.astype(bool)).all()
                            and all(all(close(a, b) for a, b in zip(K[k], K[f'{k}_s'])) for k in ['at_risk_ms', 'shares', 'vEND'] + [f'{p}{h}' for p in ('v', 'fpos', 'fneg') for h in H]))
    W = E['S'][E['S'].kind == 'live'].assign(m=lambda x: x.m.astype(int), gid=lambda x: x.gid.astype(int)).set_index(['day', 'm', 'gid']); U = E['mine_u'].set_index(['day', 'm', 'gid'])
    JJ = W.join(U, how='left')
    c['F_union_window'] = bool(all(int(a) == int(b) for a, b in zip(JJ.fills, JJ.win_n)) and all(close(a, b) for a, b in zip(JJ.shares, JJ.win_q))
                               and all(close(a, b) for a, b in zip(JJ.vEND, JJ.win_end)) and all(close(a, b) for a, b in zip(JJ.v1000, JJ.win_mo1s))
                               and all(close(a, b) for a, b in zip(JJ.v50, JJ.win_mo50)) and all(close(a, b) for a, b in zip(JJ.v100, JJ.win_mo100)))
    return c, {}


def my_cells(ms, rows):
    """cells.csv's quantities by separate groupbys per level (this checker's code)."""
    X_ = rows.merge(ms[['day', 'm', 'gid', 'side'] + list(VIEWS)], on=['day', 'm', 'gid'], how='left'); X_['side'] = X_.side.astype(str)
    X_['risk'] = X_.follow_up != 'not_at_risk'; X_['anyf'] = X_.fills > 0; X_['part'] = X_.anyf & ~X_.filled_out.astype(bool); X_['mult'] = X_.fills > 1
    for coh in ('native', 'common'):
        for h in H:
            okc = (X_[f'ok{h}'] if coh == 'native' else X_['common']).astype(bool); vv = np.where(okc, X_[f'v{h}'].fillna(0.), 0.)
            X_[f'{coh}{h}_n'] = okc.astype(int); X_[f'{coh}{h}_net'] = vv; X_[f'{coh}{h}_sp'] = np.maximum(vv, 0.); X_[f'{coh}{h}_fp'] = np.where(okc, X_[f'fpos{h}'], 0.)
            X_[f'{coh}{h}_fn'] = np.where(okc, X_[f'fneg{h}'], 0.); X_[f'{coh}{h}_q'] = np.where(okc, X_.shares, 0.)
    sums = ['risk', 'anyf', 'fills', 'shares', 'part', 'mult', 'vEND'] + [f'{coh}{h}_{x}' for coh in ('native', 'common') for h in H for x in ('n', 'net', 'sp', 'fp', 'fn', 'q')]
    out = {}
    for view in VIEWS:
        for keys, dk, sk in (([view, 'interval', 'day', 'side'], None, None), ([view, 'interval', 'side'], 'all', None), ([view, 'interval', 'day'], None, 'all'), ([view, 'interval'], 'all', 'all')):
            G = X_.groupby(keys).agg(n=('gid', 'size'), **{s: (s, 'sum') for s in sums}).reset_index()
            for r in G.to_dict('records'):
                out[(view, str(r[view]), r['interval'], dk or r['day'], sk or str(r['side']))] = r
    return out


def fam_C(E):
    ms, mi = E['mine']; mc = my_cells(ms, pd.concat([mi, E['mine_u']], ignore_index=True)); T = E['cells']; bad = 0
    keys = set(zip(T.view, T.band.astype(str), T.interval, T.day, T.side.astype(str)))
    for r in T.itertuples(index=False):
        a = mc.get((r.view, str(r.band), r.interval, r.day, str(r.side)))
        if a is None:
            bad += 1; continue
        coh = r.cohort; ok_ = (a['n'] == r.opportunities and a['risk'] == r.at_risk and a['anyf'] == r.any_fill and a['fills'] == r.fills and close(a['shares'], r.shares)
                               and a['part'] == r.partial and a['mult'] == r.multi_fill and close(a['vEND'], r.end_net))
        for h in H:
            ok_ = ok_ and a[f'{coh}{h}_n'] == getattr(r, f'h{h}_valid') and close(a[f'{coh}{h}_net'], getattr(r, f'h{h}_net')) and close(a[f'{coh}{h}_sp'], getattr(r, f'h{h}_state_positive')) \
                and close(a[f'{coh}{h}_fp'], getattr(r, f'h{h}_fill_positive')) and close(a[f'{coh}{h}_fn'], getattr(r, f'h{h}_fill_negative')) and close(a[f'{coh}{h}_q'], getattr(r, f'h{h}_valid_shares'))
        bad += int(not ok_)
    return dict(C_cells=bool(bad == 0 and keys == set(mc) and len(T) == 2 * len(mc))), dict(C=dict(rows=int(len(T)), bad=bad, keys_equal=keys == set(mc)))


def fam_R(E):
    """The USER's conditions.csv from this checker's union rows."""
    ms = E['mine'][0]; U = E['mine_u'].merge(ms[['day', 'm', 'gid', 'side'] + list(FIVE)], on=['day', 'm', 'gid']); U['com'] = U.ok1000 & U.ok5000 & U.ok30000; mine = {}
    for col in FIVE:
        for band, gg in U.groupby(col):
            for day in ['all'] + sorted(gg.day.unique()):
                for side in ('all', 0, 1):
                    g = gg if day == 'all' else gg[gg.day == day]; g = g if side == 'all' else g[g.side == side]; u = g[g.com]
                    for h in (1000, 5000, 30000):
                        v = u[f'v{h}']; q = float(u.shares.sum())
                        mine[(col, str(band), day, str(side), str(h))] = dict(rows=len(g), common=len(u), censored=len(g) - len(u), no_fill=int((u.fills == 0).sum()), q=q, net=float(v.sum()),
                                                                              positive=float(v[v > 0].sum()), negative=float(v[v < 0].sum()), per_opportunity=float(v.mean()) if len(u) else float('nan'),
                                                                              per_share=float(v.sum()) / q if q else float('nan'))
    T = E['user']; bad = 0
    for r in T.to_dict('records'):
        a = mine.get((r['condition'], str(r['band']), r['day'], str(r['side']), str(int(r['horizon_ms']))))
        bad += int(a is None or not all((a[k] == r[k]) if k in ('rows', 'common', 'censored', 'no_fill') else close(a[k], r[k]) for k in a))
    return dict(R_user_table=bool(bad == 0 and len(T) == len(mine) and len(T) > 0)), dict(R=dict(rows=int(len(T)), mine=len(mine), bad=bad))


def fam_S(E):
    ms, _ = E['mine']; U = E['mine_u']; s = E['audit']['summary']; c = {}
    vc = lambda x: {str(k): int(v) for k, v in x.value_counts().items()}
    want = dict(queue_bands=vc(ms.queue_band), first_review=vc(ms.first_status), entry=vc(ms.entry_status), recent_250=vc(ms.recent_status), venue=vc(ms.venue_status),
                gen_end_cause=vc(ms.gen_end_cause), union_follow_up=vc(U.follow_up))
    nums = dict(venue_notice_gap_agrees=int(ms.notice_gap_agrees.map(lambda v: v is True).sum()), known_remaining_agree=int(ms.R_agrees.sum()), union_common_all_horizons=int(U.common.sum()),
                union_common_1_5_30=int((U.ok1000 & U.ok5000 & U.ok30000).sum()), union_no_fill=int((U.fills == 0).sum()), pre_effect_fills=int(ms.pre_effect_fills.sum()),
                before_decision_fills=int(ms.before_decision_fills.sum()))
    c['S_summary'] = bool(all({str(k): int(v) for k, v in s[k].items()} == w for k, w in want.items()) and all(int(s[k]) == v for k, v in nums.items()))
    return c, dict(S=dict(nums, **want))


def cpu(paths):
    tot, peak = 0., 0
    for p in paths:
        for ln in Path(p).read_text(errors='replace').splitlines():
            m_ = RL.match(ln)
            if m_:
                tot += float(m_.group(6)); peak = max(peak, int(m_.group(4)))
    return tot / 3600., peak


def fam_I(E, root=None):
    own, peak = cpu(glob.glob(str((root or R) / 'logs' / '**' / '*.log'), recursive=True))
    line, _ = cpu([p for n in LINE for p in glob.glob(str(REPO / 'predictfun/reviews' / n / 'logs' / '**' / '*.log'), recursive=True)])
    rev, _ = cpu([p for n in REVIEWS for p in glob.glob(str(REPO / 'predictfun/reviews/automatic' / n / '*.log'))]); cap = float(E['gm']['cap_core_hours'])
    return dict(I_resources=bool(own <= cap and peak < 1536 and own + line + rev <= LINE_CAP)), dict(I=dict(round_core_hours=own, cap=cap, line_core_hours=own + line + rev, peak_mib=peak))


FAMS = dict(A=fam_A, B=fam_B, F=fam_F, C=fam_C, R=fam_R, S=fam_S, I=fam_I)


def run_checks(E):
    chk, det = {}, {}
    for k in FAMS:
        c, d = FAMS[k](E); chk.update(c); det.update(d)
    return dict(checks=chk, details=det, all_ok=bool(all(chk.values())))


def _fake(E):
    E['_root'] = Path(tempfile.mkdtemp(dir=str(E['out']))); (E['_root'] / 'logs').mkdir()
    (E['_root'] / 'logs' / 'fake.log').write_text('2026-11-22T00:00:00Z tag=fake rc=0 peak_rss_mib=2000 wall_s=400.0 cpu_user_s=400.0\n')


def _bump(df, col, by, where=None):
    df = df.copy(); i = df.index[where(df)][0] if where is not None else df.index[0]; df.loc[i, col] = df.loc[i, col] + by; return df


CASES = [
    ('an_output_hash_changed', 'A', 'A_outputs', lambda E: E.__setitem__('audit', dict(E['audit'], outputs=dict(E['audit']['outputs'], **{'cells.csv': '0' * 64})))),
    ('a_state_duplicated', 'B', 'B_membership', lambda E: E.__setitem__('states', pd.concat([E['states'], E['states'].iloc[[0]]], ignore_index=True))),
    ('a_band_changed', 'F', 'F_state_labels', lambda E: E.__setitem__('states', E['states'].assign(queue_band=np.where(E['states'].index == 0, 'unknown', E['states'].queue_band)))),
    ('an_effect_changed', 'F', 'F_state_values', lambda E: E.__setitem__('states', _bump(E['states'], 'effect', 1.))),
    ('a_fill_moved_interval', 'F', 'F_intervals', lambda E: E.__setitem__('intervals', _bump(E['intervals'], 'fills', 1))),
    ('a_value_changed', 'F', 'F_intervals', lambda E: E.__setitem__('intervals', _bump(E['intervals'], 'v1000', 1., lambda d: d.v1000.notna() & (d.fills > 0)))),
    ('a_union_changed', 'F', 'F_union_window', lambda E: E.__setitem__('mine_u', _bump(E['mine_u'], 'v1000', 1., lambda d: d.v1000.notna() & (d.fills > 0)))),
    ('a_cell_changed', 'C', 'C_cells', lambda E: E.__setitem__('cells', _bump(E['cells'], 'h5000_net', 1., lambda d: d.interval == 'I1'))),
    ('a_cell_dropped', 'C', 'C_cells', lambda E: E.__setitem__('cells', E['cells'].iloc[1:])),
    ('a_user_value_changed', 'R', 'R_user_table', lambda E: E.__setitem__('user', _bump(E['user'], 'net', 1.))),
    ('a_summary_count_changed', 'S', 'S_summary', lambda E: E.__setitem__('audit', dict(E['audit'], summary=dict(E['audit']['summary'], union_no_fill=E['audit']['summary']['union_no_fill'] + 1)))),
    ('resources_over_the_cap', 'I', 'I_resources', _fake),
]


def falsify(E, base):
    out = {}
    for name, fam, target, fn in CASES:
        X_ = dict(E); fn(X_)
        try:
            c, _ = FAMS[fam](X_, X_['_root']) if fam == 'I' else FAMS[fam](X_)
        finally:
            if '_root' in X_:
                shutil.rmtree(X_['_root'], ignore_errors=True)
        out[name] = dict(family=fam, target=target, clean=bool(base['checks'][target]), refused=not c[target])
    return dict(cases=out, falsifier_ok=bool(base['all_ok'] and all(v['clean'] and v['refused'] for v in out.values())))


def main(argv):
    out, man, res = (argv + [None, None, None])[:3]; E = load(out, man); E['mine'] = my_rows(E); E['mine_u'] = my_union(E['mine'][1])
    rd = Path(res or R); r_ = run_checks(E); fz = falsify(E, r_)
    for n, obj in (('checks.json', r_), ('checks_falsifier.json', fz)):
        p = rd / n; assert not p.exists(), f'{p} exists (write-once)'; p.write_text(json.dumps(obj, indent=1, default=str) + '\n')
    print(json.dumps(dict(all_ok=r_['all_ok'], failed=[k for k, v in r_['checks'].items() if not v], falsifier_ok=fz['falsifier_ok'],
                          cases={k: (v['clean'], v['refused']) for k, v in fz['cases'].items()}, details=r_['details']), indent=1, default=str)[:4000])


if __name__ == '__main__':
    main(sys.argv[1:])
