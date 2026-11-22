"""INDEPENDENT CHECKS of QUEUE-PRICE-STATE-AUDIT-v1 (declaration.md; session aea5bfb9, 2026-11-22). Every reused field, label, cell and reading is recomputed
from the hash-bound sources with this checker's own code (the module and the reader are not imported); recomputation happens once and the
falsifier re-runs only the comparisons. Closes review 72a36de9 finding 3 for this receipt.
 A identity: the frozen files committed; the ARRIVAL-QUEUE copies (raw sha256 = its committed audit.json), the feature parquets (= the
   committed day records), the audit states, the pressure ledgers (day record = its committed copy AND the gzip bytes = the record's
   sha256['gz']); the output hashes and the receipt copies;
 B membership: states.csv = ARRIVAL-QUEUE's 4,840 states once each;
 U reused fields: Q / R / o_depth / fp_shown / t_to_end / rung / price / size from the exact-key feature rows; Q / R; o_q_rel, bn_band (the
   frozen edges), pf_band from the audit live rows; first_status, venue_status and client_minus_venue from the audit first rows and the
   ledgers' venue records at the exact tick; the queue band;
 G labels: price_state, display, principal and the four view labels;
 C cells: every cells.csv row and column (all-state counts, cohort counts, rates, values, per share / per opportunity, sparse,
   censored_from_common, the follow-up and time-to-END fields) from this checker's own union and groupbys; the same key set;
 P readings: readings.json's declared readings from this checker's cells and states;
 I resources: this round <= its cap, peak < 1,536 MiB, the ladder line <= 2.0 (the reviews included).
falsify(): ten corruptions, among them the review's two demonstrated escapes (zeroed client_minus_venue; a negated h1000_per_share), must
each be refused while the clean evidence passes. Usage: qp_checks.py [outputs_dir manifest readings_dir result_dir]"""
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

REPO = Path('/home/yuqing/ctaNew'); R = REPO / 'predictfun/reviews/ladder_queue_price_20261122'; D = REPO / 'data/predict_fun/research/ladder_queue_price_v1_20261122'
AQ = 'predictfun/reviews/ladder_arrival_queue_20261122'; REST = 'predictfun/reviews/ladder_rest_20261121/outputs/days'; PRS = 'predictfun/reviews/ladder_pressure_20261121/outputs/days'
FEAT = REPO / 'data/predict_fun/research/ladder_rest_v1_20261121/days'; LED = REPO / 'data/predict_fun/research/ladder_pressure_v1_20261121/days'
AUDIT_STATES = 'predictfun/reviews/ladder_audit_20261121/outputs/states.csv.gz'; AUDIT_SHA = 'ec2eaf7cd53628170fe485bb55a9889ae77e7f85188e592b078de6f1ec9a3043'
DAYS = ('20261024', '20261025', '20261026', '20261027'); H = (0, 1000, 5000, 30000); IVS = ('I0', 'I1', 'I2', 'I3'); K = ['day', 'm', 'gid']
VIEWS = ('price_display', 'price_display_queue', 'behind_depth_queue', 'principal_pressure'); BANDS = ('front', 'within_R', 'beyond_R'); PR = 'at_touch|shown_positive'
FOLLOW = ('not_at_risk', 'at_risk_full', 'removed_in_interval', 'filled_out_in_interval', 'end_censored')
STAT = dict(removal='removed_in_interval', full_fill='filled_out_in_interval', end='end_censored')
COMMITTED = ['predictfun/research/pf_ladder_queue_price.py', 'predictfun/research/pf_ladder_queue_price_selftest.py', 'predictfun/research/pf_ladder_queue_price_mutants.py'] + \
    [f'predictfun/reviews/ladder_queue_price_20261122/{n}' for n in ('declaration.md', 'manifest.json', 'launch_queue_price.sh', 'qp_checks.py', 'tools/read_queue_price.py', 'tools/projection_qp.py')]
LINE = ['ladder_rest_20261121', 'ladder_audit_20261121', 'ladder_guard_20261121', 'ladder_pressure_20261121', 'ladder_pressure_lag_20261121', 'ladder_pressure_ablation_20261121',
        'ladder_pressure_residual_20261121', 'ladder_pressure_positive_20261121', 'ladder_arrival_queue_20261122']
REVIEWS = ['20261121_ladder_rest_review', '20261121_ladder_audit_review', '20261121_ladder_guard_review', '20261121_ladder_pressure_review', '20261121_ladder_pressure_lag_review',
           '20261121_ladder_pressure_ablation_review', '20261121_ladder_pressure_residual_review', '20261122_horizon_conditions', '20261122_queue_conditions', '20261122_arrival_queue_review']
RL = re.compile(r'^(\S+) tag=(\S+) rc=(\d+) peak_rss_mib=(\d+) wall_s=([\d.]+) cpu_user_s=([\d.]+)'); NAN = float('nan')


def sha(b):
    return hashlib.sha256(b).hexdigest()


def blob(rel):
    r = subprocess.run(['git', '-C', str(REPO), 'show', f'HEAD:{rel}'], capture_output=True); return r.stdout if r.returncode == 0 else None


def same(a, b, tol=1e-9):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return bool(a.shape == b.shape and np.allclose(a, b, rtol=tol, atol=1e-6, equal_nan=True))


# ---------------------------------------------------------------------------------------------------------------------- the sources, once
def load(out=None, man=None, rdir=None):
    E = dict(out=Path(out or D), rdir=Path(rdir or R / 'outputs')); E['gm'] = json.loads(Path(man or R / 'manifest.json').read_text())
    E['audit'] = json.loads((E['out'] / 'audit.json').read_text()); E['readings'] = json.loads((E['rdir'] / 'readings.json').read_text())
    E['states'] = pd.read_csv(E['out'] / 'states.csv', dtype={'day': str}, float_precision='round_trip')
    E['cells'] = pd.read_csv(E['out'] / 'cells.csv', dtype={'day': str, 'side': str, 'band': str}, float_precision='round_trip')
    aq_audit = json.loads(blob(f'{AQ}/outputs/audit.json')); src = {}
    for n in ('states.csv', 'intervals.csv'):
        raw = gzip.decompress(blob(f'{AQ}/outputs/{n}.gz')); src[f'aq_{n}'] = sha(raw) == aq_audit['outputs'][n]
        E[f'aq_{n}'] = pd.read_csv(io.BytesIO(raw), dtype={'day': str}, float_precision='round_trip')
    ab = blob(AUDIT_STATES); src['audit_states'] = sha(ab) == AUDIT_SHA; A_ = pd.read_csv(io.BytesIO(gzip.decompress(ab)), dtype={'day': str}, float_precision='round_trip')
    E['live'] = A_[A_.kind == 'live'].set_index(K); E['first'] = A_[A_.kind == 'first'].set_index(K); feats, venue = [], {}
    for d in DAYS:
        rb = blob(f'{REST}/{d}.json'); fb = (FEAT / f'{d}_features.parquet').read_bytes()
        src[f'feat_{d}'] = (FEAT / f'{d}.json').read_bytes() == rb and sha(fb) == json.loads(rb)['sha256']['feat']
        f = pd.read_parquet(io.BytesIO(fb)); f['day'] = f.day.astype(str); feats.append(f[f.kind == 'live'])
        pb = blob(f'{PRS}/{d}.json'); gb = (LED / f'{d}.jsonl.gz').read_bytes(); src[f'ledger_{d}'] = (LED / f'{d}.json').read_bytes() == pb and sha(gb) == json.loads(pb)['sha256']['gz']
        for ln in gzip.decompress(gb).decode().splitlines():
            if ln:
                r = json.loads(ln); venue.update({(d, int(r['m']), int(g)): v for g, v in r['disabled']['queue_venue']['first'].items()})
    E['feat'] = pd.concat(feats, ignore_index=True); E['venue'] = venue; E['src'] = src
    E['edges'] = {int(s): float(v[-1]) for s, v in json.loads(blob(f'{AQ}/manifest.json'))['edges'].items()}; return E


def fband(Q, Rr):
    return 'invalid_R' if not np.isfinite(Rr) or Rr <= 0 else ('unknown' if not np.isfinite(Q) or Q < 0 else ('front' if Q == 0 else ('within_R' if Q <= Rr else 'beyond_R')))


def my_states(E):
    """This checker's per-state fields and labels from the sources (vectorized joins; exact keys)."""
    S0 = E['aq_states.csv'][K + ['t', 'side']].copy(); F = E['feat']
    J = S0.merge(F[K + ['t', 'side', 'fp_queue', 'fp_remaining', 'o_depth', 'fp_shown', 't_to_end', 'rung', 'price', 'fp_size']], on=K + ['t', 'side'], how='left')
    lv = E['live'].reset_index()[K + ['o_q_rel', 'adv_bn_r250', 'adv_ofi250']]; fr = E['first'].reset_index()[K + ['t', 'fp_queue']].rename(columns={'t': 't_first', 'fp_queue': 'Q_first'})
    vn = pd.DataFrame([dict(day=k[0], m=k[1], gid=k[2], t_venue=float(v[0]), venue_ahead=float(v[1])) for k, v in E['venue'].items()])
    J = J.merge(lv, on=K, how='left').merge(fr, on=K, how='left').merge(vn, on=K, how='left')
    Q, Rr, dep, sh = J.fp_queue.to_numpy(float), J.fp_remaining.to_numpy(float), J.o_depth.to_numpy(float), J.fp_shown.to_numpy(float)
    x = J.adv_bn_r250.to_numpy(float); x32 = x.astype(np.float32).astype(float); last = J.side.map(E['edges']).to_numpy(float)
    M_ = pd.DataFrame(dict(day=J.day, m=J.m, gid=J.gid, t=J.t, side=J.side, rung=J.rung, price=J.price, size=J.fp_size, o_depth=dep, fp_shown=sh, Q=Q, R=Rr,
                           Q_over_R=np.where(np.isfinite(Q) & np.isfinite(Rr) & (Rr > 0), Q / np.where(Rr > 0, Rr, 1.), np.nan), o_q_rel=J.o_q_rel, t_to_end=J.t_to_end))
    M_['queue_band'] = [fband(a, b) for a, b in zip(Q, Rr)]
    M_['bn_band'] = np.where(~np.isfinite(x), 'missing', np.where(x32 <= 0, 'non_adverse', np.where(x32 >= last, 'q5', 'positive_below_q5')))
    pf = J.adv_ofi250.to_numpy(float); M_['pf_band'] = np.where(~np.isfinite(pf), 'missing', np.where(pf > 0, 'adverse', 'non_adverse'))
    M_['price_state'] = np.where(~np.isfinite(dep), 'unknown', np.where(dep < 0, 'inside_best', np.where(dep == 0, 'at_touch', 'behind_best')))
    M_['display'] = np.where(~np.isfinite(sh), 'shown_unknown', np.where(sh == 0, 'shown_zero', np.where(sh > 0, 'shown_positive', 'shown_invalid')))
    M_['principal'] = (M_.price_state == 'at_touch') & (M_.display == 'shown_positive')
    has_f, has_v = J.t_first.notna().to_numpy(), J.t_venue.notna().to_numpy(); at = has_f & has_v & (J.t_venue.to_numpy(float) == J.t_first.to_numpy(float))
    M_['first_status'] = np.where(~has_f, 'no_first_review_saved', np.where(J.t_first.to_numpy(float) == J.t.to_numpy(float), 'first_review_is_decision', 'ok'))
    M_['venue_status'] = np.where(~has_f, 'no_first_review_saved', np.where(~has_v, 'no_venue_entry', np.where(at, 'available', 'time_mismatch')))
    M_['client_minus_venue'] = np.where(at, J.Q_first.to_numpy(float) - J.venue_ahead.to_numpy(float), np.nan)
    M_['price_display'] = M_.price_state + '|' + M_.display; M_['price_display_queue'] = M_.price_display + '|' + M_.queue_band
    M_['behind_depth_queue'] = np.where((M_.price_state == 'behind_best') & (M_.display == 'shown_positive'), 'depth=' + M_.o_depth.astype(str) + '|' + M_.queue_band, 'outside')
    M_['principal_pressure'] = np.where(M_.principal, M_.queue_band + '|' + M_.bn_band, 'outside'); return M_


def my_cells(M_, I_):
    """cells.csv from this checker's own union (a pivot over the four intervals) and per-level groupbys."""
    P_ = I_.pivot(index=K, columns='interval'); idx = P_.index; ivs = list(IVS)
    at = (P_['follow_up'][ivs] != 'not_at_risk').any(axis=1).to_numpy(); cause = P_['cause']['I0'].to_numpy()
    U = dict(day=idx.get_level_values(0), m=idx.get_level_values(1), gid=idx.get_level_values(2), interval='union',
             follow_up=np.where(at, pd.Series(cause).map(STAT).to_numpy(), 'not_at_risk'), fills=P_['fills'][ivs].sum(axis=1).to_numpy(),
             shares=P_['shares'][ivs].sum(axis=1).to_numpy(), filled_out=P_['filled_out'][ivs].astype(bool).any(axis=1).to_numpy(), common=P_['common'][ivs].astype(bool).all(axis=1).to_numpy())
    for h in H:
        okh = P_[f'ok{h}'][ivs].astype(bool).all(axis=1).to_numpy(); U[f'ok{h}'] = okh
        U[f'v{h}'] = np.where(okh, P_[f'v{h}'][ivs].sum(axis=1, skipna=True).to_numpy(), np.nan)
        U[f'fpos{h}'] = P_[f'fpos{h}'][ivs].sum(axis=1).to_numpy(); U[f'fneg{h}'] = P_[f'fneg{h}'][ivs].sum(axis=1).to_numpy()
    X = pd.concat([I_, pd.DataFrame(U)], ignore_index=True).merge(M_[K + ['side', 't_to_end'] + list(VIEWS)], on=K, how='left')
    cols = dict(side=X.side.astype(str), risk=X.follow_up != 'not_at_risk', filled=X.fills > 0, fo=X.filled_out.astype(bool), lt30=X.t_to_end < 30000.)
    cols['part'] = cols['filled'] & ~cols['fo']
    for st in FOLLOW:
        cols[f's_{st}'] = X.follow_up == st
    for c in ('native', 'common'):
        for h in H:
            o = (X[f'ok{h}'] if c == 'native' else X['common']).astype(bool); v = np.where(o, X[f'v{h}'].fillna(0.), 0.)
            cols.update({f'{c}{h}_n': o, f'{c}{h}_r': o & cols['risk'], f'{c}{h}_f': o & cols['filled'], f'{c}{h}_q': np.where(o, X['shares'], 0.), f'{c}{h}_net': v,
                         f'{c}{h}_sp': np.maximum(v, 0.), f'{c}{h}_sn': np.minimum(v, 0.), f'{c}{h}_fp': np.where(o, X[f'fpos{h}'], 0.), f'{c}{h}_fn': np.where(o, X[f'fneg{h}'], 0.)})
    X = pd.concat([X.drop(columns=['side']), pd.DataFrame(cols, index=X.index)], axis=1)
    sums = ['risk', 'filled', 'fills', 'shares', 'part', 'fo', 'lt30'] + [f's_{st}' for st in FOLLOW] + [f'{c}{h}_{x}' for c in ('native', 'common') for h in H
                                                                                                            for x in ('n', 'r', 'f', 'q', 'net', 'sp', 'sn', 'fp', 'fn')]
    parts = []
    for view in VIEWS:
        for keys, dk, sk in (([view, 'interval', 'day', 'side'], None, None), ([view, 'interval', 'side'], 'all', None), ([view, 'interval', 'day'], None, 'all'), ([view, 'interval'], 'all', 'all')):
            G = X.groupby(keys); A = G[sums].sum(); A['n_'] = G.size(); A['med'] = G['t_to_end'].median(); A = A.reset_index().rename(columns={view: 'band'})
            A['view'] = view
            if dk:
                A['day'] = dk
            if sk:
                A['side'] = sk
            parts.append(A)
    A = pd.concat(parts, ignore_index=True); out = []
    for c in ('native', 'common'):
        B = {'view': A['view'], 'band': A['band'].astype(str), 'interval': A['interval'], 'day': A['day'], 'side': A['side'].astype(str), 'cohort': c, 'all_opportunities': A['n_'],
             'all_at_risk': A['risk'], 'all_filled': A['filled'], 'all_fills': A['fills'], 'all_shares': A['shares'], 'all_partial': A['part'], 'all_filled_out': A['fo'],
             'all_t_to_end_median': A['med'], 'all_t_to_end_lt30s': A['lt30'], **{f'all_n_{st}': A[f's_{st}'] for st in FOLLOW}}
        B['all_any_fill_rate'] = np.where(A['risk'] > 0, A['filled'] / A['risk'].where(A['risk'] > 0, 1), np.nan); B['censored_from_common'] = A['n_'] - A['common0_n']
        for h in H:
            n, q, r_, f_ = A[f'{c}{h}_n'], A[f'{c}{h}_q'], A[f'{c}{h}_r'], A[f'{c}{h}_f']
            for x, y in (('valid', 'n'), ('valid_risk', 'r'), ('valid_filled', 'f'), ('valid_shares', 'q'), ('net', 'net'), ('state_positive', 'sp'), ('state_negative', 'sn'),
                         ('fill_positive', 'fp'), ('fill_negative', 'fn')):
                B[f'h{h}_{x}'] = A[f'{c}{h}_{y}']
            B[f'h{h}_any_fill_rate'] = np.where(r_ > 0, f_ / r_.where(r_ > 0, 1), np.nan); B[f'h{h}_per_opportunity'] = np.where(n > 0, A[f'{c}{h}_net'] / n.where(n > 0, 1), np.nan)
            B[f'h{h}_per_share'] = np.where(q > 0, A[f'{c}{h}_net'] / q.where(q > 0, 1.), np.nan); B[f'h{h}_sparse'] = f_ < 20
        out.append(pd.DataFrame(B))
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------------------------------------------------------------- this checker's readings
def cell(C_, view, band, iv, day='all', side='all'):
    g = C_[(C_.view == view) & (C_.band == band) & (C_.interval == iv) & (C_.day == day) & (C_.side == side) & (C_.cohort == 'common')]
    return g.iloc[0] if len(g) == 1 else None


def vorder(vals):
    return '<'.join(sorted(BANDS, key=lambda b: vals[b][0])) if all(vals[b][1] for b in BANDS) else 'insufficient_support'


def val(r, h):
    return (NAN, False) if r is None else (float(r[f'h{h}_per_share']), not bool(r[f'h{h}_sparse']))


def my_readings(M_, C_):
    pr = lambda iv, h, day='all', side='all': vorder({b: val(cell(C_, 'price_display_queue', f'{PR}|{b}', iv, day, side), h) for b in BANDS})
    q2, q3 = {}, {}
    for iv in IVS + ('union',):
        q2[iv], q3[iv] = {}, {}
        for h in (1000, 30000):
            p = pr(iv, h); ds = [pr(iv, h, day=d) for d in DAYS]; ev = [x for x in ds if x != 'insufficient_support']; ss = [pr(iv, h, side=s) for s in ('0', '1')]
            q2[iv][str(h)] = dict(pooled=p, days_evaluable=len(ev), days_same_order=sum(x == p for x in ev) if p != 'insufficient_support' else 0,
                                  same_order_all_4_days=bool(p != 'insufficient_support' and len(ev) == 4 and all(x == p for x in ev)),
                                  sides_same_order=bool(p != 'insufficient_support' and all(x == p for x in ss)))
            g = C_[(C_.view == 'price_display_queue') & (C_.interval == iv) & (C_.day == 'all') & (C_.side == 'all') & (C_.cohort == 'common')]; vals = {}
            for b in BANDS:
                x = g[g.band.str.endswith('|' + b)]; q = float(x[f'h{h}_valid_shares'].sum()); vals[b] = (float(x[f'h{h}_net'].sum()) / q if q > 0 else NAN, int(x[f'h{h}_valid_filled'].sum()) >= 20)
            q3[iv][str(h)] = vorder(vals)
    persists = {str(h): ('insufficient_support' if 'insufficient_support' in (q2['union'][str(h)]['pooled'], q3['union'][str(h)]) else bool(q2['union'][str(h)]['pooled'] == q3['union'][str(h)]))
                for h in (1000, 30000)}
    deps = sorted({b.split('|')[0] for b in C_[(C_.view == 'behind_depth_queue') & (C_.band != 'outside')].band})
    q5 = {d: {str(h): vorder({b: val(cell(C_, 'behind_depth_queue', f'{d}|{b}', 'union'), h) for b in BANDS}) for h in (1000, 30000)} for d in deps}
    pp = C_[(C_.view == 'principal_pressure') & (C_.band != 'outside') & (C_.interval == 'union') & (C_.day == 'all') & (C_.side == 'all') & (C_.cohort == 'common')]
    census = {r.band: (int(r.all_opportunities), int(r.h1000_valid_filled)) for r in pp.itertuples()}
    contr = sum(int(val(cell(C_, 'principal_pressure', f'{b}|{a}', 'union'), 1000)[1] and val(cell(C_, 'principal_pressure', f'{b}|non_adverse', 'union'), 1000)[1])
                for b in BANDS for a in ('q5', 'positive_below_q5'))
    occ = {'|'.join(k): int(v) for k, v in M_.groupby(['price_state', 'display', 'queue_band']).size().items()}
    fr = M_[M_.queue_band == 'front']; front = {f'{a}|{b}': int(((fr.price_state == a) & (fr.display == b)).sum()) for a in ('inside_best', 'at_touch', 'behind_best')
                                                for b in ('shown_zero', 'shown_positive', 'shown_unknown', 'shown_invalid')}
    cen = {b: (int(r.censored_from_common), float(r.all_t_to_end_median), int(r.all_t_to_end_lt30s)) for b in BANDS for r in [cell(C_, 'price_display_queue', f'{PR}|{b}', 'union')] if r is not None}
    va = M_[M_.venue_status == 'available']
    vq = {g: (int(len(x)), int((x.client_minus_venue.abs() < 1).sum()), int((x.client_minus_venue <= -1).sum()), int((x.client_minus_venue >= 1).sum())) for g, x in va.groupby('price_display')}
    return dict(q2=q2, q3=q3, persists=persists, q5=q5, census=census, contr=contr, occ=occ, front=front, cen=cen, vq=vq)


# ---------------------------------------------------------------------------------------------------------------------- families
def fam_A(E):
    c = dict(A_committed=all(blob(p) == (REPO / p).read_bytes() for p in COMMITTED), A_inputs=bool(all(E['src'].values()) and E['gm']['inputs']['aq_audit'] == sha(blob(f'{AQ}/outputs/audit.json'))))
    c['A_outputs'] = bool(all(E['audit']['outputs'][n] == sha((E['out'] / n).read_bytes()) for n in ('states.csv', 'cells.csv'))
                          and all(gzip.decompress((E['rdir'] / f'{n}.gz').read_bytes()) == (E['out'] / n).read_bytes() for n in ('states.csv', 'cells.csv'))); return c, {}


def fam_B(E):
    a = sorted(zip(E['aq_states.csv'].day, E['aq_states.csv'].m, E['aq_states.csv'].gid)); b = sorted(zip(E['states'].day, E['states'].m, E['states'].gid))
    return dict(B_membership=bool(a == b and len(b) == len(set(b)) == 4840)), dict(B=dict(states=len(b)))


def merged(E):
    return E['mine_s'].merge(E['states'], on=K, suffixes=('', '_s'), how='left')


def fam_U(E):
    J = merged(E); num = ['t', 'side', 'rung', 'price', 'size', 'o_depth', 'fp_shown', 'Q', 'R', 'Q_over_R', 'o_q_rel', 't_to_end', 'client_minus_venue']
    bad = [c for c in num if not same(J[c], J[f'{c}_s'])] + [c for c in ('queue_band', 'bn_band', 'pf_band', 'first_status', 'venue_status') if not (J[c].astype(str) == J[f'{c}_s'].astype(str)).all()]
    return dict(U_reused_fields=bool(len(J) == len(E['mine_s']) and not bad)), dict(U=dict(mismatched=bad))


def fam_G(E):
    J = merged(E); bad = [c for c in ('price_state', 'display') + VIEWS if not (J[c].astype(str) == J[f'{c}_s'].astype(str)).all()]
    bad += [] if (J.principal.astype(bool) == J.principal_s.astype(bool)).all() else ['principal']; return dict(G_labels=not bad), dict(G=dict(mismatched=bad))


def fam_C(E):
    T, M_ = E['cells'], E['mine_c']; keys = ['view', 'band', 'interval', 'day', 'side', 'cohort']
    J = M_.merge(T, on=keys, how='outer', suffixes=('', '_s'), indicator=True); both = bool((J._merge == 'both').all()) and len(J) == len(T) == len(M_)
    cols = [c for c in M_.columns if c not in keys]; bad = ([] if both else ['key_set']) + ([] if set(T.columns) == set(M_.columns) else ['column_set'])
    for c in cols:
        if c.endswith('_sparse'):
            bad += [] if (J[c].astype(bool) == J[f'{c}_s'].astype(bool)).all() else [c]
        elif not same(J[c], J[f'{c}_s']):
            bad.append(c)
    return dict(C_cells=not bad), dict(C=dict(rows=int(len(T)), mismatched=bad[:12]))


def fam_P(E):
    m, rd = E['mine_r'], E['readings']; bad = []
    for iv, x in m['q2'].items():
        for h, y in x.items():
            z = rd['q2_principal'][iv][h]
            bad += [] if all(z[k] == y[k] for k in ('pooled', 'days_evaluable', 'days_same_order', 'same_order_all_4_days', 'sides_same_order')) else [f'q2 {iv} {h}']
            bad += [] if rd['q3_pooled'][iv][h]['order'] == m['q3'][iv][h] else [f'q3 {iv} {h}']
    bad += [] if rd['q3_persists_at_touch_occupied'] == m['persists'] else ['persists']
    bad += [] if {d: {h: v['order'] for h, v in x.items()} for d, x in rd['q5_behind_depth'].items()} == m['q5'] else ['q5']
    bad += [] if {b: (v['states'], v['valid_filled_1s']) for b, v in rd['q6_pressure_census'].items()} == m['census'] and rd['q6_contrasts_evaluable'] == m['contr'] else ['q6']
    bad += [] if rd['q1_occupancy'] == m['occ'] and {k: v for k, v in rd['q1_front'].items() if k != 'states'} == m['front'] else ['q1']
    bad += [] if {b: (v['censored_from_common'], v['t_to_end_median'], v['t_to_end_lt30s']) for b, v in rd['q7_principal_censoring'].items()} == m['cen'] else ['q7']
    bad += [] if {g: (v['n'], v['within_1'], v['below_by_1'], v['above_by_1']) for g, v in rd['q8_venue_by_price_display'].items()} == m['vq'] else ['q8']
    return dict(P_readings=not bad), dict(P=dict(mismatched=bad))


def cpu(paths):
    tot, peak = 0., 0
    for p in paths:
        for ln in Path(p).read_text(errors='replace').splitlines():
            mm = RL.match(ln)
            if mm:
                tot += float(mm.group(6)); peak = max(peak, int(mm.group(4)))
    return tot / 3600., peak


def fam_I(E, root=None):
    own, peak = cpu(glob.glob(str((root or R) / 'logs' / '**' / '*.log'), recursive=True))
    line, _ = cpu([p for n in LINE for p in glob.glob(str(REPO / 'predictfun/reviews' / n / 'logs' / '**' / '*.log'), recursive=True)])
    rev, _ = cpu([p for n in REVIEWS for p in glob.glob(str(REPO / 'predictfun/reviews/automatic' / n / '*.log'))]); cap = float(E['gm']['cap_core_hours'])
    return dict(I_resources=bool(own <= cap and peak < 1536 and own + line + rev <= 2.0)), dict(I=dict(round_core_hours=own, cap=cap, line_core_hours=own + line + rev, peak_mib=peak))


FAMS = dict(A=fam_A, B=fam_B, U=fam_U, G=fam_G, C=fam_C, P=fam_P, I=fam_I)


def run_checks(E):
    chk, det = {}, {}
    for k in FAMS:
        c, d = FAMS[k](E); chk.update(c); det.update(d)
    return dict(checks=chk, details=det, all_ok=bool(all(chk.values())))


def _fake(E):
    E['_root'] = Path(tempfile.mkdtemp(dir=str(E['out']))); (E['_root'] / 'logs').mkdir()
    (E['_root'] / 'logs' / 'fake.log').write_text('2026-11-22T00:00:00Z tag=fake rc=0 peak_rss_mib=100 wall_s=99.0 cpu_user_s=99.0\n')


def _set(df, col, fn):
    df = df.copy(); df[col] = fn(df); return df


CASES = [
    ('an_output_hash_changed', 'A', 'A_outputs', lambda E: E.__setitem__('audit', dict(E['audit'], outputs=dict(E['audit']['outputs'], **{'cells.csv': '0' * 64})))),
    ('a_state_duplicated', 'B', 'B_membership', lambda E: E.__setitem__('states', pd.concat([E['states'], E['states'].iloc[[0]]], ignore_index=True))),
    ('client_minus_venue_zeroed', 'U', 'U_reused_fields', lambda E: E.__setitem__('states', E['states'].assign(client_minus_venue=0.))),
    ('a_displayed_size_changed', 'U', 'U_reused_fields', lambda E: E.__setitem__('states', _set(E['states'], 'fp_shown', lambda d: d.fp_shown + np.where(d.index == 0, 1., 0.)))),
    ('a_price_state_relabelled', 'G', 'G_labels', lambda E: E.__setitem__('states', _set(E['states'], 'price_state', lambda d: np.where(d.index == 0, 'inside_best', d.price_state)))),
    ('h1000_per_share_negated', 'C', 'C_cells', lambda E: E.__setitem__('cells', E['cells'].assign(h1000_per_share=-E['cells'].h1000_per_share))),
    ('a_sparse_flag_flipped', 'C', 'C_cells', lambda E: E.__setitem__('cells', _set(E['cells'], 'h1000_sparse', lambda d: np.where(d.index == 0, ~d.h1000_sparse.astype(bool), d.h1000_sparse)))),
    ('a_cell_dropped', 'C', 'C_cells', lambda E: E.__setitem__('cells', E['cells'].iloc[1:])),
    ('a_reading_changed', 'P', 'P_readings', lambda E: E.__setitem__('readings', dict(E['readings'], q6_contrasts_evaluable=E['readings']['q6_contrasts_evaluable'] + 1))),
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
    out, man, rdir, res = (argv + [None] * 4)[:4]; E = load(out, man, rdir)
    E['mine_s'] = my_states(E); E['mine_c'] = my_cells(E['mine_s'], E['aq_intervals.csv']); E['mine_r'] = my_readings(E['mine_s'], E['mine_c'])
    rd = Path(res or R); r_ = run_checks(E); fz = falsify(E, r_)
    for n, obj in (('checks.json', r_), ('checks_falsifier.json', fz)):
        p = rd / n; assert not p.exists(), f'{p} exists (write-once)'; p.write_text(json.dumps(obj, indent=1, default=str) + '\n')
    print(json.dumps(dict(all_ok=r_['all_ok'], failed=[k for k, v in r_['checks'].items() if not v], falsifier_ok=fz['falsifier_ok'],
                          cases={k: (v['clean'], v['refused']) for k, v in fz['cases'].items()}, details=r_['details']), indent=1, default=str)[:3000])


if __name__ == '__main__':
    main(sys.argv[1:])
