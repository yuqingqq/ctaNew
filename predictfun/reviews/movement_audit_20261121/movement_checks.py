"""PI0-MOVEMENT-CALIBRATION-AUDIT-v1: THE DECLARED CHECKS (review 5a8ee6f5; 2026-11-21, session aea5bfb9). Declared in the freeze, before any
new table is computed. Each check recomputes from the hash-bound sources with its own loader, bins and selection code (never the module's
measurement functions), compares with the measured outputs, and is false when it cannot run.
 A identity: the module, selftest, harness, declaration and this file are the committed ones; both cohort manifests equal their committed
   receipt copies; every admission / retention partition's bytes match its record and the record its manifest; every mids record's hash
   matches its content and the audit's list; the audit's and the records' sources are the final files.
 B membership and binding: admitted keys = the manifest's selected keys, one row each; each row's weight, n_cell, cell and part equal its
   manifest row (exact key); forced keys = the exposure manifest's world x side forks; the opportunities table holds every row x stage once,
   with the manifest weights (forced 1).
 C accounting: every opportunity's status recomputed (eligibility, no_forecast_or_target, movement_missing, zero_P, the identities, the
   fill totals); on ok rows P, E[q], E[qm], V, edge, q, value, z, M, qhat, mhat and the decomposition terms recomputed from the stage
   moments and target equal the table within 1e-9 (relative), and the three identities hold within 1e-6.
 D strata: t_to_end, fp_queue, fp_remaining and the ordinal equal the stage's client inputs; hbin / qbin / obin recomputed from them;
   venue_qbin recomputed from the venue's ahead.
 E fills: per ok opportunity the counted fills recomputed (the held gid, t >= the effect start) equal fills.csv (index, t, q, price, mark,
   total, delay, class); the reference status recomputed (the mids record for the admitted, the retained edge for the forced; missing,
   unavailable, out of time); on references the mid, frame time, age, pre and post recomputed; the opportunity's pre / post sums.
 F denominators: every moments row's n / fills / weight / same_as_review, every prepost row's fills / with_mid / missing and every strata
   row's n / fills / weight recomputed from the opportunities / fills tables; the moments views complete.
 G predicates: (i)-(iv) and the recommendation per stage recomputed from the opportunities table equal audit.json's, with their numbers.
 H reproduction (real data): at the admitted entry the review's independent numbers (5a8ee6f5): bias +11.099 = -0.882 + 11.981 and
   11.981 = 0.310 + 11.671 (c), MSE(V) 4,043.27 / zero 3,982.00 / mechanical 4,098.46, movement MSE 4,336.46 / zero 4,395.20 (c^2), mean
   E[qm] -1.779 / M -13.760 (c), to the printed precision; the weighted top-5 |w D| generation share 0.520, the unweighted 0.37; the control
   233 / 299 PULL, percentile 0.806, 91 strata, 45 free with 166 rows, 299 review cells equal the admission cell.
 I resources and suites (real data): every run_light job under logs/ below 2,048 MiB, every mids record's high-water <= 1,536 MiB, the round
   within 1 core-hour; pf_mutant_summary.judge passes the harness's records against the final sources, one record per file, no other file.
FALSIFIER (falsify(), writes checks_falsifier.json): every check true on clean copies, and the named check false after one corruption of a
 real copy: a substituted partition (A); a doubled row weight, rehashed (B); a dropped admitted row, rehashed (B); V in other units (C); an
 edge sign (C); a decision queue from the venue (D); a moved fill time (E); a moved END mark (E); a moments count (F); a reading (G); a
 reproduced number (H); an over-budget job (I); a copied mutant record (I).
SELFTEST (selftest()): A-G on hand cohorts measured by the module pass clean and fail after each of the A-G corruptions; H's and I's
 comparisons on synthetic values.
Usage (run_light): movement_checks.py [selftest]  (no argument: the checks, then the falsifier)"""
import glob
import gzip
import hashlib
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent; RES = Path('/home/yuqing/ctaNew/predictfun/research'); sys.path.insert(0, str(RES))
import pf_movement_audit as M                            # noqa: E402  constants, committed_equal and sources only

pd, RT = M.pd, M.RT
FEATS, OVL, END = RT.FEATS, ('20261026', '20261027'), 5000.
LOGS = HERE / 'logs'
RL = re.compile(r'^(\S+) tag=(\S+) rc=(\d+) peak_rss_mib=(\d+) wall_s=([\d.]+) cpu_user_s=([\d.]+)')
REAL = dict(adm=M.ADM, ret=M.RET, expo_man=M.EXPO_MAN, out=M.OUT, real=True)
STR = {c: str for c in ('day', 'key', 'cell', 'side', 'admission', 'view', 'stage', 'cohort', 'subset', 'dim', 'kind', 'quantity', 'weighting', 'acc_status', 'row_status',
                        'stage_status', 'dec_status', 'hbin', 'qbin', 'obin', 'venue_qbin', 'pbin', 'mbin', 'eqbin', 'vbin', 'delay_class', 'split_status', 'ref_status')}


def gz_rows(b):
    return [json.loads(x) for x in gzip.decompress(b).decode().splitlines() if x]


def shb(b):
    return hashlib.sha256(b).hexdigest()


def kg(r):
    return json.dumps([str(r['day']), int(r['m']), int(r['t_ms']), int(r['side']), int(r['gid']), int(r['decision_ms'])])


def kf(r):
    return json.dumps([str(r['day']), int(r['m']), int(r['t_ms']), int(r['side'])])


def close(a, b, tol=1e-9):
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return False
    return bool((np.isnan(a) and np.isnan(b)) or abs(a - b) <= tol * max(1., abs(a), abs(b)))


def load(P):
    adm, ret, out = Path(P['adm']), Path(P['ret']), Path(P['out'])
    am = json.loads((adm / 'manifest.json').read_text()); em = json.loads(Path(P['expo_man']).read_text())
    pa = {int(k): ((adm / 'parts' / f'part_{int(k):03d}.jsonl.gz').read_bytes(), json.loads((adm / 'parts' / f'part_{int(k):03d}.json').read_text())) for k in am['parts']}
    pf = {k: ((ret / 'parts' / f'part_{k:03d}.jsonl.gz').read_bytes(), json.loads((ret / 'parts' / f'part_{k:03d}.json').read_text())) for k in range(len(em['parts']))}
    mids = {k: json.loads((out / 'parts' / f'mids_{k:03d}.json').read_text()) for k in pa if (out / 'parts' / f'mids_{k:03d}.json').exists()}
    rd = lambda n: pd.read_csv(out / f'{n}.csv', float_precision='round_trip', dtype=STR)
    src = {('admitted', kg(r)): r for b, _ in pa.values() for r in gz_rows(b)}; src.update({('forced', kf(r)): r for b, _ in pf.values() for r in gz_rows(b)})
    mk = {key: e for rec in mids.values() for key, e in rec['mids'].items()}
    return dict(am=am, em=em, pa=pa, pf=pf, mids=mids, mk=mk, src=src, audit=json.loads((out / 'audit.json').read_text()), O=rd('opportunities'), F=rd('fills'),
                MO=rd('moments'), PP=rd('prepost'), ST=rd('strata'), CO=rd('concentration'))


# ---------------------------------------------------------------------------------------------------------------------- the recomputations
def hb(t):
    return '(0,100]' if 0 < t <= 100 else '(100,500]' if 100 < t <= 500 else '(500,2000]' if 500 < t <= 2000 else '(2000,5000]' if 2000 < t <= 5000 else 'outside'


def qb(a, rem):
    if not (np.isfinite(a) and np.isfinite(rem)) or rem <= 0 or a < 0:
        return 'missing_or_invalid'
    x = a / rem
    return '0' if x == 0 else '(0,1]' if x <= 1 else '(1,3]' if x <= 3 else '>3'


def ob(o):
    return 'missing_or_invalid' if not np.isfinite(o) else '1' if o <= 1 else '2' if o <= 2 else '>=3'


def db(d):
    return 'negative' if d < 0 else '[0,100]' if d <= 100 else '(100,500]' if d <= 500 else '(500,2000]' if d <= 2000 else '>2000'


def counted(r, x, nm):
    h = r['held']; eff = float(h['act']) if nm == 'entry' else float(x['effect']); dmid = float(x['inputs'][FEATS.index('pf_mid')]); sg = 1. if int(r['side']) == 0 else -1.
    return [dict(index=i, t=float(f[0]), q=float(f[3]), price=float(f[4]), eff=eff, dmid=dmid, mark=float(h['mark']), sg=sg, total=sg * float(f[3]) * (float(h['mark']) - dmid))
            for i, f in enumerate(h['fills']) if float(f[0]) >= eff and int(f[1]) == int(h['gid'])]


def expect(r, nm):
    """(status, terms) of one opportunity, recomputed."""
    x = (r.get('stages') or {}).get(nm) or {}
    if r['status'] != 'ok' or x.get('status') != 'ok' or x.get('dec_status') != 'model' or not x.get('inputs'):
        return 'not_eligible', None
    mo, tg = x.get('moments') or {}, x.get('target') or {}
    if mo.get('V') is None or tg.get('q') is None:
        return 'no_forecast_or_target', None
    P_, Eq, Eqm, V, e = (float(mo[k]) for k in ('P', 'Eq', 'Eqm', 'V', 'edge')); q, v = float(tg['q']), float(tg['value']); z = float(q > 1e-12)
    if z and tg.get('qm') is None:
        return 'movement_missing', None
    Mv = float(tg['qm']) if z else 0.
    if P_ <= 0:
        return 'zero_P', None
    qh, mh = Eq / P_, Eqm / P_
    T = dict(P=P_, Eq=Eq, Eqm=Eqm, V=V, edge=e, q=q, value=v, z=z, M=Mv, qhat=qh, mhat=mh, err_V=V - v, err_q=Eq - q, err_m=Eqm - Mv, qty_term=e * (Eq - q),
             mov_prob=(P_ - z) * mh, mov_cond=z * (mh - Mv), q_prob=(P_ - z) * qh, q_cond=z * (qh - q))
    if max(abs(T['err_V'] - T['qty_term'] - T['err_m']), abs(T['err_m'] - T['mov_prob'] - T['mov_cond']), abs(T['err_q'] - T['q_prob'] - T['q_cond'])) > 1e-6:
        return 'accounting_mismatch', T
    fl = counted(r, x, nm)
    if abs(sum(f['total'] for f in fl) - Mv) > 1e-6 or bool(fl) != bool(z):
        return 'fill_mismatch', T
    return 'ok', T


def ref_of(D, cohort, key, r, f):
    """(status, mid, frame time) of one counted fill, recomputed."""
    if cohort == 'forced':
        e = (r.get('fills') or [])[f['index']].get('fill_time_edge') if f['index'] < len(r.get('fills') or []) else None
        return ('reference_missing', None, None) if e is None else ('ok', f['price'] + float(e) / (f['sg'] * f['q']), None)
    e = D['mk'].get(key)
    if e is None:
        return 'reference_unavailable:no_record', None, None
    if e['status'] != 'ok':
        return f"reference_unavailable:{e['status']}", None, None
    m = (e.get('mids') or {}).get(str(f['index']))
    if m is None:
        return 'reference_missing', None, None
    if not (float(m['frame_t']) <= f['t'] <= END):
        return 'reference_out_of_time', float(m['mid']), float(m['frame_t'])
    return 'ok', float(m['mid']), float(m['frame_t'])


# ---------------------------------------------------------------------------------------------------------------------- the checks
def check_A(P, D, R):
    ok = {}; msha, esha = shb((Path(P['adm']) / 'manifest.json').read_bytes()), shb(Path(P['expo_man']).read_bytes()); au = D['audit']
    ok['partitions'] = bool(all(shb(b) == rec['sha256'] and rec['manifest_sha256'] == msha for b, rec in D['pa'].values())
                            and all(shb(b) == rec['sha256'] and rec['manifest_sha256'] == esha for b, rec in D['pf'].values()))
    ok['audit_manifests'] = bool(au['admission_manifest_sha256'] == msha and au['exposure_manifest_sha256'] == esha)
    ok['mids'] = bool(sorted(int(k) for k in au['mids_records']) == sorted(D['mids']) and all(
        shb(json.dumps(r['mids'], sort_keys=True).encode()) == r['sha256'] == au['mids_records'][str(k)] and r['admission_manifest_sha256'] == msha for k, r in D['mids'].items()))
    if P['real']:
        ok['frozen'] = bool(M.frozen() and M.committed_equal(f'{M.RECEIPT}/movement_checks.py', HERE / 'movement_checks.py')
                            and M.committed_equal('predictfun/research/pf_movement_audit_mutants.py', RES / 'pf_movement_audit_mutants.py'))
        ok['manifests_committed'] = bool(M.committed_equal(M.ADM_MAN_REL, Path(P['adm']) / 'manifest.json') and M.committed_equal(M.EXPO_MAN_REL, P['expo_man']))
        ok['sources'] = bool(au['sources'] == M.sources() and all(r['sources'] == M.sources() for r in D['mids'].values()))
    R['A'] = ok
    return all(ok.values())


def check_B(P, D, R):
    sel = {kg(g): g for g in D['am']['selected']}; rows = [(k, r) for k, (b, _) in D['pa'].items() for r in gz_rows(b)]; keys = [kg(r) for _, r in rows]
    bind = all(kg(r) in sel and float(r['weight']) == float(sel[kg(r)]['weight']) and int(r['n_cell']) == int(sel[kg(r)]['n_cell']) and list(r['cell']) == list(sel[kg(r)]['cell'])
               and int(sel[kg(r)]['part']) == int(k) for k, r in rows)
    a_mem = len(keys) == len(set(keys)) == len(sel) and set(keys) == set(sel)
    fk = {json.dumps([str(w['day']), int(w['m']), int(w['t_ms']), s]) for p in D['em']['parts'] for w in p['worlds'] for s in (0, 1)}
    fkeys = [kf(r) for b, _ in D['pf'].values() for r in gz_rows(b)]; f_mem = len(fkeys) == len(set(fkeys)) == len(fk) and set(fkeys) == fk
    O = D['O']; oa, of = O[O.cohort == 'admitted'], O[O.cohort == 'forced']
    per = O.groupby(['cohort', 'key']).stage.apply(lambda s: sorted(s) == sorted(M.STAGES))
    o_mem = bool(len(O) == 3 * (len(sel) + len(fk)) and set(oa.key) == set(sel) and set(of.key) == fk and per.all())
    o_w = bool(all(close(w, sel[k]['weight']) for k, w in zip(oa.key, oa.weight)) and (of.weight == 1.).all())
    R['B'] = dict(binding=bool(bind), admitted=bool(a_mem), forced=bool(f_mem), opportunities=o_mem, weights=o_w, admitted_rows=len(keys), forced_rows=len(fkeys))
    return bool(bind and a_mem and f_mem and o_mem and o_w)


def check_C(P, D, R):
    bad, n_ok = [], 0
    for row in D['O'].itertuples(index=False):
        r = D['src'].get((row.cohort, row.key))
        if r is None:
            bad.append(('no_source', row.key)); continue
        st, T = expect(r, row.stage)
        if st != row.acc_status:
            bad.append(('status', row.key, row.stage, st, row.acc_status)); continue
        if st == 'ok':
            n_ok += 1; d = row._asdict()
            if not (all(close(d[k], v) for k, v in T.items()) and close(d['mov_term'], T['err_m']) and float(d['identity_residual']) <= 1e-6 and str(d['side']) == str(int(r['side']))
                    and str(d['day']) == str(r['day'])):
                bad.append(('terms', row.key, row.stage))
    R['C'] = dict(ok_rows=n_ok, failures=bad[:5], n_fail=len(bad))
    return bool(n_ok and not bad)


def check_D(P, D, R):
    bad, n = [], 0
    for row in D['O'][D['O'].acc_status != 'not_eligible'].itertuples(index=False):
        r = D['src'][(row.cohort, row.key)]; x = r['stages'][row.stage]; a = x['inputs']; g = lambda nm: float(a[FEATS.index(nm)]); n += 1
        va = (x.get('venue') or {}).get('ahead'); va = np.nan if va is None else float(va)
        if not (close(row.t_to_end, g('t_to_end')) and close(row.fp_queue, g('fp_queue')) and close(row.fp_remaining, g('fp_remaining'))
                and close(row.ordinal, g('ctx_generation_ordinal')) and row.hbin == hb(g('t_to_end')) and row.qbin == qb(g('fp_queue'), g('fp_remaining'))
                and row.obin == ob(g('ctx_generation_ordinal')) and row.venue_qbin == qb(va, g('fp_remaining'))):
            bad.append((row.key, row.stage))
    R['D'] = dict(rows=n, failures=bad[:5], n_fail=len(bad))
    return bool(n and not bad)


def check_E(P, D, R):
    O, F = D['O'], D['F']; bad = []; n_exp = 0; byk = {k: g.sort_values('index') for k, g in F.groupby(['cohort', 'key', 'stage'])}
    for row in O[O.acc_status == 'ok'].itertuples(index=False):
        r = D['src'][(row.cohort, row.key)]; x = r['stages'][row.stage]; fl = counted(r, x, row.stage); n_exp += len(fl)
        g = byk.get((row.cohort, row.key, row.stage)); g = g.to_dict('records') if g is not None else []
        if len(g) != len(fl):
            bad.append(('count', row.key, row.stage)); continue
        pre = post = 0.; full = True
        for f, h in zip(fl, g):
            st, mid, ft = ref_of(D, row.cohort, row.key, r, f)
            ok = (int(h['index']) == f['index'] and close(h['t'], f['t']) and close(h['q'], f['q']) and close(h['price'], f['price']) and close(h['mark'], f['mark'])
                  and close(h['total'], f['total']) and close(h['delay'], f['t'] - f['eff']) and h['delay_class'] == db(f['t'] - f['eff']) and h['ref_status'] == st)
            if ok and st == 'ok':
                ok = (close(h['fill_mid'], mid) and close(h['pre'], f['sg'] * f['q'] * (mid - f['dmid'])) and close(h['post'], f['sg'] * f['q'] * (f['mark'] - mid))
                      and (close(h['frame_t'], ft) and close(h['age'], f['t'] - ft) if ft is not None else bool(np.isnan(float(h['frame_t'])))))
                pre += f['sg'] * f['q'] * (mid - f['dmid']); post += f['sg'] * f['q'] * (f['mark'] - mid)
            else:
                full = False
            if not ok:
                bad.append(('fill', row.key, row.stage, f['index']))
        if not ((close(row.pre, pre) and close(row.post, post)) if full else (np.isnan(float(row.pre)) and np.isnan(float(row.post)))):
            bad.append(('split', row.key, row.stage))
    okc = len(F) == n_exp
    R['E'] = dict(fills=int(len(F)), expected=n_exp, failures=bad[:5], n_fail=len(bad))
    return bool(okc and not bad)


def view_rows(E, cohort):
    out = {('all', 'all', 'all', 'all')}
    out |= {('day', str(d), 'all', 'all') for d in E.day} | {('side', 'all', str(s), 'all') for s in E.side} | {('day_side', str(d), str(s), 'all') for d, s in zip(E.day, E.side)}
    if cohort == 'forced':
        ov = E[E.day.isin(OVL)]
        out |= ({('overlap', '20261026-27', 'all', 'all')} if len(ov) else set()) | {('admission', 'all', 'all', str(a)) for a in E.admission}
        out |= {('overlap_admission', '20261026-27', 'all', str(a)) for a in ov.admission}
    return out


def check_F(P, D, R):
    O, F, MO, PP, ST = D['O'], D['F'], D['MO'], D['PP'], D['ST']; E = O[O.acc_status == 'ok']; bad = []
    for row in MO.itertuples(index=False):
        g = E[(E.cohort == row.cohort) & (E.stage == row.stage)]
        if row.view in ('day', 'day_side'):
            g = g[g.day == row.day]
        if row.view in ('side', 'day_side'):
            g = g[g.side == row.side]
        if row.view in ('overlap', 'overlap_admission'):
            g = g[g.day.isin(OVL)]
        if row.view in ('admission', 'overlap_admission'):
            g = g[g.admission == row.admission]
        if not (len(g) == int(row.n) and int((g.z > 0).sum()) == int(row.fills) and close(g.weight.sum(), row.weight) and int(g.same_as_review.astype(bool).sum()) == int(row.same_as_review)):
            bad.append(('moments', row.cohort, row.stage, row.view, row.day, row.side, row.admission))
    for (co, nm), g in E.groupby(['cohort', 'stage']):
        have = {(v, d, s, a) for v, d, s, a in zip(*[MO[(MO.cohort == co) & (MO.stage == nm)][c] for c in ('view', 'day', 'side', 'admission')])}
        if have != view_rows(g, co):
            bad.append(('views', co, nm))
    for row in PP.itertuples(index=False):
        g = F[(F.cohort == row.cohort) & (F.stage == row.stage)]
        if row.view == 'day_side':
            g = g[(g.day == row.cell) & (g.side == row.side)]
        elif row.view == 'delay_class':
            g = g[g.delay_class == row.cell]
        elif row.view == 'overlap':
            g = g[g.day.isin(OVL)]
        if not (len(g) == int(row.fills) and int((g.ref_status == 'ok').sum()) == int(row.with_mid) and int((g.ref_status != 'ok').sum()) == int(row.missing)):
            bad.append(('prepost', row.cohort, row.stage, row.view, row.cell))
    for row in ST.itertuples(index=False):
        g = E[(E.cohort == row.cohort) & (E.stage == row.stage)]
        if row.subset == 'overlap':
            g = g[g.day.isin(OVL)]
        g = g[g[row.dim].astype(str) == row.cell]
        if not (len(g) == int(row.n) and int((g.z > 0).sum()) == int(row.fills) and close(g.weight.sum(), row.weight)):
            bad.append(('strata', row.cohort, row.subset, row.stage, row.dim, row.cell))
    R['F'] = dict(moments=int(len(MO)), prepost=int(len(PP)), strata=int(len(ST)), failures=bad[:5], n_fail=len(bad))
    return bool(len(MO) and not bad)


def preds(O, stage):
    """(i)-(iv) and the recommendation, recomputed from the opportunities table (the declared reading)."""
    E = O[(O.cohort == 'admitted') & (O.stage == stage) & (O.acc_status == 'ok')]; wmn = lambda v, w: float((v * w).sum() / w.sum())
    cells = {f'{d}/{s}': wmn(g.err_m, g.weight) for (d, s), g in E.groupby(['day', 'side'])}; pos = int(sum(v > 0 for v in cells.values()))
    ws = np.sort(np.abs(E.assign(c=E.weight * E.err_m).groupby(['day', 'm', 't_ms']).c.sum().to_numpy()))[::-1]; share = float(ws[:5].sum() / ws.sum()) if ws.sum() > 0 else None
    i_ = 'broad' if (len(cells) == 4 and pos >= 3 and share is not None and share < .5) else 'concentrated' if ((share is not None and share >= .5) or pos <= 1) else 'inconclusive'

    def rng(dim):
        v = [wmn(g[g.z > 0].c_m, g[g.z > 0].weight) for lab, g in E.groupby(dim) if int((g.z > 0).sum()) >= 10 and lab != 'no_fill']
        return float(max(v) - min(v)) if len(v) >= 2 else None
    dec = {d: rng(d) for d in M.DECISION_DIMS}; dly = rng('delay_class'); dm = max([v for v in dec.values() if v is not None], default=None)
    ii = ('inconclusive' if dm is None and dly is None else 'decision_state' if dly is None or (dm is not None and dm >= 1.5 * dly) else
          'fill_timing' if dm is None or dly >= 1.5 * dm else 'both')
    fa = E[E.z > 0]; ff = O[(O.cohort == 'forced') & (O.stage == stage) & (O.acc_status == 'ok') & O.day.isin(OVL) & (O.z > 0)]
    ca = wmn(fa.c_m, fa.weight) if len(fa) else None; cf = float(ff.c_m.mean()) if len(ff) else None
    iii = ('inconclusive' if len(fa) < 10 or len(ff) < 10 else 'material' if (np.sign(ca) != np.sign(cf) or abs(ca - cf) >= .5 * max(abs(ca), abs(cf))) else 'similar')
    pop = {d: int(sum(1 for _, g in E.groupby(d) if len(g) >= 30 and int((g.z > 0).sum()) >= 10)) for d in M.DECISION_DIMS}
    iv = 'populated' if all(v >= 2 for v in pop.values()) else 'thin'
    rec = ('calibration_candidate_defensible' if (i_ == 'broad' and iv == 'populated' and ii in ('decision_state', 'both')) else
           'representation_or_data_needed' if (i_ == 'concentrated' or iv == 'thin' or ii == 'fill_timing') else 'inconclusive')
    return dict(i=i_, ii=ii, iii=iii, iv=iv, recommendation=rec, pos=pos, share=share, dec=dec, dly=dly, ca=ca, cf=cf, pop=pop)


def check_G(P, D, R):
    au = {p['stage']: p for p in D['audit']['predicates']}; out = {}; okall = sorted(au) == sorted(M.STAGES)
    eq = lambda a, b: (a is None and b is None) or (a is not None and b is not None and close(a, b))
    for nm in M.STAGES:
        q = preds(D['O'], nm); a = au.get(nm)
        ok = bool(a and q['i'] == a['i']['answer'] and q['ii'] == a['ii']['answer'] and q['iii'] == a['iii']['answer'] and q['iv'] == a['iv']['answer']
                  and q['recommendation'] == a['recommendation'] and q['pos'] == a['i']['positive_cells'] and eq(q['share'], a['i']['top5_world_share'])
                  and all(eq(q['dec'][d], a['ii']['decision_ranges'][d]) for d in M.DECISION_DIMS) and eq(q['dly'], a['ii']['delay_range'])
                  and eq(q['ca'], a['iii']['c_admitted']) and eq(q['cf'], a['iii']['c_forced_overlap']) and q['pop'] == a['iv']['populated_cells'])
        out[nm] = dict(ok=ok, recomputed={k: q[k] for k in ('i', 'ii', 'iii', 'iv', 'recommendation')}); okall = okall and ok
    R['G'] = out
    return bool(okall)


H_EXPECT = dict(bias_V=(11.099, 3), qty_term=(-.882, 3), mov_term=(11.981, 3), mov_prob=(.310, 3), mov_cond=(11.671, 3), mse_V=(4043.27, 2), mse_zero=(3982.00, 2),
                mse_mech=(4098.46, 2), mse_m=(4336.46, 2), mse_m_zero=(4395.20, 2), mean_Eqm=(-1.779, 3), mean_M=(-13.760, 3))
H_CONTROL = dict(n_sel=233, n=299, percentile=.806, strata=91, free_strata=45, free_rows=166, cell_equal_admission=299)


def h_compare(row, conc_w, conc_u, ctl):
    """The review's numbers against one admitted entry 'all' moments row, the concentration shares and the control (header)."""
    ok = {k: bool(row is not None and round(float(row[k]), p) == round(v, p)) for k, (v, p) in H_EXPECT.items()}
    ok['top5_weighted'] = bool(conc_w is not None and round(float(conc_w), 3) == .520); ok['top5_unweighted'] = bool(conc_u is not None and round(float(conc_u), 2) == .37)
    ok['control'] = bool(ctl and all((round(float(ctl.get(k)), 3) == v) if isinstance(v, float) else (int(ctl.get(k, -1)) == v) for k, v in H_CONTROL.items()))
    return ok


def check_H(P, D, R):
    MO, CO = D['MO'], D['CO']; r = MO[(MO.cohort == 'admitted') & (MO.stage == 'entry') & (MO.view == 'all')]
    cw = CO[(CO.quantity == 'D_EXPOSURE') & (CO.weighting == 'weighted')].top5_generation_share; cu = CO[(CO.quantity == 'D_EXPOSURE') & (CO.weighting == 'unweighted')].top5_generation_share
    ok = h_compare(r.iloc[0].to_dict() if len(r) == 1 else None, cw.iloc[0] if len(cw) == 1 else None, cu.iloc[0] if len(cu) == 1 else None, D['audit'].get('control'))
    R['H'] = ok
    return all(ok.values())


def resources(logs, mids):
    peaks, cpu, bad = [], 0., []
    for p in glob.glob(str(Path(logs) / '**' / '*.log'), recursive=True):
        for ln in Path(p).read_text().splitlines():
            m = RL.match(ln)
            if m:
                peaks.append(int(m.group(4))); cpu += float(m.group(6)); bad += [m.group(2)] if int(m.group(4)) >= 2048 else []
    hw = [float(rec.get('hwm_mib', 1e9)) for rec in mids.values()]
    return dict(jobs=len(peaks), peak_mib=max(peaks) if peaks else None, core_hours=cpu / 3600., bad=bad, mids_hwm_max=max(hw) if hw else None,
                ok=bool(peaks and not bad and cpu / 3600. <= 1. and hw and max(hw) <= 1536.))


def suites(d):
    import importlib.util
    import pf_mutant_one as O_
    import pf_mutant_summary as S_
    h = 'pf_movement_audit_mutants.py'; spec = importlib.util.spec_from_file_location(h[:-3], RES / h); H = importlib.util.module_from_spec(spec); spec.loader.exec_module(H)
    files = {'control': Path(d) / 'mov_control.log', **{i: Path(d) / f'mov_{i}.log' for i in range(len(H.MUTANTS))}}
    recs = {k: S_.last_record(p) for k, p in files.items()}
    one = all(p.exists() and sum(1 for ln in p.read_text().splitlines() if ln.startswith('{') and '"cls"' in ln) == 1 for p in files.values())
    extra = sorted({x.name for x in Path(d).glob('mov_*.log')} - {p.name for p in files.values()})
    ok, bad = S_.judge(len(H.MUTANTS), recs, O_.sources(H), [m_[0] for m_ in H.MUTANTS])
    return dict(planted=len(H.MUTANTS), judge_ok=bool(ok), failures=bad[:5], one_record_per_file=bool(one), extra_files=extra, ok=bool(ok and one and not extra))


def check_I(P, D, R):
    rs = resources(P.get('logs', LOGS), D['mids']); su = suites(Path(P.get('logs', LOGS)) / 'mutants_mov')
    R['I'] = dict(resources=rs, suites=su)
    return bool(rs['ok'] and su['ok'])


CHECKS = (('A_identity', check_A), ('B_membership', check_B), ('C_accounting', check_C), ('D_strata', check_D), ('E_fills', check_E), ('F_denominators', check_F),
          ('G_predicates', check_G), ('H_reproduction', check_H), ('I_resources_suites', check_I))


def run_checks(P, names=None):
    R, ck = {}, {}
    try:
        D = load(P)
    except Exception as e:                                                                  # noqa: BLE001
        return {n: False for n, _ in CHECKS if names is None or n in names}, dict(load=repr(e)[:300])
    for n, f in CHECKS:
        if names is not None and n not in names:
            continue
        try:
            ck[n] = bool(f(P, D, R))
        except Exception as e:                                                              # noqa: BLE001
            ck[n] = False; R[n] = dict(crash=repr(e)[:300])
    return ck, R


# ---------------------------------------------------------------------------------------------------------------------- the falsifier
def copy_tree(P, d):
    d = Path(d); (d / 'adm' / 'parts').mkdir(parents=True); (d / 'ret' / 'parts').mkdir(parents=True); (d / 'out' / 'parts').mkdir(parents=True)
    shutil.copy(Path(P['adm']) / 'manifest.json', d / 'adm' / 'manifest.json'); shutil.copy(P['expo_man'], d / 'expo_manifest.json')
    for src, dst, pat in ((Path(P['adm']) / 'parts', d / 'adm' / 'parts', 'part_*'), (Path(P['ret']) / 'parts', d / 'ret' / 'parts', 'part_*'),
                          (Path(P['out']) / 'parts', d / 'out' / 'parts', 'mids_*.json')):
        for q in src.glob(pat):
            if q.is_file() and not q.name.endswith('_status.json'):
                shutil.copy(q, dst / q.name)
    for q in Path(P['out']).iterdir():
        if q.is_file():
            shutil.copy(q, d / 'out' / q.name)
    return dict(P, adm=d / 'adm', ret=d / 'ret', expo_man=d / 'expo_manifest.json', out=d / 'out')


def rewrite_part(Q, k, fn):
    gp, rp = Path(Q['adm']) / 'parts' / f'part_{k:03d}.jsonl.gz', Path(Q['adm']) / 'parts' / f'part_{k:03d}.json'
    rows = fn(gz_rows(gp.read_bytes())); nb = M.LS.gz_bytes(rows); rec = json.loads(rp.read_text()); gp.write_bytes(nb); rp.write_text(json.dumps(dict(rec, sha256=shb(nb))))


def csv_edit(Q, name, fn):
    p = Path(Q['out']) / f'{name}.csv'; df = pd.read_csv(p, float_precision='round_trip', dtype=STR); applied = fn(df); df.to_csv(p, index=False); return bool(applied)


def corruptions(real):
    """(name, the check it must flip, apply(Q) -> applied) - the A-G cases always, H-I on real data."""
    def a_sub(Q):
        ks = sorted(json.loads((Path(Q['adm']) / 'manifest.json').read_text())['parts']); gp = Path(Q['adm']) / 'parts' / f'part_{ks[0]:03d}.jsonl.gz'
        if len(ks) > 1:
            gp.write_bytes((Path(Q['adm']) / 'parts' / f'part_{ks[1]:03d}.jsonl.gz').read_bytes())
        else:
            gp.write_bytes(M.LS.gz_bytes(gz_rows(gp.read_bytes())[:-1]))
        return True

    def b_weight(Q):
        k = sorted(json.loads((Path(Q['adm']) / 'manifest.json').read_text())['parts'])[0]
        rewrite_part(Q, k, lambda rows: [dict(r, weight=2. * float(r['weight'])) if i == 0 else r for i, r in enumerate(rows)]); return True

    def b_drop(Q):
        k = sorted(json.loads((Path(Q['adm']) / 'manifest.json').read_text())['parts'])[0]; rewrite_part(Q, k, lambda rows: rows[:-1]); return True

    def first(df, m):
        idx = df.index[m]; return idx[0] if len(idx) else None

    def c_units(df):
        i = first(df, (df.acc_status == 'ok').to_numpy())
        if i is None:
            return False
        df.loc[i, 'V'] = float(df.loc[i, 'V']) * 100.; return True

    def c_sign(df):
        i = first(df, ((df.acc_status == 'ok') & (df.edge.astype(float) != 0)).to_numpy())
        if i is None:
            return False
        df.loc[i, 'edge'] = -float(df.loc[i, 'edge']); return True

    def d_venue(df):
        i = first(df, ((df.acc_status != 'not_eligible') & (df.venue_qbin != 'missing_or_invalid') & (df.venue_qbin != df.qbin)).to_numpy())
        if i is None:
            return False
        df.loc[i, 'qbin'] = df.loc[i, 'venue_qbin']; return True

    def e_time(df):
        if not len(df):
            return False
        df.loc[0, 't'] = float(df.loc[0, 't']) - 1000.; df.loc[0, 'delay'] = float(df.loc[0, 'delay']) - 1000.; return True

    def e_mark(df):
        if not len(df):
            return False
        sg = 1. if str(df.loc[0, 'side']) == '0' else -1.; df.loc[0, 'mark'] = float(df.loc[0, 'mark']) + 1.; df.loc[0, 'total'] = float(df.loc[0, 'total']) + sg * float(df.loc[0, 'q'])
        return True

    def f_count(df):
        df.loc[0, 'n'] = int(df.loc[0, 'n']) + 1; return True

    def g_reading(Q):
        p = Path(Q['out']) / 'audit.json'; j = json.loads(p.read_text()); p0 = j['predicates'][0]
        p0['recommendation'] = 'inconclusive' if p0['recommendation'] != 'inconclusive' else 'calibration_candidate_defensible'; p.write_text(json.dumps(j)); return True

    def h_number(df):
        i = first(df, ((df.cohort == 'admitted') & (df.stage == 'entry') & (df.view == 'all')).to_numpy())
        if i is None:
            return False
        df.loc[i, 'bias_V'] = float(df.loc[i, 'bias_V']) + .01; return True

    def i_budget(Q):
        (Path(Q['logs']) / 'over_budget.log').write_text('2026-11-21T00:00:00Z tag=over_budget rc=0 peak_rss_mib=100 wall_s=3700 cpu_user_s=3700.0\n'); return True

    def i_suite(Q):
        d = Path(Q['logs']) / 'mutants_mov'; shutil.copy(d / 'mov_0.log', d / 'mov_1.log'); return True
    cases = [('A_substituted', 'A_identity', a_sub), ('B_weight', 'B_membership', b_weight), ('B_drop', 'B_membership', b_drop),
             ('C_units', 'C_accounting', lambda Q: csv_edit(Q, 'opportunities', c_units)), ('C_sign', 'C_accounting', lambda Q: csv_edit(Q, 'opportunities', c_sign)),
             ('D_venue', 'D_strata', lambda Q: csv_edit(Q, 'opportunities', d_venue)), ('E_time', 'E_fills', lambda Q: csv_edit(Q, 'fills', e_time)),
             ('E_mark', 'E_fills', lambda Q: csv_edit(Q, 'fills', e_mark)), ('F_count', 'F_denominators', lambda Q: csv_edit(Q, 'moments', f_count)),
             ('G_reading', 'G_predicates', g_reading)]
    if real:
        cases += [('H_number', 'H_reproduction', lambda Q: csv_edit(Q, 'moments', h_number)), ('I_budget', 'I_resources_suites', i_budget),
                  ('I_suite', 'I_resources_suites', i_suite)]
    return cases


def falsify_on(P, td, real):
    """Clean copies pass every check of their case; each corruption flips its named check."""
    res = {}
    for name, chk, fn in corruptions(real):
        Q = copy_tree(P, Path(td) / f'{name}_clean'); Qc = copy_tree(P, Path(td) / name)
        if real:
            shutil.copytree(LOGS, Path(td) / f'{name}_clean_logs'); shutil.copytree(LOGS, Path(td) / f'{name}_logs')
            Q['logs'], Qc['logs'] = Path(td) / f'{name}_clean_logs', Path(td) / f'{name}_logs'
        clean, _ = run_checks(Q, [chk]); ap = fn(Qc); bad, det = run_checks(Qc, [chk])
        res[name] = dict(check=chk, clean=clean[chk], corrupted=bad[chk], applied=bool(ap))
        for x in (Q, Qc):
            shutil.rmtree(Path(x['out']).parent, ignore_errors=True)
            if real:
                shutil.rmtree(x['logs'], ignore_errors=True)
    return res, all(v['clean'] is True and v['corrupted'] is False and v['applied'] for v in res.values())


def falsify():
    with tempfile.TemporaryDirectory(dir=str(M.OUT)) as td:
        res, ok = falsify_on(REAL, td, True)
    (HERE / 'checks_falsifier.json').write_text(json.dumps(dict(cases=res, falsifier_ok=ok, asof=M.C.stamp()), indent=1, sort_keys=True) + '\n')
    print(json.dumps(dict(cases=res, falsifier_ok=ok)))
    return ok


def main():
    ck, R = run_checks(REAL)
    (HERE / 'checks.json').write_text(json.dumps(dict(checks=ck, all_ok=all(ck.values()), details=R, asof=M.C.stamp()), indent=1, sort_keys=True, default=str) + '\n')
    print(json.dumps(dict(checks=ck, all_ok=all(ck.values())), indent=1))
    return all(ck.values())


# ---------------------------------------------------------------------------------------------------------------------- the selftest
def hand(tmp):
    """Hand cohorts measured by the module (consistent by construction) -> paths."""
    LS, C, PA = M.LS, M.C, M.PA; vec = lambda **kw: [float(kw.get(n, 0.)) for n in FEATS]; t0 = 1792976399000

    def row(m, side, fills, mark, q, price, review=40., day='20261026', P=.3, qh=30., mh=-10.):
        sg = 1. if side == 0 else -1.; dmid = 50.5; edge = sg * (dmid - price) if q else .5; Mv = sg * q * (mark - dmid); t_ms = t0 + 1000 * m
        mo = dict(P=P, Eq=P * qh, Eqm=P * mh, V=edge * P * qh + P * mh, edge=edge)
        st = dict(status='ok', dec_status='model', action='POST', t=0., inputs=vec(pf_mid=dmid, fp_queue=10., fp_remaining=30., t_to_end=5000., ctx_generation_ordinal=1.),
                  moments=mo, target=dict(q=q, value=edge * q + Mv, qm=Mv if q else 0.), view=dict(received=1.))
        stages = dict(entry=st, first_review=dict(st, t=review, action='PULL', venue=dict(ahead=500.), effect=review + 37.), first_pull=dict(status='no_pull'))
        return dict(day=day, m=m, t_ms=t_ms, side=side, gid=0, decision_ms=0, weight=1., n_cell=1, cell=list(PA.cell_of(t_ms, 0, side)), status='ok',
                    held=dict(gid=0, act=58., mark=mark, fills=[list(f) for f in fills]), fills=[], stages=stages)
    A_ = [row(0, 0, [(100., 0, 0, 10., 50., 0)], 52., 10., 50.), row(1, 1, [(300., 0, 1, 5., 51., 0)], 52., 5., 51.), row(2, 0, [], 52., 0., 50.), row(3, 1, [], 52., 0., 51.)]
    adm = Path(tmp) / 'adm'; (adm / 'parts').mkdir(parents=True)
    sel = [dict({k: r[k] for k in ('day', 'm', 't_ms', 'side', 'gid', 'decision_ms')}, weight=1.5, n_cell=3, cell=r['cell'], part=0) for r in A_]
    for r in A_:
        r.update(weight=1.5, n_cell=3)
    (adm / 'manifest.json').write_text(json.dumps(dict(selected=sel, parts=[0]))); b = LS.gz_bytes(A_); (adm / 'parts' / 'part_000.jsonl.gz').write_bytes(b)
    (adm / 'parts' / 'part_000.json').write_text(json.dumps(dict(sha256=shb(b), manifest_sha256=C.sha(adm / 'manifest.json'))))
    F_ = [row(10, 0, [(100., 0, 0, 10., 50., 0)], 52., 10., 50., day='20261024'), row(10, 1, [], 52., 0., 51., day='20261024'),
          row(11, 0, [], 52., 0., 50.), row(11, 1, [(200., 0, 1, 4., 51., 0)], 51., 4., 51.)]
    F_[0]['fills'] = [dict(t=100., q=10., price=50., fill_time_edge=10. * (51. - 50.))]; F_[3]['fills'] = [dict(t=200., q=4., price=51., fill_time_edge=None)]
    for i, r in enumerate(F_):
        r['stages']['entry']['admission'] = 'would_POST' if i % 2 else 'would_WAIT'
    ret = Path(tmp) / 'ret'; (ret / 'parts').mkdir(parents=True); fman = Path(tmp) / 'fman.json'
    fman.write_text(json.dumps(dict(parts=[dict(worlds=[dict(day=r['day'], m=r['m'], t_ms=r['t_ms']) for r in F_[::2]])])))
    fb = LS.gz_bytes(F_); (ret / 'parts' / 'part_000.jsonl.gz').write_bytes(fb)
    (ret / 'parts' / 'part_000.json').write_text(json.dumps(dict(sha256=shb(fb), manifest_sha256=C.sha(fman))))
    out = Path(tmp) / 'out'; (out / 'parts').mkdir(parents=True)
    mids = {kg(A_[0]): dict(status='ok', mids={'0': dict(mid=51., frame_t=90., recv_t=95.)}), kg(A_[1]): dict(status='ok', mids={'0': None})}
    (out / 'parts' / 'mids_000.json').write_text(json.dumps(dict(mids=mids, sha256=shb(json.dumps(mids, sort_keys=True).encode()), admission_manifest_sha256=C.sha(adm / 'manifest.json'),
                                                                 sources=M.sources(), hwm_mib=100.)))
    M.measure(out, adm, ret, fman, out / 'parts')
    return dict(adm=adm, ret=ret, expo_man=fman, out=out, real=False)


def selftest():
    bad = []
    with tempfile.TemporaryDirectory(dir=str(M.PA.R2.I.K.IO)) as tmp:
        P = hand(tmp); ck, R = run_checks(P, [n for n, _ in CHECKS if n[0] in 'ABCDEFG'])
        if not all(ck.values()):
            bad.append(f'clean hand cohorts fail: {ck} {json.dumps(R, default=str)[:600]}')
        D = load(P); F = D['F']
        if not (len(F) == 8 and set(F.ref_status) == {'ok', 'reference_missing'} and (D['O'].acc_status == 'ok').sum() == 16):
            bad.append('the hand cohorts do not exercise both reference statuses')
        res, ok = falsify_on(P, Path(tmp) / 'fals', False)
        if not ok:
            bad.append(f'a corruption was not flagged: {res}')
    ok_h = h_compare(dict({k: v for k, (v, _) in H_EXPECT.items()}), .5204, .3712, dict(H_CONTROL))
    bad_h = h_compare(dict({k: v for k, (v, _) in H_EXPECT.items()}, bias_V=11.2), .5204, .3712, dict(H_CONTROL, percentile=.81))
    if not (all(ok_h.values()) and not bad_h['bias_V'] and not bad_h['control'] and sum(not v for v in bad_h.values()) == 2):
        bad.append(f'h_compare: {ok_h} {bad_h}')
    with tempfile.TemporaryDirectory(dir=str(M.PA.R2.I.K.IO)) as tmp:
        lg = Path(tmp); (lg / 'a.log').write_text('2026-11-21T00:00:00Z tag=a rc=0 peak_rss_mib=900 wall_s=10 cpu_user_s=100.0\n')
        good = resources(lg, {0: dict(hwm_mib=1200.)}); (lg / 'b.log').write_text('2026-11-21T00:00:00Z tag=b rc=0 peak_rss_mib=2100 wall_s=10 cpu_user_s=3600.0\n')
        over = resources(lg, {0: dict(hwm_mib=1200.)}); hw = resources(Path(tmp) / 'none', {0: dict(hwm_mib=1600.)})
        if not (good['ok'] and not over['ok'] and over['bad'] == ['b'] and not hw['ok']):
            bad.append(f'resources: {good} {over} {hw}')
    if bad:
        sys.exit('MOVEMENT CHECKS SELFTEST FAILED: ' + '; '.join(bad))
    print('movement checks selftest passed: A-G true on hand cohorts measured by the module; each A-G corruption flips its check; h_compare and resources on '
          'synthetic values')


if __name__ == '__main__':
    if sys.argv[1:] == ['selftest']:
        selftest()
    elif not sys.argv[1:]:
        a = main(); b = falsify(); sys.exit(0 if (a and b) else 1)
    else:
        sys.exit(__doc__)
