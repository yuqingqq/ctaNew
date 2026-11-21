"""PI0-MOVEMENT-SHRINK-v1: THE DECLARED CHECKS (review 7ad03681; 2026-11-21, session aea5bfb9). Declared in the freeze, before the fit. Each check
recomputes with its own loader, bins and arithmetic (only the module's constants are imported; review 7ad03681 item 2: every aggregate that drives
the receipt is recomputed, not only counts), compares with the saved artifacts, and is false when it cannot run.
 A identity: the module, selftest, harness, declaration, manifest, this file and the fit are the committed ones; both cohort manifests and every
   admission partition record equal their committed receipt copies; every partition's bytes match its record; the manifest's input hashes; the
   audit's fit_sha256, manifest_sha256 and sources equal the files.
 B membership: the manifest's TRAIN / EVAL / descriptive keys equal the eligible keys of the committed audit opportunities table (by stage and
   day); eval_rows.csv holds exactly the EVAL keys of each head with the admission manifest's weights; its P, E[q], E[qm], V, edge, q, value, z, M
   equal the audit table's and satisfy V - value = edge (E[q] - q) + (E[qm] - M) within 1e-6.
 C fit: every group's status, n, fills, weight, weighted fills, mu, lambda, lambda_raw, den, num and binding share recomputed from the TRAIN rows
   (the audit table's values, the mids from the admission partitions).
 D predictions: every EVAL row's a, bind, lambda, mu, E[qm] and V per arm and every rule's mask recomputed.
 E scores: every scores.csv row (all heads incl. the descriptive first pull, every side and arm, every column) recomputed.
 F calibration: every calibration.csv row recomputed; the cell sets equal.
 G actions: every actions.csv row (counts, weights, J, the mean among the chosen, worlds, Kish n, severe, the four top-5 shares, the changed
   count) recomputed.
 H null: the declared draws re-generated (seed, sorted strata, uniform without replacement) equal null_draws.csv; the summaries (strata, free
   strata / rows, chosen weight, J, mean, p05, p95, percentile, weight-mismatch draws) equal audit.json's.
 I readings: the prediction booleans and the decision of every head recomputed equal audit.json's.
 J resources and suites: every run_light job under logs/ below 2,048 MiB, the round within 1 core-hour (as of the check), the fit and score jobs
   below 1,536 MiB; pf_mutant_summary.judge passes the harness's final records against the final sources, one record per file, no other file.
FALSIFIER (falsify(), writes checks_falsifier.json): every check true on clean copies; the named check false after one corruption of a real copy:
 a substituted partition (A); a dropped EVAL key in the manifest (B); a fitted lambda (C); one row's candidate V (D); a conditional mean, an MSE
 and a weighted denominator in scores.csv (E, three cases); a calibration mean (F); a selection value J and a chosen weight (G, two cases); a null
 draw and the null p95 (H, two cases); a decision (I); an over-budget job and a copied mutant record (J, two cases).
SELFTEST (selftest()): C-I on hand rows scored by the module pass clean and fail after each corresponding corruption.
Usage (run_light): shrink_checks.py [selftest]  (no argument: the checks, then the falsifier)"""
import copy
import glob
import gzip
import hashlib
import io
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent; RES = Path('/home/yuqing/ctaNew/predictfun/research'); REPO = RES.parents[1]; sys.path.insert(0, str(RES))
import pf_movement_shrink as M                           # noqa: E402  constants, committed_blob / committed_equal only

pd, RT = M.pd, M.RT
LOGS = HERE / 'logs'
RL = re.compile(r'^(\S+) tag=(\S+) rc=(\d+) peak_rss_mib=(\d+) wall_s=([\d.]+) cpu_user_s=([\d.]+)')
STR = {c: str for c in ('day', 'key', 'cell', 'side', 'stage', 'cohort', 'acc_status', 'head', 'arm', 'rule', 'dim', 'kind', 'cday', 'hbin', 'qbin', 'obin', 'status')}
REAL = dict(adm=M.MA.ADM, out=M.OUT, rec=REPO / M.RECEIPT, logs=LOGS, real=True)
NUMS = ('P', 'Eq', 'Eqm', 'V', 'edge', 'q', 'value', 'z', 'M')


def close(a, b, tol=1e-9):
    if a is None or b is None or (isinstance(a, float) and np.isnan(a)) or (isinstance(b, float) and np.isnan(b)):
        an = a is None or (isinstance(a, float) and np.isnan(a)); bn = b is None or (isinstance(b, float) and np.isnan(b)); return bool(an and bn)
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return a == b
    return bool(abs(a - b) <= tol * max(1., abs(a), abs(b)))


def nn(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else x


def gz_rows(b):
    return [json.loads(x) for x in gzip.decompress(b).decode().splitlines() if x]


def gkey(r):
    return json.dumps([str(r['day']), int(r['m']), int(r['t_ms']), int(r['side']), int(r['gid']), int(r['decision_ms'])])


# ---------------------------------------------------------------------------------------------------------------------- independent arithmetic
def wmean(x, w):
    x, w = np.asarray(x, float), np.asarray(w, float); m = np.isfinite(x) & np.isfinite(w)
    return float((x[m] * w[m]).sum() / w[m].sum()) if m.any() and w[m].sum() > 0 else None


def ess(w):
    w = np.asarray(w, float); return float(w.sum() ** 2 / (w ** 2).sum()) if len(w) and (w ** 2).sum() > 0 else 0.


def proj(P, Eq, mid, side, mu):
    lo = Eq * (-mid) if int(side) == 0 else Eq * (mid - 100.); hi = Eq * (100. - mid) if int(side) == 0 else Eq * mid; raw = P * mu
    a = min(max(raw, lo), hi) if Eq != 0 else 0.
    return a, bool(raw < lo or raw > hi)


def fit_one(T, side):
    n = len(T); nf = int((T.z > 0).sum()); w = T.weight.to_numpy(float)
    base = dict(n=n, fills=nf, weight=float(w.sum()), weighted_fills=float((w * (T.z.to_numpy(float))).sum()))
    if n < 30 or nf < 10:
        return dict(base, status='insufficient_fit_support', mu=None, lam=0., lam_raw=None, den=None, num=None, binding_weight_share=None)
    mu = float((w * T.M.to_numpy(float)).sum() / (w * T.z.to_numpy(float)).sum())
    ab = [proj(p, e, m_, side, mu) for p, e, m_ in zip(T.P.astype(float), T.Eq.astype(float), T.mid.astype(float))]
    a = np.array([x[0] for x in ab]); bind = np.array([x[1] for x in ab]); b = T.Eqm.to_numpy(float); d = a - b
    den = float((w * d * d).sum()); num = float((w * d * (T.M.to_numpy(float) - b)).sum()); bw = float((w * bind).sum() / w.sum())
    if den <= 1e-12:
        return dict(base, status='zero_denominator', mu=mu, lam=0., lam_raw=None, den=den, num=num, binding_weight_share=bw)
    return dict(base, status='ok', mu=mu, lam=float(np.clip(num / den, 0., 1.)), lam_raw=num / den, den=den, num=num, binding_weight_share=bw)


def arms(E, params, head):
    E = E.copy(); rows = []
    for r in E.itertuples(index=False):
        g = params[f'{head}|{int(r.side)}']
        if g['mu'] is None:
            a, bind, lam, mu, lp = float(r.Eqm), False, 0., np.nan, 0.
        else:
            (a, bind), lam, mu, lp = proj(float(r.P), float(r.Eq), float(r.mid), r.side, float(g['mu'])), float(g['lam']), float(g['mu']), 1.
        d = a - float(r.Eqm)
        rows.append(dict(a=a, bind=bind, lam=lam, mu=mu, Eqm_original=float(r.Eqm), V_original=float(r.V), Eqm_prior=float(r.Eqm) + lp * d, V_prior=float(r.V) + lp * d,
                         Eqm_candidate=float(r.Eqm) + lam * d, V_candidate=float(r.V) + lam * d))
    X = pd.DataFrame(rows, index=E.index)
    for c in X.columns:
        E[c] = X[c]
    return E


def rule_masks(E, head):
    f = (lambda v: v > 0) if head == 'entry' else (lambda v: v >= 0)
    return {'none': np.zeros(len(E), bool), 'all': np.ones(len(E), bool), **{a: f(E[f'V_{a}'].to_numpy(float)) for a in M.ARMS}}


def top5(vals, w, keys):
    c = np.asarray(vals, float) * np.asarray(w, float)
    if not len(c):
        return None, None
    a = np.sort(np.abs(c))[::-1]; ws = {}
    for k, x in zip(keys, c):
        ws[k] = ws.get(k, 0.) + x
    aw = np.sort(np.abs(np.array(list(ws.values()))))[::-1]
    return (float(a[:5].sum() / a.sum()) if a.sum() > 0 else None), (float(aw[:5].sum() / aw.sum()) if aw.sum() > 0 else None)


def score_rows(E, head):
    out = []
    for side, g in [('all', E)] + [(str(s), E[E.side.astype(int) == s]) for s in M.SIDES]:
        if not len(g):
            continue
        w = g.weight.to_numpy(float); z = g.z.to_numpy(float); f = g[g.z > 0]; wf = f.weight.to_numpy(float); pool = wmean(z, w)
        base = dict(head=head, side=side, n=len(g), weight=float(w.sum()), ess=ess(w), worlds=len({(a, b, c) for a, b, c in zip(g.day, g.m, g.t_ms)}), fills=len(f),
                    weighted_fills=float(wf.sum()), brier=wmean((g.P - g.z) ** 2, w), brier_pooled=wmean((pool - g.z) ** 2, w), mean_P=wmean(g.P, w), fill_rate=wmean(z, w),
                    mean_M=wmean(g.M, w), mean_value=wmean(g.value, w), mse_zero=wmean(g.value ** 2, w), mse_mech=wmean((g.edge * g.Eq - g.value) ** 2, w),
                    mse_m_zero=wmean(g.M ** 2, w), binding_weight_share=wmean(g.bind.astype(float), w))
        for arm in M.ARMS:
            em = (g[f'Eqm_{arm}'] - g.M).to_numpy(float); ev = (g[f'V_{arm}'] - g.value).to_numpy(float); cm = (f[f'Eqm_{arm}'] / f.P - f.M).to_numpy(float)
            out.append(dict(base, arm=arm, mean_Eqm=wmean(g[f'Eqm_{arm}'], w), bias_m=wmean(em, w), mse_m=wmean(em ** 2, w), mae_m=wmean(np.abs(em), w),
                            mean_V=wmean(g[f'V_{arm}'], w), bias_V=wmean(ev, w), mse_V=wmean(ev ** 2, w), mae_V=wmean(np.abs(ev), w), c_m=wmean(cm, wf) if len(f) else None,
                            c_m_unweighted=float(cm.mean()) if len(f) else None, mse_c=wmean(cm ** 2, wf) if len(f) else None))
    return out


def ebin(e, edges):
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo < e <= hi:
            return f'({lo},{hi}]'
    return 'outside'


def calib_rows(E, head):
    out = []
    for arm in M.ARMS:
        G = E.assign(own_Eqm_bin=[ebin(x, M.MA.MBINS) for x in E[f'Eqm_{arm}'].astype(float)], own_V_bin=[ebin(x, M.MA.VBINS) for x in E[f'V_{arm}'].astype(float)])
        for dim in ('own_Eqm_bin', 'own_V_bin', 'hbin', 'qbin', 'obin'):
            for lab in sorted(set(G[dim].astype(str))):
                g = G[G[dim].astype(str) == lab]; w = g.weight.to_numpy(float)
                out.append(dict(head=head, arm=arm, dim=dim, cell=lab, n=len(g), fills=int((g.z > 0).sum()), weight=float(w.sum()), mean_pred_m=wmean(g[f'Eqm_{arm}'], w),
                                mean_M=wmean(g.M, w), mean_u=wmean(g[f'Eqm_{arm}'] - g.M, w), mean_pred_V=wmean(g[f'V_{arm}'], w), mean_value=wmean(g.value, w)))
    return out


def action_rows(E, head):
    mk = rule_masks(E, head); out = []
    for side, sel in [('all', np.ones(len(E), bool))] + [(str(s), (E.side.astype(int) == s).to_numpy()) for s in M.SIDES]:
        if not sel.any():
            continue
        g = E[sel]; w = g.weight.to_numpy(float); v = g.value.to_numpy(float); W = float(w.sum())
        for rule in M.RULES:
            c = mk[rule][sel]; wc, vc = w[c], v[c]; keys = list(zip(g.day[c], g.m[c], g.t_ms[c])); sev = vc < -30.
            gw, ww = top5(vc, wc, keys); gu, wu = top5(vc, np.ones(len(vc)), keys)
            out.append(dict(head=head, side=side, rule=rule, changed_vs_original=int((c != mk['original'][sel]).sum()), n=len(g), weight=W, chosen=int(c.sum()),
                            chosen_weight=float(wc.sum()), J=float((w * c * v).sum() / W) if W > 0 else None, mean_chosen=float((wc * vc).sum() / wc.sum()) if wc.sum() > 0 else None,
                            worlds=len(set(keys)), ess=ess(wc) if c.any() else 0., severe_n=int(sev.sum()), severe_sum=float((wc[sev] * vc[sev]).sum()),
                            top5_generation_share=gw, top5_world_share=ww, top5_generation_share_unweighted=gu, top5_world_share_unweighted=wu))
    return out, mk


def null_redo(E, mask, draws, seed):
    w = E.weight.to_numpy(float); v = E.value.to_numpy(float); W = float(w.sum()); rng = np.random.default_rng(seed)
    keys = sorted(set(zip(E.cday.astype(str), E.side.astype(int), E.chour.astype(int))), key=str)
    grp = [np.flatnonzero((E.cday.astype(str) == d).to_numpy() & (E.side.astype(int) == s).to_numpy() & (E.chour.astype(int) == h).to_numpy()) for d, s, h in keys]
    ks = [int(mask[g].sum()) for g in grp]; Wc = float((w * mask).sum()); Jc = float((w * mask * v).sum() / W); J = []; wm_ = 0
    for _ in range(draws):
        m = np.zeros(len(E), bool)
        for g, k in zip(grp, ks):
            if k:
                m[rng.choice(g, k, replace=False)] = True
        J.append(float((w * m * v).sum() / W)); wm_ += int(abs(float((w * m).sum()) - Wc) > 1e-9 * max(1., Wc))
    J = np.array(J); free = [(g, k) for g, k in zip(grp, ks) if 0 < k < len(g)]
    return dict(strata=len(grp), free_strata=len(free), free_rows=int(sum(len(g) for g, _ in free)), chosen=int(mask.sum()), chosen_weight=Wc, J=Jc, null_mean=float(J.mean()),
                null_p05=float(np.percentile(J, 5)), null_p95=float(np.percentile(J, 95)), percentile=float((J <= Jc + 1e-12).mean()), weight_mismatch_draws=wm_), J


def readings(params, SC, AC, NU, mk, head):
    s = {(r['side'], r['arm']): r for r in SC if r['head'] == head}; a = {(r['side'], r['rule']): r for r in AC if r['head'] == head}
    pred = {sd: dict(movement_mse_improves=bool(s[(sd, 'candidate')]['mse_m'] < s[(sd, 'original')]['mse_m']),
                     value_mse_improves=bool(s[(sd, 'candidate')]['mse_V'] < s[(sd, 'original')]['mse_V']),
                     movement_bias_reduced=bool(abs(s[(sd, 'candidate')]['bias_m']) < abs(s[(sd, 'original')]['bias_m']))) for sd in ('all', '0', '1') if (sd, 'candidate') in s}
    why = [n for n, b in (('no_fit_support', not any(params[f'{head}|{x}']['status'] == 'ok' for x in M.SIDES)),
                          ('candidate_mask_equals_original', bool((mk['candidate'] == mk['original']).all())), ('candidate_mask_equals_prior', bool((mk['candidate'] == mk['prior']).all())),
                          ('inadequate_movable_support', not (NU['free_strata'] >= 5 and NU['free_rows'] >= 30)), ('null_weight_mismatch', NU['weight_mismatch_draws'] > 0)) if b]
    J = {r: a[('all', r)]['J'] for r in M.RULES}
    dec = 'INCONCLUSIVE' if why else ('PASS' if (a[('all', 'candidate')]['chosen_weight'] > 0 and J['candidate'] > 0 and J['candidate'] > J['original'] and J['candidate'] > J['prior']
                                                 and J['candidate'] > NU['null_p95']) else 'FAIL')
    return dict(prediction=pred, decision=dec, reasons=why)


def rows_equal(got, want, keys):
    """Every recomputed row has exactly one saved row with the same keys and equal values; the row sets are equal."""
    G = {tuple(str(r[k]) for k in keys): r for r in got}; bad = []
    if len(G) != len(got) or len(want) != len(got):
        return [('count', len(got), len(want))]
    for r in want:
        k = tuple(str(r[k_]) for k_ in keys); g = G.get(k)
        if g is None:
            bad.append(('missing', k)); continue
        for c, v in r.items():
            if c not in keys and not close(nn(g.get(c)), nn(v)):
                bad.append(('value', k, c, g.get(c), v)); break
    return bad


# ---------------------------------------------------------------------------------------------------------------------- loading (real paths)
def load(P):
    adm, out, rec = Path(P['adm']), Path(P['out']), Path(P['rec'])
    am = json.loads((adm / 'manifest.json').read_text()); parts = {int(k): ((adm / 'parts' / f'part_{int(k):03d}.jsonl.gz').read_bytes(), json.loads((adm / 'parts' / f'part_{int(k):03d}.json').read_text()))
                                                                   for k in am['parts']}
    rows = [r for b, _ in parts.values() for r in gz_rows(b)]
    mid = {(gkey(r), nm): float(r['stages'][nm]['inputs'][RT.FEATS.index('pf_mid')]) for r in rows for nm in ('entry', 'first_review', 'first_pull')
           if (r.get('stages') or {}).get(nm, {}).get('inputs')}
    wsel = {gkey(g): float(g['weight']) for g in am['selected']}
    b = M.committed_blob(M.AUDIT_OPP_REL) if P.get('audit_opp') is None else Path(P['audit_opp']).read_bytes()
    A = pd.read_csv(io.BytesIO(gzip.decompress(b)), float_precision='round_trip', dtype=STR); A = A[A.cohort == 'admitted'].copy()
    A['mid'] = [mid.get((k, s), np.nan) for k, s in zip(A.key, A.stage)]
    cells = [M.PA.cell_of(int(t), float(d), int(s)) if np.isfinite(float(d)) else (None, None, None) for t, d, s in zip(A.t_ms, A.t_dec.astype(float), A.side)]
    A['cday'] = [c[0] for c in cells]; A['chour'] = [c[2] for c in cells]
    rd = lambda n: pd.read_csv(out / f'{n}.csv', float_precision='round_trip', dtype=STR)
    return dict(am=am, parts=parts, wsel=wsel, audit_opp_bytes=b, A=A, man=json.loads((rec / 'manifest.json').read_text()), fit=json.loads((rec / 'fit.json').read_text()),
                audit=json.loads((out / 'audit.json').read_text()), R=rd('eval_rows'), SC=rd('scores'), CA=rd('calibration'), AC=rd('actions'), ND=rd('null_draws'))


def eval_frame(D, head):
    E = D['A'][(D['A'].stage == head) & D['A'].key.isin(D['man']['eval'][head])].copy()
    E['weight'] = [D['wsel'][k] for k in E.key]
    order = list(D['R'][D['R']['head'] == head].key); E = E.set_index('key').loc[order].reset_index()
    return E


# ---------------------------------------------------------------------------------------------------------------------- the checks
def check_A(P, D, R):
    ok = {}; msha = hashlib.sha256((Path(P['adm']) / 'manifest.json').read_bytes()).hexdigest()
    ok['partitions'] = all(hashlib.sha256(b).hexdigest() == rec['sha256'] and rec['manifest_sha256'] == msha for b, rec in D['parts'].values())
    rec_shas = {str(k): hashlib.sha256((Path(P['adm']) / 'parts' / f'part_{k:03d}.json').read_bytes()).hexdigest() for k in D['parts']}
    ok['manifest_inputs'] = bool(D['man']['admission_manifest_sha256'] == msha and D['man']['admission_records_sha256'] == rec_shas
                                 and D['man']['audit_opportunities_sha256'] == hashlib.sha256(D['audit_opp_bytes']).hexdigest())
    ok['audit_bindings'] = bool(D['audit']['fit_sha256'] == hashlib.sha256((Path(P['rec']) / 'fit.json').read_bytes()).hexdigest()
                                and D['audit']['manifest_sha256'] == hashlib.sha256((Path(P['rec']) / 'manifest.json').read_bytes()).hexdigest() == D['fit']['manifest_sha256'])
    if P['real']:
        ok['committed'] = bool(all(M.committed_equal(f'predictfun/research/{n}', RES / n) for n in ('pf_movement_shrink.py', 'pf_movement_shrink_selftest.py', 'pf_movement_shrink_mutants.py'))
                               and all(M.committed_equal(f'{M.RECEIPT}/{n}', Path(P['rec']) / n) for n in ('declaration.md', 'manifest.json', 'fit.json'))
                               and M.committed_equal(f'{M.RECEIPT}/shrink_checks.py', HERE / 'shrink_checks.py'))
        ok['inputs_committed'] = bool(M.MA.inputs_frozen() and all(M.committed_equal(f'{M.ADM_REC_REL}/part_{k:03d}.json', Path(P['adm']) / 'parts' / f'part_{k:03d}.json') for k in D['parts'])
                                      and D['audit_opp_bytes'] == M.committed_blob(M.AUDIT_OPP_REL))
        ok['sources'] = bool(D['audit']['sources'] == M.sources() == D['fit']['sources'])
    R['A'] = ok
    return all(ok.values())


def check_B(P, D, R):
    A = D['A']; ok = {}
    pick = lambda day, nm: sorted(A[(A.acc_status == 'ok') & (A.day == day) & (A.stage == nm)].key)
    ok['manifest_keys'] = bool(D['man']['train'] == {nm: pick(M.TRAIN_DAY, nm) for nm in M.HEADS} and D['man']['eval'] == {nm: pick(M.EVAL_DAY, nm) for nm in M.HEADS}
                               and D['man']['descriptive'] == {'first_pull': pick(M.EVAL_DAY, 'first_pull')})
    Rr = D['R']; bad = []
    for nm in M.HEADS:
        g = Rr[Rr['head'] == nm]
        if sorted(g.key) != D['man']['eval'][nm] or len(set(g.key)) != len(g):
            bad.append(('keys', nm)); continue
        Aa = A[A.stage == nm].set_index('key')
        for r in g.itertuples(index=False):
            ar = Aa.loc[r.key]
            if not (close(r.weight, D['wsel'][r.key]) and all(close(getattr(r, c), ar[c]) for c in NUMS) and str(ar.acc_status) == 'ok'
                    and abs((float(r.V) - float(r.value)) - (float(r.edge) * (float(r.Eq) - float(r.q)) + float(r.Eqm) - float(r.M))) <= 1e-6):
                bad.append(('row', nm, r.key)); break
    ok['eval_rows'] = not bad; R['B'] = dict(ok, failures=bad[:3])
    return all(ok.values())


def train_frame(D, head):
    T = D['A'][(D['A'].stage == head) & D['A'].key.isin(D['man']['train'][head])].copy(); T['weight'] = [D['wsel'][k] for k in T.key]
    return T


def check_C(P, D, R):
    bad = []
    for nm in M.HEADS:
        T = train_frame(D, nm)
        for s in M.SIDES:
            got = fit_one(T[T.side.astype(int) == s], s); want = D['fit']['params'][f'{nm}|{s}']
            if not all(close(nn(got[k]), nn(want.get(k))) for k in got):
                bad.append((nm, s, {k: (got[k], want.get(k)) for k in got if not close(nn(got[k]), nn(want.get(k)))}))
    R['C'] = dict(failures=bad[:3])
    return not bad


def check_D(P, D, R):
    bad = []
    for nm in M.HEADS:
        E = arms(eval_frame(D, nm), D['fit']['params'], nm); g = D['R'][D['R']['head'] == nm].reset_index(drop=True); mk = rule_masks(E, nm)
        for c in ('a', 'lam', 'mu', 'Eqm_original', 'V_original', 'Eqm_prior', 'V_prior', 'Eqm_candidate', 'V_candidate'):
            if not all(close(nn(x), nn(y)) for x, y in zip(E[c].astype(float), g[c].astype(float))):
                bad.append((nm, c)); break
        if not all(str(bool(x)) == str(y) for x, y in zip(E.bind, g.bind)) or not all(np.array_equal(mk[r_], g[f'mask_{r_}'].astype(str).eq('True').to_numpy()) for r_ in M.RULES):
            bad.append((nm, 'bind_or_masks'))
    R['D'] = dict(failures=bad[:3])
    return not bad


def recomputed(D):
    SC, CA, AC, NU, MK = [], [], [], {}, {}
    for nm in M.HEADS:
        E = arms(eval_frame(D, nm), D['fit']['params'], nm); SC += score_rows(E, nm); CA += calib_rows(E, nm); ac, mk = action_rows(E, nm); AC += ac; MK[nm] = mk
        NU[nm] = null_redo(E, mk['candidate'], M.NULL_DRAWS, M.SEED)
    Dp = D['A'][(D['A'].stage == 'first_pull') & D['A'].key.isin(D['man']['descriptive']['first_pull'])].copy()
    if len(Dp):
        Dp['weight'] = [D['wsel'][k] for k in Dp.key]; SC += score_rows(arms(Dp, {f'first_pull|{s}': D['fit']['params'][f'first_review|{s}'] for s in M.SIDES}, 'first_pull'), 'first_pull')
    return SC, CA, AC, NU, MK


def check_E(P, D, R, RC=None):
    SC = (RC or recomputed(D))[0]; bad = rows_equal(D['SC'].to_dict('records'), SC, ('head', 'side', 'arm')); R['E'] = dict(rows=len(SC), failures=bad[:3])
    return not bad


def check_F(P, D, R, RC=None):
    CA = (RC or recomputed(D))[1]; bad = rows_equal(D['CA'].to_dict('records'), CA, ('head', 'arm', 'dim', 'cell')); R['F'] = dict(rows=len(CA), failures=bad[:3])
    return not bad


def check_G(P, D, R, RC=None):
    AC = (RC or recomputed(D))[2]; bad = rows_equal(D['AC'].to_dict('records'), AC, ('head', 'side', 'rule')); R['G'] = dict(rows=len(AC), failures=bad[:3])
    return not bad


def check_H(P, D, R, RC=None):
    NU = (RC or recomputed(D))[3]; bad = []
    for nm in M.HEADS:
        s, J = NU[nm]; saved = D['ND'][D['ND']['head'] == nm].sort_values('draw', key=lambda x: x.astype(int))
        if len(saved) != len(J) or not all(close(a, b, 1e-12) for a, b in zip(saved.J.astype(float), J)):
            bad.append((nm, 'draws'))
        au = D['audit']['null'][nm]
        if not all(close(s[k], au.get(k)) for k in s):
            bad.append((nm, {k: (s[k], au.get(k)) for k in s if not close(s[k], au.get(k))}))
    R['H'] = dict(failures=bad[:3])
    return not bad


def check_I(P, D, R, RC=None):
    SC, CA, AC, NU, MK = RC or recomputed(D); bad = []
    for nm in M.HEADS:
        q = readings(D['fit']['params'], SC, AC, NU[nm][0], MK[nm], nm); a = next(p for p in D['audit']['predicates'] if p['head'] == nm)
        if not (q['decision'] == a['decision'] and q['reasons'] == a['inconclusive_reasons'] and q['prediction'] == a['prediction']
                and all(close(q_, a['J'][r]) for r, q_ in {r: next(x['J'] for x in AC if x['head'] == nm and x['side'] == 'all' and x['rule'] == r) for r in M.RULES}.items())):
            bad.append((nm, q['decision'], a['decision']))
    R['I'] = dict(failures=bad)
    return not bad


def resources(logs):
    peaks, cpu, bad, jobs = [], 0., [], {}
    for p in glob.glob(str(Path(logs) / '**' / '*.log'), recursive=True):
        for ln in Path(p).read_text().splitlines():
            m = RL.match(ln)
            if m:
                peaks.append(int(m.group(4))); cpu += float(m.group(6)); bad += [m.group(2)] if int(m.group(4)) >= 2048 else []; jobs[m.group(2)] = int(m.group(4))
    run = {k: v for k, v in jobs.items() if k in ('shr_fit', 'shr_score')}
    return dict(jobs=len(peaks), peak_mib=max(peaks) if peaks else None, core_hours=cpu / 3600., bad=bad, run_peaks=run,
                ok=bool(peaks and not bad and cpu / 3600. <= 1. and len(run) == 2 and max(run.values()) < 1536))


def suites(d):
    import importlib.util
    import pf_mutant_one as O_
    import pf_mutant_summary as S_
    h = 'pf_movement_shrink_mutants.py'; spec = importlib.util.spec_from_file_location(h[:-3], RES / h); H = importlib.util.module_from_spec(spec); spec.loader.exec_module(H)
    files = {'control': Path(d) / 'shr_control.log', **{i: Path(d) / f'shr_{i}.log' for i in range(len(H.MUTANTS))}}
    recs = {k: S_.last_record(p) for k, p in files.items()}
    one = all(p.exists() and sum(1 for ln in p.read_text().splitlines() if ln.startswith('{') and '"cls"' in ln) == 1 for p in files.values())
    extra = sorted({x.name for x in Path(d).glob('shr_*.log')} - {p.name for p in files.values()})
    ok, bad = S_.judge(len(H.MUTANTS), recs, O_.sources(H), [m_[0] for m_ in H.MUTANTS])
    return dict(planted=len(H.MUTANTS), judge_ok=bool(ok), failures=bad[:3], one_record_per_file=bool(one), extra_files=extra, ok=bool(ok and one and not extra))


def check_J(P, D, R):
    rs = resources(P['logs']); su = suites(Path(P['logs']) / 'mutants_shr'); R['J'] = dict(resources=rs, suites=su, asof=M.C.stamp())
    return bool(rs['ok'] and su['ok'])


CHECKS = (('A_identity', check_A), ('B_membership', check_B), ('C_fit', check_C), ('D_predictions', check_D), ('E_scores', check_E), ('F_calibration', check_F),
          ('G_actions', check_G), ('H_null', check_H), ('I_readings', check_I), ('J_resources_suites', check_J))


def run_checks(P, names=None):
    R, ck = {}, {}
    try:
        D = load(P)
    except Exception as e:                                                                  # noqa: BLE001
        return {n: False for n, _ in CHECKS if names is None or n in names}, dict(load=repr(e)[:300])
    RC = None
    for n, f in CHECKS:
        if names is not None and n not in names:
            continue
        try:
            if n[0] in 'EFGHI':
                RC = RC or recomputed(D); ck[n] = bool(f(P, D, R, RC))
            else:
                ck[n] = bool(f(P, D, R))
        except Exception as e:                                                              # noqa: BLE001
            ck[n] = False; R[n] = dict(crash=repr(e)[:300])
    return ck, R


# ---------------------------------------------------------------------------------------------------------------------- the falsifier
def copy_tree(P, d):
    d = Path(d); (d / 'adm' / 'parts').mkdir(parents=True); (d / 'out').mkdir(parents=True); (d / 'rec').mkdir(parents=True)
    shutil.copy(Path(P['adm']) / 'manifest.json', d / 'adm' / 'manifest.json')
    for q in (Path(P['adm']) / 'parts').glob('part_*'):
        shutil.copy(q, d / 'adm' / 'parts' / q.name)
    for q in Path(P['out']).iterdir():
        if q.is_file():
            shutil.copy(q, d / 'out' / q.name)
    for n in ('manifest.json', 'fit.json', 'declaration.md'):
        if (Path(P['rec']) / n).exists():
            shutil.copy(Path(P['rec']) / n, d / 'rec' / n)
    (d / 'audit_opp.csv.gz').write_bytes(M.committed_blob(M.AUDIT_OPP_REL) if P.get('audit_opp') is None else Path(P['audit_opp']).read_bytes())
    Q = dict(P, adm=d / 'adm', out=d / 'out', rec=d / 'rec', audit_opp=d / 'audit_opp.csv.gz')
    if P.get('logs') is not None:
        shutil.copytree(P['logs'], d / 'logs'); Q['logs'] = d / 'logs'
    return Q


def csv_edit(Q, name, fn):
    p = Path(Q['out']) / f'{name}.csv'; df = pd.read_csv(p, float_precision='round_trip', dtype=STR); ok = fn(df); df.to_csv(p, index=False); return bool(ok)


def json_edit(p, fn):
    j = json.loads(Path(p).read_text()); ok = fn(j); Path(p).write_text(json.dumps(j, indent=1, sort_keys=True)); return bool(ok)


def first(df, m):
    idx = df.index[m.to_numpy() if hasattr(m, 'to_numpy') else m]; return idx[0] if len(idx) else None


def corruptions(real):
    def bump(col, d_, cond):
        def f(df):
            i = first(df, cond(df))
            if i is None:
                return False
            df.loc[i, col] = float(df.loc[i, col]) + d_; return True
        return f

    def a_sub(Q):
        ks = sorted(json.loads((Path(Q['adm']) / 'manifest.json').read_text())['parts']); gp = Path(Q['adm']) / 'parts' / f'part_{ks[0]:03d}.jsonl.gz'
        gp.write_bytes((Path(Q['adm']) / 'parts' / f'part_{ks[-1]:03d}.jsonl.gz').read_bytes() if len(ks) > 1 else gzip.compress(b'{}\n', mtime=0)); return True

    def b_drop(Q):
        return json_edit(Path(Q['rec']) / 'manifest.json', lambda j: bool(j['eval']['entry']) and (j['eval']['entry'].pop() is not None))

    def c_lam(Q):
        return json_edit(Path(Q['rec']) / 'fit.json', lambda j: (j['params'].__setitem__('entry|0', dict(j['params']['entry|0'], lam=float(j['params']['entry|0']['lam']) + .01)) or True))

    def i_dec(Q):
        def f(j):
            p0 = next(p for p in j['predicates'] if p['head'] == 'entry'); p0['decision'] = 'PASS' if p0['decision'] != 'PASS' else 'FAIL'; return True
        return json_edit(Path(Q['out']) / 'audit.json', f)

    def h_p95(Q):
        return json_edit(Path(Q['out']) / 'audit.json', lambda j: (j['null']['entry'].__setitem__('null_p95', float(j['null']['entry']['null_p95']) + .01) or True))

    def j_budget(Q):
        (Path(Q['logs']) / 'over_budget.log').write_text('2026-11-21T00:00:00Z tag=over_budget rc=0 peak_rss_mib=100 wall_s=3700 cpu_user_s=3700.0\n'); return True

    def j_suite(Q):
        d = Path(Q['logs']) / 'mutants_shr'; shutil.copy(d / 'shr_0.log', d / 'shr_1.log'); return True
    ok_rows = lambda df: (df['head'] == 'entry') & (df.side == 'all')
    cases = [('B_drop', 'B_membership', b_drop), ('C_lambda', 'C_fit', c_lam),
             ('D_candidate_V', 'D_predictions', lambda Q: csv_edit(Q, 'eval_rows', bump('V_candidate', 1., lambda df: df['head'] == 'entry'))),
             ('E_conditional_mean', 'E_scores', lambda Q: csv_edit(Q, 'scores', bump('c_m', 100., lambda df: ok_rows(df) & (df.arm == 'candidate')))),
             ('E_mse', 'E_scores', lambda Q: csv_edit(Q, 'scores', bump('mse_m', 1., lambda df: ok_rows(df) & (df.arm == 'candidate')))),
             ('E_weighted_denominator', 'E_scores', lambda Q: csv_edit(Q, 'scores', bump('weighted_fills', 1., lambda df: ok_rows(df)))),
             ('F_calibration_mean', 'F_calibration', lambda Q: csv_edit(Q, 'calibration', bump('mean_u', 1., lambda df: df['head'] == 'entry'))),
             ('G_selection_value', 'G_actions', lambda Q: csv_edit(Q, 'actions', bump('J', .1, lambda df: ok_rows(df) & (df.rule == 'candidate')))),
             ('G_chosen_weight', 'G_actions', lambda Q: csv_edit(Q, 'actions', bump('chosen_weight', 1., lambda df: ok_rows(df) & (df.rule == 'candidate')))),
             ('H_draw', 'H_null', lambda Q: csv_edit(Q, 'null_draws', bump('J', .01, lambda df: df['head'] == 'entry'))), ('H_p95', 'H_null', h_p95),
             ('I_decision', 'I_readings', i_dec)]
    if real:
        cases = [('A_substituted', 'A_identity', a_sub)] + cases + [('J_budget', 'J_resources_suites', j_budget), ('J_suite', 'J_resources_suites', j_suite)]
    return cases


def falsify_on(P, td, real):
    res = {}
    for name, chk, fn in corruptions(real):
        Q = copy_tree(P, Path(td) / f'{name}_clean'); Qc = copy_tree(P, Path(td) / name)
        clean, _ = run_checks(Q, [chk]); ap = fn(Qc); bad, _ = run_checks(Qc, [chk])
        res[name] = dict(check=chk, clean=clean[chk], corrupted=bad[chk], applied=bool(ap))
        shutil.rmtree(Path(td) / f'{name}_clean', ignore_errors=True); shutil.rmtree(Path(td) / name, ignore_errors=True)
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
def selftest():
    """C-I on hand rows scored by the module (B and A on real paths only)."""
    bad = []
    with tempfile.TemporaryDirectory(dir=str(M.PA.R2.I.K.IO)) as tmp:
        tmp = Path(tmp); rng = np.random.default_rng(7); rows = []
        for day, nn_ in ((M.TRAIN_DAY, 80), (M.EVAL_DAY, 80)):
            for i in range(nn_):
                for st in M.HEADS:
                    z = float(i % 3 == 0); P_ = float(rng.uniform(.1, .6)); Eq = 10.; b = float(rng.uniform(-8, 0)); edge = float(rng.uniform(.05, .3)); Mv = float(rng.uniform(-80, 20)) * z
                    rows.append(dict(key=json.dumps([day, i]), day=day, m=i, t_ms=1000 * i, side=i % 2, stage=st, cohort='admitted', acc_status='ok', weight=float(1 + i % 4),
                                     P=P_, Eq=Eq, Eqm=b, V=edge * Eq + b, edge=edge, q=10. * z, value=edge * 10. * z + Mv, z=z, M=Mv, mid=float(rng.uniform(20, 80)),
                                     cday=day, chour=i % 8, hbin='(2000,5000]', qbin='(0,1]', obin='1', t_dec=0.))
        A = pd.DataFrame(rows); man = dict(train={nm: sorted(A[(A.day == M.TRAIN_DAY) & (A.stage == nm)].key) for nm in M.HEADS},
                                           eval={nm: sorted(A[(A.day == M.EVAL_DAY) & (A.stage == nm)].key) for nm in M.HEADS}, descriptive={'first_pull': []})
        O = A.assign(status='ok'); params = M.fit_all(O[O.day == M.TRAIN_DAY], man); out = tmp / 'out'; out.mkdir()
        js = M.score_all(O, man, params, out); (out / 'audit.json').write_text(json.dumps(js, default=str))
        rd = lambda n: pd.read_csv(out / f'{n}.csv', float_precision='round_trip', dtype=STR)
        D = dict(A=A.assign(acc_status='ok'), man=man, fit=dict(params=params), wsel={k: float(w) for k, w in zip(A.key, A.weight)}, audit=json.loads((out / 'audit.json').read_text()),
                 R=rd('eval_rows'), SC=rd('scores'), CA=rd('calibration'), AC=rd('actions'), ND=rd('null_draws'))
        P0 = dict(real=False); fns = dict(C=check_C, D=check_D, E=check_E, F=check_F, G=check_G, H=check_H, I=check_I)
        for k, f in fns.items():
            if not f(P0, D, {}):
                bad.append(f'{k} false on the clean hand rows')
        def bump_df(name, col, d_):
            def f(D_):
                X = D_[name].copy(); v = X[col].astype(float); i = X.index[np.isfinite(v.to_numpy())][0]; X[col] = v; X.loc[i, col] = float(v.loc[i]) + d_; D_[name] = X
            return f

        def bump_mu(D_):
            g = dict(D_['fit']['params']['first_review|1']); g['mu'] = float(g['mu']) + 1.; D_['fit']['params']['first_review|1'] = g

        def flip(D_):
            p_ = next(p for p in D_['audit']['predicates'] if p['head'] == 'first_review'); p_['decision'] = 'PASS' if p_['decision'] != 'PASS' else 'FAIL'
        corr = [('C', bump_mu), ('D', bump_df('R', 'Eqm_candidate', 1.)), ('E', bump_df('SC', 'c_m', 100.)), ('F', bump_df('CA', 'mean_M', 1.)),
                ('G', bump_df('AC', 'mean_chosen', 1.)), ('H', bump_df('ND', 'J', .01)), ('I', flip)]
        for k, fn in corr:
            D2 = copy.deepcopy(D); fn(D2)
            if fns[k](P0, D2, {}):
                bad.append(f'{k} true after its corruption')
        if not all(p['decision'] in ('PASS', 'FAIL', 'INCONCLUSIVE') for p in D['audit']['predicates']):
            bad.append('the hand readings')
    if bad:
        sys.exit('SHRINK CHECKS SELFTEST FAILED: ' + '; '.join(bad))
    print('shrink checks selftest passed: C-I true on hand rows scored by the module; each corruption (mu, a candidate E[qm], a conditional mean, a calibration mean, '
          'a mean among the chosen, a null draw, a decision) flips its check')


if __name__ == '__main__':
    if sys.argv[1:] == ['selftest']:
        selftest()
    elif not sys.argv[1:]:
        a = main(); b = falsify(); sys.exit(0 if (a and b) else 1)
    else:
        sys.exit(__doc__)
