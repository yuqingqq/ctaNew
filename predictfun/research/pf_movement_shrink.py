"""PI0-MOVEMENT-SHRINK-v1 (P-2026-005; 2026-11-21, session aea5bfb9). Review 7ad03681's one next priority: ONE declared scalar movement-calibration
challenger, fitted on the consumed Oct 26 admitted states and measured on the consumed Oct 27 admitted states, with the prediction test and the local
action-value test kept separate. A new adaptive experiment after the movement audit (f311caae; its `representation_or_data_needed` flag is a
declared research heuristic and stays as recorded), not a retroactive pass of that gate; chronological consumed-day development, not unseen
validation. One new candidate (multiplicity in the manifest). No new date, label, branch, teacher fit, END or mid lookup.
INPUTS (immutable, bound): PI0-ADMISSION-EXPOSURE-v1's 12 partitions through the frozen pf_movement_audit.adm_rows (bytes = record sha256, record
 manifest = the manifest, each row's weight / n_cell / cell / part bound to its manifest row by exact key) plus every partition record byte-equal to
 its committed receipt copy (predictfun/reviews/pi0_admission_20261121/outputs/parts) and both cohort manifests committed (inputs_frozen); the
 opportunities through the frozen pf_movement_audit.tables_of (the audit's statuses and identities); every eligible entry / first-review / first-pull
 row reconciled with the audit's committed opportunities table (P, E[q], E[qm], V, edge, q, value, z, M within 1e-9; else audit_mismatch).
POPULATIONS: heads entry (POST) and first_review (RETAIN), sides 0 / 1 -> 4 groups. TRAIN = the Oct 26 rows with acc_status ok; EVAL = the Oct 27 rows
 with acc_status ok; first_pull (Oct 27, the first-review parameters) is descriptive only, never in training or the main scores. Keys and statuses of
 every row are in the manifest (frozen before the fit); every other row is counted with its status.
FIT (fit_group(); refuses any row not of TRAIN_DAY): with >= 30 rows and >= 10 fills (z = 1(q > 1e-12)) in the group, else insufficient_fit_support
 (mu none, lambda 0: the original forecast, the prior arm equal to it): mu = sum(w M) / sum(w z) (c per filled opportunity, quantity-weighted; M = qm,
 0 without a fill; w = the manifest weight); a = clip(P mu, E[q] L, E[q] U) with (L, U) = (-mid, 100 - mid) for bids and (mid - 100, mid) for asks,
 mid = the stage's own pf_mid input (c; physical bounds, no fitted threshold), E[q] = 0 -> a = 0; d = a - b, b = E[qm];
 lambda = clip(sum(w d (M - b)) / sum(w d^2), 0, 1) over all the group's rows; sum(w d^2) <= 1e-12 -> lambda 0, zero_denominator.
PREDICT (arms on identical rows): original (lambda 0), prior (lambda 1, a fixed control), candidate (the fitted lambda): E[qm]_arm = b + lambda (a - b),
 V_arm = V + lambda (a - b) (= edge E[q] + E[qm]_arm by V's identity; exact at lambda 0); P, E[q], edge frozen; baselines zero and edge x E[q].
SCORES (EVAL; per head, pooled and per side; weighted by w; MAE secondary): n, weight, Kish n, worlds, fills (weighted and unweighted denominators);
 movement and V bias / MSE / MAE; zero, mechanical and zero-movement baselines; Brier (unchanged) and the pooled rate; the conditional error on fills
 mhat_arm - M (c per filled opportunity, mhat_arm = E[qm]_arm / P); the projection's binding rate. Calibration: by each arm's own E[qm] and V bins
 (the audit's MBINS / VBINS) and the audit's horizon / queue / ordinal bins (diagnostics only).
LOCAL ACTIONS (EVAL): entry POST iff V_arm > 0 (a tie WAITs; value = the entry target, the neutral hold from activation to END; WAIT = 0); first review
 KEEP iff V_arm >= 0 (value = the review target, the hold after the cancel effect; PULL = 0). Rules none / all / original / prior / candidate.
 J = sum(w chosen value) / sum(w) over the slice's eligible rows (c per opportunity); the chosen count and weight, the mean value among the chosen,
 worlds, Kish n, severe (value < -SEVERE) count and sum, the top-5 generation / world shares of |w value| (weighted and unweighted) among the chosen.
 Stages are never summed. NULL: NULL_DRAWS draws (seed SEED) of the candidate's mask, uniform within decision day x side x decision-UTC-hour (the stage's
 own decision time), each stratum's chosen count preserved; every draw's chosen weight must equal the candidate's (else null_weight_mismatch); free
 strata (0 < k < n) and their rows; the null mean, p05, p95 and the candidate's percentile.
READINGS (predicates(), per head): prediction - movement_mse_improves, value_mse_improves, movement_bias_reduced (|bias|), each pooled and per side;
 decision - INCONCLUSIVE if no group of the head has fit support, the candidate's mask equals the original's or the prior's, free strata < MIN_FREE_STRATA,
 free rows < MIN_FREE_ROWS or a null weight mismatch; PASS if the chosen weight > 0, J > 0, J > J_original, J > J_prior and J > the null p95; FAIL
 otherwise. No combined verdict: a lower bias with a worse MSE or a worse selection stays visible.
Usage (run_light): pf_movement_shrink.py selftest | manifest | fit | score"""
import gzip
import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_movement_audit as MA                           # noqa: E402  frozen: the bound admitted cohort, its accounting and strata

C, pd, PA, RT = MA.C, MA.pd, MA.PA, MA.RT
VERSION = 'pi0_movement_shrink_v1'
REPO = HERE.parents[1]
OUT = PA.A.DATA / 'pi0_movement_shrink_v1_20261121'
RECEIPT = 'predictfun/reviews/movement_shrink_20261121'
ADM_REC_REL = 'predictfun/reviews/pi0_admission_20261121/outputs/parts'
AUDIT_OPP_REL = 'predictfun/reviews/movement_audit_20261121/outputs/opportunities.csv.gz'
TRAIN_DAY, EVAL_DAY = '20261026', '20261027'
HEADS, SIDES, ARMS = ('entry', 'first_review'), (0, 1), ('original', 'prior', 'candidate')
RULES = ('none', 'all', 'original', 'prior', 'candidate')
MIN_ROWS, MIN_FILLS, DEN_EPS, TOL = 30, 10, 1e-12, 1e-9
NULL_DRAWS, SEED, SEVERE = 500, 20261121, 30.
MIN_FREE_STRATA, MIN_FREE_ROWS = 5, 30
MULTIPLICITY = dict(new_candidates=1, groups=4, scalar_parameters=8, prior_candidates_on_these_days=['R3 ADV2-RICH-v1 (12 candidate heads)',
                                                                                                       'R4E ADV2-ENTRY-SMOOTH-v1 (one primary, two heads)'])
SOURCES = ('pf_movement_shrink.py', 'pf_movement_shrink_selftest.py', 'pf_movement_audit.py', 'pf_pi0_admission.py', 'pf_entry_retention.py')
COLS = ('P', 'Eq', 'Eqm', 'V', 'edge', 'q', 'value', 'z', 'M')


def sources():
    return {n: C.sha(HERE / n) for n in SOURCES}


def committed_blob(rel):
    r = subprocess.run(['git', '-C', str(REPO), 'show', f'HEAD:{rel}'], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def committed_equal(rel, local):
    b = committed_blob(rel)
    return b is not None and Path(local).exists() and b == Path(local).read_bytes()


def frozen():
    """The module, its selftest, the declaration and the manifest are the committed ones."""
    return bool(all(committed_equal(f'predictfun/research/{n}', HERE / n) for n in ('pf_movement_shrink.py', 'pf_movement_shrink_selftest.py'))
                and all(committed_equal(f'{RECEIPT}/{n}', REPO / RECEIPT / n) for n in ('declaration.md', 'manifest.json')))


def inputs_bound(adm=MA.ADM):
    """Both cohort manifests committed and every admission partition record byte-equal to its committed receipt copy."""
    man = json.loads((Path(adm) / 'manifest.json').read_text())
    return bool(MA.inputs_frozen() and all(committed_equal(f'{ADM_REC_REL}/part_{int(k):03d}.json', Path(adm) / 'parts' / f'part_{int(k):03d}.json') for k in man['parts']))


# ---------------------------------------------------------------------------------------------------------------------- the rows
def decision_cell(t_ms, t_dec, side):
    return PA.cell_of(int(t_ms), float(t_dec), int(side))


def stage_rows(ar, ref_opp=None):
    """Every admitted row x stage: the audit's opportunity (statuses, identities, strata) + the decision mid + the decision cell; eligible rows
    reconciled with the audit's committed opportunities table (audit_mismatch otherwise)."""
    O, _ = MA.tables_of(ar, 'admitted', lambda r: dict(status='no_record'))
    mid = {(json.dumps(list(MA.GKEY(r))), nm): float((r['stages'].get(nm) or {}).get('inputs')[RT.FEATS.index('pf_mid')])
           for r in ar for nm in MA.STAGES if (r.get('stages') or {}).get(nm, {}).get('inputs')}
    O = O[O.cohort == 'admitted'].copy(); O['mid'] = [mid.get((k, s), np.nan) for k, s in zip(O.key, O.stage)]
    cells = [decision_cell(t, d, s) if np.isfinite(d) else (None, None, None) for t, d, s in zip(O.t_ms, O.t_dec.astype(float), O.side)]
    O['cday'] = [c[0] for c in cells]; O['chour'] = [c[2] for c in cells]
    O['status'] = O.acc_status.astype(str)
    if ref_opp is not None:
        R = ref_opp[(ref_opp.cohort == 'admitted')].set_index(['key', 'stage'])
        for i, row in O[O.status == 'ok'].iterrows():
            k = (row.key, row.stage)
            if k not in R.index or str(R.loc[k, 'acc_status']) != 'ok' or not all(abs(float(R.loc[k, c]) - float(row[c])) <= TOL * max(1., abs(float(row[c]))) for c in COLS):
                O.loc[i, 'status'] = 'audit_mismatch'
    return O.reset_index(drop=True)


def audit_opportunities():
    """The audit's committed opportunities table (the reviewed artifact), parsed exactly."""
    b = committed_blob(AUDIT_OPP_REL)
    C.require(b is not None, 'the audit opportunities table is not committed')
    return pd.read_csv(io.BytesIO(gzip.decompress(b)), float_precision='round_trip', dtype={'day': str, 'key': str, 'acc_status': str, 'stage': str, 'cohort': str})


def populations(O):
    """TRAIN / EVAL / descriptive key lists per head (eligible = status ok), and every row's status count."""
    pick = lambda day, nm: sorted(O[(O.status == 'ok') & (O.day == day) & (O.stage == nm)].key)
    st = O.groupby(['stage', 'day', 'status']).size().rename('n').reset_index().to_dict('records')
    return dict(train={nm: pick(TRAIN_DAY, nm) for nm in HEADS}, eval={nm: pick(EVAL_DAY, nm) for nm in HEADS},
                descriptive={'first_pull': pick(EVAL_DAY, 'first_pull')}, statuses=st)


def keys_sha(keys):
    return hashlib.sha256(json.dumps(keys, sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------------------------------------------------------------- the fit
def bounds(mid, side):
    mid = np.asarray(mid, float)
    return (-mid, 100. - mid) if int(side) == 0 else (mid - 100., mid)


def prior(P, Eq, mid, side, mu):
    """a = clip(P mu, E[q] L, E[q] U) and whether the projection binds (header)."""
    P, Eq = np.asarray(P, float), np.asarray(Eq, float); L, U = bounds(mid, side); lo, hi = Eq * L, Eq * U; raw = P * float(mu)
    a = np.minimum(np.maximum(raw, lo), hi); a = np.where(Eq == 0, 0., a)
    return a, (raw < lo) | (raw > hi)


def fit_group(T, side):
    """One side x head group's (mu, lambda) on TRAIN rows (header); refuses a row of another day or side."""
    C.require(len(T) == 0 or (set(T.day.astype(str)) == {TRAIN_DAY}), 'chronology: a training row is not of the training day')
    C.require(len(T) == 0 or set(T.side.astype(int)) == {int(side)}, 'a training row of another side')
    w, M, z, b = (T[c].to_numpy(float) for c in ('weight', 'M', 'z', 'Eqm')); n, nf = int(len(T)), int((z > 0).sum())
    base = dict(n=n, fills=nf, weight=float(w.sum()), weighted_fills=float((w * z).sum()))
    if n < MIN_ROWS or nf < MIN_FILLS:
        return dict(base, status='insufficient_fit_support', mu=None, lam=0., lam_raw=None, den=None, num=None, binding_weight_share=None)
    mu = float((w * M).sum() / (w * z).sum()); a, bind = prior(T.P, T.Eq, T.mid, side, mu); d = a - b; den = float((w * d * d).sum()); num = float((w * d * (M - b)).sum())
    bw = float((w * bind).sum() / w.sum())
    if den <= DEN_EPS:
        return dict(base, status='zero_denominator', mu=mu, lam=0., lam_raw=None, den=den, num=num, binding_weight_share=bw)
    return dict(base, status='ok', mu=mu, lam=float(min(1., max(0., num / den))), lam_raw=num / den, den=den, num=num, binding_weight_share=bw)


def fit_all(O, man):
    """The four groups on the manifest's TRAIN keys (refuses keys that differ from the recomputed eligible Oct 26 rows)."""
    out = {}
    for nm in HEADS:
        T = O[(O.stage == nm) & O.key.isin(man['train'][nm])]
        C.require(sorted(T.key) == sorted(man['train'][nm]) and (T.status == 'ok').all(), f'{nm}: the training rows differ from the manifest')
        for s in SIDES:
            out[f'{nm}|{s}'] = fit_group(T[T.side == s], s)
    return out


def predict(E, params, head):
    """Arms (header) for the rows of one head, each side with its own group's parameters."""
    E = E.copy(); E['a'] = E.Eqm.astype(float); E['bind'] = False; E['lam'] = 0.; E['mu'] = np.nan
    for s in SIDES:
        g = params[f'{head}|{s}']; m = (E.side == s).to_numpy()
        if g['mu'] is not None and m.any():
            a, bind = prior(E.P[m], E.Eq[m], E.mid[m], s, g['mu']); E.loc[m, 'a'] = a; E.loc[m, 'bind'] = bind; E.loc[m, 'mu'] = g['mu']
        E.loc[m, 'lam'] = float(g['lam'])
    d = (E.a - E.Eqm).to_numpy(float)
    for arm, lam in (('original', np.zeros(len(E))), ('prior', np.where(np.isfinite(E.mu.to_numpy(float)), 1., 0.)), ('candidate', E.lam.to_numpy(float))):
        E[f'Eqm_{arm}'] = E.Eqm.to_numpy(float) + lam * d; E[f'V_{arm}'] = E.V.to_numpy(float) + lam * d
    return E


def masks(E, head):
    rule = (lambda v: v > 0) if head == 'entry' else (lambda v: v >= 0)
    out = {'none': np.zeros(len(E), bool), 'all': np.ones(len(E), bool)}
    out.update({arm: rule(E[f'V_{arm}'].to_numpy(float)) for arm in ARMS})
    return out


# ---------------------------------------------------------------------------------------------------------------------- the scores
wm, kish = MA.wm, MA.kish


def slices(E):
    return [('all', E)] + [(str(s), E[E.side == s]) for s in SIDES]


def scores(E, head):
    out = []
    for side, g in slices(E):
        if not len(g):
            continue
        w = g.weight; f = g[g.z > 0]; pool = wm(g.z, w)
        base = dict(head=head, side=side, n=int(len(g)), weight=float(w.sum()), ess=kish(w), worlds=int(g[['day', 'm', 't_ms']].drop_duplicates().shape[0]),
                    fills=int(len(f)), weighted_fills=float((f.weight).sum()), brier=wm((g.P - g.z) ** 2, w), brier_pooled=wm((pool - g.z) ** 2, w), mean_P=wm(g.P, w),
                    fill_rate=wm(g.z, w), mean_M=wm(g.M, w), mean_value=wm(g.value, w), mse_zero=wm(g.value ** 2, w), mse_mech=wm((g.edge * g.Eq - g.value) ** 2, w),
                    mse_m_zero=wm(g.M ** 2, w), binding_weight_share=wm(g.bind.astype(float), w))
        for arm in ARMS:
            em, ev = g[f'Eqm_{arm}'] - g.M, g[f'V_{arm}'] - g.value; cm = f[f'Eqm_{arm}'] / f.P - f.M
            out.append(dict(base, arm=arm, mean_Eqm=wm(g[f'Eqm_{arm}'], w), bias_m=wm(em, w), mse_m=wm(em ** 2, w), mae_m=wm(em.abs(), w), mean_V=wm(g[f'V_{arm}'], w),
                            bias_V=wm(ev, w), mse_V=wm(ev ** 2, w), mae_V=wm(ev.abs(), w), c_m=wm(cm, f.weight) if len(f) else None,
                            c_m_unweighted=float(cm.mean()) if len(f) else None, mse_c=wm(cm ** 2, f.weight) if len(f) else None))
    return out


def calibration(E, head):
    out = []
    for arm in ARMS:
        G = E.assign(pm=[MA.mbin(x) for x in E[f'Eqm_{arm}'].to_numpy(float)], pv=[MA.vbin(x) for x in E[f'V_{arm}'].to_numpy(float)])
        for dim in ('pm', 'pv', 'hbin', 'qbin', 'obin'):
            for lab, g in G.groupby(dim, sort=True):
                w = g.weight
                out.append(dict(head=head, arm=arm, dim={'pm': 'own_Eqm_bin', 'pv': 'own_V_bin'}.get(dim, dim), kind='calibration' if dim in ('pm', 'pv') else 'diagnostic',
                                cell=str(lab), n=int(len(g)), fills=int((g.z > 0).sum()), weight=float(w.sum()), mean_pred_m=wm(g[f'Eqm_{arm}'], w), mean_M=wm(g.M, w),
                                mean_u=wm(g[f'Eqm_{arm}'] - g.M, w), mean_pred_V=wm(g[f'V_{arm}'], w), mean_value=wm(g.value, w)))
    return out


def chosen_stats(g, mk):
    w = g.weight.to_numpy(float); v = g.value.to_numpy(float); c = mk.astype(bool); W = float(w.sum()); ch = g[c]; wc = w[c]; vc = v[c]
    sev = vc < -SEVERE; sh = MA.shares(vc, wc, ch[['day', 'm', 't_ms']].to_numpy()) if c.any() else dict(top5_generation_share=None, top5_world_share=None)
    shu = MA.shares(vc, np.ones(int(c.sum())), ch[['day', 'm', 't_ms']].to_numpy()) if c.any() else dict(top5_generation_share=None, top5_world_share=None)
    return dict(n=int(len(g)), weight=W, chosen=int(c.sum()), chosen_weight=float(wc.sum()), J=float((w * c * v).sum() / W) if W > 0 else None,
                mean_chosen=float((wc * vc).sum() / wc.sum()) if wc.sum() > 0 else None, worlds=int(ch[['day', 'm', 't_ms']].drop_duplicates().shape[0]),
                ess=kish(wc) if c.any() else 0., severe_n=int(sev.sum()), severe_sum=float((wc[sev] * vc[sev]).sum()), top5_generation_share=sh['top5_generation_share'],
                top5_world_share=sh['top5_world_share'], top5_generation_share_unweighted=shu['top5_generation_share'], top5_world_share_unweighted=shu['top5_world_share'])


def actions(E, head):
    mk = masks(E, head); out = []
    for side, g in slices(E):
        if not len(g):
            continue
        idx = g.index.to_numpy()
        for rule in RULES:
            out.append(dict(head=head, side=side, rule=rule, changed_vs_original=int((mk[rule][E.index.get_indexer(idx)] != mk['original'][E.index.get_indexer(idx)]).sum()),
                            **chosen_stats(g, mk[rule][E.index.get_indexer(idx)])))
    return out, mk


def null(E, mask, draws=NULL_DRAWS, seed=SEED):
    """The matched mask null (header) -> (summary, the draws' J)."""
    w = E.weight.to_numpy(float); v = E.value.to_numpy(float); W = float(w.sum()); rng = np.random.default_rng(seed)
    grp = [np.flatnonzero(m) for m in [(E.cday == d).to_numpy() & (E.side == s).to_numpy() & (E.chour == h).to_numpy()
                                       for d, s, h in sorted(set(zip(E.cday, E.side, E.chour)), key=str)]]
    ks = [int(mask[g].sum()) for g in grp]; free = [(g, k) for g, k in zip(grp, ks) if 0 < k < len(g)]
    Wc = float((w * mask).sum()); Jc = float((w * mask * v).sum() / W) if W > 0 else None; J = np.full(draws, np.nan); wmis = 0
    for i in range(draws):
        m = np.zeros(len(E), bool)
        for g, k in zip(grp, ks):
            if k:
                m[rng.choice(g, k, replace=False)] = True
        J[i] = float((w * m * v).sum() / W) if W > 0 else np.nan
        wmis += int(abs(float((w * m).sum()) - Wc) > 1e-9 * max(1., Wc))
    return dict(strata=len(grp), free_strata=len(free), free_rows=int(sum(len(g) for g, _ in free)), chosen=int(mask.sum()), chosen_weight=Wc, J=Jc,
                null_mean=float(np.nanmean(J)), null_p05=float(np.nanpercentile(J, 5)), null_p95=float(np.nanpercentile(J, 95)),
                percentile=float((J <= Jc + 1e-12).mean()) if Jc is not None else None, weight_mismatch_draws=wmis), J


def predicates(params, SC, AC, NU, mk, head):
    """The declared readings for one head (header)."""
    s = {(r['side'], r['arm']): r for r in SC if r['head'] == head}; a = {(r['side'], r['rule']): r for r in AC if r['head'] == head}
    pred = {side: dict(movement_mse_improves=bool(s[(side, 'candidate')]['mse_m'] < s[(side, 'original')]['mse_m']),
                       value_mse_improves=bool(s[(side, 'candidate')]['mse_V'] < s[(side, 'original')]['mse_V']),
                       movement_bias_reduced=bool(abs(s[(side, 'candidate')]['bias_m']) < abs(s[(side, 'original')]['bias_m'])))
            for side in ('all', '0', '1') if (side, 'candidate') in s}
    fit_ok = any(params[f'{head}|{x}']['status'] == 'ok' for x in SIDES); same_o = bool((mk['candidate'] == mk['original']).all())
    same_p = bool((mk['candidate'] == mk['prior']).all()); support = bool(NU['free_strata'] >= MIN_FREE_STRATA and NU['free_rows'] >= MIN_FREE_ROWS)
    J = {r: a[('all', r)]['J'] for r in RULES}; why = [n for n, bad in (('no_fit_support', not fit_ok), ('candidate_mask_equals_original', same_o),
                                                                          ('candidate_mask_equals_prior', same_p), ('inadequate_movable_support', not support),
                                                                          ('null_weight_mismatch', NU['weight_mismatch_draws'] > 0)) if bad]
    if why:
        dec = 'INCONCLUSIVE'
    elif (a[('all', 'candidate')]['chosen_weight'] > 0 and J['candidate'] > 0 and J['candidate'] > J['original'] and J['candidate'] > J['prior']
          and J['candidate'] > NU['null_p95']):
        dec = 'PASS'
    else:
        dec = 'FAIL'
    return dict(head=head, prediction=pred, decision=dec, inconclusive_reasons=why, J=J, null_p95=NU['null_p95'], free_strata=NU['free_strata'], free_rows=NU['free_rows'])


# ---------------------------------------------------------------------------------------------------------------------- the commands
def load(adm=MA.ADM, ref=True):
    ar, msha = MA.adm_rows(adm); O = stage_rows(ar, audit_opportunities() if ref else None)
    return ar, msha, O


def build_manifest(path=None, adm=MA.ADM):
    path = Path(path or REPO / RECEIPT / 'manifest.json'); C.require(not path.exists(), 'the manifest exists (write-once)')
    C.require(inputs_bound(adm), 'an input is not its committed copy'); ar, msha, O = load(adm)
    pop = populations(O); man = json.loads((Path(adm) / 'manifest.json').read_text())
    js = dict(version=VERSION, train_day=TRAIN_DAY, eval_day=EVAL_DAY, **pop, train_keys_sha256={k: keys_sha(v) for k, v in pop['train'].items()},
              eval_keys_sha256={k: keys_sha(v) for k, v in pop['eval'].items()}, admission_manifest_sha256=msha,
              admission_records_sha256={str(k): C.sha(Path(adm) / 'parts' / f'part_{int(k):03d}.json') for k in man['parts']},
              audit_opportunities_sha256=hashlib.sha256(committed_blob(AUDIT_OPP_REL)).hexdigest(), multiplicity=MULTIPLICITY,
              rules=dict(min_rows=MIN_ROWS, min_fills=MIN_FILLS, den_eps=DEN_EPS, null_draws=NULL_DRAWS, seed=SEED, min_free_strata=MIN_FREE_STRATA,
                         min_free_rows=MIN_FREE_ROWS, severe=SEVERE), asof=C.stamp())
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(js, indent=1, sort_keys=True, default=str) + '\n')
    print(json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in {**pop['train'], **{f'eval_{k}': v for k, v in pop['eval'].items()}}.items()}))


def check_populations(man, O):
    pop = populations(O)
    C.require(man['version'] == VERSION and pop['train'] == man['train'] and pop['eval'] == man['eval'] and pop['descriptive'] == man['descriptive'],
              'the recomputed populations differ from the manifest')


def check_inputs(man, msha, rec_shas, opp_sha):
    C.require(man['admission_manifest_sha256'] == msha and man['audit_opportunities_sha256'] == opp_sha and man['admission_records_sha256'] == rec_shas,
              'the manifest\'s input hashes differ')


def input_hashes(adm=MA.ADM):
    m_ = json.loads((Path(adm) / 'manifest.json').read_text())
    return {str(k): C.sha(Path(adm) / 'parts' / f'part_{int(k):03d}.json') for k in m_['parts']}, hashlib.sha256(committed_blob(AUDIT_OPP_REL)).hexdigest()


def check_manifest(man, O, msha, adm=MA.ADM):
    check_inputs(man, msha, *input_hashes(adm)); check_populations(man, O)


def run_fit(path=None, rec=None):
    rec = Path(rec or REPO / RECEIPT); path = Path(path or rec / 'fit.json'); C.require(not path.exists(), 'the fit exists (write-once)')
    C.require(frozen() and inputs_bound(), 'not frozen: the module / selftest / declaration / manifest or an input is not the committed one')
    man = json.loads((rec / 'manifest.json').read_text()); ar, msha, O = load(); check_manifest(man, O, msha)
    T = O[O.key.isin(set(man['train']['entry']) | set(man['train']['first_review'])) & O.stage.isin(HEADS)]
    params = fit_all(T, man)
    js = dict(version=VERSION, params=params, train_keys_sha256=man['train_keys_sha256'], manifest_sha256=C.sha(rec / 'manifest.json'), sources=sources(), asof=C.stamp())
    path.write_text(json.dumps(js, indent=1, sort_keys=True) + '\n'); print(json.dumps(params, indent=1))


def run_score(out=OUT, rec=None):
    rec = Path(rec or REPO / RECEIPT); out = Path(out); C.require(not (out / 'audit.json').exists(), 'the scores exist (immutable)')
    C.require(frozen() and inputs_bound(), 'not frozen: the module / selftest / declaration / manifest or an input is not the committed one')
    C.require(committed_equal(f'{RECEIPT}/fit.json', rec / 'fit.json'), 'the fit is not committed: no score before the fit commit')
    man = json.loads((rec / 'manifest.json').read_text()); fit = json.loads((rec / 'fit.json').read_text())
    C.require(fit['sources'] == sources() and fit['manifest_sha256'] == C.sha(rec / 'manifest.json'), 'the fit is of another identity')
    ar, msha, O = load(); check_manifest(man, O, msha); out.mkdir(parents=True, exist_ok=True)
    js = score_all(O, man, fit['params'], out)
    js.update(admission_manifest_sha256=msha, manifest_sha256=fit['manifest_sha256'], fit_sha256=C.sha(rec / 'fit.json'), sources=sources(), asof=C.stamp())
    (out / 'audit.json').write_text(json.dumps(js, indent=1, sort_keys=True, default=str) + '\n')
    print(json.dumps(dict(decisions={p['head']: p['decision'] for p in js['predicates']}, prediction={p['head']: p['prediction']['all'] for p in js['predicates']}), indent=1))


def score_all(O, man, params, out):
    """EVAL predictions, scores, calibration, actions, nulls and readings -> CSVs in out; returns the audit fields."""
    SC, CA, AC, NUL, P, ROWS, DR = [], [], [], {}, [], [], []
    for nm in HEADS:
        E = O[(O.stage == nm) & O.key.isin(man['eval'][nm])].reset_index(drop=True)
        C.require(sorted(E.key) == sorted(man['eval'][nm]) and (E.status == 'ok').all() and set(E.day) <= {EVAL_DAY}, f'{nm}: the evaluation rows differ from the manifest')
        if not len(E):
            NUL[nm] = None; P.append(dict(head=nm, prediction={}, decision='INCONCLUSIVE', inconclusive_reasons=['no_eval_rows'], J={}, null_p95=None, free_strata=0, free_rows=0))
            continue
        E = predict(E, params, nm); SC += scores(E, nm); CA += calibration(E, nm); ac, mk = actions(E, nm); AC += ac
        nu, J = null(E, mk['candidate']); NUL[nm] = nu; DR += [dict(head=nm, draw=i, J=float(j)) for i, j in enumerate(J)]
        P.append(predicates(params, SC, AC, nu, mk, nm))
        ROWS.append(E.assign(**{f'mask_{r}': mk[r] for r in RULES}))
    D = O[(O.stage == 'first_pull') & O.key.isin(man['descriptive']['first_pull'])].reset_index(drop=True)
    DS = scores(predict(D, {f'first_pull|{s}': params[f'first_review|{s}'] for s in SIDES}, 'first_pull'), 'first_pull') if len(D) else []
    R = pd.concat(ROWS, ignore_index=True) if ROWS else pd.DataFrame(columns=list(O.columns) + ['a', 'bind', 'lam', 'mu'] + [f'{p}_{arm}' for arm in ARMS for p in ('Eqm', 'V')] + [f'mask_{r}' for r in RULES])
    keep = ['head_', 'key', 'day', 'm', 't_ms', 'side', 'stage', 'weight', 'cday', 'chour', 'mid', 'status', 'hbin', 'qbin', 'obin', *COLS, 'a', 'bind', 'lam', 'mu',
            *[f'{p}_{arm}' for arm in ARMS for p in ('Eqm', 'V')], *[f'mask_{r}' for r in RULES]]
    R = R.assign(head_=R.stage)[keep].rename(columns={'head_': 'head'})
    for name, rows_ in (('scores', SC + DS), ('calibration', CA), ('actions', AC), ('null_draws', DR)):
        pd.DataFrame(rows_).to_csv(out / f'{name}.csv', index=False)
    R.to_csv(out / 'eval_rows.csv', index=False)
    st = O.groupby(['stage', 'day', 'status']).size().rename('n').reset_index().to_dict('records')
    return dict(version=VERSION, statuses=st, params=params, null=NUL, predicates=P, descriptive_first_pull=len(D))


def selftest():
    import pf_movement_shrink_selftest as ST_
    ST_.run(sys.modules[__name__])


if __name__ == '__main__':
    a = sys.argv[1:]
    if a == ['selftest']:
        selftest()
    elif a == ['manifest']:
        build_manifest()
    elif a == ['fit']:
        run_fit()
    elif a == ['score']:
        run_score()
    else:
        sys.exit(__doc__)
