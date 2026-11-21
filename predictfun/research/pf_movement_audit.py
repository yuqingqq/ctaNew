"""PI0-MOVEMENT-CALIBRATION-AUDIT-v1 (P-2026-005; 2026-11-21, session aea5bfb9). Review 5a8ee6f5's one next priority: a bounded diagnostic of
PI0's frozen movement forecast on the existing admitted and forced cohorts, with that review's four reporting / validator amendments. No fit,
no new branch, no calibration map, no threshold; consumed Oct 24-27 only; declared and frozen before any new table is computed. Teacher,
features, latency, size, cap, structural rules, terminal mark and original END are those of the saved rows: nothing is re-simulated.
COHORTS (never pooled): ADMITTED = PI0-ADMISSION-EXPOSURE-v1's 329 generations (partitions hash-bound to their records and the committed
 manifest; membership equal to the manifest's selected keys; each row's weight, n_cell and cell bound to the manifest row of its exact key,
 else weight_mismatch), inclusion-weighted. FORCED = ENTRY-RETENTION's 768 forks (384 worlds x 2 sides; 756 evaluable; partitions hash-bound
 to the committed exposure manifest; membership equal to its world x side forks), unweighted, shown for Oct 24-27, per day, for the Oct 26-27
 overlap and by the shadow entry admission (would_POST / would_WAIT; fallback = a non-model entry decision; entry:<status> without one).
 Stages entry / first_review / first_pull are separate estimands; a first pull that is its first review (same_as_review, counted) is the
 same observation. Eligible: row and stage ok, a model decision with inputs; every other row is counted (not_eligible).
ACCOUNTING (rowwise; a residual above TOL = 1e-6 c is accounting_mismatch): z = 1(q > 0), M = qm (0 without a fill), qhat = E[q] / P,
 mhat = E[qm] / P (P <= 0 is zero_P; no moments or target is no_forecast_or_target; a fill without qm is movement_missing):
 V - value = edge (E[q] - q) + (E[qm] - M);  E[qm] - M = (P - z) mhat + z (mhat - M);  E[q] - q = (P - z) qhat + z (qhat - q).
 Units: c per opportunity (unconditional), c per filled opportunity (c_m = mhat - M, c_q = qhat - q in shares), c per share (pre / post).
 qm is quantity-weighted decision-to-END movement (pre- and post-fill; review 5a8ee6f5 finding 2), not post-fill movement alone. The
 mechanical baseline is edge x E[q] (V without the movement head); the zero baselines are 0 for value, q and M; Brier's reference is the
 cohort x stage pooled fill rate. Severe: value < -SEVERE (30 c), with its signed weighted sums.
STRATA (marginal views, no Cartesian product; thin: < 30 opportunities, or < 10 fills for a conditional statistic):
 decision_state - t_to_end (ms) (0,100] (100,500] (500,2000] (2000,5000]; fp_queue / fp_remaining (client shares) 0, (0,1], (1,3], >3,
   missing_or_invalid; generation ordinal 1, 2, >=3;
 forecast - P (PBINS); E[qm] (c) (-inf,-10] (-10,-3] (-3,-1] (-1,0] (0,inf); E[q] (shares) [0,1] (1,3] (3,10] >10; V (c) (-inf,-10]
   (-10,-3] (-3,0] (0,3] (3,inf);
 outcome - the first counted fill's delay (below), no_fill;  explanatory_venue - venue-truth ahead / fp_remaining (reviews only; venue
   knowledge, never a decision input, never read by a predicate).
FILL TIMING (realized, an outcome, never a feature): counted fills are the held gid's with t >= the stage's effect start (activation at entry,
 the modeled effect at review / pull); delay class [0,100] (100,500] (500,2000] >2000 ms. PRE / POST per counted fill:
 sign q (END_mark - decision_mid) = sign q (fill_mid - decision_mid) + sign q (END_mark - fill_mid), decision_mid = the stage's own pf_mid
 input, END_mark = the row's frozen terminal mark, fill_mid = pf_adv2_replay.mid_at's convention (the latest frame with venue time <= t, both
 bests present) with its venue time, receipt time and age t - venue time. The counted fills' totals must equal M (fill_mismatch). ADMITTED
 mids: looked up on the tapes of the selected worlds (run_part: the identity-checked loader of PI0-ADMISSION, the world's canonical mark equal
 to the row's, else mark_mismatch); FORCED mids: recovered from ENTRY-RETENTION's retained fill-time edges (mid_at on the engine's frames; no
 frame time retained: age unavailable). A fill without a mid is reference_missing; a row without a record is reference_unavailable:<why>.
TABLES (measure()): moments (views all / day / side / day_side, forced also overlap, admission, overlap_admission); strata (subsets admitted,
 forced, forced overlap); prepost; concentration (weighted and unweighted top-5 generation / world shares of |D_EXPOSURE| and of the
 admitted movement error per stage); contributions (signed day x side sums); sources (entry -> first review: a new PF frame by pf_age, a new
 Binance bookTicker update by bn_age, a new Binance depth snapshot by bnd_age - each the ms age at the decision since that source's latest
 known event, new when the age grew by less than the elapsed time minus 1 ms - and a new PF receipt; the decision-mid change
 sign x (pf_mid_review - pf_mid_entry); trades carry separate timestamps and are not counted); the retained first-review control
 (PI0-ADMISSION's 500-draw selection with each review's UTC hour derived from its own decision time, the count equal to the admission cell
 recorded); membership; evidence.
EVIDENCE (predicates(), per stage; the declared reading is the admitted entry stage, weighted; u = E[qm] - M, c = mhat - M on fills):
 (i) broad: u's mean > 0 in at least 3 of the 4 day x side cells and the weighted top-5 world share of |sum w u| below 0.5; concentrated:
     that share at least 0.5 or u > 0 in at most 1 cell; inconclusive otherwise.
 (ii) over the cells with >= 10 fills, D = the largest range of mean c across one decision_state stratification, T = the range across the
     delay classes (each needs >= 2 such cells): decision_state if D >= 1.5 T (or T unavailable), fill_timing if T >= 1.5 D (or D
     unavailable), both otherwise, inconclusive if neither is available.
 (iii) c_admitted (weighted) against c_forced (Oct 26-27, unweighted): material if their signs differ or |difference| >= 0.5 max(|c|);
     similar otherwise; inconclusive below 10 fills in either.
 (iv) populated: every decision_state stratification has >= 2 cells with >= 30 opportunities and >= 10 fills; thin otherwise.
 recommendation: calibration_candidate_defensible if (i) broad, (iv) populated and (ii) decision_state or both;
 representation_or_data_needed if (i) concentrated, (iv) thin or (ii) fill_timing; inconclusive otherwise.
Usage (run_light): pf_movement_audit.py selftest | run <part> | measure"""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_pi0_admission as PA                            # noqa: E402  frozen: the admitted cohort, its loader and the first-review control
import pf_entry_retention as RT                          # noqa: E402  frozen: the forced cohort
import pf_entry_exposure as EX                           # noqa: E402  frozen: the forced cohort's manifest

C, pd, LS, L, KP, X, AS, AE, BE = PA.C, PA.pd, PA.LS, PA.L, PA.KP, PA.X, PA.AS, PA.AE, PA.BE
VERSION = 'pi0_movement_audit_v1'
OUT = PA.A.DATA / 'pi0_movement_audit_v1_20261121'
REPO = HERE.parents[1]
RECEIPT = 'predictfun/reviews/movement_audit_20261121'
ADM, RET, EXPO_MAN = Path(PA.OUT), Path(RT.OUT), Path(RT.EXPO) / 'manifest.json'
ADM_MAN_REL, EXPO_MAN_REL = f'{PA.RECEIPT}/manifest.json', EX.RECEIPT_REL
STAGES = ('entry', 'first_review', 'first_pull')
OVERLAP, OVERLAP_LABEL = ('20261026', '20261027'), '20261026-27'
HBINS = ((0., 100.), (100., 500.), (500., 2000.), (2000., 5000.))
DBINS = ((0., 100.), (100., 500.), (500., 2000.), (2000., np.inf))
MBINS = (-np.inf, -10., -3., -1., 0., np.inf)
VBINS = (-np.inf, -10., -3., 0., 3., np.inf)
DIMS = dict(hbin='decision_state', qbin='decision_state', obin='decision_state', pbin='forecast', mbin='forecast', eqbin='forecast', vbin='forecast',
            delay_class='outcome', venue_qbin='explanatory_venue')
DECISION_DIMS = ('hbin', 'qbin', 'obin')
THIN_N, THIN_FILLS, TOL, END, SEVERE, RATIO = 30, 10, 1e-6, PA.END, PA.SEVERE, 1.5
SOURCES = ('pf_movement_audit.py', 'pf_movement_audit_selftest.py', 'pf_pi0_admission.py', 'pf_entry_retention.py', 'pf_adv2_replay.py')
GKEY = lambda r: (str(r['day']), int(r['m']), int(r['t_ms']), int(r['side']), int(r['gid']), int(r['decision_ms']))
FKEY = lambda r: (str(r['day']), int(r['m']), int(r['t_ms']), int(r['side']))


def sources():
    return {n: C.sha(HERE / n) for n in SOURCES}


def committed_equal(rel, local):
    r = subprocess.run(['git', '-C', str(REPO), 'show', f'HEAD:{rel}'], capture_output=True)
    return r.returncode == 0 and r.stdout == Path(local).read_bytes()


def frozen():
    """The module, its selftest and the declaration are the committed ones (no measurement before the freeze)."""
    return bool(all(committed_equal(f'predictfun/research/{n}', HERE / n) for n in ('pf_movement_audit.py', 'pf_movement_audit_selftest.py'))
                and committed_equal(f'{RECEIPT}/declaration.md', REPO / RECEIPT / 'declaration.md'))


def inputs_frozen(adm_man=ADM / 'manifest.json', expo_man=EXPO_MAN):
    """Both cohorts' manifests are their committed receipts' copies."""
    return bool(committed_equal(ADM_MAN_REL, adm_man) and committed_equal(EXPO_MAN_REL, expo_man))


# ---------------------------------------------------------------------------------------------------------------------- the cohorts
def adm_rows(adm=ADM):
    """The admitted rows, each partition hash-bound to its record and the manifest; the manifest's weight / n_cell / cell bound by exact key."""
    adm = Path(adm); man = json.loads((adm / 'manifest.json').read_text()); msha = C.sha(adm / 'manifest.json'); sel = {GKEY(g): g for g in man['selected']}; rows = []
    for k in man['parts']:
        rec = json.loads((adm / 'parts' / f'part_{k:03d}.json').read_text()); b = (adm / 'parts' / f'part_{k:03d}.jsonl.gz').read_bytes()
        C.require(hashlib.sha256(b).hexdigest() == rec['sha256'] and rec['manifest_sha256'] == msha, f'admission part {k}: identity')
        for r in LS.gz_rows(b):
            g = sel.get(GKEY(r)); bound = bool(g is not None and float(r.get('weight', np.nan)) == float(g['weight']) and int(r.get('n_cell', -1)) == int(g['n_cell'])
                                                and list(r.get('cell') or []) == list(g['cell']) and int(g['part']) == int(k))
            rows.append(dict(r, part=int(k), status=r['status'] if bound else 'weight_mismatch', weight=float(g['weight']) if g else np.nan))
    return rows, msha


def forced_rows(ret=RET, man_path=EXPO_MAN):
    """The forced rows (ENTRY-RETENTION), each partition hash-bound to its record and the exposure manifest."""
    ret = Path(ret); man_path = Path(man_path); man = json.loads(man_path.read_text()); rows = []
    for k in range(len(man['parts'])):
        rec = json.loads((ret / 'parts' / f'part_{k:03d}.json').read_text()); b = (ret / 'parts' / f'part_{k:03d}.jsonl.gz').read_bytes()
        C.require(hashlib.sha256(b).hexdigest() == rec['sha256'] and rec['manifest_sha256'] == C.sha(man_path), f'retention part {k}: identity')
        rows += [dict(r, weight=1.) for r in LS.gz_rows(b)]
    return rows


def membership(ar, fr, adm=ADM, expo_man=EXPO_MAN):
    """Complete membership: admitted rows = the manifest's selected keys; forced rows = the exposure manifest's world x side forks."""
    man = json.loads((Path(adm) / 'manifest.json').read_text()); em = json.loads(Path(expo_man).read_text())
    ak = {GKEY(g) for g in man['selected']}; fk = {(str(w['day']), int(w['m']), int(w['t_ms']), s) for p in em['parts'] for w in p['worlds'] for s in (0, 1)}
    a_, f_ = [GKEY(r) for r in ar], [FKEY(r) for r in fr]
    return dict(admitted_rows=len(a_), admitted_keys=len(ak), admitted_equal=bool(len(a_) == len(set(a_)) == len(ak) and set(a_) == ak),
                forced_rows=len(f_), forced_keys=len(fk), forced_equal=bool(len(f_) == len(set(f_)) == len(fk) and set(f_) == fk),
                weight_mismatch=int(sum(1 for r in ar if r['status'] == 'weight_mismatch')), admitted_weight_sum=float(np.nansum([r['weight'] for r in ar])))


# ---------------------------------------------------------------------------------------------------------------------- the accounting
def hbin(t):
    for lo, hi in HBINS:
        if lo < t <= hi:
            return f'({lo:.0f},{hi:.0f}]'
    return 'outside'


def qbin(queue, remaining):
    if not (np.isfinite(queue) and np.isfinite(remaining)) or remaining <= 0 or queue < 0:
        return 'missing_or_invalid'
    r = queue / remaining
    return '0' if r == 0 else '(0,1]' if r <= 1 else '(1,3]' if r <= 3 else '>3'


def obin(o):
    return 'missing_or_invalid' if not np.isfinite(o) else '1' if o <= 1 else '2' if o <= 2 else '>=3'


def dbin(d):
    if d < 0:
        return 'negative'
    if d <= 100.:
        return '[0,100]'
    for lo, hi in DBINS[1:]:
        if lo < d <= hi:
            return f'>{lo:.0f}' if not np.isfinite(hi) else f'({lo:.0f},{hi:.0f}]'
    return 'outside'


def edges_bin(e, edges):
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo < e <= hi:
            return f'({lo},{hi}]'
    return 'outside'


def mbin(e):
    return edges_bin(e, MBINS)


def vbin(v):
    return edges_bin(v, VBINS)


def eqbin(e):
    return 'negative' if e < 0 else '[0,1]' if e <= 1 else '(1,3]' if e <= 3 else '(3,10]' if e <= 10 else '>10'


def pbin(p):
    for lo, hi in zip(RT.PBINS[:-1], RT.PBINS[1:]):
        if (lo <= p < hi) or (hi == 1. and p == 1.):
            return f'[{lo},{hi})'
    return 'outside'


def accounting(x):
    """One stage's decomposition (header) -> dict with acc_status."""
    mo, tg = x.get('moments') or {}, x.get('target') or {}
    if mo.get('V') is None or tg.get('q') is None:
        return dict(acc_status='no_forecast_or_target')
    P, Eq, Eqm, V, edge = (float(mo[k]) for k in ('P', 'Eq', 'Eqm', 'V', 'edge')); q, value = float(tg['q']), float(tg['value'])
    z = 1. if q > 1e-12 else 0.; M = (float(tg['qm']) if tg.get('qm') is not None else np.nan) if z else 0.
    base = dict(P=P, Eq=Eq, Eqm=Eqm, V=V, edge=edge, q=q, value=value, z=z, M=M)
    if not np.isfinite(M):
        return dict(base, acc_status='movement_missing')
    if P <= 0:
        return dict(base, acc_status='zero_P')
    qh, mh = Eq / P, Eqm / P
    out = dict(base, acc_status='ok', qhat=qh, mhat=mh, err_V=V - value, err_q=Eq - q, err_m=Eqm - M, qty_term=edge * (Eq - q), mov_term=Eqm - M, mov_prob=(P - z) * mh,
               mov_cond=z * (mh - M), q_prob=(P - z) * qh, q_cond=z * (qh - q), c_m=(mh - M) if z else np.nan, c_q=(qh - q) if z else np.nan, mech=edge * Eq)
    out['identity_residual'] = max(abs(out['err_V'] - (out['qty_term'] + out['mov_term'])), abs(out['mov_term'] - (out['mov_prob'] + out['mov_cond'])),
                                   abs(out['err_q'] - (out['q_prob'] + out['q_cond'])))
    if out['identity_residual'] > TOL:
        out['acc_status'] = 'accounting_mismatch'
    return out


def stage_eff(r, x, nm):
    return float(r['held']['act']) if nm == 'entry' else float(x['effect'])


def fill_rows(r, x, nm, mref, cohort, key):
    """The counted fills of one stage (the held gid's, t >= the effect start) with delay, mids and the pre / post split (header)."""
    s = int(r['side']); sg = 1. if s == 0 else -1.; mark = float(r['held']['mark']); eff = stage_eff(r, x, nm); dmid = float(x['inputs'][RT.FEATS.index('pf_mid')])
    ok_ref = mref.get('status') == 'ok'; out = []
    for i, f in enumerate(r['held']['fills']):
        t, q, p = float(f[0]), float(f[3]), float(f[4])
        if t < eff or int(f[1]) != int(r['held']['gid']):
            continue
        m = (mref.get('mids') or {}).get(i) if ok_ref else None; fm = None if m is None else float(m['mid'])
        st = 'ok' if m is not None else 'reference_missing' if ok_ref else f"reference_unavailable:{mref.get('status')}"
        ft = None if m is None else m.get('frame_t')
        d = dict(cohort=cohort, key=key, day=str(r['day']), side=s, stage=nm, weight=float(r['weight']), index=i, t=t, q=q, price=p, eff=eff, delay=t - eff, delay_class=dbin(t - eff),
                 decision_mid=dmid, mark=mark, fill_mid=fm, frame_t=ft, recv_t=None if m is None else m.get('recv_t'), age=None if ft is None else t - float(ft),
                 total=sg * q * (mark - dmid), pre=np.nan if fm is None else sg * q * (fm - dmid), post=np.nan if fm is None else sg * q * (mark - fm), ref_status=st)
        if fm is not None and abs(d['pre'] + d['post'] - d['total']) > TOL:
            d['ref_status'] = 'split_mismatch'
        if ft is not None and not (float(ft) <= t <= END):
            d['ref_status'] = 'reference_out_of_time'
        out.append(d)
    return out


def forced_mids(r):
    """FORCED fill mids recovered from the retained fill-time edges (aligned with held.fills): mid = price + edge / (sign q)."""
    s = int(r['side']); sg = 1. if s == 0 else -1.; mids = {}
    for i, f in enumerate(r.get('fills') or []):
        if f.get('fill_time_edge') is not None and float(f['q']) > 0:
            mids[i] = dict(mid=float(f['price']) + float(f['fill_time_edge']) / (sg * float(f['q'])), frame_t=None, recv_t=None)
    return dict(status='ok', mids=mids)


def admission_of(r, cohort):
    if cohort == 'admitted':
        return 'admitted'
    ent = (r.get('stages') or {}).get('entry') or {}
    if ent.get('status') != 'ok':
        return f"entry:{ent.get('status')}"
    return ent.get('admission') if ent.get('dec_status') == 'model' else 'fallback'


def tables_of(rows, cohort, mref_of):
    """Opportunity rows (every row x stage, statuses kept) and counted fill rows of one cohort."""
    opp, fills = [], []
    for r in rows:
        key = json.dumps(list(GKEY(r) if cohort == 'admitted' else FKEY(r))); st = r.get('stages') or {}; adm = admission_of(r, cohort)
        for nm in STAGES:
            x = st.get(nm) or {}
            base = dict(cohort=cohort, key=key, day=str(r['day']), m=int(r['m']), t_ms=int(r['t_ms']), side=int(r['side']), stage=nm, weight=float(r['weight']), row_status=r['status'],
                        stage_status=x.get('status', r['status']), dec_status=x.get('dec_status'), admission=adm, same_as_review=bool(x.get('same_as_review')))
            if r['status'] != 'ok' or x.get('status') != 'ok' or x.get('dec_status') != 'model' or not x.get('inputs'):
                opp.append(dict(base, acc_status='not_eligible')); continue
            a = x['inputs']; va = (x.get('venue') or {}).get('ahead')
            d = dict(base, t_dec=float(x['t']), t_to_end=float(a[RT.FEATS.index('t_to_end')]), fp_queue=float(a[RT.FEATS.index('fp_queue')]),
                     fp_remaining=float(a[RT.FEATS.index('fp_remaining')]), ordinal=float(a[RT.FEATS.index('ctx_generation_ordinal')]),
                     gen_age=float(a[RT.FEATS.index('ctx_generation_age_ms')]), venue_ahead=np.nan if va is None else float(va), **accounting(x))
            d.update(hbin=hbin(d['t_to_end']), qbin=qbin(d['fp_queue'], d['fp_remaining']), obin=obin(d['ordinal']),
                     venue_qbin=qbin(d['venue_ahead'], d['fp_remaining']))
            if d['acc_status'] in ('ok', 'accounting_mismatch', 'zero_P', 'movement_missing'):
                d.update(pbin=pbin(d['P']), mbin=mbin(d['Eqm']), eqbin=eqbin(d['Eq']), vbin=vbin(d['V']))
            fl = fill_rows(r, x, nm, mref_of(r), cohort, key) if d['acc_status'] == 'ok' else []
            full = all(f['ref_status'] == 'ok' for f in fl); first = min(fl, key=lambda f: f['t']) if fl else None
            d.update(n_fills=len(fl), fills_with_mid=int(sum(f['ref_status'] == 'ok' for f in fl)), split_status='no_fill' if not fl else 'ok' if full else 'reference_incomplete',
                     pre=float(sum(f['pre'] for f in fl)) if full else np.nan, post=float(sum(f['post'] for f in fl)) if full else np.nan,
                     delay_class=first['delay_class'] if first else 'no_fill', first_delay=first['delay'] if first else np.nan)
            if d['acc_status'] == 'ok' and (abs(sum(f['total'] for f in fl) - d['M']) > TOL or bool(fl) != bool(d['z'])):
                d['acc_status'] = 'fill_mismatch'
            opp.append(d); fills += fl
    return pd.DataFrame(opp), pd.DataFrame(fills)


# ---------------------------------------------------------------------------------------------------------------------- the tapes (ADMITTED mids)
def mid_lookup(frames, t):
    """pf_adv2_replay.mid_at's convention with its frame: the latest frame with venue time <= t and both bests -> (mid, venue time, receipt)."""
    j = int(np.searchsorted(frames[:, 0], t, side='right')) - 1
    if j < 0:
        return None
    b, a = frames[j, 2], frames[j, 12]
    return dict(mid=float((b + a) / 2.), frame_t=float(frames[j, 0]), recv_t=float(frames[j, 1])) if b >= 0 and a >= 0 else None


def run_part(k, out=OUT):
    """The ADMITTED partition k's held-fill mids from the identity-checked tapes -> out/parts/mids_<k>.json (one record)."""
    out = Path(out); (out / 'parts').mkdir(parents=True, exist_ok=True); rp = out / 'parts' / f'mids_{int(k):03d}.json'
    C.require(frozen(), 'the module / selftest / declaration is not committed in HEAD: no measurement before the freeze')
    C.require(inputs_frozen(), 'a cohort manifest is not its committed receipt copy')
    if rp.exists():
        C.require(json.loads(rp.read_text()).get('sources') == sources(), f'part {k}: an existing record of another identity'); return 'DONE'
    rows, msha = adm_rows(); rows = [r for r in rows if r['part'] == int(k) and r['status'] == 'ok' and r['held']['fills']]
    plan = json.loads((AE.EVAL_DIR / 'plan.json').read_text()); C.require(C.sha(AE.EVAL_DIR / 'plan.json') == AE.EVAL_PLAN_SHA, 'not the frozen evaluation plan')
    byw = {}
    for r in rows:
        byw.setdefault((int(r['m']), int(r['t_ms'])), []).append(r)
    days = sorted({str(r['day']) for r in rows}); C.require(len(days) <= 1, f'part {k}: more than one day'); day = days[0] if days else None
    res = {}; G = LS.Guard(600. + 120. * max(1, len(byw)), 120.); mon = LS.RN.Sampler(out / 'parts' / f'mids_{int(k):03d}_status.json', LS.GUARD); mon.start()
    try:
        if byw:
            kp = Path(plan['kp_dir']); src = KP.Tape(); ff = AS.guarded_files; members = [dict(day=day, m=m, t_ms=t) for (m, t) in sorted(byw)]
            C.require(LS.identity_now(day, kp, src, ff, LS.hours_tracked) == plan['sample'][day]['inputs'], f"{day}: the inputs differ from the plan's")
            allw = {LS.ckey(LS.wkey(x)): x for x in LS.worlds_all(day, kp, keys=[(x['m'], x['t_ms']) for x in members])}
            C.require(sorted(allw) == sorted(LS.ckey(LS.wkey(x)) for x in members), "a member world is not one of the day's worlds")
            mk = sorted({x['m'] for x in members}); fair, depth, hrs, trk = X.real_inputs(day, mk)
            C.require(LS.day_identity(day, kp, src, hrs, trk, ff) == plan['sample'][day]['inputs'], "real_inputs' hours / tracked differ from the plan")
            data = src.load(day, mk, src.offset(day)); prs = {int(W_[0]['m']): W_ for W_ in src.pairs(data)}; data = None; G.check('load')
            for x in members:
                aw = allw[LS.ckey(LS.wkey(x))]; tw = time.time()
                world, wst = L.world_bind(day, dict(m=x['m'], t=aw['t'], o=aw['o']), prs, fair, depth, X.real_features)
                for r in byw[(int(x['m']), int(x['t_ms']))]:
                    key = json.dumps(list(GKEY(r)))
                    if world is None:
                        res[key] = dict(status=f'world:{wst}'); continue
                    if BE.canon_mark(world['path'].get('mark')) != BE.canon_mark(r['held']['mark']):
                        res[key] = dict(status='mark_mismatch'); continue
                    fr = np.asarray(world['path']['frames'], float)
                    res[key] = dict(status='ok', mids={str(i): (mid_lookup(fr, float(f[0])) if float(f[0]) <= END else None) for i, f in enumerate(r['held']['fills'])})
                world = None; G.check('pair', time.time() - tw); G.check('world')
        b = json.dumps(res, sort_keys=True).encode()
        LS.RN.atomic_json(rp, dict(version=VERSION, part=int(k), day=day, rows=len(res), mids=res, sha256=hashlib.sha256(b).hexdigest(), admission_manifest_sha256=msha,
                                   admission_record_sha256=C.sha(ADM / 'parts' / f'part_{int(k):03d}.json'), sources=sources(), elapsed_s=time.time() - G.t0, hwm_mib=LS.hwm(),
                                   timing=LS.timing(G.log), asof=C.stamp()))
        return 'PART_DONE'
    except LS._Latch as e:
        LS.RN.atomic_json(out / 'parts' / f'mids_{int(k):03d}_status.json', dict(state='STOP_RESOURCE', part=int(k), latch=e.args[0], asof=C.stamp())); return 'STOP_RESOURCE'
    finally:
        mon.live = False


def adm_mref(mids):
    def of(r):
        e = mids.get(json.dumps(list(GKEY(r))))
        if e is None:
            return dict(status='no_record')
        return dict(status=e['status'], mids={int(i): v for i, v in (e.get('mids') or {}).items() if v is not None})
    return of


# ---------------------------------------------------------------------------------------------------------------------- the measurements
def wm(x, w):
    x, w = np.asarray(x, float), np.asarray(w, float); m = np.isfinite(x) & np.isfinite(w)
    return float((x[m] * w[m]).sum() / w[m].sum()) if m.any() and w[m].sum() > 0 else None


def kish(w):
    w = np.asarray(w, float); return float(w.sum() ** 2 / (w ** 2).sum()) if len(w) and (w ** 2).sum() > 0 else 0.


def views(g0, cohort):
    v = [('all', 'all', 'all', 'all', g0)] + [('day', str(d), 'all', 'all', g) for d, g in g0.groupby('day', sort=True)]
    v += [('side', 'all', str(s), 'all', g) for s, g in g0.groupby('side', sort=True)] + [('day_side', str(d), str(s), 'all', g) for (d, s), g in g0.groupby(['day', 'side'], sort=True)]
    if cohort == 'forced':
        ov = g0[g0.day.isin(OVERLAP)]
        v += [('overlap', OVERLAP_LABEL, 'all', 'all', ov)] + [('admission', 'all', 'all', str(a), g) for a, g in g0.groupby('admission', sort=True)]
        v += [('overlap_admission', OVERLAP_LABEL, 'all', str(a), g) for a, g in ov.groupby('admission', sort=True)]
    return [x for x in v if len(x[4])]


def moments(O):
    out = []; E = O[O.acc_status == 'ok']
    for (co, nm), g0 in E.groupby(['cohort', 'stage'], sort=True):
        pool = wm(g0.z, g0.weight)
        for view, day, side, adm, g in views(g0, co):
            w = g.weight; f = g[g.z > 0]; sv = (g.value < -SEVERE).to_numpy()
            out.append(dict(cohort=co, stage=nm, view=view, day=day, side=side, admission=adm, n=int(len(g)), weight=float(w.sum()), ess=kish(w),
                            worlds=int(g[['day', 'm', 't_ms']].drop_duplicates().shape[0]), fills=int(len(f)), same_as_review=int(g.same_as_review.sum()),
                            fill_rate=wm(g.z, w), mean_P=wm(g.P, w), brier=wm((g.P - g.z) ** 2, w), brier_pooled=wm((pool - g.z) ** 2, w),
                            mean_V=wm(g.V, w), mean_value=wm(g.value, w), bias_V=wm(g.err_V, w), mse_V=wm(g.err_V ** 2, w), mse_zero=wm(g.value ** 2, w),
                            mse_mech=wm((g.mech - g.value) ** 2, w), mae_V=wm(g.err_V.abs(), w), mae_zero=wm(g.value.abs(), w),
                            mean_Eq=wm(g.Eq, w), mean_q=wm(g.q, w), bias_q=wm(g.err_q, w), mse_q=wm(g.err_q ** 2, w), mse_q_zero=wm(g.q ** 2, w),
                            mean_Eqm=wm(g.Eqm, w), mean_M=wm(g.M, w), bias_m=wm(g.err_m, w), mse_m=wm(g.err_m ** 2, w), mse_m_zero=wm(g.M ** 2, w), mae_m=wm(g.err_m.abs(), w),
                            qty_term=wm(g.qty_term, w), mov_term=wm(g.mov_term, w), mov_prob=wm(g.mov_prob, w), mov_cond=wm(g.mov_cond, w), q_prob=wm(g.q_prob, w),
                            q_cond=wm(g.q_cond, w), c_m=wm(f.c_m, f.weight) if len(f) else None, c_q=wm(f.c_q, f.weight) if len(f) else None,
                            mean_mhat_fill=wm(f.mhat, f.weight) if len(f) else None, mean_M_fill=wm(f.M, f.weight) if len(f) else None,
                            sum_err_V=float((w * g.err_V).sum()), sum_err_m=float((w * g.err_m).sum()), sum_value=float((w * g.value).sum()), severe_n=int(sv.sum()),
                            severe_sum_value=float((w[sv] * g.value[sv]).sum()), severe_sum_err_V=float((w[sv] * g.err_V[sv]).sum()),
                            severe_sum_err_m=float((w[sv] * g.err_m[sv]).sum())))
    return out


def strata(O):
    out = []; E = O[O.acc_status == 'ok']
    for co, sub, g1 in (('admitted', 'all', E[E.cohort == 'admitted']), ('forced', 'all', E[E.cohort == 'forced']),
                        ('forced', 'overlap', E[(E.cohort == 'forced') & E.day.isin(OVERLAP)])):
        for nm, g0 in g1.groupby('stage', sort=True):
            for dim, kind in DIMS.items():
                for lab, g in g0.groupby(dim, sort=True):
                    f = g[g.z > 0]; w = g.weight
                    out.append(dict(cohort=co, subset=sub, stage=nm, dim=dim, kind=kind, cell=str(lab), n=int(len(g)), fills=int(len(f)), weight=float(w.sum()),
                                    mean_P=wm(g.P, w), fill_rate=wm(g.z, w), mean_Eq=wm(g.Eq, w), mean_q=wm(g.q, w), mean_Eqm=wm(g.Eqm, w), mean_M=wm(g.M, w),
                                    mean_V=wm(g.V, w), mean_value=wm(g.value, w), bias_V=wm(g.err_V, w), mean_u=wm(g.err_m, w), sum_u=float((w * g.err_m).sum()),
                                    c_m=wm(f.c_m, f.weight) if len(f) else None, split_n=int(g.pre.notna().sum()), mean_pre=wm(g.pre, w), mean_post=wm(g.post, w),
                                    thin=bool(len(g) < THIN_N), thin_conditional=bool(len(f) < THIN_FILLS)))
    return out


def prepost(F):
    out = []
    if not len(F):
        return out
    for (co, nm), g0 in F.groupby(['cohort', 'stage'], sort=True):
        vs = [('all', 'all', 'all', g0)] + [('day_side', str(d), str(s), g) for (d, s), g in g0.groupby(['day', 'side'], sort=True)]
        vs += [('delay_class', str(c), 'all', g) for c, g in g0.groupby('delay_class', sort=True)]
        if co == 'forced':
            vs.append(('overlap', OVERLAP_LABEL, 'all', g0[g0.day.isin(OVERLAP)]))
        for view, cell, side, g in vs:
            if not len(g):
                continue
            ok = g[g.ref_status == 'ok']; wq = float((ok.weight * ok.q).sum())
            out.append(dict(cohort=co, stage=nm, view=view, cell=cell, side=side, fills=int(len(g)), with_mid=int(len(ok)), missing=int((g.ref_status != 'ok').sum()),
                            q=float(g.q.sum()), q_with_mid=float(ok.q.sum()), sum_total=float((g.weight * g.total).sum()), sum_total_with_mid=float((ok.weight * ok.total).sum()),
                            sum_pre=float((ok.weight * ok.pre).sum()), sum_post=float((ok.weight * ok.post).sum()),
                            total_per_share=float((ok.weight * ok.total).sum()) / wq if wq > 0 else None, pre_per_share=float((ok.weight * ok.pre).sum()) / wq if wq > 0 else None,
                            post_per_share=float((ok.weight * ok.post).sum()) / wq if wq > 0 else None,
                            median_age=float(ok.age.astype(float).median()) if len(ok) and ok.age.notna().any() else None,
                            max_age=float(ok.age.astype(float).max()) if len(ok) and ok.age.notna().any() else None))
    return out


def shares(vals, w, keys):
    """|w x| top-5 generation and world shares (the world sums signed first)."""
    c = np.asarray(vals, float) * np.asarray(w, float); a = np.sort(np.abs(c))[::-1]
    wk = pd.DataFrame(dict(k=[tuple(x) for x in np.asarray(keys)], c=c)).groupby('k').c.sum().to_numpy() if len(c) else np.zeros(0); aw = np.sort(np.abs(wk))[::-1]
    return dict(n=int(len(c)), worlds=int(len(wk)), top5_generation_share=float(a[:5].sum() / a.sum()) if a.sum() > 0 else None,
                top5_world_share=float(aw[:5].sum() / aw.sum()) if aw.sum() > 0 else None, sum=float(c.sum()))


def concentration(G, O):
    out = []; ok = G[G.status == 'ok']
    for wt in ('weighted', 'unweighted'):
        w = ok.weight.astype(float) if wt == 'weighted' else np.ones(len(ok))
        out.append(dict(quantity='D_EXPOSURE', weighting=wt, **shares(ok.D_EXPOSURE.astype(float), w, ok[['day', 'm', 't_ms']].to_numpy())))
        for nm in STAGES:
            E = O[(O.cohort == 'admitted') & (O.stage == nm) & (O.acc_status == 'ok')]
            out.append(dict(quantity=f'movement_error_{nm}', weighting=wt, **shares(E.err_m, E.weight if wt == 'weighted' else np.ones(len(E)), E[['day', 'm', 't_ms']].to_numpy())))
    return out


def contributions(G, O):
    ok = G[G.status == 'ok']; tot = float((ok.weight * ok.D_EXPOSURE).sum()); E = O[(O.cohort == 'admitted') & (O.stage == 'entry') & (O.acc_status == 'ok')]
    te = float((E.weight * E.err_m).sum()); out = []
    for (d, s), g in ok.groupby(['day', 'side'], sort=True):
        e = E[(E.day == str(d)) & (E.side == int(s))]; sd = float((g.weight * g.D_EXPOSURE).sum()); se = float((e.weight * e.err_m).sum())
        out.append(dict(day=str(d), side=int(s), n=int(len(g)), weighted_sum_D=sd, share_D=sd / tot if tot else None, entry_n=int(len(e)), weighted_sum_entry_u=se,
                        share_entry_u=se / te if te else None, weighted_sum_entry_err_V=float((e.weight * e.err_V).sum())))
    return out


def sources_table(rows, cohort):
    """Entry -> first review: the three clocks (ms ages), the PF receipt and the decision-mid change (header)."""
    out = []
    for r in rows:
        st = r.get('stages') or {}; e, v = st.get('entry') or {}, st.get('first_review') or {}
        if r['status'] != 'ok' or e.get('dec_status') != 'model' or v.get('dec_status') != 'model' or not e.get('inputs') or not v.get('inputs'):
            continue
        a0, a1 = e['inputs'], v['inputs']; dt = float(v['t']) - float(e['t']); g = lambda a, n: float(a[RT.FEATS.index(n)]); sg = 1. if int(r['side']) == 0 else -1.
        out.append(dict(cohort=cohort, day=str(r['day']), side=int(r['side']), weight=float(r['weight']), dt=dt, new_pf_frame=bool(g(a1, 'pf_age') < g(a0, 'pf_age') + dt - 1.),
                        new_bookticker=bool(g(a1, 'bn_age') < g(a0, 'bn_age') + dt - 1.), new_depth=bool(g(a1, 'bnd_age') < g(a0, 'bnd_age') + dt - 1.),
                        new_pf_receipt=bool((v.get('view') or {}).get('received') != (e.get('view') or {}).get('received')), mid_change=sg * (g(a1, 'pf_mid') - g(a0, 'pf_mid'))))
    S = pd.DataFrame(out)
    if not len(S):
        return [], S
    S['any_new'] = S.new_pf_frame | S.new_bookticker | S.new_depth | S.new_pf_receipt
    S['move_class'] = np.where(S.mid_change > 0, 'toward_order_value', np.where(S.mid_change < 0, 'against', 'unchanged'))
    summ = []
    for nm, g in [('all', S)] + [(f'mid {c}', S[S.move_class == c]) for c in ('against', 'unchanged', 'toward_order_value')] + [('no new observation', S[~S.any_new])]:
        summ.append(dict(cohort=cohort, subset=nm, n=int(len(g)), weight=float(g.weight.sum()), **{k: wm(g[k].astype(float), g.weight) if len(g) else None for k in
                                                                                                   ('new_pf_frame', 'new_bookticker', 'new_depth', 'new_pf_receipt', 'any_new')},
                         mean_mid_change=wm(g.mid_change, g.weight) if len(g) else None, mean_abs_mid_change=wm(g.mid_change.abs(), g.weight) if len(g) else None))
    return summ, S


def control(ar):
    """The retained first-review control (PI0-ADMISSION's selection: 500 draws, its seed) with each review's UTC hour derived from its own
    decision time; the count whose derived (day, side, hour) cell equals the admission cell (review 5a8ee6f5 item 4)."""
    S = PA.stage_rows(ar); m = ((S.stage == 'first_review') & S.t.notna()).to_numpy()
    der = [PA.cell_of(t_ms, t, s) for t_ms, t, s in zip(S.t_ms[m], S.t[m], S.side[m])]
    keys = [(str(d), int(mm), int(t), int(s), int(gg)) for d, mm, t, s, gg in zip(S.day[m], S.m[m], S.t_ms[m], S.side[m], S.gid[m])]
    adm = {GKEY(r)[:5]: tuple(r.get('cell') or ()) for r in ar}; same = int(sum(tuple(dc) == tuple(adm.get(k_, ())) for dc, k_ in zip(der, keys)))
    S.loc[m, 'hour'] = [int(dc[2]) for dc in der]
    return dict(PA.selection(S), reviews_with_time=int(m.sum()), cell_equal_admission=same)


def predicates(O, ST, CONC, stage='entry'):
    """The declared evidence (header) for one stage."""
    E = O[(O.cohort == 'admitted') & (O.stage == stage) & (O.acc_status == 'ok')]
    cells = {f'{d}/{s}': wm(g.err_m, g.weight) for (d, s), g in E.groupby(['day', 'side'], sort=True)}; pos = sum(1 for v in cells.values() if v is not None and v > 0)
    share = next((c['top5_world_share'] for c in CONC if c['quantity'] == f'movement_error_{stage}' and c['weighting'] == 'weighted'), None)
    i_ = ('broad' if (len(cells) == 4 and pos >= 3 and share is not None and share < .5) else
          'concentrated' if ((share is not None and share >= .5) or pos <= 1) else 'inconclusive')
    S = pd.DataFrame([s for s in ST if s['cohort'] == 'admitted' and s['subset'] == 'all' and s['stage'] == stage])
    q_ = lambda d: S[(S.dim == d) & (S.fills >= THIN_FILLS) & S.c_m.notna() & (S.cell != 'no_fill')] if len(S) else S
    rng = lambda df: float(df.c_m.max() - df.c_m.min()) if len(df) >= 2 else None
    dec = {d: rng(q_(d)) for d in DECISION_DIMS}; dly = rng(q_('delay_class')); dmax = max([v for v in dec.values() if v is not None], default=None)
    ii = ('inconclusive' if (dmax is None and dly is None) else 'decision_state' if (dly is None or (dmax is not None and dmax >= RATIO * dly)) else
          'fill_timing' if (dmax is None or dly >= RATIO * dmax) else 'both')
    fa = E[E.z > 0]; FO = O[(O.cohort == 'forced') & (O.stage == stage) & (O.acc_status == 'ok')]; ff = FO[FO.day.isin(OVERLAP) & (FO.z > 0)]; fall = FO[FO.z > 0]
    ca = wm(fa.c_m, fa.weight) if len(fa) else None; cf = wm(ff.c_m, ff.weight) if len(ff) else None
    iii = ('inconclusive' if (len(fa) < THIN_FILLS or len(ff) < THIN_FILLS) else
           'material' if (np.sign(ca) != np.sign(cf) or abs(ca - cf) >= .5 * max(abs(ca), abs(cf))) else 'similar')
    pop = {d: int(((S.dim == d) & (S.n >= THIN_N) & (S.fills >= THIN_FILLS)).sum()) if len(S) else 0 for d in DECISION_DIMS}
    iv = 'populated' if all(v >= 2 for v in pop.values()) else 'thin'
    rec = ('calibration_candidate_defensible' if (i_ == 'broad' and iv == 'populated' and ii in ('decision_state', 'both')) else
           'representation_or_data_needed' if (i_ == 'concentrated' or iv == 'thin' or ii == 'fill_timing') else 'inconclusive')
    return dict(stage=stage, i=dict(answer=i_, positive_cells=pos, cells=cells, top5_world_share=share),
                ii=dict(answer=ii, decision_ranges=dec, delay_range=dly), iii=dict(answer=iii, c_admitted=ca, c_forced_overlap=cf, c_forced_all=wm(fall.c_m, fall.weight) if len(fall) else None,
                                                                         fills_admitted=int(len(fa)), fills_forced_overlap=int(len(ff))),
                iv=dict(answer=iv, populated_cells=pop), recommendation=rec)


def evidence_rows(P):
    return [dict(stage=P['stage'], question=q, answer=P[q]['answer'], detail=json.dumps({k: v for k, v in P[q].items() if k != 'answer'}, sort_keys=True, default=str))
            for q in ('i', 'ii', 'iii', 'iv')] + [dict(stage=P['stage'], question='recommendation', answer=P['recommendation'], detail='')]


def measure(out=OUT, adm=ADM, ret=RET, forced_man=EXPO_MAN, mids_dir=None):
    out = Path(out); C.require(not (out / 'audit.json').exists(), 'the measurement exists (immutable)'); out.mkdir(parents=True, exist_ok=True)
    ar, msha = adm_rows(adm); fr = forced_rows(ret, forced_man); mem = membership(ar, fr, adm, forced_man)
    C.require(mem['admitted_equal'] and mem['forced_equal'], f'incomplete membership: {mem}')
    mids_dir = Path(mids_dir or out / 'parts'); mids, mrec = {}, {}
    for k in sorted({r['part'] for r in ar}):
        p = mids_dir / f'mids_{k:03d}.json'
        if p.exists():
            rec = json.loads(p.read_text())
            C.require(rec['sources'] == sources() and rec['admission_manifest_sha256'] == msha and
                      hashlib.sha256(json.dumps(rec['mids'], sort_keys=True).encode()).hexdigest() == rec['sha256'], f'mids {k}: identity')
            mids.update(rec['mids']); mrec[int(k)] = rec['sha256']
    O1, F1 = tables_of(ar, 'admitted', adm_mref(mids)); O2, F2 = tables_of(fr, 'forced', forced_mids)
    O, F = pd.concat([O1, O2], ignore_index=True), pd.concat([F1, F2], ignore_index=True)
    G = PA.gen_table(ar); MO = moments(O); ST = strata(O); PP = prepost(F); CO = concentration(G, O); CT = contributions(G, O)
    s1, S1 = sources_table(ar, 'admitted'); s2, S2 = sources_table(fr, 'forced'); CTL = control(ar); P = [predicates(O, ST, CO, nm) for nm in STAGES]
    EV = [e for p in P for e in evidence_rows(p)]
    for name, rows_ in (('moments', MO), ('strata', ST), ('prepost', PP), ('concentration', CO), ('contributions', CT), ('sources', s1 + s2), ('evidence', EV)):
        pd.DataFrame(rows_).to_csv(out / f'{name}.csv', index=False)
    O.to_csv(out / 'opportunities.csv', index=False); F.to_csv(out / 'fills.csv', index=False); pd.concat([S1, S2], ignore_index=True).to_csv(out / 'transitions.csv', index=False)
    st = O.groupby(['cohort', 'stage', 'acc_status']).size().rename('n').reset_index().to_dict('records')
    fst = F.groupby(['cohort', 'ref_status']).size().rename('n').reset_index().to_dict('records') if len(F) else []
    js = dict(version=VERSION, statuses=st, fill_statuses=fst, membership=mem, control=CTL, predicates=P, admission_manifest_sha256=msha, exposure_manifest_sha256=C.sha(forced_man),
              mids_records=mrec, parts_without_mids=sorted(set(int(r['part']) for r in ar) - set(mrec)), sources=sources(), asof=C.stamp())
    (out / 'audit.json').write_text(json.dumps(js, indent=1, sort_keys=True, default=str) + '\n')
    print(json.dumps(dict(statuses=st, fill_statuses=fst, membership=mem, recommendation=[p['recommendation'] for p in P]), indent=1, default=str))


def selftest():
    import pf_movement_audit_selftest as ST_
    ST_.run(sys.modules[__name__])


if __name__ == '__main__':
    a = sys.argv[1:]
    if a == ['selftest']:
        selftest()
    elif a[:1] == ['run'] and len(a) == 2:
        print(run_part(int(a[1])))
    elif a == ['measure']:
        C.require(frozen() and inputs_frozen(), 'not frozen: the module / selftest / declaration or a cohort manifest is not the committed one')
        measure()
    else:
        sys.exit(__doc__)
