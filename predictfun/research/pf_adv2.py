"""ADV2-RICH v1: THE FIXED DIRECT PAIRED-ADVANTAGE CANDIDATE, ITS FAIR / MECH INPUT CONTROLS AND THE CONTROLLER (plan v2 R3; P-2026-005;
2026-11-21, session aea5bfb9). Review 315979d2's next priority, items 1-5, 8 and 12: "train the declared four-head ADV2-RICH direct regressor
and FAIR/MECH controls on full Oct24-25 ...". Declared and frozen (freeze 2) before any fit; this module never reads an Oct 26-27 label.

CANDIDATES: ADV2-RICH-v1 (primary) and the fixed input controls ADV2-FAIR-v1 / ADV2-MECH-v1. Each = four independent squared-error regressors,
 bid / ask x POST / KEEP, of the paired label Y in cents: R(POST now) - R(WAIT this tick) and R(KEEP this tick) - R(PULL now), both arms under
 the frozen pi0_50_v1 continuation to the original END (+5,000 ms), 50 ms grid, 58 / 37 / 100 ms, gross (zero fees). 12 heads; RICH stays the
 primary whatever the rankings; the teacher's 16 heads are counted separately (multiplicity: one primary, two controls).
INPUTS (ordered; names checked unique and inside the stored 107; the sha256 of each sorted-key JSON list is frozen in INPUT_SHA):
 - RICH = the stored 107 (pf_r2_teacher.FEATS: pf_r2_shadow.INPUTS + CTX), the order of the training rows' 'inputs';
 - FAIR = fair_edge, t_to_end and the 16 pf_r2_shadow.CTX (18);
 - MECH = pf_flow_price_common.MECHANICAL, t_to_end and the 16 CTX (40).
 float32 as stored, native NaN (LightGBM's missing-value handling), existing units, no extra normalization. No ID, source label, day / time
 identity, hash, branch outcome or teacher / future score is an input; keys are joins only.
TRAINING ROWS: every ok row of the accepted R2-F dataset (TRAIN), streamed one part at a time. The accepted parts list must hash to the
 dataset sha256 and every part's bytes to its listed hash; every row's 107 float32 bytes must equal its stored inputs_sha256 (a mismatch
 refuses the extraction: an artifact error, never a status). Zeros and complete tails are kept: no outcome-based selection, balancing,
 clipping or transform. Exclusions are statuses with counts: pair_not_ok, feature_unavailable, empty_pool, no_additional_post, world:*,
 and crosses_day_boundary (t0 + END after the next UTC midnight; purged, plan v2).
WEIGHTS: the stored w = 1 / (pi N q); per side / head ONE constant n / sum(w), so they average 1 with the relative weights unchanged (the
 teacher's declared L2 convention, pf_r2_teacher.norm_weights), identical in the three input fits; raw weights kept for reporting.
LEARNER: pf_pipeline_bench.PARAMS with objective 'regression' (resolved; PARAMS_SHA), 200 rounds, no early stopping or search, an
 in-memory Dataset (at most 6,084 rows a head). Every saved model is reloaded from its file and must predict bit-equal to the fitted booster
 on its own training rows; its file sha256 is recorded.
CANDIDATE SUPPORT (outcome-independent; computed from the training rows only, common to the three inputs; separate from the teacher's
 support, and not a calibration), per side / head, checked in this order:
 - unsupported_horizon: the horizon bin of END - decision (pf_r2_teacher.hbin) not observed in training ('nan' outside the bins);
 - insufficient_candidate_support: fewer than 200 distinct training originals in the side / head;
 - unsupported_state_category: a value of the six category dimensions (pf_r2_teacher.categories) never observed;
 - outside_training_range: a pf_r2_shadow.NUMERIC field outside its observed [lo, hi] +- 1e-9 (NaN is outside; never clipped).
CONTROLLER (policy layer; plan v2 section 2): absent -> POST iff the supported prediction > 0, else WAIT; live -> PULL iff < 0, else KEEP;
 an exact tie keeps the state (WAIT / KEEP). Structural first, as the teacher's: cap_blocked and near_end -> WAIT; the engine's own no-book
 fallback; then feature_unavailable, the support statuses and nonfinite_prediction, each falling back to WAIT (absent) / KEEP (live) with its
 own reason. No timer, no hold, no inventory penalty; pending and terminal states are the engine's.
POLICIES (the bounded R4 diagnostic): N0 (never enters), PI0 (the frozen teacher), RICH / FAIR / MECH, ENTRY_ONLY (RICH entry, PI0
 retention), RETAIN_ONLY (PI0 entry, RICH retention), ENTRY_PA (RICH entry; PULL at every live review, which is each generation's first
 eligible review). Every policy is bound to its own branch; candidate inputs are rebuilt on that branch by the teacher's own Teacher.inputs
 (pf_r2_shadow.rebuild_row), and pf_r2_labels.ArmTeacher's binding check runs at every callback.
Usage (run_light): pf_adv2.py selftest | extract | fit
"""
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_flow_price_common as FPC                       # noqa: E402
import pf_pipeline_bench as BX                           # noqa: E402
import pf_r2_label_sample as LS                          # noqa: E402  frozen (4cdeff0c): its readers only
import pf_r2_labels as L                                 # noqa: E402  frozen: ArmTeacher, teacher_assets
import pf_r2_shadow as R2                                # noqa: E402
import pf_r2_teacher as T                                # noqa: E402

C, pd, KP = R2.C, R2.pd, R2.KP
VERSION = 'adv2_rich_v1'
CANDS = ('RICH', 'FAIR', 'MECH')
INPUTS = dict(RICH=list(T.FEATS), FAIR=list(dict.fromkeys(['fair_edge', 't_to_end'] + list(R2.CTX))),
              MECH=list(dict.fromkeys(list(FPC.MECHANICAL) + ['t_to_end'] + list(R2.CTX))))
INPUT_SHA = dict(RICH='43c9598dfb3bc2475d6b98c2b1107cfa57cfed1a8644992569e32c8a4d9ff1f9', FAIR='c7e58a5ea0415656f694ad99805481750646b973caf0d168b79488bdc1ae85fc',
                 MECH='04bb6d81e06e6960e75dc8a7e02876d740df415963fffe6685f8e81a5868b8c8')
COUNTS = dict(RICH=107, FAIR=18, MECH=40)
HEADS = ((0, 'post'), (0, 'keep'), (1, 'post'), (1, 'keep'))
PARAMS, ROUNDS = dict(BX.PARAMS, objective='regression'), 200
PARAMS_SHA = 'ae93f1d4e35259c5b193fe15ff5d41ba9ea9e66bfdecc3f41a21579ccbf53cf4'
END, STEP, DAY_MS, TOL, MIN_ORIG = float(R2.END), float(L.STEP), 86400000, R2.INV_TOL, 200
DATA = Path(R2.I.K.IO)
TRAIN = dict(dir=DATA / 'r2_label_sample_v2_20261120' / 'production_v1', dataset_sha256='2c62fc9baf696e6cd8913c5c294dea2c9513640714ae34b413dc37ef8c0acb2e',
             days=('20261024', '20261025'))
OUT = DATA / 'adv2_rich_v1_20261121'
STATUSES = ('cap_blocked', 'near_end', 'feature_unavailable', 'unsupported_horizon', 'insufficient_candidate_support', 'unsupported_state_category',
            'outside_training_range', 'nonfinite_prediction', 'model')
FALLBACK = dict(post='WAIT', keep='KEEP')
POLICIES = ('N0', 'PI0', 'RICH', 'FAIR', 'MECH', 'ENTRY_ONLY', 'RETAIN_ONLY', 'ENTRY_PA')
GEN_I = T.FEATS.index('ctx_generation_ordinal')


def ck(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'), default=str)


def sha_of(x):
    return hashlib.sha256(ck(x).encode()).hexdigest()


def check_inputs():
    """The declared lists: counts, uniqueness, membership in the stored 107, and the frozen hashes."""
    for c in CANDS:
        f = INPUTS[c]
        C.require(len(f) == COUNTS[c] and len(set(f)) == len(f) and set(f) <= set(T.FEATS), f'{c}: the input list is not {COUNTS[c]} unique stored inputs')
        C.require(sha_of(f) == INPUT_SHA[c], f'{c}: the input list hash changed')
    C.require(sha_of(PARAMS) == PARAMS_SHA and ROUNDS == 200, 'the resolved parameters changed')
    bad = [x for x in ('day', 'm', 't_ms', 'oid', 'source', 'Y', 'w', 'row_rank') if any(x == f for f in INPUTS['RICH'])]
    C.require(not bad, f'an identity / outcome field is an input: {bad}')


# ---------------------------------------------------------------- the accepted datasets -> rows
def dataset_rows(d, want_sha):
    """Every row of an accepted label dataset, part by part: the parts list hashes to the dataset and each part's bytes to its hash."""
    d = Path(d); acc = json.loads((d / 'accepted.json').read_text())
    C.require(acc.get('accepted') is True and acc['dataset_sha256'] == want_sha, f'{d.name}: not the accepted dataset {want_sha[:12]}')
    C.require(hashlib.sha256(LS.ckey(acc['parts']).encode()).hexdigest() == acc['dataset_sha256'], f'{d.name}: the parts list does not hash to the dataset')
    for p in acc['parts']:
        b = LS.part_paths(d, p['part'])[1].read_bytes()
        C.require(hashlib.sha256(b).hexdigest() == p['sha256'], f"{d.name} part {p['part']}: its bytes differ from the accepted hash")
        rows = LS.gz_rows(b); C.require(len(rows) == p['rows'], f"{d.name} part {p['part']}: row count")
        for r in rows:
            yield r


def crosses(day, t_ms):
    d0 = int(pd.Timestamp(f'{day[:4]}-{day[4:6]}-{day[6:]}', tz='UTC').value // 10 ** 6)
    return not (d0 <= int(t_ms) and int(t_ms) + END <= d0 + DAY_MS)


def extract(rows, days):
    """rows -> per (side, head): the 107 inputs, Y, raw w and the row metadata, plus the status counts (exclusions with their reasons)."""
    E = {h: dict(X=[], y=[], w=[], meta=[]) for h in HEADS}; st = {}
    for r in rows:
        C.require(r['day'] in days, f"a row of {r['day']}, outside the declared days {days}")
        status = r['status']
        if status == 'ok' and crosses(r['day'], r['t_ms']):
            status = 'crosses_day_boundary'
        st[f"{r['day']} {r['head']} {status}"] = st.get(f"{r['day']} {r['head']} {status}", 0) + 1
        if status != 'ok':
            continue
        s, h = int(r['side']), str(r['head']); C.require((s, h) in E, f'an undeclared side / head {s} {h}')
        fk, res = r['fork'], r['result']
        x = np.asarray(fk['inputs'], np.float32)
        C.require(x.shape == (107,) and hashlib.sha256(x.tobytes()).hexdigest() == fk['inputs_sha256'], 'a row whose 107 inputs differ from their stored hash')
        y, w = float(res['Y']), float(r['w'])
        C.require(np.isfinite(y) and np.isfinite(w) and w > 0 and res['status'] == 'ok' and res['kind'] == h and int(res['side']) == s,
                  'a row with a nonfinite label / weight or a result of another side / head')
        k_ = r['row_key']
        E[(s, h)]['X'].append(x); E[(s, h)]['y'].append(y); E[(s, h)]['w'].append(w)
        E[(s, h)]['meta'].append(dict(day=r['day'], m=int(r['m']), t_ms=int(r['t_ms']), side=s, head=h, oid=int(r['oid']), pick=r['pick'], source=r['source'],
                                      decision_ms=float(k_['decision_ms']), generation=float(x[GEN_I]), price=float(fk['price']), pi=float(r['pi']),
                                      hour=int(r['hour']), row_key=k_))
    for h in HEADS:
        e = E[h]; e['X'] = np.asarray(e['X'], np.float32).reshape(-1, 107); e['y'] = np.asarray(e['y'], np.float64); e['w'] = np.asarray(e['w'], np.float64)
    return E, dict(sorted(st.items()))


def norm(w):
    """One constant per training set: n / sum(w) (the teacher's convention)."""
    return T.norm_weights(w)


def manifest_of(E, st):
    """The extraction manifest: per side / head the rows, raw weight total and the hashes of keys, feature bytes, labels, raw / normalized
    weights; the status counts."""
    out = dict(statuses=st, heads={})
    for (s, h), e in E.items():
        keys = [ck(dict(day=m['day'], m=m['m'], t_ms=m['t_ms'], side=m['side'], head=m['head'], pick=m['pick'], row_key=m['row_key'])) for m in e['meta']]
        out['heads'][f'{s} {h}'] = dict(rows=len(e['y']), originals=len({(m['day'], m['m'], m['t_ms'], m['side']) for m in e['meta']}), w_raw_sum=float(e['w'].sum()),
                                        keys_sha256=hashlib.sha256('\n'.join(keys).encode()).hexdigest(), x_sha256=hashlib.sha256(e['X'].tobytes()).hexdigest(),
                                        y_sha256=hashlib.sha256(e['y'].tobytes()).hexdigest(), w_sha256=hashlib.sha256(e['w'].tobytes()).hexdigest(),
                                        w_norm_sha256=hashlib.sha256(norm(e['w']).tobytes()).hexdigest() if len(e['w']) else None)
    return out


# ---------------------------------------------------------------- the candidate support
def support_of(E):
    """The candidate support manifest from training rows (header): per side / head the observed horizon bins, category values, NUMERIC ranges
    and distinct originals."""
    S = {}
    for (s, h), e in E.items():
        df = pd.DataFrame(e['X'], columns=T.FEATS); df['price'] = [m['price'] for m in e['meta']]; df['generation'] = [m['generation'] for m in e['meta']]
        cats = T.categories(df); dec = np.array([m['decision_ms'] for m in e['meta']], float)
        rng = {}
        for f in R2.NUMERIC:
            v = df[f].astype(float).values; v = v[np.isfinite(v)]
            rng[f] = [float(v.min()), float(v.max())] if len(v) else None
        S[f'{s} {h}'] = dict(horizon_bins=sorted(set(str(x) for x in T.hbin(END - dec))), categories={d: sorted(set(cats[d])) for d in R2.DIMS},
                             ranges=rng, originals=len({(m['day'], m['m'], m['t_ms'], m['side']) for m in e['meta']}))
    return S


class Support:
    """The candidate support adapter (header's order of checks)."""

    def __init__(self, S):
        self.S = S

    def check(self, s, head, row):
        """row: the 107 inputs + price + generation + decision_ms -> None (supported) or the first failing status."""
        m = self.S[f'{int(s)} {head}']; hb = str(T.hbin(END - float(row['decision_ms']))[0])
        if hb == 'nan' or hb not in m['horizon_bins']:
            return 'unsupported_horizon'
        if m['originals'] < MIN_ORIG:
            return 'insufficient_candidate_support'
        cats = T.categories(pd.DataFrame([{**{c: row[c] for c in T.FEATS}, 'price': row['price'], 'generation': row['generation']}]))
        if any(str(cats[d][0]) not in set(m['categories'][d]) for d in R2.DIMS):
            return 'unsupported_state_category'
        for f in R2.NUMERIC:
            lohi = m['ranges'][f]; x = float(row[f])
            if lohi is None or not (lohi[0] - TOL <= x <= lohi[1] + TOL):
                return 'outside_training_range'
        return None


# ---------------------------------------------------------------- the fits
def lgb():
    import lightgbm
    return lightgbm


def fit_head(X, y, w, cand, rounds=ROUNDS):
    """One head: the candidate's columns of the 107, the normalized weights, the pinned parameters, 200 rounds."""
    cols = [T.FEATS.index(c) for c in INPUTS[cand]]; lg = lgb()
    ds = lg.Dataset(np.ascontiguousarray(X[:, cols]), label=np.asarray(y, float), weight=norm(w), feature_name=list(INPUTS[cand]), free_raw_data=False)
    return lg.train(dict(PARAMS), ds, num_boost_round=rounds)


def predict(b, cand, X):
    """Predictions of one head on rows of the 107 inputs (float32, the candidate's columns, one thread)."""
    cols = [T.FEATS.index(c) for c in INPUTS[cand]]
    return np.asarray(b.predict(np.ascontiguousarray(np.asarray(X, np.float32)[:, cols]), num_threads=1), float)


def model_path(out, cand, s, h):
    return Path(out) / 'models' / f'{cand}_{s}_{h}.txt'


def fit_all(E, out, rounds=ROUNDS):
    """The 12 heads -> saved models, each reloaded and bit-equal on its training rows; returns the fit manifest."""
    out = Path(out); (out / 'models').mkdir(parents=True, exist_ok=True); man = {}; lg = lgb()
    for cand in CANDS:
        for (s, h) in HEADS:
            e = E[(s, h)]; t0 = time.time(); b = fit_head(e['X'], e['y'], e['w'], cand, rounds); p = model_path(out, cand, s, h)
            C.require(not p.exists(), f'{p.name}: a model file exists (never overwritten)'); b.save_model(str(p))
            b2 = lg.Booster(model_file=str(p)); p1, p2 = predict(b, cand, e['X']), predict(b2, cand, e['X'])
            C.require(np.array_equal(p1, p2), f'{p.name}: the reloaded model does not predict bit-equal to the fitted booster')
            wn = norm(e['w']); ym = float(np.sum(wn * e['y']) / np.sum(wn))
            man[f'{cand} {s} {h}'] = dict(file=p.name, sha256=C.sha(p), rows=int(len(e['y'])), trees=int(b2.num_trees()), seconds=round(time.time() - t0, 2),
                                          train_weighted_mean_y=ym, insample_wmse=float(np.sum(wn * (p1 - e['y']) ** 2) / np.sum(wn)),
                                          insample_wmse_zero=float(np.sum(wn * e['y'] ** 2) / np.sum(wn)), insample_wmse_mean=float(np.sum(wn * (e['y'] - ym) ** 2) / np.sum(wn)))
    return man


def load_heads(out, fz):
    """The 12 boosters by their frozen hashes -> {cand: {(side, head): booster}}."""
    lg = lgb(); H = {c: {} for c in CANDS}
    for cand in CANDS:
        for (s, h) in HEADS:
            p = model_path(out, cand, s, h); C.require(C.sha(p) == fz['models'][f'{cand} {s} {h}']['sha256'], f'{p.name}: not the frozen model')
            H[cand][(s, h)] = lg.Booster(model_file=str(p))
    return H


# ---------------------------------------------------------------- the policies
class Candidate(L.ArmTeacher):
    """One candidate's controller on its own branch: the teacher's inputs (rebuilt on the bound branch) and binding check; the decisions from
    the candidate's heads (header's CONTROLLER)."""

    def __init__(self, heads, support, cand, causal=None):
        super().__init__(None, None, causal=causal); self.H, self.csup, self.cand, self.name = heads, support, cand, cand

    def decide(self, v, B):
        s, g, st = int(v['side']), float(v['t']), v['state']; head = 'post' if st == 'absent' else 'keep'; fb = FALLBACK[head]
        rec = dict(t=g, side=s, state=st, head=head, gid=v['gid'], policy=self.name)
        if st == 'absent' and v['room'] < -1e-9:
            return dict(rec, status='cap_blocked', action='WAIT')
        if st == 'absent' and g + KP.PLACE >= B.end:
            return dict(rec, status='near_end', action='WAIT')
        row = self.inputs(v, B, s, g)
        if row is None:
            return dict(rec, status='feature_unavailable', action=fb)
        ns = self.csup.check(s, head, row)
        if ns is not None:
            return dict(rec, status=ns, action=fb, row=row)
        a_ = float(predict(self.H[(s, head)], self.cand, np.asarray([[row[c] for c in T.FEATS]], np.float32))[0])
        if not np.isfinite(a_):
            return dict(rec, status='nonfinite_prediction', action=fb, row=row, A=a_)
        a = ('POST' if a_ > 0 else 'WAIT') if head == 'post' else ('PULL' if a_ < 0 else 'KEEP')
        return dict(rec, status='model', action=a, row=row, A=a_)


class PullAtReview(L.ArmTeacher):
    """ENTRY_PA's retention: PULL at every live review (each generation's first eligible review; a pulled order is cancel-pending after it)."""

    def __init__(self):
        super().__init__(None, None); self.name = 'PA'

    def decide(self, v, B):
        C.require(v['state'] == 'live', 'the review rule asked outside a live state')
        return dict(t=float(v['t']), side=int(v['side']), state='live', head='keep', gid=v['gid'], policy=self.name, status='first_review_pull', action='PULL')


class Never(L.ArmTeacher):
    """N0: never enters (absent -> WAIT); a live state cannot occur."""

    def __init__(self):
        super().__init__(None, None); self.name = 'N0'

    def decide(self, v, B):
        C.require(v['state'] == 'absent', 'N0 asked in a live state: it never posted')
        return dict(t=float(v['t']), side=int(v['side']), state='absent', head='post', gid=None, policy=self.name, status='n0', action='WAIT')


class Teacher0(L.ArmTeacher):
    """PI0 inside a policy: the frozen teacher's decision, tagged."""

    def __init__(self, M, sup):
        super().__init__(M, sup); self.name = 'PI0'

    def decide(self, v, B):
        return dict(super().decide(v, B), policy=self.name)


class Hybrid(L.ArmTeacher):
    """Entry from one policy, retention from another, on ONE branch (both bound to it): absent -> entry.decide, live -> retain.decide."""

    def __init__(self, entry, retain, name, causal=None):
        super().__init__(None, None, causal=causal); self.entry, self.retain, self.name = entry, retain, name

    def bind(self, B, world):
        super().bind(B, world); self.entry.bind(B, world); self.retain.bind(B, world)

    def decide(self, v, B):
        return (self.entry if v['state'] == 'absent' else self.retain).decide(v, B)


def make_policy(name, heads, csup, M, sup, causal=None):
    """A new, unbound policy object of the declared name (header's POLICIES)."""
    C.require(name in POLICIES, f'an undeclared policy {name}')
    if name == 'N0':
        p = Never()
    elif name == 'PI0':
        p = Teacher0(M, sup)
    elif name in CANDS:
        p = Candidate(heads[name], csup, name)
    elif name == 'ENTRY_ONLY':
        p = Hybrid(Candidate(heads['RICH'], csup, 'RICH'), Teacher0(M, sup), name)
    elif name == 'RETAIN_ONLY':
        p = Hybrid(Teacher0(M, sup), Candidate(heads['RICH'], csup, 'RICH'), name)
    else:
        p = Hybrid(Candidate(heads['RICH'], csup, 'RICH'), PullAtReview(), name)
    p.causal = causal
    return p


def factory(name, heads, csup, M, sup, world, every=None):
    """A policy factory for pf_r2_labels.bound_fork / pair_bound: each branch gets its own policy bound to it and to the world."""
    made = []

    def make(B):
        probe, st = T.causal_probe(every) if every else (None, None)
        p = make_policy(name, heads, csup, M, sup, causal=probe); p.bind(B, world); made.append((p, st)); return p
    make.made = made
    return make


# ---------------------------------------------------------------- the commands
def cmd_extract(out=OUT):
    """The training extraction (Oct 24-25 accepted rows) -> out/train/{side}_{head}.npz + extract.json + support.json."""
    check_inputs(); out = Path(out); dst = out / 'train'; C.require(not (dst / 'extract.json').exists(), 'the training extraction exists')
    dst.mkdir(parents=True, exist_ok=True)
    E, st = extract(dataset_rows(TRAIN['dir'], TRAIN['dataset_sha256']), TRAIN['days'])
    for (s, h), e in E.items():
        np.savez(dst / f'{s}_{h}.npz', X=e['X'], y=e['y'], w=e['w'])
        (dst / f'{s}_{h}.meta.json').write_text(json.dumps(e['meta'], sort_keys=True) + '\n')
    man = dict(version=VERSION, dataset_sha256=TRAIN['dataset_sha256'], days=list(TRAIN['days']), inputs={c: dict(n=len(INPUTS[c]), sha256=INPUT_SHA[c]) for c in CANDS},
               **manifest_of(E, st), asof=C.stamp())
    (dst / 'extract.json').write_text(json.dumps(man, indent=1, sort_keys=True) + '\n')
    S = support_of(E); (dst / 'support.json').write_text(json.dumps(S, indent=1, sort_keys=True) + '\n')
    print(json.dumps(dict(heads={k: v['rows'] for k, v in man['heads'].items()}, statuses=st, support_sha256=C.sha(dst / 'support.json')), indent=1))


def load_train(out=OUT):
    dst = Path(out) / 'train'; E = {}
    for (s, h) in HEADS:
        z = np.load(dst / f'{s}_{h}.npz'); E[(s, h)] = dict(X=z['X'], y=z['y'], w=z['w'], meta=json.loads((dst / f'{s}_{h}.meta.json').read_text()))
    return E


def cmd_fit(out=OUT):
    """The 12 fits on the frozen extraction -> out/models + fit.json (the model hashes; written before any Oct 26-27 label is read)."""
    check_inputs(); out = Path(out); C.require(not (out / 'fit.json').exists(), 'the fit exists')
    ex = json.loads((out / 'train' / 'extract.json').read_text()); E = load_train(out)
    C.require(manifest_of(E, ex['statuses'])['heads'] == ex['heads'], 'the training files differ from their extraction manifest')
    import lightgbm as lg_
    import platform
    t0 = time.time(); man = fit_all(E, out)
    fz = dict(version=VERSION, extract_sha256=C.sha(out / 'train' / 'extract.json'), support_sha256=C.sha(out / 'train' / 'support.json'), params=PARAMS,
              params_sha256=PARAMS_SHA, rounds=ROUNDS, inputs={c: dict(n=len(INPUTS[c]), sha256=INPUT_SHA[c]) for c in CANDS}, models=man,
              libraries=dict(lightgbm=lg_.__version__, numpy=np.__version__, pandas=pd.__version__, python=platform.python_version()),
              source_sha256=C.sha(Path(__file__)), seconds=round(time.time() - t0, 1), asof=C.stamp())
    (out / 'fit.json').write_text(json.dumps(fz, indent=1, sort_keys=True) + '\n')
    print(json.dumps({k: dict(rows=v['rows'], trees=v['trees'], sha=v['sha256'][:12]) for k, v in man.items()}, indent=1))


def frozen_fit(out=OUT):
    """fit.json, its extraction and support hashes and every model hash verified -> (fit manifest, heads, Support)."""
    out = Path(out); fz = json.loads((out / 'fit.json').read_text())
    C.require(C.sha(out / 'train' / 'extract.json') == fz['extract_sha256'] and C.sha(out / 'train' / 'support.json') == fz['support_sha256'], 'the extraction / support changed after the fit')
    import lightgbm as lg_
    C.require(lg_.__version__ == fz['libraries']['lightgbm'] and np.__version__ == fz['libraries']['numpy'], 'library versions differ from the fit')
    return fz, load_heads(out, fz), Support(json.loads((out / 'train' / 'support.json').read_text()))


def selftest():
    import pf_adv2_selftest as ST
    ST.run(sys.modules[__name__])


if __name__ == '__main__':
    a = sys.argv[1:]
    if a == ['selftest']:
        selftest()
    elif a == ['extract']:
        cmd_extract()
    elif a == ['fit']:
        cmd_fit()
    else:
        sys.exit(__doc__)
