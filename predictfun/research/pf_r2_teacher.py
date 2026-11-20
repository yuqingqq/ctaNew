"""THE SUPPORTED TEACHER pi0-50, VERSION pi0_50_v1 (plan v2 stage R2; P-2026-005; 2026-11-20, session aea5bfb9).
USER 2026-11-19: "Check my latest commit, next, can you ask codex research session for next priority. continue the implementing and next
priority loop"; review 3d7eae6a's next priority: "Implement, fit and freeze the supported pi0-50 teacher on this accepted Oct23 dataset, with
one bounded teacher-policy receipt." Declared before any fit, prediction or rollout of this module is read. Consumed Oct 23 only (the plan's
teacher warm-up); nothing is chosen on results: no threshold, horizon, leaf, round, weight or input choice after reading; one teacher version.

DATA: the census accepted by review 3d7eae6a: plan 249161127c39d5a51cc7c374262073af4f38c9245ef08fb505f0ec3e646dd56b, the 477 completion
 records in the committed order (reviews/r2b_full_day_20261120/records_full.sha256), every record = its manifest line and every part's
 parquet = its record, verified before anything is written. Population: EVERY sampled ok row (status ok, sampled); zero, no-fill and
 post-only-refusal outcomes kept; no outcome-selected downsampling. Groups: bid / ask x POST / RETAIN (the census head), fitted separately.
INPUTS: the census's 107 in its order (pf_r2_shadow.INPUTS + CTX), float32 as stored; identity, label and bookkeeping fields are never inputs
 (checked); the sha256 of the ordered list is frozen.
J ARCHITECTURE (pf_pipeline_bench.fit 'J', unchanged semantics): heads p (binary: the fill indicator q > 1e-12, all rows), q and qm (regression
 on the filled rows q > 1e-12), severe (regression, all rows); pf_pipeline_bench.PARAMS resolved and hashed, each head's objective, 200
 rounds, no early stopping, no tuning: 4 groups x 4 heads = 16 boosters.
VERSIONED PLUMBING (not the legacy fit code, which is unweighted on 91 inputs and in memory):
 (1) the 107 inputs;
 (2) weights: the census weight (equal total per original x side x head over its sampled ok rows), subsetted to the filled rows for q and
     qm. Each training set's weights are multiplied by ONE constant, n / sum(w), so that they average 1; the relative weights are exactly the
     census's. Reason, fixed before any fit: the pinned lambda_l2 (10) and LightGBM's min_sum_hessian_in_leaf act on hessian sums. At the
     census scale (sum = the number of originals, ~1/157 of the retain rows) a 200-row leaf would hold a hessian of ~1.3 against lambda
     10, shrinking every leaf ~10x relative to the unweighted legacy fit: a parameter change in disguise;
 (3) the Dataset is built from a file-backed lgb.Sequence: rows read by pread in 4,096-row batches, so the group matrix is never in memory.
     LightGBM bins from its own deterministic 200,000-row sample. At <= 200,000 rows the streamed trees are identical to an in-memory fit
     (selftest, and measured before this module). Above that, the two constructions draw different deterministic bin samples (both valid,
     not equal), so the streamed construction is declared, never claimed equal to an in-memory fit.
MAPPING (unchanged): pf_pipeline_bench.predict on the 107 inputs (SETS key 'T107', registered at import). It gives P; E[q | fill] clipped to
 the known remaining; E[q], E[q move]; V = edge E[q] + E[q move], with edge = s (pf_mid - p); and the two-atom round-trip coherence (ok).
SUPPORT (the census support JSON, sha256 frozen; reviews c68a072c / 3d7eae6a), evaluated in this order, each a distinct status. Fallbacks:
 absent -> WAIT, live -> KEEP.
 - Structural first, in this order: pending / terminal states process notices (the engine asks nothing). An absent side is cap_blocked
   (room < 0) or near_end (activation at or after END) -> WAIT. No valid book: the engine's own fallback. feature_unavailable: no grid
   point, a nonfinite pf_mid, or no rebuilt row.
 - Then the support checks, in this order:
   * unsupported_horizon: END - decision outside the declared bins;
   * insufficient_training_support: the coarse cell is absent or not SUPPORTED;
   * unsupported_state_category: any of the six dims' value never observed for the side / head, 'n/a' included;
   * outside_training_range: a mandatory numeric context field outside its observed [lo, hi] +- 1e-9, the declared inventory-sign
     tolerance; NaN is outside; never clipped;
   * incoherent_forecast: the mapping's ok false, or a nonfinite V.
POLICY pi0-50 (the E1 convention, plan v2), on the 50 ms grid in the engine's order (pre_tick, alternating sides, the second side's view
 after the first side's command):
 - absent, supported and coherent: POST iff V_post > 0, else WAIT (a tie WAITs);
 - live, supported and coherent: PULL iff V_retain < 0, else KEEP (a tie KEEPs).
 The inputs are rebuilt by pf_r2_shadow.rebuild_row, unchanged, on the ACTUAL branch state (B given) - never a stored shadow row: the
 client context; the generation's price / activation; known_steps from the fills with the notice delay; the market features at the
 decision's grid index; the archived fair path; t_to_end to the original END. Each side's inputs use its own original of the shared
 (market, time) world. No timer, no severe-risk penalty, no inventory utility; the cap and timing are the engine's (58 / 37 / 100 ms,
 cap 30).
FREEZE: freeze.json binds:
 - the actually imported research sources;
 - the data identities and the extraction manifest;
 - the ordered inputs, the weight rule and the resolved parameters;
 - the library versions;
 - the 16 model hashes, each reloaded and predicting equal to its fitted booster;
 - the support JSON hash and this policy contract;
 - multiplicity: one teacher version, no candidates.
 diag and rollout refuse unless every frozen hash still holds.
DIAGNOSTICS (in-sample, training only, not skill and not gates), per side / head / horizon bin, weighted by the normalized census weights:
 rows, originals, fill calibration, conditional quantity and movement errors on the filled rows, the severe error, coherent / incoherent
 fractions, and predicted vs realized value (cents). Also: the support adapter's categories and cells recomputed on every training row,
 against the support JSON.
ROLLOUT (bounded, after the freeze): the fixed first 128 originals per side of consumed Oct 23, gate2A's population, which forms 128
 (market, time) worlds with both sides. Each world is one 5 s episode with the teacher on both sides.
 Reported per side: decisions by status and action; the engine's fallbacks and constraints; originals with an entry at offset 0, a delayed
 first entry, or none; later generations (ordinal >= 2); PULLs; fills.
 Ledger invariants: the engine's position check every tick; cash / position / value recomputed from the fills; every generation's fills
 <= its size; every decision the engine asked of the policy logged.
 Causality on a deterministic sample of decisions (every 7th policy call): hidden_copy leaves the inputs, status and action unchanged;
 the delivered copy is the positive control.
 If delayed entry is absent or fallbacks dominate, that is reported, never tuned.
Usage (run_light): pf_r2_teacher.py selftest | extract | fit <group 0-3> | freeze | diag | rollout
"""
import copy
import hashlib
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_branch_engine as BE                             # noqa: E402
import pf_pipeline_bench as BX                            # noqa: E402
import pf_r2_shadow as R2                                 # noqa: E402

C, pd, I, KP, X, QP = R2.C, R2.pd, R2.I, R2.KP, R2.X, R2.QP
VERSION, DAY, MARK = 'pi0_50_v1', '20261023', 'R2 TEACHER'
PLAN_SHA = '249161127c39d5a51cc7c374262073af4f38c9245ef08fb505f0ec3e646dd56b'
REAL = dict(census=Path(R2.OUT) / f'census_{DAY}_full', support=Path(R2.OUT) / f'support_{DAY}_full.json', plan_sha=PLAN_SHA, n_parts=477,
            records=HERE.parent / 'reviews' / 'r2b_full_day_20261120' / 'records_full.sha256', out=Path(I.K.IO) / 'r2_teacher_20261120',
            population=HERE.parent / 'reviews' / 'r2a_launch_gate_20261120' / 'gate2A_checkpoint' / 'plan.json')
FEATS = list(R2.INPUTS) + list(R2.CTX)
GROUPS = [(0, 'post'), (0, 'retain'), (1, 'post'), (1, 'retain')]
HEADS = ('p', 'q', 'qm', 'severe')
OBJ = dict(p='binary', q='regression', qm='regression', severe='regression')
PARAMS, ROUNDS = dict(BX.PARAMS), BX.ROUNDS
FILLED, BATCH, TOL, END, STEP = 1e-12, 4096, R2.INV_TOL, float(BE.END), float(BE.STEP)
LAB = ['q', 'qm', 'severe', 'value', 'weight', 'price', 'decision_ms', 'offset_ms', 'generation', 'original_episode_id', 'instrument_id']
IDENT = ('day', 'instrument_id', 'original_episode_id', 'side', 't0', 'source', 'head', 'offset_ms', 'decision_ms', 'generation', 'gid', 'status',
         'sampled', 'weight', 'q', 'value', 'qm', 'severe', 'refused', 'price', 'activation_ms', 'effective_ms', 'end_mark', 'end_abs_ms',
         'decision_abs_ms', 'phase_post', 'phase_retain', 'first_live', 'pf_mid_ref', 'mark_source')
OTHER = ('absent', 'post_pending', 'live', 'cancel_pending', 'terminal')
STATUSES = ('cap_blocked', 'near_end', 'feature_unavailable', 'unsupported_horizon', 'insufficient_training_support', 'unsupported_state_category',
            'outside_training_range', 'incoherent_forecast', 'model')
GUARD_MIB = 1536                                           # the census's checkpoint high-water: a fit / rollout above it stops (never raised)
BX.SETS['T107'] = FEATS                                    # the unchanged mapping on the 107 inputs


def sh_of(s, head):
    return f"{'bid' if int(s) == 0 else 'ask'} {head}"


def hbin(t_to_end):
    """The census's horizon bin label (pf_r2_shadow.support_part: pd.cut, right-closed, the interval's string) of END - decision; 'nan' outside
    the bins (explicitly from the codes: pandas 3 keeps a missing label as NaN under astype(str))."""
    c = pd.cut(np.atleast_1d(np.asarray(t_to_end, float)), R2.HORIZON_BINS, right=True); lab_ = np.array([str(x) for x in c.categories] + ['nan'])
    return lab_[np.where(np.asarray(c.codes) >= 0, np.asarray(c.codes), len(lab_) - 1)]


def categories(df):
    """The six support dims of rows (columns fp_remaining, fp_queue, ctx_inventory_shares, ctx_other_state_*, generation, price), with the census's
    own np.select rules (pf_r2_shadow.support_part), the inputs as their stored float32 values."""
    rem = df.fp_remaining.astype(float); qu = df.fp_queue.astype(float); inv = df.ctx_inventory_shares.astype(float); gen = df.generation.astype(float)
    out = dict(c_rem=np.select([rem <= 0, rem <= 10, rem <= 20, rem <= 30], ['0', '(0,10]', '(10,20]', '(20,30]'], 'n/a'),
               c_queue=np.select([~np.isfinite(qu), qu <= 0, qu <= 30, qu <= 150], ['unknown', '0', '(0,30]', '(30,150]'], '>150'),
               c_gen=np.select([gen <= 1, gen <= 2], ['1', '2'], '>=3'),
               c_inv=np.select([inv > TOL, inv < -TOL], ['long', 'short'], 'flat'),
               c_other=np.select([df[f'ctx_other_state_{z}'].astype(float) > 0 for z in OTHER], list(OTHER), 'n/a'),
               c_price=np.select([df.price <= 5, df.price <= 20, df.price <= 80, df.price <= 95], ['[1,5]', '(5,20]', '(20,80]', '(80,95]'], '(95,99]'))
    return {k: np.asarray(v).astype(str) for k, v in out.items()}


class Support:
    """The support adapter: the coarse cells, the observed category values and the numeric ranges of the support JSON."""

    def __init__(self, S):
        self.cells = {k: v['support'] for k, v in S['cells'].items()}
        self.present = {sh: {d: {x for x, n in vals.items() if int(n) > 0} for d, vals in S['presence'][sh].items()} for sh in R2.SIDE_HEADS}
        self.ranges = {sh: {f: (float(v[0]), float(v[1])) for f, v in r.items()} for sh, r in S['ranges'].items()}

    def check(self, s, head, row):
        """row: the 107 float32 inputs + price + generation + decision_ms -> None (supported) or the first failing status."""
        sh = sh_of(s, head); hb = hbin(END - float(row['decision_ms']))[0]
        if hb == 'nan':
            return 'unsupported_horizon'
        if self.cells.get(f'{sh} {hb}') != 'SUPPORTED':
            return 'insufficient_training_support'
        cats = categories(pd.DataFrame([row]))
        if any(str(cats[d][0]) not in self.present.get(sh, {}).get(d, set()) for d in R2.DIMS):
            return 'unsupported_state_category'
        for f in R2.NUMERIC:
            lo, hi = self.ranges[sh][f]; x = float(row[f])
            if not (lo - TOL <= x <= hi + TOL):
                return 'outside_training_range'
        return None


# ---------------------------------------------------------------- the accepted census -> the group files
def census_parts(cfg):
    """The accepted census's parts in the committed order, each verified: the plan hash, every record = its line of the committed records
    manifest, every parquet = its record's hash -> [(record, parquet path)]."""
    C.require(C.sha(cfg['census'] / 'plan.json') == cfg['plan_sha'], 'the census plan is not the accepted one')
    lines = [x for x in Path(cfg['records']).read_text().splitlines() if x.strip()]; out = []
    C.require(len(lines) == cfg['n_parts'], f"the records manifest holds {len(lines)} lines, not {cfg['n_parts']}")
    for i, ln in enumerate(lines):
        h, rel = ln.split('  '); rp = cfg['census'] / rel.split('/', 1)[1]
        C.require(rp.name == f'part_{i:04d}.json' and C.sha(rp) == h, f'record {i} differs from the committed manifest')
        rec = json.loads(rp.read_text()); f = rp.parent / rec['file']
        C.require(rec['plan_sha256'] == cfg['plan_sha'] and rec['part'] == i and C.sha(f) == rec['sha256'], f'part {i}: record or parquet differs')
        out.append((rec, f))
    return out


def extract(cfg=REAL):
    """Every sampled ok row of the census into per-group files (written to train.partial, promoted when complete): X_all.f32 / X_fil.f32
    (row-major float32, the 107 in order) and one float64 file per LAB column for all / filled rows; manifest.json with the counts and hashes.
    The census is never concatenated: one part at a time."""
    fin, part = cfg['out'] / 'train', cfg['out'] / 'train.partial'
    C.require(not fin.exists(), 'the training files exist (immutable: never rewritten)')
    parts = census_parts(cfg); part.mkdir(parents=True, exist_ok=False)
    dirs = {g: part / f'{g[0]}_{g[1]}' for g in GROUPS}; fh = {}
    for g, d in dirs.items():
        d.mkdir()
        for nm in ['X_all.f32', 'X_fil.f32'] + [f'{c}_{k}.f64' for c in LAB for k in ('all', 'fil')]:
            fh[(g, nm)] = open(d / nm, 'wb')
    cols = list(dict.fromkeys(FEATS + LAB + ['side', 'head', 'status', 'sampled'])); n = {g: [0, 0] for g in GROUPS}; per_part = []
    try:
        for rec, f in parts:
            D = pd.read_parquet(f, columns=cols); D = D[(D.status == 'ok') & D.sampled.astype(bool)]
            C.require(len(D) == int(rec['sampled_ok']), f"part {rec['part']}: {len(D)} sampled ok rows, the record says {rec['sampled_ok']}")
            C.require(not any(c in FEATS for c in IDENT), 'an identity / label field among the inputs')
            row = []
            for g in GROUPS:
                G = D[(D.side.astype(int) == g[0]) & (D['head'] == g[1])]; Xa = np.ascontiguousarray(G[FEATS].to_numpy(np.float32))
                fil = G.q.to_numpy(float) > FILLED
                fh[(g, 'X_all.f32')].write(Xa.tobytes()); fh[(g, 'X_fil.f32')].write(np.ascontiguousarray(Xa[fil]).tobytes())
                for c in LAB:
                    v = G[c].to_numpy(np.float64); fh[(g, f'{c}_all.f64')].write(v.tobytes()); fh[(g, f'{c}_fil.f64')].write(np.ascontiguousarray(v[fil]).tobytes())
                n[g][0] += len(G); n[g][1] += int(fil.sum()); row.append(len(G))
            C.require(sum(row) == len(D), f"part {rec['part']}: {len(D) - sum(row)} sampled ok rows in no side / head group")
            per_part.append(row)
    finally:
        for x in fh.values():
            x.close()
    man = dict(version=VERSION, asof=C.stamp(), plan_sha256=cfg['plan_sha'], records_sha256=C.sha(cfg['records']), features=FEATS,
               features_sha256=hashlib.sha256(json.dumps(FEATS).encode()).hexdigest(), lab=LAB,
               groups={f'{g[0]}_{g[1]}': dict(rows=n[g][0], filled=n[g][1]) for g in GROUPS}, per_part_rows=per_part,
               files={f'{g[0]}_{g[1]}/{p.name}': C.sha(p) for g in GROUPS for p in sorted(dirs[g].iterdir())})
    (part / 'manifest.json').write_text(json.dumps(man, indent=1)); part.rename(fin)
    return man


def train_manifest(cfg, verify=True):
    """The extraction manifest, every group file re-hashed (verify)."""
    man = json.loads((cfg['out'] / 'train' / 'manifest.json').read_text())
    if verify:
        for k, h in man['files'].items():
            C.require(C.sha(cfg['out'] / 'train' / k) == h, f'training file {k} changed since its manifest')
    C.require(man['features'] == FEATS, 'the extracted input order differs')
    return man


# ---------------------------------------------------------------- the streamed J fit
def _seq_base():
    import lightgbm as lgb
    return lgb


class Rows:
    """A file-backed row source for lgb.Dataset: rows of a row-major float32 file read by pread (int -> one row, float64 as the sampler
    requires, exact; slice -> a batch)."""
    batch_size = BATCH

    def __init__(self, path, n, ncol=len(FEATS)):
        self.path, self.n, self.ncol, self.rb = str(path), int(n), int(ncol), int(ncol) * 4
        C.require(os.path.getsize(self.path) == self.n * self.rb, f'{path}: size {os.path.getsize(self.path)} != {self.n} rows x {self.rb} bytes')
        self.fd = os.open(self.path, os.O_RDONLY)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            a, b, st = idx.indices(self.n); C.require(st == 1, 'a strided batch')
            return np.frombuffer(os.pread(self.fd, (b - a) * self.rb, a * self.rb), np.float32).reshape(b - a, self.ncol)
        i = int(idx); i = i + self.n if i < 0 else i
        return np.frombuffer(os.pread(self.fd, self.rb, i * self.rb), np.float32).astype(np.float64)

    def block(self, a, b):
        return self[a:b]

    def close(self):
        os.close(self.fd)


def as_sequence(rows):
    """Rows wrapped as an lgb.Sequence (LightGBM requires the subclass)."""
    lgb = _seq_base()

    class Seq(lgb.Sequence):
        batch_size = BATCH

        def __len__(self):
            return len(rows)

        def __getitem__(self, idx):
            return rows[idx]
    return Seq()


def norm_weights(w):
    """The census weights of one training set times the one constant n / sum(w) (mean 1; relative weights unchanged)."""
    w = np.asarray(w, float); C.require(len(w) > 0 and np.all(np.isfinite(w)) and np.all(w > 0), 'a training weight is not finite and positive')
    return w * (len(w) / w.sum())


def train_one(rows, y, w, objective, rounds=None):
    """One head: the pinned parameters with its objective, the streamed Dataset, 200 rounds."""
    lgb = _seq_base()
    ds = lgb.Dataset([as_sequence(rows)], label=np.asarray(y, float), weight=norm_weights(w), feature_name=FEATS, free_raw_data=True)
    return lgb.train({**PARAMS, 'objective': objective}, ds, num_boost_round=rounds or ROUNDS)


def lab(d, col, kind, n):
    v = np.fromfile(d / f'{col}_{kind}.f64', np.float64); C.require(len(v) == n, f'{d.name} {col}_{kind}: {len(v)} values, not {n}'); return v


def fit_group(k, cfg=REAL, rounds=None):
    """Group k (GROUPS) -> its four boosters saved as models/<side>_<head>_<head name>.txt and fit_<k>.json (rows, hashes, time, high-water)."""
    import pf_r2_runner as RN
    s, h = GROUPS[k]; man = train_manifest(cfg); d = cfg['out'] / 'train' / f'{s}_{h}'; mdir = cfg['out'] / 'models'; mdir.mkdir(exist_ok=True)
    na, nf = man['groups'][f'{s}_{h}']['rows'], man['groups'][f'{s}_{h}']['filled']; t0 = time.time()
    qa, qf = lab(d, 'q', 'all', na), lab(d, 'q', 'fil', nf); C.require(int((qa > FILLED).sum()) == nf and np.all(qf > FILLED), 'the filled subset differs')
    wa, wf = lab(d, 'weight', 'all', na), lab(d, 'weight', 'fil', nf)
    ra, rf = Rows(d / 'X_all.f32', na), Rows(d / 'X_fil.f32', nf); out = {}
    try:
        spec = dict(p=(ra, (qa > FILLED).astype(float), wa), q=(rf, qf, wf), qm=(rf, lab(d, 'qm', 'fil', nf), wf), severe=(ra, lab(d, 'severe', 'all', na), wa))
        for head in HEADS:
            r_, y_, w_ = spec[head]; b = train_one(r_, y_, w_, OBJ[head], rounds)
            iters = int(b.current_iteration())                                              # LightGBM ends early only when no split exists
            C.require(b.feature_name() == FEATS and 1 <= iters <= (rounds or ROUNDS), f'{head}: the booster lost its inputs, or {iters} iterations')
            p_ = mdir / f'{s}_{h}_{head}.txt'; b.save_model(str(p_)); Xs = r_.block(0, min(2000, len(r_)))
            eq = bool(np.array_equal(b.predict(Xs, num_threads=1), _seq_base().Booster(model_file=str(p_)).predict(Xs, num_threads=1)))
            out[head] = dict(sha256=C.sha(p_), rows=len(y_), reload_equal=eq, iterations=iters)
            C.require(RN.rss()[1] <= GUARD_MIB, f'the high-water passed {GUARD_MIB} MiB')
    finally:
        ra.close(); rf.close()
    info = dict(group=f'{s}_{h}', rows=na, filled=nf, heads=out, rounds=rounds or ROUNDS, seconds=round(time.time() - t0, 1), hwm_mib=RN.rss()[1], asof=C.stamp())
    (cfg['out'] / f'fit_{k}.json').write_text(json.dumps(info, indent=1)); return info


# ---------------------------------------------------------------- the freeze
def research_sources():
    """sha256 of every research module actually imported (the closure of this process)."""
    out = {}
    for m in list(sys.modules.values()):
        f = getattr(m, '__file__', None)
        if f and Path(f).resolve().parent == HERE and f.endswith('.py'):
            out[Path(f).name] = C.sha(Path(f).resolve())
    return dict(sorted(out.items()))


def load_models(cfg, freeze=None):
    """The 16 boosters; with a freeze, every model file must hash to its frozen value first."""
    lgb = _seq_base(); M = {}
    for s, h in GROUPS:
        for head in HEADS:
            p_ = cfg['out'] / 'models' / f'{s}_{h}_{head}.txt'
            if freeze is not None:
                C.require(C.sha(p_) == freeze['models'][f'{s}_{h}_{head}'], f'model {p_.name} differs from the freeze')
            M.setdefault((s, h), {})[head] = lgb.Booster(model_file=str(p_))
    return M


def freeze(cfg=REAL):
    """freeze.json: sources, data identities, inputs, weight rule, parameters, libraries, the 16 model hashes (each saved model reloaded in its
    fitting process and predicting exactly as the fitted booster on the first 2,000 training rows of its set), the support JSON and the policy
    contract."""
    import lightgbm as lgb
    import pyarrow
    man = train_manifest(cfg); fits = [json.loads((cfg['out'] / f'fit_{k}.json').read_text()) for k in range(len(GROUPS))]
    models = {f'{s}_{h}_{head}': C.sha(cfg['out'] / 'models' / f'{s}_{h}_{head}.txt') for s, h in GROUPS for head in HEADS}
    C.require(all(models[f"{f_['group']}_{hd}"] == f_['heads'][hd]['sha256'] for f_ in fits for hd in HEADS), 'a model file differs from its fit record')
    reload_ok = all(f_['heads'][hd]['reload_equal'] for f_ in fits for hd in HEADS)
    C.require(reload_ok, 'a saved model does not predict identically to its fitted booster')
    fz = dict(version=VERSION, asof=C.stamp(), sources=research_sources(), plan_sha256=cfg['plan_sha'], records_sha256=C.sha(cfg['records']),
              train_manifest_sha256=C.sha(cfg['out'] / 'train' / 'manifest.json'), support_sha256=C.sha(cfg['support']), features=FEATS,
              features_sha256=man['features_sha256'], params=PARAMS, params_sha256=hashlib.sha256(json.dumps(PARAMS, sort_keys=True).encode()).hexdigest(),
              objectives=OBJ, rounds=fits[0]['rounds'], weight_rule='census weight x (n / sum w) per training set; q, qm on q > 1e-12 with the same weights subsetted',
              construction='file-backed lgb.Sequence (pread, 4,096-row batches); LightGBM bin sample of the streamed path', models=models, fits=fits,
              reload_equal=reload_ok, libraries=dict(lightgbm=lgb.__version__, numpy=np.__version__, pandas=pd.__version__, pyarrow=pyarrow.__version__,
                                                     python=platform.python_version()),
              policy=dict(rule='E1: absent POST iff supported, coherent and V > 0, else WAIT; live PULL iff supported, coherent and V < 0, else KEEP',
                          statuses=list(STATUSES), fallback=dict(R2.FALLBACK), timing=dict(place_ms=KP.PLACE, cancel_ms=KP.CANCEL, fill_notice_ms=BE.FILL_NOTICE,
                          step_ms=STEP, end_ms=END, cap=BE.CAP, size=BE.SIZE), costs=dict(R2.COSTS)),
              multiplicity=dict(teacher_versions=1, candidates=0))
    (cfg['out'] / 'freeze.json').write_text(json.dumps(fz, indent=1, default=str)); return fz


def frozen(cfg=REAL):
    """The freeze, verified: every model, the training manifest, the support JSON and every source it names still hash as frozen."""
    fz = json.loads((cfg['out'] / 'freeze.json').read_text())
    C.require(C.sha(cfg['out'] / 'train' / 'manifest.json') == fz['train_manifest_sha256'] and C.sha(cfg['support']) == fz['support_sha256'],
              'the training manifest or the support JSON differs from the freeze')
    for n_, h_ in fz['sources'].items():
        C.require(C.sha(HERE / n_) == h_, f'source {n_} changed since the freeze')
    return fz


# ---------------------------------------------------------------- the teacher policy
def predict_rows(models_g, df, side):
    """The unchanged mapping on rows of one group: df holds the 107 inputs and the order price -> pf_pipeline_bench.predict's dict."""
    d_ = df[FEATS].astype(np.float32).copy(); d_['side'] = int(side); d_['p'] = df['price'].astype(float).values
    return BX.predict(models_g, d_, 'T107')


class Teacher(BE.Policy):
    """pi0-50: a frozen causal function of the branch's client state, through pf_r2_shadow.rebuild_row on the actual branch."""

    def __init__(self, models, support, causal=None):
        super().__init__(self.fn, VERSION); self.M, self.sup = models, support; self.log = []; self.causal = causal; self.B = None; self.w = None

    def bind(self, B, world):
        self.B, self.w = B, world

    def inputs(self, v, B, s, g):
        """The 107 inputs (float32), the order price and the generation of side s at g on branch B (v: its view), or None (feature_unavailable)."""
        w = self.w; o = w['o'][s]; tg = np.asarray(w['W'][s]['tg'], float); d = float(o.t) + g; kk = min(int(np.searchsorted(tg, d)), len(tg) - 1)
        mk = w['market'][s].get(kk) if abs(tg[kk] - d) <= 1e-6 else None
        if mk is None:
            return None
        x = R2.rebuild_row(w['path'], w['W'], w['fv'], o, w['day'], {'decision_ms': g}, w['fair_m'], w['depth'], w['feat_fn'], B=B, market=mk)
        if x is None or not np.isfinite(x.get('pf_mid', np.nan)):
            return None
        gid = B.open[s]
        price = (v['bid'] if s == 0 else v['ask']) if gid is None else B.E.o[gid]['price']        # rebuild_row's JOIN rule / the order's price
        row = {c: np.float32(x[c]) for c in FEATS}; row.update(price=float(price), generation=float(row['ctx_generation_ordinal']), decision_ms=float(g))
        return row

    def decide(self, v, B):
        s, g, st = int(v['side']), float(v['t']), v['state']; head = 'post' if st == 'absent' else 'retain'; fb = R2.FALLBACK[head]
        rec = dict(t=g, side=s, state=st, head=head, gid=v['gid'])
        if st == 'absent' and v['room'] < -1e-9:
            return dict(rec, status='cap_blocked', action='WAIT')
        if st == 'absent' and g + KP.PLACE >= B.end:
            return dict(rec, status='near_end', action='WAIT')
        row = self.inputs(v, B, s, g)
        if row is None:
            return dict(rec, status='feature_unavailable', action=fb)
        ns = self.sup.check(s, head, row)
        if ns is not None:
            return dict(rec, status=ns, action=fb, row=row)
        pr = predict_rows(self.M[(s, head)], pd.DataFrame([row]), s); V = float(pr['V'][0])
        if not (bool(pr['ok'][0]) and np.isfinite(V)):
            return dict(rec, status='incoherent_forecast', action=fb, row=row, V=V)
        a = ('POST' if V > 0 else 'WAIT') if head == 'post' else ('PULL' if V < 0 else 'KEEP')
        return dict(rec, status='model', action=a, row=row, V=V, P=float(pr['P'][0]))

    def fn(self, v):
        dec = self.decide(v, self.B); self.log.append(dec)
        if self.causal is not None:
            self.causal(self, v, dec)
        return dec['action']


def same_decision(a, b):
    """Two decisions equal in status, action and every input (float32 bits) and price."""
    if a['status'] != b['status'] or a['action'] != b['action']:
        return False
    ra, rb = a.get('row'), b.get('row')
    if (ra is None) != (rb is None):
        return False
    return ra is None or all(np.float32(ra[c]).tobytes() == np.float32(rb[c]).tobytes() for c in FEATS) and ra['price'] == rb['price']


def causal_probe(every, seed=0):
    """A deterministic sample of decisions (every `every`-th call): the teacher's decision on hidden_copy (unreceived frames, unnotified fills,
    engine remainders / queues, future trades perturbed) must equal the actual one; the delivered copy (the last received frame perturbed) is
    the positive control (counted when the inputs or the status change). Parity: the decision's inputs and price must equal an independent
    rebuild (pf_r2_shadow.rebuild_row computing its own market features at the decision's grid index; the price by the JOIN / order rule)."""
    st = dict(n=0, checked=0, mismatch=0, delivered_changed=0, parity_checked=0, parity_mismatch=0, by_state={}, details=[])

    def probe(T, v, dec):
        st['n'] += 1
        if (st['n'] - 1) % every:
            return
        rng = np.random.default_rng([seed, st['n']]); g, s = float(v['t']), int(v['side'])
        B2 = R2.hidden_copy(T.B, g, rng); d2 = T.decide(B2.view(g, s), B2)
        B3 = R2.hidden_copy(T.B, g, rng, delivered=True); d3 = T.decide(B3.view(g, s), B3)
        st['checked'] += 1; ok_ = same_decision(dec, d2); ch_ = not same_decision(dec, d3); st['mismatch'] += int(not ok_); st['delivered_changed'] += int(ch_)
        bs = st['by_state'].setdefault(v['state'], dict(checked=0, mismatch=0, delivered_changed=0)); bs['checked'] += 1; bs['mismatch'] += int(not ok_)
        bs['delivered_changed'] += int(ch_)
        row = dec.get('row')
        if row is not None:                                                                 # the independent rebuild
            w = T.w; o = w['o'][s]; x = R2.rebuild_row(w['path'], w['W'], w['fv'], o, w['day'], {'decision_ms': g}, w['fair_m'], w['depth'], w['feat_fn'], B=T.B)
            gid = T.B.open[s]; pr_ = (v['bid'] if s == 0 else v['ask']) if gid is None else T.B.E.o[gid]['price']
            par = x is not None and all(np.float32(x[c]).tobytes() == np.float32(row[c]).tobytes() for c in FEATS) and float(pr_) == row['price']
            st['parity_checked'] += 1; st['parity_mismatch'] += int(not par)
        if not ok_ and len(st['details']) < 5:
            st['details'].append(dict(t=g, side=s, status=(dec['status'], d2['status']), action=(dec['action'], d2['action'])))
    return probe, st


# ---------------------------------------------------------------- diagnostics (in-sample) and the adapter reconciliation
def diag(cfg=REAL, chunk=200000):
    """Per side / head / horizon bin training diagnostics of the frozen models (weighted by the normalized census weights), and the adapter's
    categories / cells recomputed on every training row against the support JSON -> diag.json + diag_horizon.csv in out/diag."""
    fz = frozen(cfg); man = train_manifest(cfg); M = load_models(cfg, fz); S = json.loads(Path(cfg['support']).read_text()); out = cfg['out'] / 'diag'
    out.mkdir(exist_ok=True); rows = []; cat_cnt = {}; cell_cnt = {}
    for s, h in GROUPS:
        d = cfg['out'] / 'train' / f'{s}_{h}'; n = man['groups'][f'{s}_{h}']['rows']; sh = sh_of(s, h); r = Rows(d / 'X_all.f32', n)
        L = {c: lab(d, c, 'all', n) for c in LAB}; wn = norm_weights(L['weight']); hb_all = hbin(END - L['decision_ms']); acc = {}
        orig = L['original_episode_id'].astype(np.int64); nzall = L['q'] > FILLED
        for a in range(0, n, chunk):
            b = min(a + chunk, n); Xn = r.block(a, b); Xc = pd.DataFrame(Xn, columns=FEATS); Xc['price'] = L['price'][a:b]; Xc['generation'] = L['generation'][a:b]
            pr = predict_rows(M[(s, h)], Xc, s); cats = categories(Xc); o_, nz, hb, w_ = orig[a:b], nzall[a:b], hb_all[a:b], wn[a:b]
            qf = np.full(b - a, np.nan); mq = np.full(b - a, np.nan)
            if nz.any():                                                                    # the conditional heads on the filled rows only
                qf[nz] = np.clip(M[(s, h)]['q'].predict(Xn[nz], num_threads=1), 0., Xc.fp_remaining.to_numpy(float)[nz])
                mq[nz] = M[(s, h)]['qm'].predict(Xn[nz], num_threads=1)
            for dim in R2.DIMS:
                for val in np.unique(cats[dim]):
                    _cnt(cat_cnt.setdefault((sh, dim, str(val)), [0, set(), set()]), cats[dim] == val, o_, nz)
            for hv in np.unique(hb):
                m_ = hb == hv; _cnt(cell_cnt.setdefault(f'{sh} {hv}', [0, set(), set()]), m_, o_, nz)
                e = acc.setdefault(hv, dict(n=0, w=0., P=0., fill=0., wf=0., qp=0., qr=0., qae=0., mp=0., mr=0., mae=0., sp=0., sr=0., sae=0., coh=0, zq=0,
                                            Vp=0., Vr=0., Vpw=0., Vrw=0.))
                wm, okm, f_ = w_[m_], pr['ok'][m_], nz[m_]; wf_ = wm[f_]
                e['n'] += int(m_.sum()); e['w'] += wm.sum(); e['P'] += (wm * pr['P'][m_]).sum(); e['fill'] += (wm * f_).sum(); e['wf'] += wf_.sum()
                qr_, mr_ = L['q'][a:b][m_][f_], L['qm'][a:b][m_][f_]; qp_, mp_ = qf[m_][f_], mq[m_][f_]
                e['qp'] += (wf_ * qp_).sum(); e['qr'] += (wf_ * qr_).sum(); e['qae'] += (wf_ * np.abs(qp_ - qr_)).sum()
                e['mp'] += (wf_ * mp_).sum(); e['mr'] += (wf_ * mr_).sum(); e['mae'] += (wf_ * np.abs(mp_ - mr_)).sum()
                sv = L['severe'][a:b][m_]; e['sp'] += (wm * pr['S'][m_]).sum(); e['sr'] += (wm * sv).sum(); e['sae'] += (wm * np.abs(pr['S'][m_] - sv)).sum()
                e['coh'] += int(okm.sum()); e['zq'] += int(pr['zq'][m_].sum()); vr = L['value'][a:b][m_]
                e['Vp'] += float(pr['V'][m_][okm].sum()); e['Vr'] += float(vr[okm].sum()); e['Vpw'] += float((wm[okm] * pr['V'][m_][okm]).sum()); e['Vrw'] += float((wm[okm] * vr[okm]).sum())
        r.close()
        for hv, e in sorted(acc.items()):
            sw, swf = e['w'], e['wf']; ce = cell_cnt[f'{sh} {hv}']
            rows.append(dict(side_head=sh, horizon=hv, rows=e['n'], originals=len(ce[1]), nonzero_originals=len(ce[2]), weight=sw, pred_fill=e['P'] / sw,
                             real_fill=e['fill'] / sw, cond_q_pred=e['qp'] / swf if swf else np.nan, cond_q_real=e['qr'] / swf if swf else np.nan,
                             cond_q_mae=e['qae'] / swf if swf else np.nan, cond_qm_pred=e['mp'] / swf if swf else np.nan,
                             cond_qm_real=e['mr'] / swf if swf else np.nan, cond_qm_mae=e['mae'] / swf if swf else np.nan, severe_pred=e['sp'] / sw,
                             severe_real=e['sr'] / sw, severe_mae=e['sae'] / sw, coherent_frac=e['coh'] / e['n'], incoherent=e['n'] - e['coh'],
                             roundtrip_refused=e['zq'], V_pred_sum_c=e['Vp'], V_real_sum_c=e['Vr'], V_pred_wmean_c=e['Vpw'] / sw, V_real_wmean_c=e['Vrw'] / sw))
    rec = reconcile({k: (v[0], len(v[1]), len(v[2])) for k, v in cat_cnt.items()}, {k: (v[0], len(v[1]), len(v[2])) for k, v in cell_cnt.items()}, S)
    pd.DataFrame(rows).to_csv(out / 'diag_horizon.csv', index=False)
    (out / 'diag.json').write_text(json.dumps(dict(asof=C.stamp(), freeze_sha256=C.sha(cfg['out'] / 'freeze.json'), adapter=rec, rows=rows), indent=1, default=str))
    return rec, rows


def reconcile(got_c, got_l, S):
    """The adapter's (rows, originals, nonzero originals) per category and per cell against the support JSON's tables; a declared empty cell or
    value counts (0, 0, 0)."""
    z3 = (0, 0, 0); got_c, got_l = dict(got_c), dict(got_l)
    want_c = {(c['sh'], c['dim'], c['value']): (c['rows'], c['originals'], c['nonzero_originals']) for c in S['categories']}
    want_l = {k: (v['rows'], v['episodes'], v['nonzero_episodes']) for k, v in S['cells'].items()}
    for dct, keys in ((got_c, want_c), (want_c, got_c), (got_l, want_l), (want_l, got_l)):
        for k in keys:
            dct.setdefault(k, z3)
    return dict(categories_equal=got_c == want_c, cells_equal=got_l == want_l, category_keys=len(got_c), cell_keys=len(got_l),
                category_diff=sorted(set(got_c.items()) ^ set(want_c.items()))[:10], cell_diff=sorted(set(got_l.items()) ^ set(want_l.items()))[:10])


def _cnt(e, m_, orig, nz):
    """Add the masked rows to a (rows, distinct originals, distinct nonzero originals) counter."""
    e[0] += int(np.asarray(m_).sum()); e[1].update(orig[m_].tolist()); e[2].update(orig[m_ & nz].tolist())


# ---------------------------------------------------------------- the bounded rollout
def worlds_of(cfg, kp):
    """The rollout population: gate2A's first 128 originals per side, grouped into (market, time) worlds with both sides."""
    P = json.loads(Path(cfg['population']).read_text()); C.require(P['population']['kind'] == 'first_per_side', 'not the frozen first-per-side population')
    sel = set(map(tuple, P['population']['selected_ids'])); O = pd.read_parquet(Path(kp) / cfg.get('day', DAY) / 'orders.parquet')
    O = O[[(int(a), int(b)) in sel for a, b in zip(O.side, O.oid)]]; C.require(len(O) == len(sel), 'a selected original is missing from the orders')
    W = []
    for (m, t), g in O.groupby(['m', 't'], sort=True):
        C.require(sorted(g.side.astype(int)) == [0, 1], f'world {m} {t}: not one original per side')
        W.append(dict(m=int(m), t=float(t), o={int(r.side): r for r in g.itertuples()}))
    return W, P


def rollout(cfg=REAL, src=None, inputs=None, feat_fn=None, kp=None, every=7):
    """The bounded chronological rollout of the frozen teacher (see the module header) -> rollout.json + CSVs in out/rollout."""
    import pf_r2_runner as RN
    fz = frozen(cfg); M = load_models(cfg, fz); sup = Support(json.loads(Path(cfg['support']).read_text())); out = cfg['out'] / 'rollout'
    C.require(not out.exists(), 'the rollout output exists'); out.mkdir()
    P0 = json.loads(Path(cfg['population']).read_text()); kp = Path(kp or P0['inputs']['kp_dir']); worlds, _ = worlds_of(cfg, kp)
    day = cfg.get('day', DAY); src = src or KP.Tape(); mk = sorted({w['m'] for w in worlds}); fair, depth, _, _ = (inputs or X.real_inputs)(day, mk)
    feat_fn = feat_fn or X.real_features; data = src.load(day, mk, src.offset(day)); pairs = {int(W_[0]['m']): W_ for W_ in src.pairs(data)}
    probe, cst = causal_probe(every); T = Teacher(M, sup, causal=probe); res, per = [], []
    nt = int(np.ceil(END / STEP))
    for w in worlds:
        W_ = pairs.get(w['m']); o0 = w['o'][0]
        if W_ is None:
            res.append(dict(m=w['m'], t=w['t'], status='missing_source:pair')); continue
        path, pst = QP.build_path(W_, w['t'], w['t'] + END)
        if path is None:
            res.append(dict(m=w['m'], t=w['t'], status=f'no_path:{pst}')); continue
        fv = np.array([float(getattr(o0, f'f{j:02d}')) for j in range(KP.NSTEP + 1)])
        chosen = {s: sorted({int(min(np.searchsorted(np.asarray(W_[s]['tg'], float), w['t'] + k * STEP), len(W_[s]['tg']) - 1)) for k in range(nt)}) for s in (0, 1)}
        F = feat_fn(W_, fair.get(w['m']), depth, chosen)
        B = BE.Branch(path, ident=R2.episode_ident(day, o0)); n0 = len(T.log)
        T.bind(B, dict(path=path, W=W_, fv=fv, o=w['o'], day=day, fair_m=fair.get(w['m']), depth=depth, feat_fn=feat_fn, market={s: F.get(s, {}) for s in (0, 1)}))
        for k in range(nt):
            BE.step(B, k, T)
        L = BE.finish(B); dec = T.log[n0:]
        res.append(dict(m=w['m'], t=w['t'], status=L['status'], **invariants(B, L, dec)))
        per.extend(per_side(w, B, L, dec))
        C.require(RN.rss()[1] <= GUARD_MIB, f'the high-water passed {GUARD_MIB} MiB')
    D = pd.DataFrame([dict({k: v for k, v in d_.items() if k != 'row'}) for d_ in T.log]); D.to_csv(out / 'decisions.csv', index=False)
    pd.DataFrame(per).to_csv(out / 'per_side.csv', index=False); pd.DataFrame(res).to_csv(out / 'worlds.csv', index=False)
    summ = summarize(D, pd.DataFrame(per), pd.DataFrame(res), cst)
    (out / 'rollout.json').write_text(json.dumps(dict(asof=C.stamp(), freeze_sha256=C.sha(cfg['out'] / 'freeze.json'), summary=summ, causal=cst,
                                                      hwm_mib=RN.rss()[1]), indent=1, default=str))
    return summ


def invariants(B, L, dec):
    """Ledger checks of one finished world: cash / position / value from the fills, fills within every generation's size, every policy call
    logged once per engine decision that reached the policy."""
    if L['status'] != 'ok':
        return dict(ledger_ok=None)
    sg = lambda s: 1. if s == 0 else -1.
    cash = -sum(sg(f[2]) * f[3] * f[4] for f in L['fills']); pos = sum(sg(f[2]) * f[3] for f in L['fills']); val = cash + pos * B.mark
    per_gen = {}
    for f in L['fills']:
        per_gen[f[1]] = per_gen.get(f[1], 0.) + f[3]
    within = all(q_ <= B.E.o[g_]['size'] + 1e-9 for g_, q_ in per_gen.items())
    asked = [d_ for d_ in L['decisions'] if d_[2] in ('absent', 'live') and d_[3] is not None]
    fb = len(L['fallbacks']); logged = len(dec)
    ok = abs(cash - L['cash']) <= 1e-6 and abs(pos - L['position']) <= 1e-9 and abs(val - L['value']) <= 1e-6 and within and logged == len(asked) - fb
    return dict(ledger_ok=bool(ok), cash=L['cash'], position=L['position'], value=L['value'], fills=len(L['fills']), generations=L['generations'],
                logged=logged, engine_asked=len(asked), engine_fallbacks=fb)


def per_side(w, B, L, dec):
    """One row per original (side) of a world: entries, delayed first entry, later generations, PULLs, fills and statuses."""
    out = []
    for s in (0, 1):
        gens = [o for o in B.E.o if o['side'] == s]; first = gens[0] if gens else None; ds = [d_ for d_ in dec if d_['side'] == s]
        out.append(dict(m=w['m'], t=w['t'], side=s, oid=int(w['o'][s].oid), orig_status=str(w['o'][s].status), generations=len(gens),
                        first_submit_ms=float(first['submit']) if first else np.nan, entry=('none' if first is None else 'offset_zero' if first['submit'] == 0 else 'delayed'),
                        later_generations=max(len(gens) - 1, 0), refused=sum(int(bool(o['refused'])) for o in gens),
                        pulls=sum(1 for d_ in ds if d_['action'] == 'PULL'), posts=sum(1 for d_ in ds if d_['action'] == 'POST'),
                        fills=sum(1 for f in L.get('fills', []) if f[2] == s), filled_shares=sum(f[3] for f in L.get('fills', []) if f[2] == s),
                        **{f'n_{st}': sum(1 for d_ in ds if d_['status'] == st) for st in STATUSES}))
    return out


def summarize(D, Pq, R, cst):
    """Computed readings of the rollout."""
    by = lambda col: {str(k): int(v) for k, v in D.groupby(['side', col]).size().items()} if len(D) else {}
    if not len(D):                                                                          # the full structure, zero decisions
        D = pd.DataFrame({c: [] for c in ('side', 'status', 'action')})
    model = D[D.status == 'model']; nfb = int((D.status != 'model').sum()); Pq = Pq if len(Pq) else pd.DataFrame({c: [] for c in ('side', 'entry', 'later_generations', 'pulls', 'fills')})
    return dict(worlds=int(len(R)), worlds_ok=int((R.status == 'ok').sum()), ledger_ok=bool(R.ledger_ok.dropna().astype(bool).all()) if 'ledger_ok' in R else None,
                decisions=int(len(D)), by_status=by('status'), by_action=by('action'), model_decisions=int(len(model)), non_model=nfb,
                non_model_frac=nfb / len(D) if len(D) else None, entries={str(k): int(v) for k, v in Pq.groupby(['side', 'entry']).size().items()},
                later_generations=int(Pq.later_generations.sum()), originals_with_later_generation=int((Pq.later_generations > 0).sum()),
                pulls=int(Pq.pulls.sum()), fills=int(Pq.fills.sum()), delayed_entries=int((Pq.entry == 'delayed').sum()),
                causal=dict(checked=cst['checked'], mismatch=cst['mismatch'], delivered_changed=cst['delivered_changed'], by_state=cst['by_state'],
                            parity_checked=cst['parity_checked'], parity_mismatch=cst['parity_mismatch']))


def selftest():
    import pf_r2_teacher_selftest as ST
    return ST.run(sys.modules[__name__])


if __name__ == '__main__':
    a = sys.argv[1:]
    if a == ['selftest']:
        selftest()
    elif a == ['extract']:
        print(json.dumps(extract()['groups']))
    elif len(a) == 2 and a[0] == 'fit':
        print(json.dumps(fit_group(int(a[1]))))
    elif a == ['freeze']:
        fz = freeze(); print(json.dumps(dict(models=fz['models'], reload_equal=fz['reload_equal'])))
    elif a == ['diag']:
        print(json.dumps(diag()[0], default=str))
    elif a == ['rollout']:
        print(json.dumps(rollout(), default=str))
    else:
        sys.exit('usage: pf_r2_teacher.py selftest | extract | fit <group 0-3> | freeze | diag | rollout')
