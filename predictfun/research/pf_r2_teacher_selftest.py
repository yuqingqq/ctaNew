"""SELFTEST of pf_r2_teacher (synthetic only; no market data). run(T) checks the module object T (the file, or a mutated copy) and exits with
'R2 TEACHER SELFTEST FAILED' on any failure. Each block is a computed predicate with a known answer:
 (1) the 107 inputs: the census order, unique, no identity / label field; extraction stores the census's float32 values in that order;
 (2) the streamed construction: trees identical to an in-memory fit at <= 200,000 rows; two streamed fits identical;
 (3) whole-original weighting and the conditional subsets, on a controlled census with constant inputs, where a booster cannot split and each
     head predicts its weighted training mean exactly. p = 0.75 (unweighted 0.95), q | fill = 23.333 (unweighted 28.947, all rows 17.5),
     qm | fill = 0.3333, severe = 1.5;
 (4) the normalization constant: an informative input with census-scale weights; after 10 rounds the fitted spread is > 2.5 (normalized,
     about 3.9), against about 1.4 at the raw scale;
 (5) the census identity: a changed plan, record, parquet or sampled-ok count is refused; extraction never rewrites;
 (6) the support adapter: every status from crafted rows; the inventory tolerance; NaN outside the range;
 (7) the decision rule on stub forecasts: signs, ties, incoherence, unsupported states and the structural statuses;
 (8) a synthetic world end to end: extract -> fit (5 rounds) -> freeze -> diag -> rollout. The adapter's categories / cells equal the
     synthetic census's support tables; the ledgers hold; every policy call is logged; the causal probe checks decisions with zero
     mismatches and its delivered control fires; a frozen model, source or support file changed is refused;
 (9) the causal probe's falsifier: a teacher reading the engine-truth remainder is caught.
PF_TEACHER_SYN (optional): a directory holding a built synthetic world (from build_world), reused across mutation runs."""
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_branch_engine as BE                             # noqa: E402
import pf_r2_shadow as R2                                 # noqa: E402

C, pd, I, KP, X, QP = R2.C, R2.pd, R2.I, R2.KP, R2.X, R2.QP
FAIL = 'R2 TEACHER SELFTEST FAILED'


def write_census(root, groups, extra_unsampled=3):
    """A controlled one-part census: groups = {(side, head): DataFrame of FEATS + LAB rows} -> cfg for the teacher (plan, part, record, manifest)."""
    T_ = sys.modules.get('pf_r2_teacher_under_test')
    root.mkdir(parents=True); (root / 'parts').mkdir(); (root / 'plan.json').write_text(json.dumps(dict(test=str(root))))
    frames = []
    for (s, h), G in groups.items():
        G = G.copy(); G['side'] = s; G['head'] = h; G['status'] = 'ok'; G['sampled'] = True; frames.append(G)
        U = G.head(extra_unsampled).copy(); U['status'] = 'unsampled'; U['sampled'] = False; frames.append(U)
        N = G.head(2).copy(); N['status'] = 'near_end'; N['sampled'] = True; frames.append(N)                   # a sampled status row
        K = G.head(2).copy(); K['status'] = 'ok'; K['sampled'] = False; frames.append(K)                        # an ok row outside the sample
    D = pd.concat(frames, ignore_index=True); feats = T_.FEATS
    for c in feats:
        D[c] = D[c].astype(np.float32)
    f = root / 'parts' / 'part_0000.parquet'; D.to_parquet(f)
    ok = int(((D.status == 'ok') & D.sampled).sum())
    rec = dict(plan_sha256=C.sha(root / 'plan.json'), part=0, file=f.name, sha256=C.sha(f), sampled_ok=ok)
    rp = root / 'parts' / 'part_0000.json'; rp.write_text(json.dumps(rec)); man = root.parent / f'{root.name}_records.sha256'
    man.write_text(f'{C.sha(rp)}  {root.name}/parts/part_0000.json\n')
    return dict(census=root, support=root / 'support.json', plan_sha=rec['plan_sha256'], n_parts=1, records=man, out=root.parent / f'{root.name}_out',
                population=root / 'plan.json'), D


def frame(T, n, **cols):
    """n rows: every input 0 (constant), the given input / label columns, the defaults for the rest."""
    base = {c: np.zeros(n, np.float32) for c in T.FEATS}
    base.update(q=np.zeros(n), qm=np.zeros(n), severe=np.zeros(n), value=np.zeros(n), weight=np.ones(n), price=np.full(n, 50.), decision_ms=np.zeros(n),
                offset_ms=np.zeros(n), generation=np.ones(n), original_episode_id=np.zeros(n), instrument_id=np.ones(n))
    for k, v in cols.items():
        base[k] = np.asarray(v, np.float32 if k in T.FEATS else float) if np.ndim(v) else np.full(n, v, np.float32 if k in T.FEATS else float)
    return pd.DataFrame(base)


def controlled(T):
    """(3) / (4): group (0, post) - original 1: 900 filled rows (q 30, qm 1, severe 0), weight 1/900; original 2: 100 rows, 50 filled (q 10,
    qm -1) and 50 unfilled, severe 3, weight 1/100; constant inputs. Group (0, retain) - 10 originals x 100 rows, weight 1/100, input 0 in
    {0, 1} by halves, severe = 10 x input. The other groups: 40 rows with one filled row."""
    a = frame(T, 900, q=30., qm=1., severe=0., weight=1 / 900., original_episode_id=1.)
    b = frame(T, 100, q=np.r_[np.full(50, 10.), np.zeros(50)], qm=np.r_[np.full(50, -1.), np.zeros(50)], severe=3., weight=1 / 100., original_episode_id=2.)
    x0 = np.tile(np.r_[np.zeros(50), np.ones(50)], 10)
    r = frame(T, 1000, severe=10. * x0, q=np.where(np.arange(1000) % 3 == 0, 5., 0.), weight=1 / 100., original_episode_id=np.repeat(np.arange(10), 100) + 10.,
              decision_ms=1000.); r[T.FEATS[0]] = x0.astype(np.float32)
    small = frame(T, 40, q=np.r_[30., np.zeros(39)], weight=1 / 40., original_episode_id=50.)
    return {(0, 'post'): pd.concat([a, b], ignore_index=True), (0, 'retain'): r, (1, 'post'): small, (1, 'retain'): small.assign(original_episode_id=51.)}


def must(fn, what):
    """A call on VALID synthetic input: the module's own refusal (ValueError from a guard) is a check failure, reported at once."""
    try:
        return fn()
    except ValueError as e:
        raise SystemExit(f'{FAIL}: {what} refused valid input ({str(e)[:200]})')


def stub(vals):
    class S:
        def __init__(self, v):
            self.v = np.asarray(v, float)

        def predict(self, X, num_threads=1):
            return np.resize(self.v, len(X))
    return S(vals)


def world_probe(T, cls, models, wcfg, wj, src):
    """A teacher class with the given models and an admit-all adapter over the synthetic worlds, every decision causally probed -> the probe stats."""
    probe, st = T.causal_probe(1); tch = cls(models, _Open(None), causal=probe)
    wl, _ = T.worlds_of(wcfg, wj['kp']); mk = sorted({w_['m'] for w_ in wl}); fair, depth, _, _ = X.syn_inputs(wj['day'], mk)
    data = src.load(wj['day'], mk, src.offset(wj['day'])); pairs = {int(P_[0]['m']): P_ for P_ in src.pairs(data)}; ff = X.syn_features()
    for w_ in wl:
        P_ = pairs[w_['m']]; path, _ = QP.build_path(P_, w_['t'], w_['t'] + T.END)
        if path is None:
            continue
        fv = np.array([float(getattr(w_['o'][0], f'f{j:02d}')) for j in range(KP.NSTEP + 1)])
        ch = {s: sorted({int(min(np.searchsorted(np.asarray(P_[s]['tg'], float), w_['t'] + k * T.STEP), len(P_[s]['tg']) - 1)) for k in range(100)}) for s in (0, 1)}
        F = ff(P_, fair.get(w_['m']), depth, ch); B = BE.Branch(path, ident=R2.episode_ident(wj['day'], w_['o'][0]))
        tch.bind(B, dict(path=path, W=P_, fv=fv, o=w_['o'], day=wj['day'], fair_m=fair.get(w_['m']), depth=depth, feat_fn=ff, market={s: F.get(s, {}) for s in (0, 1)}))
        n0 = len(tch.log)
        for k in range(100):
            BE.step(B, k, tch)
        last = (B, BE.finish(B), tch.log[n0:])
    st['last'] = last
    return st


def build_world(root):
    """The R2 synthetic world (KP.Syn tapes, the keep / pull run, a 4-per-side census through the real runner, no causality) -> paths."""
    import pf_queue_place_ledger as QL
    src = KP.Syn(); days = list(KP.DAYS); d0 = days[0]; root = Path(root); root.mkdir(parents=True, exist_ok=True)
    with QL._v1_on(src, root):
        for d in days:
            QP.day_run(d)
    with contextlib.redirect_stdout(io.StringIO()):
        for d in days:
            KP.day_run(d, src=src, v1=root / 'v1', out=root / 'kp')
        cd_, S, R = R2.census(d0, per_side=4, out=root / 'r2', src=src, kp=root / 'kp', inputs=X.syn_inputs, feat_fn=X.syn_features(), causal=False)
    (root / 'world.json').write_text(json.dumps(dict(census=str(cd_), kp=str(root / 'kp'), day=d0)))
    return root


def run(T):  # noqa: C901
    sys.modules['pf_r2_teacher_under_test'] = T
    bad = []
    import lightgbm as lgb
    # ---- (1) the inputs
    if not (T.FEATS == list(R2.INPUTS) + list(R2.CTX) and len(T.FEATS) == 107 and len(set(T.FEATS)) == 107 and not set(T.FEATS) & set(T.IDENT)):
        bad.append('inputs: not the census 107 in order, or an identity / label field among them')
    with tempfile.TemporaryDirectory(dir=str(I.K.IO)) as tmp:
        tmp = Path(tmp)
        # ---- (2) the streamed construction
        rng = np.random.default_rng(3); Xr = rng.normal(size=(3000, 107)).astype(np.float32); Xr[rng.random(Xr.shape) < .05] = np.nan
        y = (Xr[:, 0] + rng.normal(size=3000) > 0).astype(float); w = rng.uniform(.5, 2., 3000); Xr.tofile(tmp / 'x.f32'); rows = T.Rows(tmp / 'x.f32', 3000)
        b1 = T.train_one(rows, y, w, 'binary', rounds=20); b2 = T.train_one(rows, y, w, 'binary', rounds=20)
        bm = lgb.train({**T.PARAMS, 'objective': 'binary'}, lgb.Dataset(Xr, label=y, weight=T.norm_weights(w), feature_name=T.FEATS), num_boost_round=20)
        tr = lambda b: b.model_to_string()[b.model_to_string().index('Tree=0'):b.model_to_string().index('end of trees')]
        if not (tr(b1) == tr(bm) and b1.model_to_string() == b2.model_to_string()):
            bad.append('construction: the streamed fit differs from the in-memory fit at <= 200,000 rows, or is not deterministic')
        if not np.array_equal(rows[5], Xr[5].astype(np.float64), equal_nan=True) or not np.array_equal(rows[10:20], Xr[10:20], equal_nan=True):
            bad.append('construction: the file-backed rows differ from the matrix')
        rows.close()
        # ---- (3) / (4) / (5) the controlled census
        cfg, D = write_census(tmp / 'cen', controlled(T))
        man = must(lambda: T.extract(cfg), 'extract (controlled census)')
        rd = T.Rows(cfg['out'] / 'train' / '0_post' / 'X_all.f32', man['groups']['0_post']['rows']); G0 = D[(D.status == 'ok') & D.sampled.astype(bool) & (D.side == 0) & (D['head'] == 'post')]
        if not np.array_equal(rd[0:len(G0)], G0[T.FEATS].to_numpy(np.float32), equal_nan=True):
            bad.append('extract: the stored inputs are not the census values in the declared order')
        rd.close()
        if not (man['groups']['0_post'] == dict(rows=1000, filled=950) and man['groups']['0_retain']['rows'] == 1000 and sum(v['rows'] for v in man['groups'].values()) == 2080):
            bad.append(f"extract: group counts {man['groups']} (the unsampled rows must be dropped, every sampled ok row kept)")
        f0, f1 = must(lambda: T.fit_group(0, cfg, rounds=5), 'fit group 0'), must(lambda: T.fit_group(1, cfg, rounds=10), 'fit group 1')
        M = {hd: lgb.Booster(model_file=str(cfg['out'] / 'models' / f'0_post_{hd}.txt')) for hd in T.HEADS}; z = np.zeros((3, 107), np.float32)
        got = {hd: float(M[hd].predict(z, num_threads=1)[0]) for hd in T.HEADS}; want = dict(p=.75, q=70 / 3., qm=1 / 3., severe=1.5)
        if not all(abs(got[k] - want[k]) <= 1e-6 for k in want):
            bad.append(f'weights: the heads predict {got}, not the weighted / conditional means {want} (the census weights, subsetted to the filled rows)')
        if not all(f_['heads'][hd]['reload_equal'] for f_ in (f0, f1) for hd in T.HEADS):
            bad.append('reload: a saved model does not predict as its fitted booster')
        sv = lgb.Booster(model_file=str(cfg['out'] / 'models' / '0_retain_severe.txt')); z1 = np.zeros((2, 107), np.float32); z1[1, 0] = 1.
        spread = float(np.diff(sv.predict(z1, num_threads=1))[0])
        if not spread > 2.5:
            bad.append(f'weights: the 10-round spread is {spread:.3f}, not > 2.5 (the per-set constant n / sum w is not applied)')
        try:
            T.extract(cfg); bad.append('extract: the training files were rewritten')
        except ValueError:
            pass
        for what, change in (('plan', lambda c: dict(c, plan_sha='0' * 64)), ('record', None), ('parquet', None), ('sampled ok', None), ('group', None)):
            c2, D2 = write_census(tmp / f'cen_{what.replace(" ", "_")}', controlled(T))
            if what == 'record':
                rp = c2['census'] / 'parts' / 'part_0000.json'; rp.write_text(rp.read_text().replace('"part": 0', '"part": 0 '))
            elif what == 'parquet':                                                        # one value changed, every count kept: only the hash can refuse
                fp = c2['census'] / 'parts' / 'part_0000.parquet'; D2.loc[0, T.FEATS[0]] = np.float32(D2.loc[0, T.FEATS[0]] + 1.); D2.to_parquet(fp)
            elif what == 'group':                                                          # a sampled ok row outside every side / head group, consistently recorded
                fp = c2['census'] / 'parts' / 'part_0000.parquet'; D2.loc[0, 'head'] = 'other'; D2.to_parquet(fp)
                rp = c2['census'] / 'parts' / 'part_0000.json'; r_ = json.loads(rp.read_text()); r_['sha256'] = C.sha(fp); rp.write_text(json.dumps(r_))
                (c2['records']).write_text(f"{C.sha(rp)}  {c2['census'].name}/parts/part_0000.json\n")
            elif what == 'sampled ok':
                rp = c2['census'] / 'parts' / 'part_0000.json'; r_ = json.loads(rp.read_text()); r_['sampled_ok'] += 1; rp.write_text(json.dumps(r_))
                (c2['records']).write_text(f"{C.sha(rp)}  {c2['census'].name}/parts/part_0000.json\n")
            try:
                T.extract(change(c2) if change else c2); bad.append(f'identity: a changed {what} was not refused')
            except ValueError:
                pass
        # ---- (6) the support adapter
        S_ = dict(cells={'bid post (0.0, 100.0]': dict(support='INSUFFICIENT_TRAINING_SUPPORT'), 'bid post (500.0, 1000.0]': dict(support='SUPPORTED')},
                  presence={sh: {d: {x: 5 for x in R2.DECLARED[d]} for d in R2.DIMS} for sh in R2.SIDE_HEADS},
                  ranges={sh: {f: [0., 30.] for f in R2.NUMERIC} for sh in R2.SIDE_HEADS})
        S_['presence']['bid post']['c_inv']['long'] = 0; S_['ranges']['bid post']['t_to_end'] = [100., 5000.]; S_['ranges']['bid post']['ctx_inventory_shares'] = [-30., 0.]
        sup = T.Support(S_)
        base = {c: np.float32(0.) for c in T.FEATS}; base.update(fp_remaining=np.float32(30.), fp_queue=np.float32(40.), t_to_end=np.float32(4200.),
                                                                   ctx_other_state_absent=np.float32(1.), price=50., generation=1., decision_ms=4200.)
        cases = [('fine', dict(decision_ms=4200., t_to_end=np.float32(800.)), None),
                 ('horizon', dict(decision_ms=5000.), 'unsupported_horizon'), ('cell', dict(decision_ms=4950.), 'insufficient_training_support'),
                 ('cell missing', dict(decision_ms=0.), 'insufficient_training_support'),
                 ('long bid', dict(decision_ms=4200., t_to_end=np.float32(800.), ctx_inventory_shares=np.float32(5.)), 'unsupported_state_category'),
                 ('n/a', dict(decision_ms=4200., t_to_end=np.float32(800.), ctx_other_state_absent=np.float32(0.)), 'unsupported_state_category'),
                 ('range', dict(decision_ms=4200., t_to_end=np.float32(800.), ctx_generation_age_ms=np.float32(31.)), 'outside_training_range'),
                 ('NaN', dict(decision_ms=4200., t_to_end=np.float32(800.), ctx_bid_reserved_shares=np.float32(np.nan)), 'outside_training_range'),
                 ('round-off', dict(decision_ms=4200., t_to_end=np.float32(800.), ctx_inventory_shares=np.float32(1e-12)), None),
                 ('t_to_end below', dict(decision_ms=4200., t_to_end=np.float32(99.)), 'outside_training_range')]
        for nm, ch, want_ in cases:
            got_ = sup.check(0, 'post', dict(base, **ch))
            if got_ != want_:
                bad.append(f'support: case {nm} gives {got_}, not {want_}')
        cat = T.categories(pd.DataFrame([dict(base, ctx_inventory_shares=np.float32(1e-12)), dict(base, ctx_inventory_shares=np.float32(2e-9)),
                                         dict(base, ctx_inventory_shares=np.float32(-2e-9))]))
        if list(cat['c_inv']) != ['flat', 'long', 'short']:
            bad.append(f"support: the inventory-sign tolerance is not 1e-9 ({list(cat['c_inv'])})")
        # ---- (7) the decision rule on stub forecasts
        row = dict(base, decision_ms=4200., t_to_end=np.float32(800.), pf_mid=np.float32(50.), price=49.)          # bid: edge +1
        Bst = types.SimpleNamespace(end=5000.)
        def dec(head, P_, q_, qm_, room=5., t=4200., supp=False):
            t_ = T.Teacher({(0, head): dict(p=stub([P_]), q=stub([q_]), qm=stub([qm_]), severe=stub([0.]))}, sup if supp else _Open(sup))
            t_.inputs = lambda v, B, s, g: dict(row, decision_ms=float(g))
            return t_.decide(dict(side=0, t=t, state='absent' if head == 'post' else 'live', gid=None if head == 'post' else 0, room=room), Bst)
        rule = [('post V>0', ('post', .5, 10., 0.), 'model', 'POST'), ('post V<0', ('post', .5, 10., -20.), 'model', 'WAIT'),
                ('post tie', ('post', 0., 10., 0.), 'model', 'WAIT'), ('retain V<0', ('retain', .5, 10., -20.), 'model', 'PULL'),
                ('retain V>0', ('retain', .5, 10., 0.), 'model', 'KEEP'), ('retain tie', ('retain', 0., 10., 0.), 'model', 'KEEP'),
                ('incoherent post', ('post', 1.5, 10., 0.), 'incoherent_forecast', 'WAIT'), ('incoherent retain', ('retain', -.2, 10., 0.), 'incoherent_forecast', 'KEEP')]
        for nm, a_, st_, ac_ in rule:
            d_ = dec(*a_)
            if (d_['status'], d_['action']) != (st_, ac_):
                bad.append(f"rule: {nm} gives {d_['status']} {d_['action']}, not {st_} {ac_}")
        for nm, kw, st_ in (('cap', dict(room=-1.), 'cap_blocked'), ('near END', dict(t=4950.), 'near_end')):
            d_ = dec('post', .5, 10., 0., **kw)
            if (d_['status'], d_['action']) != (st_, 'WAIT'):
                bad.append(f"rule: {nm} gives {d_['status']} {d_['action']}, not {st_} WAIT")
        t_ = T.Teacher({(0, 'post'): dict(p=stub([.5]), q=stub([10.]), qm=stub([0.]), severe=stub([0.]))}, sup)
        t_.inputs = lambda v, B, s, g: dict(row, ctx_inventory_shares=np.float32(5.))
        d_ = t_.decide(dict(side=0, t=4200., state='absent', gid=None, room=5.), Bst)
        if (d_['status'], d_['action']) != ('unsupported_state_category', 'WAIT'):
            bad.append(f"rule: an unsupported state with V > 0 gives {d_['status']} {d_['action']}, not the WAIT fallback")
        t_.inputs = lambda v, B, s, g: None
        d_ = t_.decide(dict(side=0, t=4200., state='live', gid=0, room=5.), Bst)
        if (d_['status'], d_['action']) != ('feature_unavailable', 'KEEP'):
            bad.append(f"rule: missing inputs give {d_['status']} {d_['action']}, not feature_unavailable KEEP")
        # ---- (8) / (9) the synthetic world end to end
        wroot = Path(os.environ['PF_TEACHER_SYN']) if os.environ.get('PF_TEACHER_SYN') else build_world(tmp / 'world')
        wj = json.loads((wroot / 'world.json').read_text()); cd_ = Path(wj['census'])
        recs = sorted((cd_ / 'parts').glob('part_*.json')); mf = tmp / 'world_records.sha256'
        mf.write_text(''.join(f'{C.sha(p_)}  {cd_.name}/parts/{p_.name}\n' for p_ in recs))
        sup_p = Path(cd_.parent) / f"support_{wj['day']}.json"
        wcfg = dict(census=cd_, support=sup_p, plan_sha=C.sha(cd_ / 'plan.json'), n_parts=len(recs), records=mf, out=tmp / 'wteach', population=cd_ / 'plan.json',
                    day=wj['day'])
        with contextlib.redirect_stdout(io.StringIO()):
            wm = must(lambda: T.extract(wcfg), 'extract (synthetic world)')
            if not all(v['filled'] > 0 for v in wm['groups'].values()):
                bad.append(f"world: a synthetic group without a filled row ({wm['groups']}) - the end-to-end check cannot run"); raise SystemExit(f'{FAIL}: {bad}')
            for k in range(4):
                must(lambda: T.fit_group(k, wcfg, rounds=5), f'fit world group {k}')
            must(lambda: T.freeze(wcfg), 'freeze'); rec, _ = must(lambda: T.diag(wcfg), 'diag')
            if not (rec['categories_equal'] and rec['cells_equal']):
                bad.append(f"adapter: categories / cells differ from the census support ({rec['category_diff'][:2]}, {rec['cell_diff'][:2]})")
            src = KP.Syn(); summ = must(lambda: T.rollout(wcfg, src=src, inputs=X.syn_inputs, feat_fn=X.syn_features(), kp=wj['kp'], every=1), 'rollout')
        W_ = pd.read_csv(wcfg['out'] / 'rollout' / 'worlds.csv')
        if not (summ.get('decisions', 0) > 0 and summ['worlds_ok'] == len(W_) > 0 and bool(W_.ledger_ok.astype(bool).all())):
            bad.append(f"rollout: ledgers / logging fail or no decision ({summ.get('worlds_ok')}, {len(W_)}, {W_.get('ledger_ok')})")
        cz = summ.get('causal', {})
        if not (cz.get('checked', 0) > 0 and cz.get('mismatch', 1) == 0 and cz.get('delivered_changed', 0) > 0):
            bad.append(f"rollout: causal probe {cz} (checks must run, none may mismatch, the delivered control must fire)")
        mp = wcfg['out'] / 'models' / '0_post_p.txt'; keep = mp.read_text(); mp.write_text(keep + '\n')
        try:
            T.load_models(wcfg, T.frozen(wcfg)); bad.append('freeze: a changed model file was not refused')
        except ValueError:
            pass
        mp.write_text(keep)
        sp_keep = sup_p.read_text(); sup_p.write_text(sp_keep + ' ')
        try:
            T.frozen(wcfg); bad.append('freeze: a changed support JSON was not refused')
        except ValueError:
            pass
        sup_p.write_text(sp_keep)
        fz = json.loads((wcfg['out'] / 'freeze.json').read_text()); fz['sources']['pf_r2_shadow.py'] = '0' * 64; (wcfg['out'] / 'freeze_bad.json').write_text(json.dumps(fz))
        shutil.copy(wcfg['out'] / 'freeze.json', wcfg['out'] / 'freeze_ok.json'); shutil.copy(wcfg['out'] / 'freeze_bad.json', wcfg['out'] / 'freeze.json')
        try:
            T.frozen(wcfg); bad.append('freeze: a changed source was not refused')
        except ValueError:
            pass
        shutil.copy(wcfg['out'] / 'freeze_ok.json', wcfg['out'] / 'freeze.json')

        stubs = {g: dict(p=stub([.5]), q=stub([10.]), qm=stub([0.]), severe=stub([0.])) for g in T.GROUPS}      # V > 0 at the JOIN: orders go live

        class LeakRem(T.Teacher):                                                           # (9a) reads the engine-truth remainder (live)
            def inputs(self, v, B, s, g):
                r_ = super().inputs(v, B, s, g)
                if r_ is not None and B.open[s] is not None:
                    r_['fp_remaining'] = np.float32(float(r_['fp_remaining']) + 1e-3 * float(B.E.o[B.open[s]]['rem']))
                return r_

        class LeakFrame(T.Teacher):                                                         # (9b) reads the next unreceived book frame (any state)
            def inputs(self, v, B, s, g):
                r_ = super().inputs(v, B, s, g)
                if r_ is not None and B.E.known + 1 < len(B.E.f):
                    r_['fp_queue'] = np.float32(float(r_['fp_queue']) + 1e-3 * float(B.E.f[B.E.known + 1, 7]))
                return r_
        honest = world_probe(T, T.Teacher, stubs, wcfg, wj, src); lr_ = world_probe(T, LeakRem, stubs, wcfg, wj, src); lf_ = world_probe(T, LeakFrame, stubs, wcfg, wj, src)
        if not (honest['mismatch'] == 0 and honest['by_state'].get('live', {}).get('checked', 0) > 0 and honest['by_state'].get('absent', {}).get('checked', 0) > 0
                and honest['parity_checked'] > 0 and honest['parity_mismatch'] == 0 and cz.get('parity_mismatch', 1) == 0):
            bad.append(f"causality: the honest teacher on live and absent states {honest['by_state']}, parity {honest['parity_mismatch']} / {honest['parity_checked']}, "
                       f"rollout parity {cz.get('parity_mismatch')} (both states checked, no mismatch, inputs = the independent rebuild)")
        if not (lr_['by_state'].get('live', {}).get('mismatch', 0) > 0 and lf_['mismatch'] > 0):
            bad.append(f"causality: a teacher reading the engine-truth remainder ({lr_['by_state']}) or an unreceived frame ({lf_['mismatch']}) is not caught")
        cst = dict(lr_, frame=lf_['mismatch'])
        Bl, Ll, dl = honest['last']; inv0 = T.invariants(Bl, Ll, dl)                       # the ledger instrument and its known-bad ledgers
        f0_ = Ll['fills'][0] if Ll['fills'] else None
        bads = dict(cash=dict(Ll, cash=Ll['cash'] + 1.), logged=None,
                    size=(dict(Ll, fills=Ll['fills'] + [(f0_[0], f0_[1], f0_[2], 31., f0_[4], f0_[5])], cash=Ll['cash'] - (1. if f0_[2] == 0 else -1.) * 31. * f0_[4],
                               position=Ll['position'] + (1. if f0_[2] == 0 else -1.) * 31., value=Ll['value'] + (1. if f0_[2] == 0 else -1.) * 31. * (Bl.mark - f0_[4]))
                          if f0_ else None))
        got_b = {k: (T.invariants(Bl, v, dl)['ledger_ok'] if v is not None else T.invariants(Bl, Ll, dl[:-1])['ledger_ok']) for k, v in bads.items() if k != 'size' or v is not None}
        if not (inv0['ledger_ok'] is True and f0_ is not None and all(x is False for x in got_b.values())):
            bad.append(f"ledger: the instrument passes a clean ledger {inv0['ledger_ok']} and must fail cash +1 / a 31-share fill / a missing log entry {got_b} (a fill must exist: {f0_ is not None})")
        Sw = json.loads(sup_p.read_text()); okc = T.reconcile({(c['sh'], c['dim'], c['value']): (c['rows'], c['originals'], c['nonzero_originals']) for c in Sw['categories']},
                                                              {k: (v['rows'], v['episodes'], v['nonzero_episodes']) for k, v in Sw['cells'].items()}, Sw)
        c0 = Sw['categories'][0]; offc = T.reconcile({(c0['sh'], c0['dim'], c0['value']): (c0['rows'] + 1, c0['originals'], c0['nonzero_originals'])},
                                                    {k: (v['rows'], v['episodes'], v['nonzero_episodes']) for k, v in Sw['cells'].items()}, Sw)
        k0 = next(iter(Sw['cells'])); offl = T.reconcile({(c['sh'], c['dim'], c['value']): (c['rows'], c['originals'], c['nonzero_originals']) for c in Sw['categories']},
                                                         {k0: (Sw['cells'][k0]['rows'] + 1, 0, 0)}, Sw)
        if not (okc['categories_equal'] and okc['cells_equal'] and not offc['categories_equal'] and not offl['cells_equal']):
            bad.append('adapter: the reconciliation does not separate equal tables from a count off by one')
    if bad:
        raise SystemExit(f'{FAIL}: ' + ' | '.join(bad))
    print(f"r2 teacher selftests passed (synthetic only): inputs, streamed = in-memory at <= 200k, weights / subsets / constant, identity refusals, "
          f"{len(cases)} support cases, {len(rule) + 2} rule cases, world: {summ.get('decisions')} decisions, {cz.get('checked')} causal checks, "
          f"adapter = census support, honest live / absent checks {honest['by_state']}, leaks caught (remainder {cst['mismatch']}, frame {cst['frame']})")


class _Open:
    """A support adapter that admits everything (for the rule cases)."""

    def __init__(self, sup):
        self.sup = sup

    def check(self, s, head, row):
        return None


if __name__ == '__main__':
    if sys.argv[1:2] == ['build_world']:
        print(build_world(sys.argv[2]))
    else:
        import pf_r2_teacher as T_
        run(T_)
