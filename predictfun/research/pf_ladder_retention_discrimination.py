"""LONG-VALUE-RETENTION-DISCRIMINATION-v1 (P-2026-005; 2026-11-22, session aea5bfb9). Review 838f4be2's ONE next priority; USER 2026-11-22 approved
the new label-and-discrimination scope: <= 1.25 additional core-hours including a 0.05 h review reserve, ladder line ceiling 4.00. Question: can
information available at an order's first review identify orders whose FULL joint continuation value is better if retained than cancelled and
re-entered? Primary target: the accepted paired full-book Y to the common supported END (pf_ladder_long_continuation, frozen 7407bdf1); first-fill
loss, 1 s markouts and FIFO components never replace it. Not a strategy, a composed policy replay, a validation or a profitability claim.
POPULATION (census frozen before any fork): CURRENT_ASYNC_FAST10 on the 19 complete consumed markets Oct 24-27 (ASYNC-EXECUTION-SENSITIVITY-v1's
 hash-bound pilot + rest records); REVIEW = per generation its first policy-consulted recorded live KEEP. Per day x side the 32 states with the
 smallest sha256(arm|day|m|REVIEW|side|gid) (the pilot's key): up to 256 forks; inclusion probability 32 / n(day, side); the exact overlap with
 the pilot's REVIEW labels recorded. ADMISSION stays deferred.
FORKS: the pilot's frozen fork, ONE forced KEEP / PULL at the callback, the common focal-lineage continuation to END, its gates (base parity,
 unused exemption, exactly-once at equal contexts, pre-fork identity, reconciliation, queue reset, no lineage pull) - through FeatureGuard, which
 records at the callback the CAUSAL input vector FEATURES from the client view, the order view, the received book and the guard's receipt-clocked
 pressure cache (no venue queue, no future fill); both arms must record identical features.
CANDIDATE (ONE, fixed before any label): ridge on the 17 FEATURES - each winsorized at its training 1st / 99th percentile, standardized by the
 training mean / sd (sd 0 -> 1); target Y winsorized at its training 5th / 95th percentile; lambda = 1.0 x n_train (heavy shrinkage); intercept
 = the clipped training mean. Train on the ok pairs of Oct 24-25, score the ok pairs of Oct 26-27 ONCE. Decision: retain iff predicted Y > 0.
MEASUREMENT (evaluation days): V_rule = sum of Y over retained (zero for the alternative), also per day / side / market and inclusion-weighted;
 controls always-retain (sum Y), always-alternative (0), the training constant (retain all iff the training mean Y > 0); N_DRAWS = 1000 matched
 draws (seed 20261122) retaining the rule's count per evaluation day at random -> one-sided p; MSE vs zero and vs the training constant;
 calibration by predicted quartile; the negative tail and top-5 concentration of the selected; movable (non-tie) support.
READING (declared): INSUFFICIENT if training or evaluation has < 30 non-tie ok pairs, or the rule retains < 5 or > n - 5 on evaluation;
 useful_local_discrimination if V_rule > 0, V_rule > every control and the draw p <= 0.05 pooled, AND on each evaluation day V_rule_day > 0 and
 above that day's controls; otherwise no_useful_discrimination. A new adaptive development candidate (count 1); independent fork decisions,
 not a composed policy; no threshold search, feature race or tuning after scoring. A failed base parity makes the reading not_reliable;
 pairs failing a gate keep their status and leave the training / evaluation sets (counted).
PARTS: prefix = Oct 24 (7 markets, 64 forks) -> the write-once projection (spent + 3.3 x the prefix + a 120 s tail + the reserve <= 1.25; the
 line + those <= 4.00; peak < 1,536 MiB) -> rest Oct 25 / 26 / 27 in parallel -> measure.
Usage (run_light): pf_ladder_retention_discrimination.py selftest | manifest | prefix | project | rest <day> | measure [out]"""
import gzip
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_ladder_long_continuation as LC                  # noqa: E402  frozen (7407bdf1): the paired fork, its guard and gates

AF, P, M, LB, BE, C, pd = LC.AF, LC.P, LC.M, LC.LB, LC.BE, LC.C, LC.pd
VERSION = 'long_value_retention_discrimination_v1'
ARM, PER_CELL, TOL = LC.ARM, 32, 1e-6
TRAIN_DAYS, EVAL_DAYS, PREFIX_DAY = ('20261024', '20261025'), ('20261026', '20261027'), '20261024'
FEATURES = ('age_ms', 'rung', 'dist_ticks', 'spread_ticks', 'size', 'known_rem', 'shown_at_price', 'position_toward', 'reservation_side', 'reservation_opp',
            'n_side', 'n_opp', 'pressure_x', 'pressure_ok', 'time_left_ms', 'mid_change_toward_1s', 'restricted')
LAMBDA_PER_N, WINSOR_Q, Y_CLIP_Q, N_DRAWS, SEED, MIN_NONTIE, MIN_ACTIONS, P_MAX = 1.0, (0.01, 0.99), (0.05, 0.95), 1000, 20261122, 30, 5, 0.05
REPO = HERE.parents[1]
RECEIPT = 'predictfun/reviews/ladder_retention_discrimination_20261122'
OUT = REPO / 'data/predict_fun/research/ladder_retention_discrimination_v1_20261122'
CAP_BATCH, REVIEW_RESERVE, LINE_CEILING, TAIL_S, FACTOR = 1.25, 0.05, 4.00, 120., 3.3
SOURCES = ('pf_ladder_retention_discrimination.py', 'pf_ladder_retention_discrimination_selftest.py')
FROZEN_RECEIPT = ('declaration.md', 'manifest.json', 'census.csv', 'launch_retention.sh', 'rd_checks.py')


def need(cond, msg):
    C.require(cond, msg)


def sha_b(b):
    return hashlib.sha256(b).hexdigest()


def committed(rel):
    r = subprocess.run(['git', '-C', str(REPO), 'show', f'HEAD:{rel}'], capture_output=True); return r.stdout if r.returncode == 0 else None


def sources():
    return {n: sha_b((HERE / n).read_bytes()) for n in SOURCES}


def frozen():
    return bool(all(committed(f'predictfun/research/{n}') == (HERE / n).read_bytes() for n in SOURCES)
                and all(committed(f'{RECEIPT}/{n}') == (REPO / RECEIPT / n).read_bytes() for n in FROZEN_RECEIPT))


# ---------------------------------------------------------------------------------------------------------------------- the causal features
def features(B, pol, v, o):
    """The causal input vector at a live review (header): client view, order view, the received book, the receipt-clocked pressure cache."""
    t, s = float(v['t']), int(v['side']); sg = 1. if s == 0 else -1.; price = float(o['price']); best = float(v['bid'] if s == 0 else v['ask'])
    fk = B.E.f[B.E.known]; p0 = 2 + 10 * s; shown = float(sum(fk[p0 + 5 + k] for k in range(5) if fk[p0 + k] >= 0 and fk[p0 + k] == price))
    x, st, _ = pol.value(v); jp = int(np.searchsorted(B.E.f[:, 1], t - 1000., side='right')) - 1; f0 = B.E.f[jp] if jp >= 0 else None
    mid_now = (float(v['bid']) + float(v['ask'])) / 2.; mid_prev = (f0[2] + f0[12]) / 2. if (f0 is not None and f0[2] >= 0 and f0[12] >= 0) else mid_now
    return dict(age_ms=t - float(o['act']), rung=float(o['rung']), dist_ticks=sg * (best - price), spread_ticks=float(v['ask']) - float(v['bid']), size=float(o['size']),
                known_rem=float(o['size']) - float(o['known_filled']), shown_at_price=shown, position_toward=sg * float(v['position']),
                reservation_side=float(v['B'] if s == 0 else v['A']), reservation_opp=float(v['A'] if s == 0 else v['B']), n_side=float(len(v['orders'])),
                n_opp=float(len(v['opp_prices'])), pressure_x=float(x) if (st == 'ok' and np.isfinite(x)) else 0., pressure_ok=float(st == 'ok'),
                time_left_ms=float(v['end']) - t, mid_change_toward_1s=sg * (mid_now - mid_prev), restricted=float(v['mode'] != 'UNRESTRICTED'))


class FeatureGuard(LC.LineageGuard):
    """The pilot's LineageGuard plus the feature record at the forced REVIEW callback (no other change)."""

    def __init__(self, B, cache, edges):
        super().__init__(B, cache, edges); self.features = None

    def live(self, v, o):
        f = self.fork
        if f is not None and f['kind'] == 'REVIEW' and float(v['t']) == f['t'] and int(v['side']) == f['side'] and int(o['gid']) == f['gid'] and not self.applied:
            self.features = features(self.B, self, v, o)
        return super().live(v, o)


# ---------------------------------------------------------------------------------------------------------------------- the candidate and the measurement
def fit(X, y):
    """The ONE frozen candidate (header): winsorize, standardize, clipped target, ridge with lambda = LAMBDA_PER_N x n."""
    X, y = np.asarray(X, float), np.asarray(y, float); lo, hi = np.quantile(X, WINSOR_Q[0], axis=0), np.quantile(X, WINSOR_Q[1], axis=0); Xc = np.clip(X, lo, hi)
    mu, sd = Xc.mean(0), Xc.std(0); sd = np.where(sd > 0, sd, 1.); Z = (Xc - mu) / sd; yl, yh = np.quantile(y, Y_CLIP_Q[0]), np.quantile(y, Y_CLIP_Q[1]); yc = np.clip(y, yl, yh)
    ym = float(yc.mean()); w = np.linalg.solve(Z.T @ Z + LAMBDA_PER_N * len(y) * np.eye(Z.shape[1]), Z.T @ (yc - ym))
    return dict(lo=lo.tolist(), hi=hi.tolist(), mu=mu.tolist(), sd=sd.tolist(), w=w.tolist(), ym=ym, y_clip=[float(yl), float(yh)], n=int(len(y)))


def predict(m, X):
    Z = (np.clip(np.asarray(X, float), np.asarray(m['lo']), np.asarray(m['hi'])) - np.asarray(m['mu'])) / np.asarray(m['sd'])
    return m['ym'] + Z @ np.asarray(m['w'])


def draws(Ev, seed=SEED, n=N_DRAWS):
    """Matched draws: per evaluation day the rule's retain count at random -> the draws' V (header)."""
    rng = np.random.default_rng(seed); out = np.zeros(n)
    for _, g in Ev.groupby('day'):
        y = g.Y.values; k = int((g.pred > 0).sum())
        for i in range(n):
            out[i] += y[rng.choice(len(y), size=k, replace=False)].sum() if k else 0.
    return out


def evaluate(Tr, Ev, seed=SEED):
    """The declared measurement and reading (header) on the training / evaluation ok pairs with columns Y, pred, day, side, m, weight."""
    retain = Ev.pred > 0; V = float(Ev.Y[retain].sum()); n = len(Ev); const_retain = bool(Tr.Y.mean() > 0); D_ = draws(Ev, seed)
    res = dict(n_train=int(len(Tr)), n_eval=int(n), train_nontie=int((Tr.Y.abs() > TOL).sum()), eval_nontie=int((Ev.Y.abs() > TOL).sum()), retained=int(retain.sum()),
               V_rule=V, V_rule_mean=V / n if n else np.nan, V_always_retain=float(Ev.Y.sum()), V_always_alternative=0., V_constant=float(Ev.Y.sum()) if const_retain else 0.,
               constant_retains=const_retain, V_rule_weighted=float((Ev.Y * Ev.weight)[retain].sum()), draws_mean=float(D_.mean()), draws_q95=float(np.quantile(D_, .95)),
               p_draw=float((1 + (D_ >= V).sum()) / (1 + len(D_))), mse_model=float(((Ev.pred - Ev.Y) ** 2).mean()), mse_zero=float((Ev.Y ** 2).mean()),
               mse_constant=float(((Tr.Y.mean() - Ev.Y) ** 2).mean()), selected_min=float(Ev.Y[retain].min()) if retain.any() else np.nan,
               selected_negative_sum=float(Ev.Y[retain & (Ev.Y < 0)].sum()))
    sa = Ev.Y[retain].abs().sort_values(ascending=False); res['selected_top5_share'] = float(sa.head(5).sum() / sa.sum()) if sa.sum() > TOL else np.nan
    q = pd.qcut(Ev.pred.rank(method='first'), 4, labels=False) if n >= 4 else pd.Series(np.zeros(n, int), index=Ev.index)
    res['calibration'] = [dict(q=int(k), n=int(len(g)), pred=float(g.pred.mean()), Y=float(g.Y.mean())) for k, g in Ev.groupby(q)]
    per = {}
    for d, g in Ev.groupby('day'):
        r_ = g.pred > 0; cr = bool(Tr.Y.mean() > 0)
        per[d] = dict(V_rule=float(g.Y[r_].sum()), always_retain=float(g.Y.sum()), constant=float(g.Y.sum()) if cr else 0., retained=int(r_.sum()), n=int(len(g)))
    res['per_day'] = per
    res['per_side'] = {str(s): float(g.Y[g.pred > 0].sum()) for s, g in Ev.groupby('side')}
    res['per_market'] = {str(m): float(g.Y[g.pred > 0].sum()) for m, g in Ev.groupby('m')}
    ctrl = max(res['V_always_retain'], 0., res['V_constant'])
    insufficient = res['train_nontie'] < MIN_NONTIE or res['eval_nontie'] < MIN_NONTIE or res['retained'] < MIN_ACTIONS or res['retained'] > n - MIN_ACTIONS
    daily = all(x['V_rule'] > 0 and x['V_rule'] > max(x['always_retain'], 0., x['constant']) for x in per.values()) and len(per) == len(EVAL_DAYS)
    res['reading'] = 'INSUFFICIENT' if insufficient else ('useful_local_discrimination' if (V > 0 and V > ctrl and res['p_draw'] <= P_MAX and daily) else 'no_useful_discrimination')
    return res


# ---------------------------------------------------------------------------------------------------------------------- census, inputs, the day run
def af_records(gm_af):
    out = {}
    for part in ('pilot', 'rest'):
        for day in gm_af['days']:
            if not gm_af['parts'][part][day]:
                continue
            p_ = AF.OUT / part / f'{day}.json'; r = json.loads(p_.read_text())
            need(sha_b((AF.OUT / part / f'{day}.jsonl.gz').read_bytes()) == r['sha256']['gz'] and sha_b((AF.OUT / part / f'{day}_arrays.npz').read_bytes()) == r['sha256']['npz'],
                 f'{part} {day}: bytes differ from the record')
            out[f'{part}/{day}'] = sha_b(p_.read_bytes())
    return out


def census(gm_af):
    """Every REVIEW first policy-consulted live KEEP on the 19 markets with its key; per day x side the PER_CELL smallest keys selected."""
    rows = []
    for part in ('pilot', 'rest'):
        for day in gm_af['days']:
            ms = gm_af['parts'][part][day]
            if not ms:
                continue
            z = np.load(AF.OUT / part / f'{day}_arrays.npz')
            for m in ms:
                cm = z[f'{int(m)}__{ARM}__commands']; lk = cm[(cm[:, 2] == LC.LIVE) & (cm[:, 7] == LC.KEEP) & (cm[:, 8] == LC.KEEP_REASON)]
                for gid in np.unique(lk[:, 3]).astype(int):
                    x = lk[lk[:, 3] == gid]; i = int(np.argmin(x[:, 0])); s = int(x[i, 1])
                    rows.append(dict(day=day, m=int(m), part=part, kind='REVIEW', side=s, gid=int(gid), rung=float(x[i, 4]), t=float(x[i, 0]),
                                     key=hashlib.sha256(f'{ARM}|{day}|{int(m)}|REVIEW|{s}|{int(gid)}'.encode()).hexdigest()))
    D_ = pd.DataFrame(rows); D_['n_cell'] = D_.groupby(['day', 'side']).key.transform('size'); D_['rank'] = D_.groupby(['day', 'side']).key.rank(method='first').astype(int)
    D_['selected'] = D_['rank'] <= PER_CELL; D_['p_incl'] = np.minimum(PER_CELL / D_.n_cell, 1.); return D_.sort_values(['day', 'side', 'rank']).reset_index(drop=True)


def saved_record(day, m, part):
    for ln in gzip.decompress((AF.OUT / part / f'{day}.jsonl.gz').read_bytes()).decode().splitlines():
        r = json.loads(ln)
        if int(r['m']) == int(m):
            return r[ARM], np.load(AF.OUT / part / f'{day}_arrays.npz')[f'{int(m)}__{ARM}__commands']
    raise ValueError(f'{day} {m}: no saved record')


def run_market(day, m, part, path, ident, cache, edges, states):
    """One market: the roll-in, a REVIEW pair per state with its causal features, the base completed to END (parity)."""
    end = float(ident['end_rel']); B = AF.ProfileBranch(path, end, ident, False, LC.PROFILE); pol = FeatureGuard(B, cache, edges); obs = P.QueueObs(B, pol)
    rec, cm = saved_record(day, m, part); pairs, ledgers, k_prev = [], [], 0
    for st in sorted(states, key=lambda s: (s['t'], s['side'], s['gid'])):
        k = int(round(st['t'] / BE.STEP)); need(abs(k * BE.STEP - st['t']) < 1e-9 and k >= k_prev, f'{day} {m}: a state off the grid or out of order')
        LB.run_episode(B, pol, k0=k_prev, until=k, observe=obs); k_prev = k; ctx = (B, pol, obs); key0 = LC.fork_key(B, pol); out, feats = {}, {}
        for name, action in zip(('retain', 'alternative'), LC.ACTIONS['REVIEW']):
            Bx, px, ox = LC.fork_copy(ctx); need(LC.fork_key(Bx, px) == key0, 'a fork copy differs from its context'); px.arm(st, action)
            LB.run_episode(Bx, px, k0=k, observe=ox); out[name] = LC.arm_result(Bx, px, ox, st, end); feats[name] = px.features
        pr = LC.pair_result(st, out['retain'], out['alternative']); R_ = out['retain']
        expect = bool(len(R_['applied']) == 1 and R_['applied'][0][:4] == ('live', st['t'], st['gid'], 'KEEP') and len(out['alternative']['applied']) == 1
                      and out['alternative']['applied'][0][:4] == ('live', st['t'], st['gid'], 'PULL'))
        pr['gate_expected_callback'] = expect; pr['gate_features_equal'] = bool(feats['retain'] is not None and feats['retain'] == feats['alternative'])
        if R_['suppressed'] == 0:
            ss = LC.same_as_saved(R_['ledger'], R_['cmds'], rec, cm) if R_['cmds'] is not None else dict(commands=False); pr['gate_unused_exemption'] = bool(all(ss.values()))
        else:
            pr['gate_unused_exemption'] = None
        if not (expect and pr['gate_features_equal'] and pr['gate_unused_exemption'] is not False):
            pr['status'] = 'gate_failed' if pr['status'] == 'ok' else pr['status']
        pr.update(part=part, p_incl=st['p_incl'], **{f'f_{k_}': v for k_, v in (feats['retain'] or {}).items()}); pairs.append(pr)
        ledgers.append(dict(key=st['key'], day=day, m=m, kind='REVIEW', t=st['t'], gid=st['gid'], features=feats['retain'], features_alt=feats['alternative'], **{name: {k_: out[name][k_] for k_ in ('value', 'Q', 'I_fork', 'cash_fork', 'M_fork',
                            'mark', 'lineage', 'focal', 'applied', 'suppressed', 'pre_fills', 'post_raw', 'pp_log', 'replacements')} for name in ('retain', 'alternative')}))
    LB.run_episode(B, pol, k0=k_prev, observe=obs); Lb = LB.finish(B); ld = M.ledger_record(B, Lb); par = LC.same_as_saved(ld, P.command_arrays(B)['commands'], rec, cm)
    par.update(guard_log=[list(x) for x in pol.log] == rec['guard_log']); par = dict(day=day, m=m, **par, all=bool(all(par.values())))
    ledgers.append(dict(key=f'BASE|{day}|{m}', day=day, m=m, kind='BASE', base=dict(fills=ld['fills'], orders=ld['orders'], value=ld['value'], fingerprint=ld['fingerprint'])))
    return pairs, par, ledgers


# ---------------------------------------------------------------------------------------------------------------------- manifest, parts, projection, measurement
def build_manifest(path=None, census_path=None):
    path = Path(path or REPO / RECEIPT / 'manifest.json'); cpath = Path(census_path or REPO / RECEIPT / 'census.csv')
    need(not path.exists() and not cpath.exists(), 'the manifest / census exists (write-once)'); gm_af = LC.af_manifest(); D_ = census(gm_af); cpath.parent.mkdir(parents=True, exist_ok=True)
    D_.to_csv(cpath, index=False); sel = D_[D_.selected]; lcm = json.loads((REPO / LC.RECEIPT / 'manifest.json').read_text())
    pilot_keys = {s['key'] for s in lcm['selected'] if s['kind'] == 'REVIEW'}; tm = json.loads((REPO / 'predictfun/reviews/ladder_turnover_attribution_20261122/manifest.json').read_text())
    man = dict(version=VERSION, review='838f4be2', approval='USER 2026-11-22: approve 1.25 h / line 4.00 (AskUserQuestion)', arm=ARM, per_cell=PER_CELL, features=list(FEATURES),
               candidate=dict(model='ridge', lambda_per_n=LAMBDA_PER_N, winsor_q=list(WINSOR_Q), y_clip_q=list(Y_CLIP_Q), decision='retain iff predicted Y > 0'),
               train_days=list(TRAIN_DAYS), eval_days=list(EVAL_DAYS), draws=dict(n=N_DRAWS, seed=SEED), reading=dict(min_nontie=MIN_NONTIE, min_actions=MIN_ACTIONS, p_max=P_MAX),
               census_sha256=sha_b(cpath.read_bytes()), census_rows=int(len(D_)), selected=sel[['day', 'm', 'part', 'kind', 'side', 'gid', 'rung', 't', 'key', 'p_incl']].to_dict('records'),
               pilot_overlap=sorted(set(sel.key) & pilot_keys), af_records=af_records(gm_af), parts=dict(prefix=[PREFIX_DAY], rest=[d for d in gm_af['days'] if d != PREFIX_DAY]),
               markets={d: dict(pilot=[int(m) for m in gm_af['parts']['pilot'][d]], rest=[int(m) for m in gm_af['parts']['rest'][d]]) for d in gm_af['days']}, sources=sources(),
               line_receipts=tm['line_receipts'] + ['ladder_turnover_attribution_20261122'], line_reviews=tm['line_reviews'] + ['20261122_turnover_attribution_review'],
               prior_unlogged_s=tm['prior_unlogged_s'], caps=dict(batch_core_hours=CAP_BATCH, review_reserve=REVIEW_RESERVE, line_ceiling=LINE_CEILING, tail_s=TAIL_S, factor=FACTOR),
               asof=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    path.write_text(json.dumps(man, indent=1, sort_keys=True) + '\n'); return man


def day_paths(part, day, out=None):
    d = Path(out or OUT) / part
    return dict(rec=d / f'{day}.json', pairs=d / f'{day}_pairs.csv', parity=d / f'{day}_parity.json', ledgers=d / f'{day}_ledgers.jsonl.gz')


def run_part(part, day, out=None, require_frozen=True, man_path=None, proj_path=None):
    """One date: every selected state of its markets -> pairs (with features) / parity / ledgers, write-once."""
    if require_frozen:
        need(frozen(), 'the discrimination builder / selftest / declaration / manifest / census / launcher / checker are not the committed ones')
    gm = json.loads(Path(man_path or REPO / RECEIPT / 'manifest.json').read_text()); need(gm['sources'] == sources(), 'a source differs from the frozen one')
    need(day in gm['parts'][part], f'{day} is not in the {part} part'); gm_af = LC.af_manifest(); need(af_records(gm_af) == gm['af_records'], 'an ASYNC record differs from the manifest')
    pp = Path(proj_path or REPO / RECEIPT / 'logs' / 'projection.json')
    need(part == 'prefix' or (pp.exists() and json.loads(pp.read_text()).get('run_rest') is True), 'the rest needs the write-once projection with run_rest true')
    P_ = day_paths(part, day, out); need(not P_['rec'].exists(), f'{day}: the output exists (write-once)'); P_['rec'].parent.mkdir(parents=True, exist_ok=True); t0 = time.time()
    pairs, par, lgs = [], [], []
    for mp in ('pilot', 'rest'):
        mans = gm['markets'][day][mp]
        if not mans:
            continue
        inp, edges = LC.market_inputs(day, mans, gm_af)
        for m in mans:
            path, ident, cache = inp[m]; states = [dict(s) for s in gm['selected'] if s['day'] == day and int(s['m']) == m]
            if not states:
                continue
            a_, b_, c_ = run_market(day, m, mp, path, ident, cache, edges, states); pairs += a_; par.append(b_); lgs += c_
            need(AF.LS.hwm() < 1450., f'{day} {m}: memory above the self-guard')
        del inp
    pd.DataFrame(pairs).to_csv(P_['pairs'], index=False); P_['parity'].write_text(json.dumps(par, indent=1, default=str) + '\n')
    P_['ledgers'].write_bytes(gzip.compress('\n'.join(json.dumps(x, default=str) for x in lgs).encode(), mtime=0))
    P_['rec'].write_text(json.dumps(dict(version=VERSION, part=part, day=day, pairs=len(pairs), sha256={k: sha_b(P_[k].read_bytes()) for k in ('pairs', 'parity', 'ledgers')}, sources=sources(),
                                         elapsed_s=time.time() - t0, hwm_mib=AF.LS.hwm(), asof=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())), indent=1) + '\n')
    print(json.dumps(dict(part=part, day=day, pairs=len(pairs), parity_all=all(x['all'] for x in par), statuses=pd.Series([x['status'] for x in pairs]).value_counts().to_dict()), indent=1))


def project(path=None):
    """The write-once projection: spent + FACTOR x the prefix + the tail + the reserve <= the batch cap; the line + those <= the ceiling; peak guard."""
    path = Path(path or REPO / RECEIPT / 'logs' / 'projection.json'); need(not path.exists(), 'the projection exists (write-once)')
    gm = json.loads((REPO / RECEIPT / 'manifest.json').read_text()); own, peak = LC.cpu_of([str(REPO / RECEIPT / 'logs' / '**' / '*.log')])
    line, _ = LC.cpu_of([str(REPO / 'predictfun/reviews' / n / 'logs' / '**' / '*.log') for n in gm['line_receipts']])
    rev, _ = LC.cpu_of([str(REPO / 'predictfun/reviews/automatic' / n / '*.log') for n in gm['line_reviews']]); line += rev + gm['prior_unlogged_s'] / 3600.
    pre, _ = LC.cpu_of([str(REPO / RECEIPT / 'logs' / 'rd_prefix.log')]); need(pre > 0, 'the prefix has not run'); cp = gm['caps']
    proj = own + cp['factor'] * pre + cp['tail_s'] / 3600.
    res = dict(spent=own, prefix_core_hours=pre, projected_batch=proj, line_before=line, projected_line=line + proj + cp['review_reserve'], peak_mib=peak,
               run_rest=bool(proj + cp['review_reserve'] <= cp['batch_core_hours'] and line + proj + cp['review_reserve'] <= cp['line_ceiling'] and peak < 1536))
    path.write_text(json.dumps(res, indent=1) + '\n'); print(json.dumps(res, indent=1)); return res


def measure(out=None, require_frozen=True, man_path=None):
    """All parts -> pairs.csv, the fitted candidate, scores and the evaluation -> audit.json (write-once)."""
    if require_frozen:
        need(frozen(), 'the discrimination builder / selftest / declaration / manifest / census / launcher / checker are not the committed ones')
    gm = json.loads(Path(man_path or REPO / RECEIPT / 'manifest.json').read_text()); od = Path(out or OUT); md = od / 'measure'; need(not (md / 'audit.json').exists(), 'the audit exists (write-once)')
    md.mkdir(parents=True, exist_ok=True); tabs, par = [], []
    for part, days in gm['parts'].items():
        for day in days:
            P_ = day_paths(part, day, out); r = json.loads(P_['rec'].read_text()); need(all(r['sha256'][k] == sha_b(P_[k].read_bytes()) for k in ('pairs', 'parity', 'ledgers')), f'{day}: bytes differ')
            tabs.append(pd.read_csv(P_['pairs'], dtype={'day': str})); par += json.loads(P_['parity'].read_text())
    Pr = pd.concat(tabs, ignore_index=True); need(sorted(Pr.key) == sorted(s['key'] for s in gm['selected']), 'the pairs are not the selected states')
    ok = Pr[Pr.status == 'ok'].copy(); ok['weight'] = 1. / ok.p_incl; Xc = [f'f_{f}' for f in FEATURES]
    Tr, Ev = ok[ok.day.isin(TRAIN_DAYS)].copy(), ok[ok.day.isin(EVAL_DAYS)].copy(); model = fit(Tr[Xc].values, Tr.Y.values)
    Tr['pred'] = predict(model, Tr[Xc].values); Ev['pred'] = predict(model, Ev[Xc].values); res = evaluate(Tr, Ev)
    res.update(parity_all=bool(all(x['all'] for x in par)), statuses=Pr.status.value_counts().to_dict(), pilot_overlap=len(gm['pilot_overlap']))
    if not res['parity_all']:
        res['reading'] = 'not_reliable'
    pd.concat([Tr.assign(split='train'), Ev.assign(split='eval')]).to_csv(md / 'scored.csv', index=False); Pr.to_csv(md / 'pairs.csv', index=False)
    pd.DataFrame(par).to_csv(md / 'parity.csv', index=False); (md / 'model.json').write_text(json.dumps(model, indent=1) + '\n')
    (md / 'audit.json').write_text(json.dumps(dict(version=VERSION, evaluation=res, outputs={n: sha_b((md / n).read_bytes()) for n in ('scored.csv', 'pairs.csv', 'parity.csv', 'model.json')},
                                                   sources=sources(), asof=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())), indent=1, default=str) + '\n')
    print(json.dumps({k: v for k, v in res.items() if k not in ('calibration', 'per_market')}, indent=1, default=str))


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'selftest':
        import pf_ladder_retention_discrimination_selftest as ST
        ST.main(sys.modules[__name__])
    elif cmd == 'manifest':
        print(json.dumps({k: v for k, v in build_manifest().items() if k not in ('selected', 'pilot_overlap')}, indent=1)[:2000])
    elif cmd == 'prefix':
        run_part('prefix', PREFIX_DAY)
    elif cmd == 'project':
        project()
    elif cmd == 'rest':
        run_part('rest', sys.argv[2])
    elif cmd == 'measure':
        measure(*(sys.argv[2:3] or [None]))
    else:
        sys.exit('usage: pf_ladder_retention_discrimination.py selftest | manifest | prefix | project | rest <day> | measure [out]')
