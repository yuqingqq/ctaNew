"""The frozen chase line priced on the live port's rule: decide at the tick frame's receipt f plus the port's compute, on what has been
received by f (USER 2026-11-28: "Check Tokyo response, it decides based on the information received, don't manually wait a few ms"; Tokyo's
response pmmmrust df8e942 asks research to price this rule beside addendum 1's D = max(t + 3.5 ms, f + 2 ms)). P-2026-005, public data, no
orders. Declared before the run. A re-pricing of the frozen candidate on the forward days it was tested on (11-20 .. 11-26, read by this line
on 11-27), not a new test: nothing is chosen, there is no PASS rule; the selections, limits and models are the frozen run's, as in
chase_decision_delay.py, whose design, covariate and strata this reuses.
Timing (research's days): the at-frame decision is X = f + c - (t + FEED + DECIDE) after the replay's instant, negative when the frame
arrives early; lambda = f - t from Tokyo's stage-1 hour by stratum (chase_decision_delay's n_tr_20 quintiles); c = the port's receipt ->
decision latency, from the live shadow's published quantiles (median 0.34, p90 1.25, p99 2.65, max 3.75 ms; linear between, 0 at the bottom)
and, as a sensitivity, the pilot dry run's (0.40, 1.61, 2.84, 3.77), at K_C quantile midpoints. Exits at GRID_NEG = (-3, -2, -1, 0) ms
(x = 0 must equal chase_forward_bigmove's stored exits on the same rows), joined with chase_decision_delay's (0, 2, 8, 30, 120); linear between
grid points, clamped at both ends (counted). Rows: the primary selection's actions EARLY at some x >= -3 (cut_minus_act_ms - x >= EARLY_MS);
the scope re-applied at each x.
Readings (chase_forward_bigmove's R1..R5): D (must reproduce chase_decision_delay.json's MATCHED), AT_FRAME (shadow c; the estimate),
AT_FRAME_PILOT (pilot c), D_X2 and AT_FRAME_X2 (the frame's delay beyond FEED doubled: lambda' = lambda + max(0, lambda - FEED); D_X2 must
reproduce chase_decision_delay's MATCHED_X2), AT_FRAME - D; CAL_DRAWS block-bootstrap draws of Tokyo's hour (1-minute blocks) for R1 and R5:
AT_FRAME, D and their paired difference.
Information (Tokyo's stage-1 hour; research's frozen rows on it, the frozen model and threshold): the at-frame inputs lack the trades with
E <= t received after f. Tokyo's stage-1 replay decided at the frame on what it had. Research's rows, at each (t, d) matched to Tokyo's row
at the millisecond's last frame after Tokyo's warm-up, with:
  H1     its / ag volumes (mv_its_btc, mv_ag_btc, its_100_btc, ag_100_btc) from Tokyo's row: exact at f;
  H2     H1 + the best-level flows (x_*) from Tokyo's row: the stage-1 port never re-netted a late trade, which over-states what fix 1's
         netting leaves (a pessimistic bound);
  H_AGE  research's inputs with pf_age_ms at the at-frame instant (f + 0.34 ms) if the known Predict frame were the same (approximate; the
         rows where it might not be are counted);
rescored with the frozen model, one action per move (the first finite MOVE >= threshold), the forward scope: actions kept / lost / added and
moves acting at another tick against research's (H0, which must reproduce research's stored scores and actions), and how MOVE changes.
Known-bad: its_100_btc zeroed must change the scores. Tokyo's stage-1 actions on a frame that is not its millisecond's last (a provisional
state, which the at-frame rule can act on) are counted. Exact: Tokyo's replay with fix 1 and fix 2 (inputs at the decision and at D per row).
Limits declared: one Tokyo hour; c independent of the burst (live, the port's queue can grow in bursts); on research's days the Predict inputs
and the limit stay at t + 3.5 ms's known frame and the action set is the frozen run's (the information check sizes what that leaves out).
Stages: selftest | tokyo <dir holding tokyo-chase-stage1-20261128> | exits <day> | read.  Outputs (write-once): chase_decision_atframe_tokyo.json,
OUT/chase_forward/exits_atframe/<day>.parquet + <day>.json, chase_decision_atframe.json.
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/chase_decision_atframe.py <stage> ...
"""
import datetime as dt
import hashlib
import json
import pickle
import resource
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import chase_decision_delay as CDD  # noqa: E402
import chase_forward as CF  # noqa: E402
import chase_forward_bigmove as CFB  # noqa: E402
import chase_forward_latency as CFL  # noqa: E402
import chase_forward_tokyo as CFT  # noqa: E402
import chase_move_size as CMS  # noqa: E402

EF, C, TR, CT, TM, DC = CF.EF, CF.C, CF.TR, CF.CT, CF.TM, CF.DCm
OUT_A = CF.FOUT / 'exits_atframe'
TOKYO, RESULT = H / 'chase_decision_atframe_tokyo.json', H / 'chase_decision_atframe.json'
SEL, READINGS, COV, NSTRATA = CDD.SEL, CDD.READINGS, CDD.COV, CDD.NSTRATA
GRID_NEG = (-3., -2., -1., 0.)
I0 = GRID_NEG.index(0.)
GRID_ALL = tuple(sorted(set(GRID_NEG) | set(CDD.GRID)))
C_SHADOW = ((0., 0.), (.5, .34), (.9, 1.25), (.99, 2.65), (1., 3.75))
C_PILOT = ((0., 0.), (.5, .40), (.9, 1.61), (.99, 2.84), (1., 3.77))
K_C, C_CHECK, WARM_MS = 100, .34, 60000.
H_VOL = ['mv_its_btc', 'mv_ag_btc', 'its_100_btc', 'ag_100_btc']
CAL_DRAWS, SEED = 200, 20261201
RULES = ('D', 'D_X2', 'AT_FRAME', 'AT_FRAME_PILOT', 'AT_FRAME_X2')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------------------------------ the delay samples
def c_quantiles(qf, k=K_C):
    """The compute latency's quantile function through the published points (u, c), at k midpoints."""
    u = (np.arange(k) + .5) / k
    U, Cq = zip(*qf)
    C.require(np.all(np.diff(U) > 0) and np.all(np.diff(Cq) >= 0), 'a quantile function must increase')
    return np.interp(u, U, Cq)


def stress(lam):
    lam = np.asarray(lam, float)
    return lam + np.maximum(lam - TM.FEED, 0.)


def rule_sample(rule, lam, sb):
    """A rule's delay sample X (against the replay's t + FEED + DECIDE) and its strata, from Tokyo's points (lambda, stratum)."""
    lam, sb = np.asarray(lam, float), np.asarray(sb)
    lead = TM.FEED + TM.DECIDE
    if rule in ('D', 'D_X2'):
        lm = lam if rule == 'D' else stress(lam)
        return np.maximum(lm - TM.FEED, 0.), sb
    lm = stress(lam) if rule == 'AT_FRAME_X2' else lam
    cq = c_quantiles(C_PILOT if rule == 'AT_FRAME_PILOT' else C_SHADOW)
    return (lm[:, None] + cq[None, :] - lead).ravel(), np.repeat(sb, len(cq))


def close(a, b, rel=1e-9):
    """Numbers within rel x max(1, |b|), lists elementwise, anything else equal."""
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(close(x, y, rel) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return abs(a - b) <= rel * max(1., abs(b))
    return a == b


def weights_for(xs, sb, grid=GRID_ALL, fallback=None):
    """Per stratum the hat weights of its sample on the grid (clamped at both ends; an empty stratum takes fallback's row)
    -> (W, sample points below the first grid point, above the last, empty strata)."""
    rows, lo, hi, empty = [], 0, 0, 0
    for b in range(NSTRATA):
        x = xs[sb == b]
        if not len(x):
            C.require(fallback is not None, f'stratum {b} has no sample points')
            rows.append(fallback[b])
            empty += 1
            continue
        w, a = CDD.hat_weights(x, grid)
        rows.append(w)
        lo += int(np.sum(x < grid[0]))
        hi += a
    return np.stack(rows), lo, hi, empty


# ------------------------------------------------------------------------------------------------------------------ tokyo
def hybrid(Xs, T, ok, cols):
    """Research's rows with the columns cols taken from Tokyo's matched rows where ok, in research's dtypes."""
    Z = Xs.copy()
    for c in cols:
        Z[c] = np.where(ok, T[c].to_numpy(float), Xs[c].to_numpy(float)).astype(Xs[c].dtype)
    return Z


def stage_tokyo(src):
    C.require(not TOKYO.exists(), f'{TOKYO.name} exists (write-once)')
    resource.setrlimit(resource.RLIMIT_CPU, (900, 901))
    hd = Path(src) / 'tokyo-chase-stage1-20261128'
    T0 = json.loads(CDD.TOKYO.read_text())
    rows_fp, dec_fp = CDD.SRC / 'research_rows.parquet', hd / 'replay' / 'decisions.csv.gz'
    C.require(sha(rows_fp) == T0['inputs']['research_rows_sha256'], "research's rows changed since chase_decision_delay's Tokyo stage")
    X = pd.read_parquet(rows_fp)
    bt, tr, _ = CDD.tokyo_frames(CDD.SRC / 'data' / 'raw')
    t = X.t.to_numpy(float)
    f = X.t.map(bt.groupby('E').r.max()).to_numpy(float) / 1e6
    C.require(np.isfinite(f).all(), "a decision point without its tick ms's bookTicker frame in Tokyo's recording")
    lam = f - t
    sb = CDD.strata_of(CDD.count_in(tr.E.to_numpy(float), t, float(COV.split('_')[-1])), T0['strata']['edges'])
    blk = np.floor((t - t.min()) / CDD.BLOCK_MS).astype(int)
    same = (np.allclose(np.round(np.maximum(lam - TM.FEED, 0.), 6), np.array(T0['samples']['X']), rtol=0., atol=1e-9)
            and sb.tolist() == T0['samples']['stratum'] and blk.tolist() == T0['samples']['block'])
    C.require(same, "the points' delays, strata or blocks differ from chase_decision_delay's sample")
    # the information check
    R = json.loads((H / 'chase_forward_refit.json').read_text())
    kf, thr = R['inputs'], float(R['thresholds']['MOVE_top10'])
    with open(CF.FOUT / 'refit' / 'models.pkl', 'rb') as fh:
        Mo = pickle.load(fh)
    Tk = pd.read_csv(dec_fp, float_precision='round_trip')
    ev = [json.loads(ln) for ln in open(hd / 'hour1' / 'events.jsonl') if ln.strip()]
    st = [e for e in ev if e.get('kind') == 'start']
    C.require(len(st) == 1, "expected one start event in Tokyo's events.jsonl")
    start_ms = st[0]['recv_ns'] / 1e6
    Tk = Tk.assign(t=Tk.t.astype(float), d=Tk.d.astype(float))
    last = Tk.sort_values(['t', 'd', 'recv_ns'], kind='stable').groupby(['t', 'd'], as_index=False).tail(1)
    acts = Tk[Tk.decision.astype(str) == 'act']
    prov = acts.merge(last[['t', 'd', 'recv_ns']], on=['t', 'd', 'recv_ns'], how='left', indicator=True)
    Xs = X.assign(f_ms=f).sort_values(['day', 'd', 't'], kind='stable').reset_index(drop=True)
    xcols = [c for c in kf if c.startswith('x_')]
    M = Xs[['t', 'd']].merge(last[['t', 'd'] + H_VOL + xcols], on=['t', 'd'], how='left', validate='1:1', indicator=True)
    after = Xs.t.to_numpy(float) >= start_ms + WARM_MS
    ok = (M._merge == 'both').to_numpy() & after
    cost = CF.cost_of(Xs)
    early = Xs.early.to_numpy(bool)
    key = Xs.key.to_numpy()

    def run(Z):
        pred = Mo['move'].predict(Z[kf])
        mv = pred - cost
        above = np.isfinite(mv) & (mv >= thr)
        act = np.zeros(len(Z), bool)
        act[CT.first_per_move(above, key)] = True
        return mv, above, act
    mv0, ab0, act0 = run(Xs)
    C.require(np.array_equal(mv0, Xs.MOVE.to_numpy(float), equal_nan=True) and np.array_equal(act0, Xs.act.to_numpy(bool)),
              "the frozen model does not reproduce research's stored scores and actions")
    A0 = act0 & early & after

    def versus(nm, Z, note=None):
        mv, ab, act = run(Z)
        A = act & early & after
        k0, k1 = set(key[A0]), set(key[A])
        t0_ = dict(zip(key[A0], Xs.t.to_numpy(float)[A0]))
        t1_ = dict(zip(key[A], Xs.t.to_numpy(float)[A]))
        dm = (mv - mv0)[ok]
        out = dict(rows_replaced=int(ok.sum()), rows_move_changed=int(np.sum(np.abs(dm) > 0)), threshold_side_flips=int(np.sum((ab != ab0) & ok)),
                   abs_move_change=CDD.quant(np.abs(dm[np.abs(dm) > 0])), actions=int(A.sum()), actions_research=int(A0.sum()),
                   kept_same_tick=int(np.sum(A & A0)), lost=int(np.sum(A0 & ~A)), added=int(np.sum(A & ~A0)),
                   moves_acting_at_another_tick=int(sum(1 for k_ in k0 & k1 if t0_[k_] != t1_[k_])), moves_no_longer_acting=int(len(k0 - k1)),
                   moves_newly_acting=int(len(k1 - k0)))
        if note:
            out.update(note)
        return out
    Mm = M.reset_index(drop=True)
    H1 = hybrid(Xs, Mm, ok, H_VOL)
    H2 = hybrid(H1, Mm, ok, xcols)
    delta = Xs.f_ms.to_numpy(float) + C_CHECK - (Xs.t.to_numpy(float) + TM.FEED + TM.DECIDE)
    age = Xs.pf_age_ms.to_numpy(float)
    kmin = CF.I.PF_MIN + json.loads((CDD.SRC / 'data' / 'step1' / '20261128' / 'clock_rl_d2.json').read_text())['d_dec']
    Ha = Xs.copy()
    Ha['pf_age_ms'] = np.where(after, age + delta, age).astype(Xs.pf_age_ms.dtype)
    info = dict(H1_volumes=versus('H1', H1), H2_volumes_and_flows=versus('H2', H2),
                H_AGE=versus('H_AGE', Ha, dict(rows_maybe_older_known_frame=int(np.sum(after & (delta < 0) & (age + delta < kmin))),
                                                rows_maybe_newer_known_frame=int(np.sum(after & (delta > 0))), delta_ms=CDD.quant(delta[after]))))
    kb = Xs.copy()
    kb['its_100_btc'] = np.where(ok, 0., kb.its_100_btc.to_numpy(float)).astype(Xs.its_100_btc.dtype)
    kbv = versus('known_bad', kb)
    C.require(kbv['rows_move_changed'] > 0.01 * ok.sum(), f'the known-bad hybrid (its_100_btc zeroed) changed too few scores: {kbv["rows_move_changed"]}')
    aa = X.act.to_numpy(bool) & X.early.to_numpy(bool)
    out = dict(asof=dt.datetime.now(dt.timezone.utc).isoformat(),
               inputs=dict(research_rows_sha256=sha(rows_fp), tokyo_decisions_sha256=sha(dec_fp), tokyo_recording='pmmmrust data/tokyo-chase-stage1-20261128 (0fec30b)',
                           decision_delay_tokyo=CDD.TOKYO.name),
               consistent_with_decision_delay=bool(same), compute_quantiles=dict(shadow=C_SHADOW, pilot=C_PILOT, k=K_C),
               lam=CDD.quant(lam), x_at_frame_points=CDD.quant(rule_sample('AT_FRAME', lam, sb)[0]),
               x_at_frame_actions=CDD.quant(rule_sample('AT_FRAME', lam[aa], sb[aa])[0]),
               information=dict(start_ms=start_ms, warm_ms=WARM_MS, matched_after_warmup=int(ok.sum()), research_actions_in_scope=int(A0.sum()), **info,
                                known_bad_its_100_zeroed=kbv, tokyo_actions=int(len(acts)), tokyo_actions_on_a_provisional_frame=int((prov._merge == 'left_only').sum())),
               samples=dict(lam=[round(float(x), 6) for x in lam], stratum=[int(b) for b in sb], block=[int(b) for b in blk]))
    TR.dump(TOKYO, out)
    print(json.dumps(TR.clean({k: {x: v[x] for x in ('rows_move_changed', 'threshold_side_flips', 'actions', 'kept_same_tick', 'lost', 'added', 'moves_acting_at_another_tick')}
                               for k, v in info.items()})), 'provisional', out['information']['tokyo_actions_on_a_provisional_frame'], 'of', len(acts),
          'X at frame', {k: round(v, 2) for k, v in out['x_at_frame_points'].items() if k in ('mean', 'p10', 'p50', 'p90', 'p99')}, flush=True)


# ------------------------------------------------------------------------------------------------------------------ exits
def configs_neg(ms):
    nk = len(ms['Le'])
    ent = [dict(name=f'D{k}Q{i}', Le=float(ms['Le'][k] + x)) for i, x in enumerate(GRID_NEG) for k in range(nk)]
    ext = [dict(name=f'D{k}Q{i}_S{j}', entry=f'D{k}Q{i}', Sa=s, Sb=0., Lx=ms['Lx'], pols=('TAKE_C1000',))
           for i in range(len(GRID_NEG)) for k in range(nk) for j, s in enumerate(ms['S'])]
    names = [x['name'] for x in ext]
    C.require(len(set(names)) == len(names) and len({e['name'] for e in ent}) == len(ent), 'config names')
    C.require(all(e['Le'] == ms['Le'][k] for k, e in enumerate(ent[I0 * nk:(I0 + 1) * nk])), 'the x = 0 entries are not the big-move delays exactly')
    C.require(min(e['Le'] for e in ent) > 0., 'an entry delay below zero')
    return ent, ext


def parity_q0(new, old, nk, nS):
    """The x = 0 columns (Q{I0}) of new against the big-move columns of old (same rows), exactly (NaN = NaN). -> mismatched names."""
    pairs = [('cut_minus_act_ms', 'cut_minus_act_ms'), ('t', 't'), ('d', 'd'), ('mid', 'mid'), ('lim', 'lim')]
    for k in range(nk):
        for s in (3, 10):
            pairs.append((f'D{k}Q{I0}_AB_s{s}_fill', f'D{k}_AB_s{s}_fill'))
            pairs += [(f'D{k}Q{I0}_S{j}_AB_s{s}_TAKE_C1000_pnl', f'D{k}_S{j}_AB_s{s}_TAKE_C1000_pnl') for j in range(nS)]
    pairs += [(f'mv_up100_Q{I0}', 'mv_up100'), (f'mv_n_post200_Q{I0}', 'mv_n_post200')]
    if len(new) != len(old):
        return ['rows differ']
    return [a for a, b in pairs if a not in new.columns or b not in old.columns or not CFL.same_col(new[a].reset_index(drop=True), old[b].reset_index(drop=True))]


def stage_exits(day):
    C.require(day in CF.FWD, f'{day} is not a declared forward day')
    fp, fj = OUT_A / f'{day}.parquet', OUT_A / f'{day}.json'
    C.require(not fp.exists() and not fj.exists(), f'{fp.name} exists (write-once)')
    resource.setrlimit(resource.RLIMIT_CPU, (1800, 1801))
    tk = json.loads(TOKYO.read_text())
    C.require(tk['consistent_with_decision_delay'], "the Tokyo stage's sample is not chase_decision_delay's")
    CF.frozen()
    ms = CFB.measured()
    nk, nS = len(ms['Le']), len(ms['S'])
    ent, ext = configs_neg(ms)
    CF.forward_mode((day,))
    S = pd.read_parquet(CF.FOUT / 'forward_selections.parquet')
    S = S[S[SEL].to_numpy(bool) & (S.day == day).to_numpy()].reset_index(drop=True)
    old = pd.read_parquet(CF.FOUT / 'exits_bigmove' / f'{day}.parquet')
    C.require(np.array_equal(old.t.to_numpy(float), S.t.to_numpy(float)) and np.array_equal(old.d.to_numpy(float), S.d.to_numpy(float)),
              "the stored big-move exits are not the selection's rows")
    keep = old.cut_minus_act_ms.to_numpy(float) >= EF.EARLY_MS + GRID_NEG[0]
    Se, oe = S[keep].reset_index(drop=True), old[keep].reset_index(drop=True)
    X, st = CFT.exits_day_tokyo(day, Se, ent, ext)
    C.require(np.array_equal(X.t.to_numpy(float), Se.t.to_numpy(float)) and np.array_equal(X.d.to_numpy(float), Se.d.to_numpy(float)), 'row order')
    t, dd = Se.t.to_numpy(float), Se.d.to_numpy(float)
    Eb, Mb = CT.mid_series(day)
    for i, x in enumerate(GRID_NEG):
        mv = CMS.move_stats(Eb, Mb, t + x, dd)
        X[f'mv_up100_Q{i}'], X[f'mv_n_post200_Q{i}'] = mv['up100'], mv['n_post200']
    bad = parity_q0(X, oe, nk, nS)
    C.require(not bad, f'x = 0 parity with the stored big-move exits failed on {day}: {bad[:8]}')
    tr = DC.read_trades(day).drop_duplicates('id')
    tr = tr[(tr.p > 0) & (tr.q > 0)]
    X[COV] = CDD.count_in(np.sort(tr.E.to_numpy(float)), t, float(COV.split('_')[-1]))
    del tr
    cols = list(Se.columns) + ['cut_minus_act_ms', COV] + [f'mv_{k}_Q{i}' for i in range(len(GRID_NEG)) for k in ('up100', 'n_post200')]
    cols += [f'D{k}Q{i}_AB_s3_fill' for i in range(len(GRID_NEG)) for k in range(nk)]
    cols += [f'D{k}Q{i}_S{j}_AB_s3_TAKE_C1000_pnl' for i in range(len(GRID_NEG)) for k in range(nk) for j in range(nS)]
    X = X[list(dict.fromkeys(cols))]
    st.update(rows_selected=int(len(S)), rows_scope=int(len(Se)), rows_early_at_0=int(np.sum(oe.cut_minus_act_ms.to_numpy(float) >= EF.EARLY_MS)),
              parity=dict(columns=4 + 2 * nk * (1 + nS) + 2, mismatched=bad), grid=list(GRID_NEG), grid_drops=dict(CF.GRID_DROPS))
    OUT_A.mkdir(parents=True, exist_ok=True)
    X.to_parquet(fp, index=False)
    TR.dump(fj, st)
    print(day, json.dumps(TR.clean({k: v for k, v in st.items() if k in ('actions', 'rows_scope', 'rows_early_at_0', 'parity', 'unloaded')})), flush=True)


# ------------------------------------------------------------------------------------------------------------------ read
def stage_read():
    C.require(not RESULT.exists(), f'{RESULT.name} exists (write-once)')
    resource.setrlimit(resource.RLIMIT_CPU, (1200, 1201))
    rc = CF.frozen()
    p, sel_p, s_p = CFL.primary(rc)
    C.require(sel_p == SEL, f"the receipt's primary selection is {sel_p}, the rows here are {SEL}")
    s = int(s_p)
    tk = json.loads(TOKYO.read_text())
    R0 = json.loads(CDD.RESULT.read_text())
    ms = CFB.measured()
    days = tuple(d for d in CF.FWD if (OUT_A / f'{d}.parquet').exists())
    C.require(days == CF.FWD, f'exits_atframe has {len(days)} of {len(CF.FWD)} days')
    A = pd.concat([pd.read_parquet(OUT_A / f'{d}.parquet') for d in days], ignore_index=True)
    B = pd.concat([pd.read_parquet(CDD.OUT_D / f'{d}.parquet') for d in days], ignore_index=True)
    nk, nS = len(ms['A']), len(ms['S'])
    bcols = [c for c in B.columns if (c.startswith('D') and 'X' in c.split('_')[0]) or (c.startswith('mv_') and '_X' in c)]
    J = A.merge(B[['day', 't', 'd'] + [c for c in bcols if c not in A.columns]], on=['day', 't', 'd'], how='left', validate='1:1', indicator=True)
    inb = (J._merge == 'both').to_numpy()
    C.require(int(inb.sum()) == len(B), "a chase_decision_delay row is missing from the at-frame rows")
    z0 = [c for k in range(nk) for c in ([f'D{k}Q{I0}_AB_s{s}_fill', f'D{k}X0_AB_s{s}_fill']
                                         + [x for j in range(nS) for x in (f'D{k}Q{I0}_S{j}_AB_s{s}_TAKE_C1000_pnl', f'D{k}X0_S{j}_AB_s{s}_TAKE_C1000_pnl')])]
    bad0 = [a for a, b in zip(z0[::2], z0[1::2]) if not np.array_equal(J[a].to_numpy(float)[inb], J[b].to_numpy(float)[inb], equal_nan=True)]
    C.require(not bad0, f"the x = 0 columns differ from chase_decision_delay's: {bad0[:4]}")
    keys, dayk = [], []
    for d in days:
        reg = CF.MM.registry(d).sort_values('ws')
        keys += [f'{d}:{int(m)}' for m in reg.mid]
        dayk += [d] * len(reg)
    keys, dayk = np.array(keys), np.array(dayk)
    CF.SX.DAYS = days
    per_day = len(keys) / len(days)
    wk = (J.day.astype(str) + ':' + J.mid.astype(np.int64).astype(str)).to_numpy()
    big = [k for k, r in enumerate(ms['rows']) if r['probe'] == 'big']
    allk = list(range(nk))
    cov = {k_: np.array([r[k_] for r in ms['rows']], float) for k_ in ('up100', 'n_post200')}
    cma = J.cut_minus_act_ms.to_numpy(float)
    m_all = np.ones(len(J), bool)
    every = lambda ks: np.tile(np.isin(np.arange(nk), ks), (len(J), 1))  # noqa: E731
    ng = len(GRID_ALL)
    V = {r: np.zeros((ng, len(J))) for r in READINGS}
    Fv = {r: np.zeros((ng, len(J))) for r in READINGS}
    scope = {}
    for gi, x in enumerate(GRID_ALL):
        tag = f'Q{GRID_NEG.index(x)}' if x in GRID_NEG else f'X{CDD.GRID.index(x)}'
        P = np.stack([np.stack([J[f'D{k}{tag}_S{j}_AB_s{s}_TAKE_C1000_pnl'].to_numpy(float) for j in range(nS)]) for k in range(nk)]).mean(axis=1)
        F = np.stack([(np.nan_to_num(J[f'D{k}{tag}_AB_s{s}_fill'].to_numpy(float)) > 0).astype(float) for k in range(nk)])
        mx = (cma - x) >= EF.EARLY_MS
        C.require(x <= 0 or not np.any(mx & ~inb), f'a row outside chase_decision_delay\'s set is in scope at x = {x:g}')
        scope[f'x{x:g}'] = int(mx.sum())
        up, npost = J[f'mv_up100_{tag}'].to_numpy(float), J[f'mv_n_post200_{tag}'].to_numpy(float)
        sets = dict(R1_BIG=every(big), R2_BIG_MOVE=CFB.strata_sets(up, cov['up100'][big], big, nk, up[mx])[0],
                    R3_BIG_BURST=CFB.strata_sets(npost, cov['n_post200'][big], big, nk, npost[mx])[0], R4_BOTH=every(allk),
                    R5_BOTH_BURST=CFB.strata_sets(npost, cov['n_post200'], allk, nk, npost[mx])[0])
        for r, st_ in sets.items():
            C.require(st_ is not None, f'{r} is not computable at x = {x:g}')
            pn, ff = CFB.set_mean(P, F, st_)
            V[r][gi] = np.where(mx, np.nan_to_num(pn), 0.)
            Fv[r][gi] = np.where(mx, ff, 0.)
    stats = lambda v, f: CFT.stats_of(v, m_all, f, wk, keys, dayk, per_day)  # noqa: E731
    lam, sb, blk = (np.array(tk['samples'][k]) for k in ('lam', 'stratum', 'block'))
    bA = CDD.strata_of(J[COV].to_numpy(float), json.loads(CDD.TOKYO.read_text())['strata']['edges'])
    W, clamp = {}, {}
    xd = np.array(json.loads(CDD.TOKYO.read_text())['samples']['X'])                   # D and D_X2 on chase_decision_delay's own sample
    for ru in RULES:
        xs, sbs = (xd, sb) if ru == 'D' else (2. * xd, sb) if ru == 'D_X2' else rule_sample(ru, lam, sb)
        W[ru], lo, hi, _ = weights_for(xs, sbs)
        clamp[ru] = dict(below_first=lo, above_last=hi, sample=int(len(xs)))
    readings, par = {}, {}
    for r in READINGS:
        rr = {ru: stats(np.sum(W[ru][bA].T * V[r], axis=0), np.sum(W[ru][bA].T * Fv[r], axis=0)) for ru in RULES}
        for ru, ref in (('D', 'MATCHED'), ('D_X2', 'MATCHED_X2')):
            u = json.loads(json.dumps(TR.clean(rr[ru])))
            b0 = R0['readings'][r][ref]
            par[f'{r}|{ru}'] = [k for k in ('total_usd', 'days_positive', 'sharpe_day_equiv', 'filled', 'daily_usd') if not close(u[k], b0[k])]
        rr['AT_FRAME_minus_D_usd'] = rr['AT_FRAME']['total_usd'] - rr['D']['total_usd']
        rr['AT_FRAME_X2_minus_D_X2_usd'] = rr['AT_FRAME_X2']['total_usd'] - rr['D_X2']['total_usd']
        rr['uniform_x'] = {f'x{x:g}': stats(V[r][gi], Fv[r][gi])['total_usd'] for gi, x in enumerate(GRID_ALL)}
        readings[r] = rr
    C.require(not any(par.values()), f"D / D_X2 do not reproduce chase_decision_delay.json's MATCHED / MATCHED_X2: {par}")
    rng = np.random.default_rng(SEED)
    nb = int(blk.max() + 1)
    draws = {r: dict(AT_FRAME=[], D=[], diff=[]) for r in ('R1_BIG', 'R5_BOTH_BURST')}
    empty = 0
    for _ in range(CAL_DRAWS):
        pick = rng.integers(0, nb, nb)
        idx = np.concatenate([np.flatnonzero(blk == b) for b in pick])
        Wd = {}
        for ru in ('AT_FRAME', 'D'):
            xs, sbs = rule_sample(ru, lam[idx], sb[idx])
            Wd[ru], _, _, e = weights_for(xs, sbs, fallback=W[ru])
            empty += e > 0
        for r, dr in draws.items():
            a = float(np.sum(np.sum(Wd['AT_FRAME'][bA].T * V[r], axis=0)) / 100.)
            b = float(np.sum(np.sum(Wd['D'][bA].T * V[r], axis=0)) / 100.)
            dr['AT_FRAME'].append(a)
            dr['D'].append(b)
            dr['diff'].append(a - b)
    q = lambda v: dict(p5=float(np.quantile(v, .05)), median=float(np.median(v)), p95=float(np.quantile(v, .95)), share_positive=float(np.mean(np.array(v) > 0)))  # noqa: E731
    exp_x = {ru: float(np.mean([np.dot(W[ru][b], GRID_ALL) for b in bA])) for ru in RULES}
    out = dict(asof=dt.datetime.now(dt.timezone.utc).isoformat(), receipt=CF.receipt_path().name, primary=p, days=list(days), grid_ms=list(GRID_ALL),
               rows=int(len(J)), rows_in_decision_delay_set=int(inb.sum()), scope_by_x=scope, expected_x_ms_on_the_grid=exp_x, clamped=clamp,
               information=tk['information'], readings=readings,
               calibration_draws=dict(n=CAL_DRAWS, seed=SEED, empty_stratum_draws=int(empty), **{r: {k: q(v) for k, v in dr.items()} for r, dr in draws.items()}),
               parity=dict(x0_vs_decision_delay=[], rules_vs_decision_delay={k: v for k, v in par.items()}))
    TR.dump(RESULT, out)
    print(json.dumps(TR.clean({r: dict(D=round(v['D']['total_usd'], 1), AT_FRAME=round(v['AT_FRAME']['total_usd'], 1), PILOT=round(v['AT_FRAME_PILOT']['total_usd'], 1),
                                       D_X2=round(v['D_X2']['total_usd'], 1), AT_X2=round(v['AT_FRAME_X2']['total_usd'], 1), days=v['AT_FRAME']['days_positive'],
                                       sharpe=round(v['AT_FRAME']['sharpe_day_equiv'], 3)) for r, v in readings.items()})),
          'draws', json.dumps(TR.clean(out['calibration_draws'])), 'expected x', {k: round(v, 2) for k, v in exp_x.items()}, flush=True)


# ------------------------------------------------------------------------------------------------------------------ selftest
def selftest():
    cq = c_quantiles(C_SHADOW)
    C.require(len(cq) == K_C and np.all(np.diff(cq) >= 0) and 0 < cq[0] < .34 < cq[-1] <= 3.75 and abs(np.median(cq) - .34) < .02, f'compute quantiles {cq[:3]} {cq[-3:]}')
    try:
        c_quantiles(((0., 0.), (.5, 1.), (.4, 2.)))
        raise AssertionError('a decreasing quantile function was accepted')
    except ValueError as e_:
        assert 'must increase' in str(e_), e_
    lam = np.array([1., 1.5, 4.])
    xd, sd = rule_sample('D', lam, np.array([0, 1, 2]))
    C.require(np.allclose(xd, [0., 0., 2.5]) and sd.tolist() == [0, 1, 2], f'D sample {xd}')
    xd2, _ = rule_sample('D_X2', lam, np.array([0, 1, 2]))
    C.require(np.allclose(xd2, [0., 0., 5.]), f'D_X2 sample {xd2}')
    xa, sa = rule_sample('AT_FRAME', lam, np.array([0, 1, 2]))
    C.require(len(xa) == 3 * K_C and np.allclose(xa[:K_C], 1. + cq - 3.5) and sa[:K_C].tolist() == [0] * K_C and sa[-1] == 2, 'AT_FRAME sample')
    xa2, _ = rule_sample('AT_FRAME_X2', lam, np.array([0, 1, 2]))
    C.require(np.allclose(xa2[-K_C:], 6.5 + cq - 3.5), 'AT_FRAME_X2 sample')
    w, lo, hi, e = weights_for(np.array([-4., -2.5, 0., 200.]), np.array([0, 0, 0, 0]), fallback=np.tile(CDD.hat_weights([0.], GRID_ALL)[0], (NSTRATA, 1)))
    exp = np.zeros(len(GRID_ALL))
    exp[0] += 1. + .5
    exp[1] += .5
    exp[GRID_ALL.index(0.)] += 1.
    exp[-1] += 1.
    C.require(np.allclose(w[0], exp / 4.) and lo == 1 and hi == 1 and e == NSTRATA - 1, f'weights_for {w[0]} {lo} {hi} {e}')
    C.require(abs(np.dot(CDD.hat_weights([-1.5], GRID_ALL)[0], GRID_ALL) + 1.5) < 1e-12, 'interpolation reproduces a negative x')
    C.require(close(1., 1. + 1e-12) and not close(1., 1.001) and close([1., 2], [1., 2]) and not close([1.], [1., 2.]) and close(None, None) and close(3, 3.0),
              'close')
    ms = CFB.measured()
    ent, ext = configs_neg(ms)
    C.require(len(ent) == len(GRID_NEG) * 30 and len(ext) == len(GRID_NEG) * 30 * 35, f'config counts {len(ent)} {len(ext)}')
    a = pd.DataFrame({'cut_minus_act_ms': [1., 2.], 't': [1., 2.], 'd': [1., -1.], 'mid': [5, 6], 'lim': [40., 50.], f'mv_up100_Q{I0}': [1., 2.], f'mv_n_post200_Q{I0}': [3., 4.]})
    b = a.rename(columns={f'mv_up100_Q{I0}': 'mv_up100', f'mv_n_post200_Q{I0}': 'mv_n_post200'})
    for s_ in (3, 10):
        a[f'D0Q{I0}_AB_s{s_}_fill'] = [1., 0.]
        b[f'D0_AB_s{s_}_fill'] = [1., 0.]
        a[f'D0Q{I0}_S0_AB_s{s_}_TAKE_C1000_pnl'] = [np.nan, 2.]
        b[f'D0_S0_AB_s{s_}_TAKE_C1000_pnl'] = [np.nan, 2.]
    C.require(parity_q0(a, b, 1, 1) == [], f'parity_q0 on equal frames: {parity_q0(a, b, 1, 1)}')
    b2 = b.copy()
    b2.loc[1, 'D0_S0_AB_s10_TAKE_C1000_pnl'] = 2.5
    C.require(parity_q0(a, b2, 1, 1) == [f'D0Q{I0}_S0_AB_s10_TAKE_C1000_pnl'], 'parity_q0 missed a changed value')
    C.require(parity_q0(a.iloc[:1], b, 1, 1) == ['rows differ'], 'parity_q0 missed a row count')
    Xs = pd.DataFrame(dict(a=np.array([1., 2.], np.float32), b=[3., 4.]))
    Z = hybrid(Xs, pd.DataFrame(dict(a=[9., 9.])), np.array([False, True]), ['a'])
    C.require(Z.a.tolist() == [1., 9.] and Z.a.dtype == np.float32 and Xs.a.tolist() == [1., 2.], 'hybrid')
    try:
        stage_exits('20261101')
        raise AssertionError('stage_exits accepted a consumed day')
    except ValueError as e_:
        assert 'not a declared forward day' in str(e_), e_
    global RESULT, TOKYO
    keep = RESULT, TOKYO
    with tempfile.NamedTemporaryFile(suffix='.json') as tf:
        RESULT = TOKYO = Path(tf.name)
        try:
            for fn in (stage_read, lambda: stage_tokyo('.')):
                try:
                    fn()
                    raise AssertionError('a stage overwrote its result')
                except ValueError as e_:
                    assert 'write-once' in str(e_), e_
        finally:
            RESULT, TOKYO = keep
    print('chase_decision_atframe selftest PASS')


if __name__ == '__main__':
    cmd, a_ = sys.argv[1], sys.argv[2:]
    dict(selftest=selftest, tokyo=lambda: stage_tokyo(a_[0]), read=stage_read, exits=lambda: stage_exits(a_[0]))[cmd]()
