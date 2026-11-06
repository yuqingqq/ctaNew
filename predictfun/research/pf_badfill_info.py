"""Can we avoid more bad fills with the information we have, or do we need more? (declared in STATUS.yml 2026-11-04 01:57Z,
part B). Descriptive, 8 read days, nothing selected.

Unit (rule 1/2: decision-time exposures on a neutral grid, no own orders): rows of rows_rl_d2 every 500 ms, live 0-240 s, known
mid; for each side s (bid +1 / ask -1) an order resting at the KNOWN best p (Tokyo clock, frames known per pf_clock_rl). If kept,
it is exposed from tau + 22 ms (our cancel would land then) to tau + 522 ms; it is swept when a market leg on that side trades
at a worse price than p in that interval (first such leg at t_sw). Value v = s (m(t_sw + 1 s) - p) c/sh if swept, else 0; bad =
swept and v <= -2.
Scores (lower = pull first), each pulling 5/10/20/30% of exposures per side per test day:
  base     s * pred_1000 of the current model (preds_o3 read days, preds_c3 11-01/02) - the current guard's input
  M1       LightGBM regression of v (clipped +-10) on the current features (pf_model_spec ALL), one model per side,
           pinned PARAMS / ROUNDS, leave one day out
  M2       M1 + outside feeds known at tau (pf_venue_feeds): Binance spot price move and aggressor flow (100 / 250 / 1000 ms,
           known +3.5 ms), Hyperliquid mid move and flow (same windows, +5 ms assumed), Chainlink relay vs the Binance perp mid
           (bp), relay age, relay move over 2 s
  ideal    s * y1000 (positive control: must remove most of the bad value); random (falsifier: ~ the pulled share of each)
Per day and score: bad $ avoided (share of the day's bad value), good $ lost, net = -(sum of v pulled) / |bad value|; AUC for
bad. Differences vs base with a day-clustered t interval (G = 8). Statuses: rows with no known book, no mid after a sweep.
Usage: PF_RL_DDEC=2 python3 -m predictfun.research.pf_badfill_info
"""
import os
import numpy as np, pandas as pd, lightgbm as lgb
from . import pf_io as io
from . import pf_clock_rl as CK
from . import pf_venue_feeds as VF
from .pf_model_spec import SETS, PARAMS, ROUNDS

assert CK.D_DEC == 2, 'run with PF_RL_DDEC=2 (rows_rl_d2)'
DAYS = ['20261023', '20261024', '20261025', '20261026', '20261027', '20261031', '20261101', '20261102']; FRESH = ('20261101', '20261102')
FE = SETS['ALL']; L_CXL, W = 22.0, 500.0; PULLS = (0.05, 0.10, 0.20, 0.30)
OUTD = f'{io.OUT}/step4/badfill_info'; os.makedirs(OUTD, exist_ok=True)
NEWF = [f'{f}_{k}{w}' for f in ('spot', 'hl') for k in ('mv', 'v') for w in (100, 250, 1000)] + ['cl_dev', 'cl_age', 'cl_mv2000']
st = dict(rows=0, no_book=0, no_mid_after=0)


def build(d):
    S1 = f'{io.OUT}/step1/{d}'; shift = CK.load_shift(S1)
    r = pd.read_parquet(f'{io.OUT}/step5/{d}/rows_rl_d2.parquet', columns=list(dict.fromkeys(['mid', 'tau', 't_rel', 'y1000'] + FE)))
    pr = pd.read_parquet(f"{io.OUT}/step5/{d}/{'preds_c3' if d in FRESH else 'preds_o3'}.parquet", columns=['pred_1000'])
    r['pred_1000'] = pr.pred_1000.values; del pr
    keep = ((r.t_rel < 240) & r.pf_mid.notna() & r.y1000.notna() & r.pred_1000.notna()).values & ((np.floor(r.tau.values / 50.0).astype(np.int64) % 10) == 0)
    r = r[keep].reset_index(drop=True); st['rows'] += len(r)
    bk = pd.concat([pd.read_parquet(f'{S1}/books/{f}', columns=['mid', 't_ms', 'recv_ns', 'b1_p', 'a1_p']) for f in sorted(os.listdir(f'{S1}/books'))])
    bk.sort_values(['mid', 't_ms', 'recv_ns'], inplace=True, kind='stable')
    lt = pd.read_parquet(f'{S1}/legs_timed2.parquet', columns=['mid', 'side', 'ptick', 't_ms', 'how', 't_e2'])
    lt = lt[(lt.how.values > 0) | np.isfinite(lt.t_e2.values)].copy(); lt['ta'] = np.where(lt.how.values > 0, lt.t_ms.values, lt.t_e2.values)
    LB = {k: (x.ta.values[np.argsort(x.ta.values, kind='stable')].astype(float), x.ptick.values[np.argsort(x.ta.values, kind='stable')]) for k, x in lt.groupby(['mid', 'side'], sort=False)}
    BKD = {m: g for m, g in bk.groupby('mid', sort=False)}; del bk, lt
    v = np.zeros((len(r), 2)); bad = np.zeros((len(r), 2), bool); ok = np.ones(len(r), bool)
    for mid, idx in r.groupby('mid', sort=False).indices.items():
        g = BKD.get(mid)
        if g is None:
            ok[idx] = False; st['no_book'] += len(idx); continue
        kf = np.maximum.accumulate(CK.pf_known(g.t_ms.values, g.recv_ns.values, shift)); b1 = g.b1_p.values; a1 = g.a1_p.values
        tf = g.t_ms.values.astype(float); vm = (b1 >= 0) & (a1 > b1); tv = tf[vm]; mv = ((b1 + a1) / 2.0)[vm]
        tau = r.tau.values[idx].astype(float); j = np.searchsorted(kf, tau, side='right') - 1
        good = (j >= 0); jj = np.clip(j, 0, None); pb = np.where(good, b1[jj], -1); pa = np.where(good, a1[jj], -1)
        good &= (pb >= 0) & (pa > pb); st['no_book'] += int((~good).sum()); ok[idx[~good]] = False
        for si, (s, p) in enumerate(((1.0, pb), (-1.0, pa))):
            L = LB.get((mid, si))
            if L is None:
                continue
            T, PT = L; i0 = np.searchsorted(T, tau + L_CXL, side='right'); i1 = np.searchsorted(T, tau + L_CXL + W, side='right')
            for n in np.flatnonzero(good & (i1 > i0)):
                seg = PT[i0[n]:i1[n]]; hit = np.flatnonzero(seg < p[n]) if s > 0 else np.flatnonzero(seg > p[n])
                if len(hit):
                    tsw = T[i0[n] + hit[0]]; k = np.searchsorted(tv, tsw + 1000.0, side='right') - 1
                    if k < 0:
                        st['no_mid_after'] += 1; continue
                    v[idx[n], si] = s * (mv[k] - p[n]); bad[idx[n], si] = v[idx[n], si] <= -2
    r['v_bid'] = v[:, 0]; r['v_ask'] = v[:, 1]; r['bad_bid'] = bad[:, 0]; r['bad_ask'] = bad[:, 1]
    r = r[ok].reset_index(drop=True)
    F = VF.load(d, DAYS); t = r.tau.values.astype(float)
    for f, lag in (('spot', 3.5), ('hl', 5.0)):
        for w in (100, 250, 1000):
            r[f'{f}_mv{w}'] = VF.move(F[f], t - lag, float(w)).astype(np.float32); r[f'{f}_v{w}'] = VF.vol(F[f + '_tr'], t - lag, float(w)).astype(np.float32)
    cl, cl_age = VF.last(F['cl'], t); bn, _ = VF.last(F['bnf'], t - 3.5)
    r['cl_dev'] = ((cl - bn) * 1e4).astype(np.float32); r['cl_age'] = cl_age.astype(np.float32); r['cl_mv2000'] = VF.move(F['cl'], t, 2000.0).astype(np.float32)
    r['day'] = d; del F
    io.mem_check(f'build {d}')
    return r


if os.environ.get('PF_BF_SMOKE'):
    x = build(os.environ['PF_BF_SMOKE'])
    print(f"smoke {os.environ['PF_BF_SMOKE']}: statuses {st}; rows {len(x)}; swept bid {np.mean(x.v_bid != 0):.1%} ask {np.mean(x.v_ask != 0):.1%}; bad bid {x.bad_bid.mean():.2%} "
          f"ask {x.bad_ask.mean():.2%}; mean v swept bid {x.v_bid[x.v_bid != 0].mean():+.2f} ask {x.v_ask[x.v_ask != 0].mean():+.2f}; NaN share new features "
          + str({c: round(float(x[c].isna().mean()), 3) for c in NEWF}))
    raise SystemExit(0)
R = pd.concat([build(d) for d in DAYS], ignore_index=True)
R[list(dict.fromkeys(['day', 'mid', 'tau', 't_rel', 'y1000', 'pred_1000', 'v_bid', 'v_ask', 'bad_bid', 'bad_ask'] + FE + NEWF))].to_parquet(f'{OUTD}/grid.parquet', index=False)
print(f'statuses {st}; exposures per side {len(R)}; swept share bid {np.mean(R.v_bid != 0):.1%} ask {np.mean(R.v_ask != 0):.1%}; '
      f'bad share bid {R.bad_bid.mean():.2%} ask {R.bad_ask.mean():.2%}; mean v bid {R.v_bid.mean():+.4f} ask {R.v_ask.mean():+.4f} c', flush=True)
RNG = np.random.default_rng(7)
for side, sg in (('bid', 1.0), ('ask', -1.0)):
    R[f'base_{side}'] = sg * R.pred_1000.values; R[f'ideal_{side}'] = sg * R.y1000.values; R[f'rand_{side}'] = RNG.random(len(R))
    for name, cols in (('M1', FE), ('M2', FE + NEWF)):
        pred = np.full(len(R), np.nan)
        for d in DAYS:
            tr = (R.day != d).values; te = ~tr
            ds = lgb.Dataset(R.loc[tr, cols].values.astype(np.float32), R.loc[tr, f'v_{side}'].clip(-10, 10).values, params=PARAMS, free_raw_data=True).construct()
            m = lgb.train(PARAMS, ds, ROUNDS); del ds
            pred[te] = m.predict(R.loc[te, cols].values.astype(np.float32), num_threads=1); del m
        R[f'{name}_{side}'] = pred
        io.mem_check(f'{name} {side}'); print(f'fitted {name} {side}', flush=True)


def auc(score, y):
    o = np.argsort(score, kind='stable'); rk = np.empty(len(score)); rk[o] = np.arange(1, len(score) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return float((n1 * (len(y) + 1) - rk[y].sum()) / (n1 * n0)) if n1 and n0 else np.nan   # P(bad ranked lower = pulled first)


SC = ['base', 'M1', 'M2', 'ideal', 'rand']; rows = []
for d in DAYS:
    x = R[R.day == d]
    for sc in SC:
        rec = dict(day=d, score=sc)
        vb = np.r_[x.v_bid.values, x.v_ask.values]; sco = np.r_[x[f'{sc}_bid'].values, x[f'{sc}_ask'].values]
        badv = vb[vb < 0].sum(); goodv = vb[vb > 0].sum(); isbad = np.r_[x.bad_bid.values, x.bad_ask.values]
        rec['auc_bad'] = auc(sco, isbad)
        for k in PULLS:
            pulled = np.zeros(len(vb), bool)
            for half in (slice(0, len(x)), slice(len(x), 2 * len(x))):                # per side
                s_ = sco[half]; n = int(round(k * len(s_))); pulled[np.arange(len(vb))[half][np.argsort(s_, kind='stable')[:n]]] = True
            rec[f'bad_avoided_{k}'] = float(vb[pulled & (vb < 0)].sum() / badv); rec[f'good_lost_{k}'] = float(vb[pulled & (vb > 0)].sum() / goodv)
            rec[f'net_{k}'] = float(-vb[pulled].sum() / -badv)
        rows.append(rec)
D = pd.DataFrame(rows); D.to_parquet(f'{OUTD}/by_day.parquet', index=False)
print('\nper score, mean over 8 days (share of the day\'s bad value avoided / good value lost / net gain as share of bad value), AUC for bad')
for k in PULLS:
    print(f'  pull {k:.0%} of exposures per side')
    for sc in SC:
        x = D[D.score == sc]
        line = f"    {sc:6s} bad avoided {x[f'bad_avoided_{k}'].mean():6.1%}  good lost {x[f'good_lost_{k}'].mean():6.1%}  net {x[f'net_{k}'].mean():+6.1%}"
        if sc in ('M1', 'M2'):
            dd = (x[f'net_{k}'].values - D[D.score == 'base'][f'net_{k}'].values); se = dd.std(ddof=1) / np.sqrt(len(dd))
            line += f"  | net vs base {dd.mean():+6.1%} (95% CI {dd.mean() - 2.365 * se:+.1%}..{dd.mean() + 2.365 * se:+.1%}, days better {int((dd > 0).sum())}/8)"
        if sc == 'M2':
            d2 = (x[f'net_{k}'].values - D[D.score == 'M1'][f'net_{k}'].values); se2 = d2.std(ddof=1) / np.sqrt(len(d2))
            line += f"  | vs M1 {d2.mean():+6.1%} ({d2.mean() - 2.365 * se2:+.1%}..{d2.mean() + 2.365 * se2:+.1%}, {int((d2 > 0).sum())}/8)"
        print(line)
print('AUC for bad (higher = bad exposures ranked to be pulled first): ' + ', '.join(f"{sc} {D[D.score == sc].auc_bad.mean():.3f}" for sc in SC))
cur = float(np.mean(np.r_[R.base_bid.values <= -0.25, R.base_ask.values <= -0.25]))
print(f'reference: the current M025 guard pulls {cur:.1%} of exposures (prediction 0.25 c or more against the side)')
io.write_json(f'{OUTD}/report.json', dict(statuses=st, current_guard_pull_share=cur, means={sc: D[D.score == sc].drop(columns=['day', 'score']).mean().round(4).to_dict() for sc in SC}))
