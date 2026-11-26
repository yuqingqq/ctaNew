"""A predictor of the take's own outcome on Predict (USER 2026-11-26: "can we train a predictor to estimate 150ms later price based on the
factors we have found so far, then we use the predictor to fire on predict, or we use predict price as the target based on our factors").
Development evidence on the four consumed days (all seen): leave-one-day-out, in-sample in the sense of rule 11; no strategy freeze.
Target (declared before the run): the value of a 50-share marketable IOC decided at the moment (decision g = moment + 1.5 ms feed + 2 ms,
the score_replay convention), limited at the Predict best known at g (the harness's known clock), arriving after the network and matched
after the 150 ms hold against the venue book at act (frame clock), filled up to 50 shares across the shown levels at or better than the
limit (+ k c for the pay-through variant), valued 1 s after act (take-marks mid) after the 2% x min(p, 100 - p) fee; no fill = 0; c per
intended share. Variants: total latency T = 152 / 162 / 172 ms (network 2 / 12 / 22 ms), k = 0 (k = 3 at T = 172). The fill rule is
simpler than the harness (no queue or public-trade debit): checked against the harness's own outcomes for score_replay's decisions (same
limit, same act). Also recorded: the Predict mid's move over the same span (unconditional) and the value at the limit if filled.
Population: public_score3's random moments (both directions) inside the complete markets' recorded paths with a valid known book and mid
1 s after act; statuses counted.
Features: public_score3's state and v3 factors (the held-out scores excluded), Predict's book at the decision (mid, spread, best and 5-level
sizes on the side taken and the other, the mid's move its way over 1 s, distance from 50), NORM_WINDOWS' raw rates and the imbalance against
its last 30 s. Model: gradient-boosted trees with public_score's pinned settings, leave-one-day-out.
Readings (train): per held-out day and pooled, the realised target among the moments ranked top 1 / 0.3 / 0.1 % by the prediction, the
share predicted > 0 and their realised value, Spearman between prediction and realised; the same rankings by public_score3's held-out
Binance-move score and at random; per variant. A harness replay of the model's top moments follows only if the realised top-1 % value is
positive on the simplified target.
Outputs (write-once): OUT/take_model/<day>_labels.parquet (stage labels), take_model.json + OUT/take_model/predictions.parquet (train).
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/take_model.py \
       selftest | labels <day> [<dir> <n markets>] | train [<dir>]
"""
import collections
import gzip
import json
import resource
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import score_replay as SRy  # noqa: E402
import public_score3 as P3  # noqa: E402

VR, MM, TR, C, F = SRy.VR, SRy.MM, SRy.TR, SRy.C, SRy.F
WT = P3.WT
OUT = WT.OUT / 'take_model'
DAYS = SRy.DAYS
FEED, DECIDE, HOLD, Q = 1.5, 2.0, 150., 50.
NETS = (2., 12., 22.)
PAY_K = 3.
QTOP = (.01, .003, .001)
HGB = dict(max_iter=200, learning_rate=.05, max_leaf_nodes=15, min_samples_leaf=200, l2_regularization=1.)
EXCL = {'day', 't', 'd', 'kind', 'y', 'next_ms', 's_logit_v2', 's_gbm_v2', 's_logit_v3', 's_gbm_v3'}
NW_COLS = ['far_new|raw', 'far_cxl|raw', 'far_trade|raw', 'near_new|raw', 'near_cxl|raw', 'near_trade|raw', 'activity|raw', 'imbalance|z_30s']


def fee(p):
    return .02 * np.minimum(p, 100. - p)


def walk(row, side, d, lim, q):
    """Fill up to q shares on book side (0 bid, 1 ask) of one frame row at prices at or better than lim -> (filled, vwap)."""
    b = 2 + 10 * side
    got, cost = 0., 0.
    for k in range(5):
        p, s = row[b + k], row[b + 5 + k]
        if p < 0 or s <= 0:
            break
        if (d > 0 and p > lim + 1e-9) or (d < 0 and p < lim - 1e-9):
            break
        x = min(q - got, s)
        got += x; cost += x * p
        if got >= q - 1e-9:
            break
    return got, (cost / got if got > 0 else np.nan)


def mid_of(row):
    b, a = row[2], row[12]
    return (b + a) / 2. if (0 <= b < a <= 100) else np.nan


def labels(day, od=None, limit=None):
    resource.setrlimit(resource.RLIMIT_CPU, (1500, 1501))
    C.require(day in DAYS, f'{day} is not a declared day')
    od = Path(od) if od else OUT
    fp = od / f'{day}_labels.parquet'
    C.require(not fp.exists(), f'{fp.name} exists (write-once)')
    Z = pd.read_parquet(WT.OUT / 'public_score3_points.parquet', columns=['day', 't', 'd', 'kind'])
    Z['row'] = np.arange(len(Z))
    Z = Z[(Z.day == day) & (Z.kind == 'random')].reset_index(drop=True)
    vs = json.loads((VR.OUT / 'summary.json').read_text())
    comp = {int(r['mid']) for r in vs['rows'] if r['day'] == day and r['status'] == 'complete'}
    reg = MM.registry(day).set_index('mid')
    ws_to_mid = {int(float(reg.loc[m, 'ws'])): int(m) for m in reg.index}
    Z['mid'] = (np.floor(Z.t / 300000.) * 300).astype(int).map(ws_to_mid)
    Z = Z[Z.mid.isin(comp)].reset_index(drop=True)
    by_mid = Z.groupby('mid').indices
    rows, status, parity = [], collections.Counter(), collections.Counter()
    src = MM.KP.Tape(); off = src.offset(day); n_m = 0
    for h, g in reg.groupby('hour'):
        mids = [int(m) for m in g.index if int(m) in by_mid]
        if limit is not None:
            mids = mids[:max(0, limit - n_m)]
        if not mids:
            continue
        for m, W, lst in VR.worlds_all(day, mids, src, off, reg):
            C.require(lst == 'loaded', f'{m} {lst}')
            ws = float(reg.loc[m, 'ws']) * 1000.
            p, st, info = VR.vm_path(W, ws, 50.)
            C.require(st == 'complete', f'{m} {st}')
            n_m += 1
            t0, end = float(info['t_start']), float(info['end_rel'])
            f = np.asarray(p['frames'], float)
            for i in by_mid[m]:
                t, d = float(Z.t.iat[i]), float(Z.d.iat[i])
                g_ = t + FEED + DECIDE - t0
                if g_ < 1000. or g_ + max(NETS) + HOLD + 1000. > end:
                    status['outside_path'] += 1
                    continue
                jk = int(np.searchsorted(f[:, 1], g_, 'right') - 1)
                j1 = int(np.searchsorted(f[:, 1], g_ - 1000., 'right') - 1)
                if jk < 0 or j1 < 0:
                    status['no_known_frame'] += 1
                    continue
                rk = f[jk]
                bid, ask = rk[2], rk[12]
                if not (0 <= bid < ask <= 100):
                    status['invalid_known_book'] += 1
                    continue
                side = 1 if d > 0 else 0
                lim = ask if d > 0 else bid
                md = (bid + ask) / 2.
                m1 = mid_of(f[j1])
                bs = np.where(rk[2:7] >= 0, rk[7:12], 0.); as_ = np.where(rk[12:17] >= 0, rk[17:22], 0.)
                r = dict(row=int(Z.row.iat[i]), day=day, mid=m, t=t, d=d, pf_mid=md, pf_spread=ask - bid, pf_far1=as_[0] if d > 0 else bs[0], pf_near1=bs[0] if d > 0 else as_[0],
                         pf_far5=as_.sum() if d > 0 else bs.sum(), pf_near5=bs.sum() if d > 0 else as_.sum(), pf_mid_ret_1000=d * (md - m1) if np.isfinite(m1) else np.nan,
                         pf_extreme=abs(md - 50.), pf_its_price=md if d > 0 else 100. - md)
                ok = True
                for net in NETS:
                    act = g_ + net + HOLD
                    jv = int(np.searchsorted(f[:, 0], act, 'right') - 1)
                    ja = int(np.searchsorted(f[:, 0], act + 1000., 'right') - 1)
                    if jv < 0 or ja < 0 or (act + 1000.) - f[ja, 0] > 1000.:
                        ok = False; break
                    ma = mid_of(f[ja])
                    if not np.isfinite(ma):
                        ok = False; break
                    T = int(net + HOLD)
                    for k in ((0., PAY_K) if net == max(NETS) else (0.,)):
                        q, vw = walk(f[jv], side, d, lim + (k if d > 0 else -k), Q)
                        r[f'v_{T}_{int(k)}'] = q * (d * (ma - vw) - fee(vw)) / Q if q > 0 else 0.
                        r[f'fill_{T}_{int(k)}'] = q / Q
                    r[f'mv_{T}'] = d * (ma - md)
                    r[f'wb_{T}'] = d * (ma - lim) - fee(lim)
                if not ok:
                    status['no_mid_after'] += 1
                    continue
                status['ok'] += 1
                rows.append(r)
            # parity with the harness: score_replay's decisions, same limit and act
            for cfg in SRy.CONFIGS:
                rec = json.loads(gzip.decompress((SRy.OUT / f'{m}_{cfg}.json.gz').read_bytes()))
                for x in rec['ioc_results']:
                    s_ = int(x['side']); dd = 1. if s_ == 0 else -1.
                    jv = int(np.searchsorted(f[:, 0], float(x['act']), 'right') - 1)
                    if jv < 0:
                        continue
                    q, vw = walk(f[jv], 1 - s_, dd, float(x['limit']), float(x['q']))
                    hf = float(x['filled']) > 1e-9
                    parity['checked'] += 1
                    parity['same_fill_status'] += int((q > 1e-9) == hf)
                    parity['we_fill_harness_not'] += int(q > 1e-9 and not hf)
                    parity['harness_fills_we_not'] += int(hf and q <= 1e-9)
                    if hf and q > 1e-9:
                        parity['both_filled'] += 1
                        parity['same_price'] += int(abs(vw - float(x['price'])) < .51)
            C.require(F.rss_mib() < F.RSS_GUARD_MIB, 'RSS over the self-guard')
    L = pd.DataFrame(rows)
    od.mkdir(parents=True, exist_ok=True)
    L.to_parquet(fp, index=False)
    TR.dump(od / f'{day}_labels.json', dict(asof='2026-11-26', day=day, markets=n_m, status=dict(status), parity=dict(parity)))
    print(day, n_m, dict(status), dict(parity), flush=True)


def topk_eval(pred, real, fill, q):
    o = np.argsort(-pred, kind='stable')
    k = max(1, int(round(q * len(pred))))
    s = o[:k]
    return dict(n=int(k), realised_c=float(np.mean(real[s])), fill=float(np.mean(fill[s])), pred_mean=float(np.mean(pred[s])))


def train(od=None):
    resource.setrlimit(resource.RLIMIT_CPU, (1800, 1801))
    od = Path(od) if od else None
    fn = (od or H) / 'take_model.json'
    fpp = (od or OUT) / 'predictions.parquet'
    C.require(not fn.exists() and not fpp.exists(), 'an output exists (write-once)')
    L = pd.concat([pd.read_parquet(OUT / f'{d}_labels.parquet') for d in DAYS], ignore_index=True)
    Pz = pd.read_parquet(WT.OUT / 'public_score3_points.parquet')
    Pz['row'] = np.arange(len(Pz))
    Pz = Pz[Pz.row.isin(set(L.row))]
    NWp = pd.concat([pd.read_parquet(WT.OUT / 'norm_windows' / f'{d}.parquet', columns=['day', 't', 'd', 'kind'] + NW_COLS) for d in DAYS], ignore_index=True)
    NWp = NWp[NWp.kind == 'random'].drop_duplicates(['day', 't', 'd'])
    X = L.merge(Pz, on=['row'], how='inner', suffixes=('', '_p'))
    C.require(len(X) == len(L) and np.allclose(X.t.to_numpy(), X.t_p.to_numpy()), 'labels and features do not align')
    X = X.merge(NWp.drop(columns=['kind']), on=['day', 't', 'd'], how='left')
    feats = [c for c in Pz.columns if c not in EXCL and c != 'row'] + [c for c in L.columns if c.startswith('pf_')] + NW_COLS
    targets = [c for c in L.columns if c.startswith('v_')]
    out = dict(asof='2026-11-26', rows=int(len(X)), features=feats, targets={}, base={})
    P = X[['row', 'day', 't', 'd', 'mid', 's_gbm_v3'] + targets + [c for c in L.columns if c.startswith('fill_') or c.startswith('mv_') or c.startswith('wb_')]].copy()
    for tg in targets:
        fl = tg.replace('v_', 'fill_')
        pred = np.full(len(X), np.nan)
        for dh in DAYS:
            tr_, te = X.day != dh, X.day == dh
            if te.sum() == 0:
                continue
            mdl = HistGradientBoostingRegressor(random_state=0, **HGB).fit(X.loc[tr_, feats], X.loc[tr_, tg])
            pred[te.to_numpy()] = mdl.predict(X.loc[te, feats])
        P[f'pred_{tg}'] = pred
        real, fill = X[tg].to_numpy(float), X[fl].to_numpy(float)
        rng = np.random.default_rng(20261126)
        o = dict(all=dict(n=int(len(real)), realised_c=float(real.mean()), fill=float(fill.mean())), model={}, binance_score={}, random={}, by_day={})
        for q in QTOP:
            o['model'][str(q)] = topk_eval(pred, real, fill, q)
            o['binance_score'][str(q)] = topk_eval(X.s_gbm_v3.to_numpy(float), real, fill, q)
            o['random'][str(q)] = topk_eval(rng.random(len(real)), real, fill, q)
        pos = pred > 0
        o['predicted_positive'] = dict(share=float(pos.mean()), n=int(pos.sum()), realised_c=float(real[pos].mean()) if pos.any() else None)
        o['spearman'] = float(spearmanr(pred, real).correlation)
        for dh in DAYS:
            m_ = (X.day == dh).to_numpy()
            if m_.sum() > 200:
                o['by_day'][dh] = dict(n=int(m_.sum()), top1=topk_eval(pred[m_], real[m_], fill[m_], .01), all_c=float(real[m_].mean()), spearman=float(spearmanr(pred[m_], real[m_]).correlation))
        out['targets'][tg] = o
    for c in [c for c in L.columns if c.startswith('mv_') or c.startswith('wb_')]:
        out['base'][c] = float(X[c].mean())
    P.to_parquet(fpp, index=False)
    TR.dump(fn, out)
    for tg, o in out['targets'].items():
        print(tg, 'all', round(o['all']['realised_c'], 3), 'model top1', {k: round(v, 3) for k, v in o['model']['0.01'].items()}, 'top0.1', round(o['model']['0.001']['realised_c'], 3),
              'binance top1', round(o['binance_score']['0.01']['realised_c'], 3), 'pos', o['predicted_positive'], 'rho', round(o['spearman'], 3))


def selftest():
    row = np.full(22, -1.); row[12:15] = [55., 56., 58.]; row[17:20] = [20., 20., 50.]; row[2] = 54.; row[7] = 10.
    q, vw = walk(row, 1, 1., 55., 50.)
    assert q == 20. and vw == 55., (q, vw)
    q, vw = walk(row, 1, 1., 58., 50.)
    assert q == 50. and abs(vw - (20 * 55 + 20 * 56 + 10 * 58) / 50.) < 1e-12, (q, vw)
    q, vw = walk(row, 0, -1., 54., 50.)
    assert q == 10. and vw == 54.
    assert mid_of(row) == 54.5
    print('take_model selftest PASS')


if __name__ == '__main__':
    cmd, a = sys.argv[1], sys.argv[2:]
    if cmd == 'labels':
        labels(a[0], a[1] if len(a) > 1 else None, int(a[2]) if len(a) > 2 else None)
    elif cmd == 'train':
        train(a[0] if a else None)
    else:
        dict(selftest=selftest)[cmd]()
