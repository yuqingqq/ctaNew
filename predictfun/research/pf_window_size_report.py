"""Second-review stress runs and the window / size grid on the fixed pipeline (declared in STATUS.yml 2026-11-05 01:49Z).
Base: preds_h60k (rebuilt guard, start withhold), PF_TXTIME=ws, XV, V0, pull 60%. Money = mark@stop $ / day WITHOUT the rebate (rebate
UNVERIFIED, own column); +1 s c/sh without rebate; 8 read days, 11-03 apart.
(A) S60 C1-C4 under rbtxE (E2-only transactions 300 ms earlier) and rbtxS50 (all trades 50 ms earlier): R_E2 / R_S50 = 8-day sum > 0.
(B) stop 60 / 120 / 180 / 240 x AV:5:25:0.06 / AV:20:100:0.06 / AV:40:200:0.06 (rbw) and the same under PF_QACT=max + PF_TXSHIFT=-25 (rbwX).
    A longer stop or bigger size ADDS iff the day-clustered 95% CI (G = 8, t 2.365) of its delta excludes 0 AND its stress (rbwX) delta
    is > 0. Per 60 s segment of fill time against the top 5 in the same segment. Parity: rbw S60 AV:20:100 / AV:5:25 = rbtx C3 / C4.
Usage: python3 -m predictfun.research.pf_window_size_report
"""
import json
import numpy as np, pandas as pd
from . import pf_io as io

D8 = ['20261023', '20261024', '20261025', '20261026', '20261027', '20261031', '20261101', '20261102']; FR = ['20261103']
S4 = f'{io.OUT}/step4'; B = 'MICRO+G4+G2+G6+G5+G3+G1'; SEG = ((0, 60), (60, 120), (120, 180), (180, 240))
stem = lambda n: n.replace('/', '__').replace('+', '_').replace('@', '_sz').replace(':', '-')
arm = lambda s, sz: f'{B}+V0+XNA+XMIN1+XV+S{s}/mid@{sz}'
seg_of = lambda t: np.minimum((np.asarray(t) // 60).astype(int), 3)


def load(tag, s, sz, days):
    base = f'{S4}/live_tokyoreal_custom_e2_preds_h60k_{tag}'; a = arm(s, sz); mk, se, rb, net, rq, thr, F = [], [], [], [], [], 0, []
    for d in days:
        x = json.load(open(f'{base}/{d}/summary.json'))['arms'][a]
        mk.append(x['mtm_stop_usd']); se.append(x['gross_usd']); rb.append(x['rebate_usd_unverified']); net.append(x['abs_net_stop_mean'])
        thr += int(x['live'].get('counters', {}).get('throttled', 0) or 0)
        F.append(pd.read_parquet(f'{base}/{d}/fills/{stem(a)}.parquet', columns=['t', 'q', 'mo1']))
        R = pd.read_parquet(f'{base}/{d}/requests/{stem(a)}.parquet', columns=['t_ms', 'mid']).merge(pd.read_parquet(f'{base}/{d}/win/{stem(a)}.parquet', columns=['mid', 'ws']), on='mid')
        rel = (R.t_ms - R.ws) / 60_000.0; R = R[(rel >= 0) & (rel < s / 60.0)]
        rq.append(R.groupby([R.mid, (R.t_ms - R.ws) // 60_000]).size().values)
    f = pd.concat(F); rq = np.concatenate(rq) if rq else np.zeros(1)

    def q_(g):
        c = 100 * g.mo1 / g.q; Q = g.q.sum()
        return dict(shares_day=Q / len(days), mo1_c=100 * g.mo1.sum() / Q, bad=g.q[c <= -2].sum() / Q, ratio=-g.mo1[c <= -2].sum() / g.mo1[c >= 0].sum())
    r = q_(f); r.update(mark_day=mk, settle_day=se, mark=float(np.mean(mk)), settle=float(np.mean(se)), rebate=float(np.mean(rb)), net_stop=float(np.mean(net)),
                        req_p99=float(np.quantile(rq, .99)), thr_day=thr / len(days), worst=float(min(mk)),
                        loo=float(min(np.mean([v for j, v in enumerate(mk) if j != i]) for i in range(len(mk)))) if len(mk) > 1 else float('nan'),
                        seg={int(i): q_(g) for i, g in f.groupby(seg_of(f.t.values))})
    return r


def ci(dd):
    dd = np.asarray(dd, float); se = dd.std(ddof=1) / np.sqrt(len(dd))
    return dict(mean=float(dd.mean()), lo=float(dd.mean() - 2.365 * se), hi=float(dd.mean() + 2.365 * se), better=int((dd > 0).sum()))


top = json.load(open(f'{S4}/report_size_stop.json'))['market']; out, pred = {}, {}
H = f"{'':26s} {'sh/day':>7s} {'+1s c/sh':>8s} {'bad':>6s} {'ratio':>6s} {'mark':>7s} {'settle':>7s} {'rebate':>6s} {'days>0':>6s} {'LOO min':>7s} {'worst':>7s} {'|net|':>6s} {'req p99/min':>11s} {'thr/day':>7s}"
line = lambda lab, r, n: (f"{lab:26s} {r['shares_day']:7.0f} {r['mo1_c']:+8.3f} {r['bad']:6.1%} {r['ratio']:6.0%} {r['mark']:7.1f} {r['settle']:7.1f} {r['rebate']:6.1f} "
                          f"{sum(v > 0 for v in r['mark_day'])}/{n:<4d} {r['loo']:7.1f} {r['worst']:7.1f} {r['net_stop']:6.1f} {r['req_p99']:11.0f} {r['thr_day']:7.0f}")
print('(A) stress at S60, 8 read days ($ / day mark@stop, no rebate)')
print(H)
CAND = dict(C1='A1:5:25:0', C2='A1:20:100:0', C3='AV:20:100:0.06', C4='AV:5:25:0.06')
for c, sz in CAND.items():
    for tag in ('rbtx', 'rbtxE', 'rbtxS50'):
        r = load(tag, 60, sz, D8); out[f'A|{c}|{tag}'] = r; print(line(f'{c} {tag}', r, 8))
        f1 = load(tag, 60, sz, FR); out[f'A|{c}|{tag}|1103'] = f1
    pred[f'{c}_R_E2'] = bool(sum(out[f'A|{c}|rbtxE']['mark_day']) > 0); pred[f'{c}_R_S50'] = bool(sum(out[f'A|{c}|rbtxS50']['mark_day']) > 0)
    print(f"   11-03 mark: rbtx {out[f'A|{c}|rbtx|1103']['mark']:+.1f}, rbtxE {out[f'A|{c}|rbtxE|1103']['mark']:+.1f}, rbtxS50 {out[f'A|{c}|rbtxS50|1103']['mark']:+.1f}; "
          f"R_E2 {pred[f'{c}_R_E2']}, R_S50 {pred[f'{c}_R_S50']}")
SZ = ('AV:5:25:0.06', 'AV:20:100:0.06', 'AV:40:200:0.06'); ST = (60, 120, 180, 240)
par = []
for sz, c in (('AV:20:100:0.06', 'C3'), ('AV:5:25:0.06', 'C4')):
    a = load('rbw', 60, sz, D8); b = out[f'A|{c}|rbtx']; par.append(bool(np.allclose(a['mark_day'], b['mark_day'], atol=1e-6)))
print(f'\nparity rbw S60 = rbtx (C3, C4): {par}')
print('\n(B) window x size, 8 read days; then the stress variant (PF_QACT=max + trades 25 ms earlier)')
print(H)
for tag in ('rbw', 'rbwX'):
    for sz in SZ:
        for s in ST:
            r = load(tag, s, sz, D8); out[f'B|{tag}|{s}|{sz}'] = r; out[f'B|{tag}|{s}|{sz}|1103'] = load(tag, s, sz, FR)
            print(line(f"{tag} S{s} {sz}", r, 8))
    print()
print('longer stop vs S60 (same size) and bigger size (same stop): delta mark $ / day, 95% CI, days better | stress delta | 11-03 delta -> adds')
for sz in SZ:
    for s in ST[1:]:
        k0, k1 = f'B|rbw|60|{sz}', f'B|rbw|{s}|{sz}'
        c_ = ci(np.array(out[k1]['mark_day']) - out[k0]['mark_day']); dx = out[f'B|rbwX|{s}|{sz}']['mark'] - out[f'B|rbwX|60|{sz}']['mark']
        d3 = out[k1 + '|1103']['mark'] - out[k0 + '|1103']['mark']; adds = bool(c_['lo'] > 0 and dx > 0); pred[f'stop_S{s}_vs_S60_{sz}_adds'] = adds
        print(f"   {sz:15s} S{s} vs S60: {c_['mean']:+7.1f} ({c_['lo']:+.1f}..{c_['hi']:+.1f}) {c_['better']}/8 | stress {dx:+7.1f} | 11-03 {d3:+7.1f} -> adds {adds}")
for s in ST:
    for big, ref in (('AV:40:200:0.06', 'AV:20:100:0.06'), ('AV:20:100:0.06', 'AV:5:25:0.06')):
        k0, k1 = f'B|rbw|{s}|{ref}', f'B|rbw|{s}|{big}'
        c_ = ci(np.array(out[k1]['mark_day']) - out[k0]['mark_day']); dx = out[f'B|rbwX|{s}|{big}']['mark'] - out[f'B|rbwX|{s}|{ref}']['mark']
        d3 = out[k1 + '|1103']['mark'] - out[k0 + '|1103']['mark']; adds = bool(c_['lo'] > 0 and dx > 0); pred[f'size_{big}_vs_{ref}_S{s}_adds'] = adds
        print(f"   S{s} {big} vs {ref}: {c_['mean']:+7.1f} ({c_['lo']:+.1f}..{c_['hi']:+.1f}) {c_['better']}/8 | stress {dx:+7.1f} | 11-03 {d3:+7.1f} -> adds {adds}")
print('\nby segment of fill time (S240 arms, rbw) vs top 5 in the same segment: shares/day, +1 s c/sh, bad, ratio')
for sz in SZ:
    for i in range(4):
        r = out[f'B|rbw|240|{sz}']['seg'].get(i); m = top[f'top 5|{i}']
        if r is None:
            continue
        print(f"   {sz:15s} {SEG[i][0]:3d}-{SEG[i][1]:3d} s  {r['shares_day']:7.0f} {r['mo1_c']:+7.3f} {r['bad']:6.1%} {r['ratio']:5.0%}   top 5 {m['mo1_c']:+.3f} {m['bad']:.1%} {m['ratio']:.0%}")
io.write_json(f'{S4}/report_window_size.json', dict(results=out, predicates=pred, parity=par))
