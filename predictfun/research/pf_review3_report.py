"""Rebuild 3 report (declared in STATUS.yml 2026-11-05 03:01Z). Real taker-order key (pf_txmap), guard refit on its labels (preds_t60k),
tx ws, XV, V0, pull 60%, Tokyo clock. Variants: kJ central (match time uniform within its frame), kX bound (one full frame early +
pessimistic queue), kS frame stamps (old timing, real key), kE (kJ + E2-only transactions 300 ms earlier), kM (earliest leg of either
kind + jitter). Money = mark@stop $ / day WITHOUT the rebate (rebate UNVERIFIED, own column); 8 read days; 11-03 beside, not a test.
Predicates per candidate (S60): P_DAYS (kJ > 0 on >= 6/8 days and every leave-one-day-out mean > 0), P_BOUND (kX 8-day sum > 0),
P_E2 (kE sum > 0), P_MIN (kM sum > 0), P_TOP5 (kJ +1 s c/sh >= top 5 in 0-60 s), ALL. Longer stop / bigger size ADDS iff the
day-clustered 95% CI (G = 8, t 2.365) of its kJ delta excludes 0 AND its kX delta > 0. All on the selection days: screens, not tests.
Usage: python3 -m predictfun.research.pf_review3_report
"""
import json
import numpy as np, pandas as pd
from . import pf_io as io

D8 = ['20261023', '20261024', '20261025', '20261026', '20261027', '20261031', '20261101', '20261102']; FR = ['20261103']
S4 = f'{io.OUT}/step4'; B = 'MICRO+G4+G2+G6+G5+G3+G1'; SEG = ((0, 60), (60, 120), (120, 180), (180, 240))
stem = lambda n: n.replace('/', '__').replace('+', '_').replace('@', '_sz').replace(':', '-')
arm = lambda s, sz: f'{B}+V0+XNA+XMIN1+XV+S{s}/mid@{sz}'
seg_of = lambda t: np.minimum((np.asarray(t) // 60).astype(int), 3)
DIR = lambda tag: f'{S4}/live_tokyoreal_custom_e2_preds_t60k_{tag}' if tag[0] == 'k' else f'{S4}/live_tokyoreal_custom_e2_preds_h60k_{tag}'
TXS = {}


def load(tag, s, sz, days):
    a = arm(s, sz); mk, se, rb, rq, thr, F = [], [], [], [], 0, []
    for d in days:
        j = json.load(open(f'{DIR(tag)}/{d}/summary.json')); x = j['arms'][a]
        if tag[0] == 'k':
            assert j['env'].get('PF_TXKEY') == 'tk' and j['env'].get('PF_PREDS') == 'preds_t60k', (tag, d, j['env'])
            TXS[(tag, d)] = dict(j['tx_timing'], input_bounds=j['input_bounds'])
        mk.append(x['mtm_stop_usd']); se.append(x['gross_usd']); rb.append(x['rebate_usd_unverified'])
        thr += int(x['live'].get('counters', {}).get('throttled', 0) or 0)
        F.append(pd.read_parquet(f'{DIR(tag)}/{d}/fills/{stem(a)}.parquet', columns=['t', 'q', 'mo1']))
        R = pd.read_parquet(f'{DIR(tag)}/{d}/requests/{stem(a)}.parquet', columns=['t_ms', 'mid']).merge(
            pd.read_parquet(f'{DIR(tag)}/{d}/win/{stem(a)}.parquet', columns=['mid', 'ws']), on='mid')
        rel = (R.t_ms - R.ws) / 1000.0; R = R[(rel >= 0) & (rel < s)]
        rq.append(R.groupby([R.mid, (R.t_ms - R.ws) // 60_000]).size().values)
    f = pd.concat(F); rq = np.concatenate(rq) if rq else np.zeros(1)

    def q_(g):
        c = 100 * g.mo1 / g.q; Q = g.q.sum()
        return dict(shares_day=float(Q / len(days)), mo1_c=float(100 * g.mo1.sum() / Q), bad=float(g.q[c <= -2].sum() / Q),
                    ratio=float(-g.mo1[c <= -2].sum() / g.mo1[c >= 0].sum()))
    r = q_(f)
    r.update(mark_day=mk, settle_day=se, mark=float(np.mean(mk)), settle=float(np.mean(se)), rebate=float(np.mean(rb)),
             req_p99=float(np.quantile(rq, .99)), thr_day=thr / len(days), worst=float(min(mk)), days_pos=int(sum(v > 0 for v in mk)),
             loo=float(min(np.mean([v for j_, v in enumerate(mk) if j_ != i]) for i in range(len(mk)))) if len(mk) > 1 else float('nan'),
             seg={int(i): q_(g) for i, g in f.groupby(seg_of(f.t.values))})
    return r


def ci(dd):
    dd = np.asarray(dd, float); se = dd.std(ddof=1) / np.sqrt(len(dd))
    return dict(mean=float(dd.mean()), lo=float(dd.mean() - 2.365 * se), hi=float(dd.mean() + 2.365 * se), better=int((dd > 0).sum()))


top = json.load(open(f'{S4}/report_size_stop.json'))['market']; TOP5 = top['top 5|0']['mo1_c']; out, pred = {}, {}
H = (f"{'':24s} {'sh/day':>7s} {'+1s c/sh':>8s} {'bad':>6s} {'ratio':>6s} {'mark':>7s} {'settle':>7s} {'rebate':>6s} {'days>0':>6s} "
     f"{'LOO min':>7s} {'worst':>7s} {'req p99':>7s} {'thr/day':>7s} {'11-03':>7s}")
line = lambda lab, r, r3: (f"{lab:24s} {r['shares_day']:7.0f} {r['mo1_c']:+8.3f} {r['bad']:6.1%} {r['ratio']:6.0%} {r['mark']:7.1f} {r['settle']:7.1f} "
                           f"{r['rebate']:6.1f} {r['days_pos']}/8    {r['loo']:7.1f} {r['worst']:7.1f} {r['req_p99']:7.0f} {r['thr_day']:7.0f} {r3['mark']:+7.1f}")
CAND = dict(C1='A1:5:25:0', C2='A1:20:100:0', C3='AV:20:100:0.06', C4='AV:5:25:0.06')
print(f'$ / day mark@stop WITHOUT rebate, 8 read days (11-03 last column, not a test); top 5 +1 s in 0-60 s {TOP5:+.3f} c/sh')
print('(A) candidates at S60'); print(H)
for c, sz in CAND.items():
    for tag in ('rbtx', 'kS', 'kJ', 'kX', 'kE', 'kM'):
        out[f'A|{c}|{tag}'] = load(tag, 60, sz, D8); out[f'A|{c}|{tag}|1103'] = load(tag, 60, sz, FR)
        print(line(f'{c} {tag}', out[f'A|{c}|{tag}'], out[f'A|{c}|{tag}|1103']))
    J = out[f'A|{c}|kJ']
    pred[c] = dict(P_DAYS=bool(J['days_pos'] >= 6 and J['loo'] > 0), P_BOUND=bool(sum(out[f'A|{c}|kX']['mark_day']) > 0),
                   P_E2=bool(sum(out[f'A|{c}|kE']['mark_day']) > 0), P_MIN=bool(sum(out[f'A|{c}|kM']['mark_day']) > 0), P_TOP5=bool(J['mo1_c'] >= TOP5))
    pred[c]['ALL'] = all(pred[c].values())
    kd = ci(np.array(out[f'A|{c}|kS']['mark_day']) - out[f'A|{c}|rbtx']['mark_day']); jd = ci(np.array(J['mark_day']) - out[f'A|{c}|kS']['mark_day'])
    out[f'A|{c}|key_effect'] = kd; out[f'A|{c}|jitter_effect'] = jd
    print(f"   {c}: per day kJ {[round(v) for v in J['mark_day']]}; key + refit effect (kS - rbtx) {kd['mean']:+.1f} ({kd['lo']:+.1f}..{kd['hi']:+.1f}); "
          f"jitter effect (kJ - kS) {jd['mean']:+.1f} ({jd['lo']:+.1f}..{jd['hi']:+.1f}); predicates {pred[c]}")
SZ = ('AV:5:25:0.06', 'AV:20:100:0.06', 'AV:40:200:0.06'); ST = (60, 120, 180, 240)
print('\n(B) window x size (kJ central, then kX bound)'); print(H)
for tag in ('kJ', 'kX'):
    for sz in SZ:
        for s in ST:
            out[f'B|{tag}|{s}|{sz}'] = load(tag, s, sz, D8); out[f'B|{tag}|{s}|{sz}|1103'] = load(tag, s, sz, FR)
            print(line(f'{tag} S{s} {sz}', out[f'B|{tag}|{s}|{sz}'], out[f'B|{tag}|{s}|{sz}|1103']))
    print()
print('delta mark $ / day (kJ), 95% CI, days better | kX delta | 11-03 kJ delta -> adds')
for sz in SZ:
    for s in ST[1:]:
        k0, k1 = f'B|kJ|60|{sz}', f'B|kJ|{s}|{sz}'
        c_ = ci(np.array(out[k1]['mark_day']) - out[k0]['mark_day']); dx = out[f'B|kX|{s}|{sz}']['mark'] - out[f'B|kX|60|{sz}']['mark']
        d3 = out[k1 + '|1103']['mark'] - out[k0 + '|1103']['mark']; adds = bool(c_['lo'] > 0 and dx > 0); pred[f'stop_S{s}_vs_S60_{sz}_adds'] = adds
        print(f"   {sz:15s} S{s} vs S60: {c_['mean']:+7.1f} ({c_['lo']:+.1f}..{c_['hi']:+.1f}) {c_['better']}/8 | {dx:+7.1f} | {d3:+7.1f} -> adds {adds}")
for s in ST:
    for big, ref in (('AV:20:100:0.06', 'AV:5:25:0.06'), ('AV:40:200:0.06', 'AV:20:100:0.06')):
        k0, k1 = f'B|kJ|{s}|{ref}', f'B|kJ|{s}|{big}'
        c_ = ci(np.array(out[k1]['mark_day']) - out[k0]['mark_day']); dx = out[f'B|kX|{s}|{big}']['mark'] - out[f'B|kX|{s}|{ref}']['mark']
        d3 = out[k1 + '|1103']['mark'] - out[k0 + '|1103']['mark']; adds = bool(c_['lo'] > 0 and dx > 0); pred[f'size_{big}_vs_{ref}_S{s}_adds'] = adds
        print(f"   S{s} {big} vs {ref}: {c_['mean']:+7.1f} ({c_['lo']:+.1f}..{c_['hi']:+.1f}) {c_['better']}/8 | {dx:+7.1f} | {d3:+7.1f} -> adds {adds}")
print('\nby 60 s segment of fill time (kJ, S240 arms) vs top 5 in the same segment: shares/day, +1 s c/sh, bad, ratio')
for sz in SZ:
    for i in range(4):
        r = out[f'B|kJ|240|{sz}']['seg'].get(i); m = top[f'top 5|{i}']
        if r is not None:
            print(f"   {sz:15s} {SEG[i][0]:3d}-{SEG[i][1]:3d} s {r['shares_day']:7.0f} {r['mo1_c']:+7.3f} {r['bad']:6.1%} {r['ratio']:5.0%}   top 5 {m['mo1_c']:+.3f} {m['bad']:.1%} {m['ratio']:.0%}")
print('\ntiming instruments per day (kJ): tk statuses (legs), groups old key -> taker orders, jitter mean ms, book frames dropped (input bounds)')
for d in D8 + FR:
    t = TXS[('kJ', d)]
    print(f"   {d} {t['tk_status_legs']} {t['groups_old_key']} -> {t['groups_tk']}; before {t['tx_multi_time_before']}; jitter {t['jitter_mean_ms']:.1f} "
          f"(zero-gap legs {t['jitter_zero_gap_legs']}); dropped frames {t['input_bounds']['book_frames_dropped']}")
io.write_json(f'{S4}/report_review3.json', dict(results=out, predicates=pred, top5_mo1_c=TOP5, tx={f'{k[0]}|{k[1]}': v for k, v in TXS.items()}))
