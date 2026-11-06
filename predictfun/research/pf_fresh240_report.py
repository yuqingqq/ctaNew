"""Report of run_fresh240.sh: per-market feed refresh of the guard over the whole window vs the 50 ms grid, fixed live engine round i
(declared in STATUS 2026-11-06). Exploratory, IN-SAMPLE (guard fit on the 8 read days; 11-03 beside). Refreshed = preds_t60kF (every
Predict frame's known time in [ws, ws + 240 s) merged with the grid); grid = the invwin runs (preds_t60k), same arm names and engine.
Per arm and day: MM = mark@stop (mtm_stop_usd), +1 s $ (mo1_usd), settle (gross_usd), shares, +1 s c/sh; per minute of the window the
+1 s c/sh from the fills. Comparison refreshed minus grid: day-clustered mean over the 8 read days, t interval (7 df, 2.365), 11-03 beside;
ADDS = MM interval above 0, ADDS_BOTH = at central (J) and bound (X) timing. A missing run is a status, never a zero.
Usage: python3 -m predictfun.research.pf_fresh240_report [ctrl]
"""
import json, os, sys
import numpy as np, pandas as pd
from . import pf_io as io

S4 = f'{io.OUT}/step4'; ENG = f'{S4}/live_tokyoreal_custom_e2_'
D8 = ['20261023', '20261024', '20261025', '20261026', '20261027', '20261031', '20261101', '20261102']; BES = '20261103'; D9 = D8 + [BES]
PRE = 'MICRO+G4+G2+G6+G5+G3+G1+V0+GS+XNA+XMIN1+XV+'
SIZES = {'s5': 'AV:5:25:0.06', 's20': 'A1:20:100:0'}
WINS = ['S60', 'S120', 'S180', 'S240']; BATCH = {'S60': 'a', 'S120': 'a', 'S180': 'b', 'S240': 'b'}
KEYS = ('mtm_stop_usd', 'gross_usd', 'fills', 'shares', 'carry_stop_usd', 'mo1_usd', 'windows')
T7 = 2.365; NB = 4; BLAB = ['0-60', '60-120', '120-180', '180-240']


def stem(n):
    return n.replace('/', '__').replace('+', '_').replace('@', '_sz').replace(':', '-')


def arm(e, sz):
    return f'{PRE}{e}/mid@{SIZES[sz]}'


def paths(src, k, e, sz, d):
    if src == 'F':
        return f'{ENG}preds_t60kF_fw{k}/{d}'
    if src == 'E':
        return f'{ENG}preds_t60kE_fwEctrl/{d}'
    return f'{ENG}preds_t60k_iw{k}_{sz}{BATCH[e]}/{d}'


def load(src, k, e, sz, d):
    base = paths(src, k, e, sz, d); f = f'{base}/summary.json'
    if not os.path.exists(f):
        return None
    s = json.load(open(f))
    assert s['env'].get('PF_PREDS') == {'F': 'preds_t60kF', 'E': 'preds_t60kE', 'G': 'preds_t60k'}[src], (base, s['env'].get('PF_PREDS'))
    a = s['arms'].get(arm(e, sz))
    if a is None:
        return None
    fl = pd.read_parquet(f'{base}/fills/{stem(arm(e, sz))}.parquet', columns=['t', 'q', 'mo1'])
    b = np.clip((fl.t.values // 60).astype(int), 0, NB - 1)
    a = dict(a, _b={i: (float(fl.q.values[b == i].sum()), float(np.nansum(fl.mo1.values[b == i]))) for i in range(NB)}, _base=base,
             _env={k_: v_ for k_, v_ in s['env'].items() if k_ not in ('PF_PREDS', 'PF_ARMS', 'PF_TAG')}, _sha=s['code_sha256'])
    return a


def ci(x):
    x = np.asarray(x, float); m = x.mean(); h = T7 * x.std(ddof=1) / np.sqrt(len(x))
    return dict(mean=round(m, 2), lo=round(m - h, 2), hi=round(m + h, 2), excludes_0=bool(m - h > 0 or m + h < 0), above_0=bool(m - h > 0),
                days_pos=int((x > 0).sum()))


def same(a, b, e, sz):
    diff = {k: (a[k], b[k]) for k in KEYS if a[k] != b[k]}
    wa = pd.read_parquet(f"{a['_base']}/win/{stem(arm(e, sz))}.parquet"); wb = pd.read_parquet(f"{b['_base']}/win/{stem(arm(e, sz))}.parquet")
    return (not diff) and wa.reset_index(drop=True).equals(wb.reset_index(drop=True)), diff


def ctrl():
    bad = 0
    for d in ('20261024', '20261101'):
        for sz in SIZES:
            f, e_ = load('F', 'J', 'S60', sz, d), load('E', 'J', 'S60', sz, d)
            if f is None or e_ is None:
                print(d, sz, 'MISSING run'); bad += 1; continue
            ok, diff = same(f, e_, 'S60', sz); bad += int(not ok)
            print(d, sz, 'S60 refreshed-240 vs refreshed-65:', 'EQUAL ok' if ok else f'DIFF {diff} BAD')
            f2, g2 = load('F', 'J', 'S240', sz, d), load('G', 'J', 'S240', sz, d)
            if f2 is None or g2 is None:
                print(d, sz, 'MISSING run'); bad += 1; continue
            eq, diff = same(f2, g2, 'S240', sz); bad += int(eq)
            print(d, sz, 'S240 refreshed vs grid (must differ):', 'EQUAL BAD' if eq else f'DIFF ok {diff}')
    print('CONTROLS', 'PASS' if bad == 0 else f'FAIL ({bad})')
    return 0 if bad == 0 else 1


def main():
    if sys.argv[1:] == ['ctrl']:
        return ctrl()
    out = dict(note='exploratory, in-sample; fixed engine round i; refreshed (preds_t60kF) minus grid (preds_t60k); $ / day without rebate',
               arms={}, cmp={}, status={})
    for sz in SIZES:
        for e in WINS:
            key = f'{sz}/{e}'; out['cmp'][key] = {}
            for k in ('J', 'X'):
                R = {src: {d: load(src, k, e, sz, d) for d in D9} for src in ('F', 'G')}
                miss = [f'{src}/{d}' for src in R for d in D9 if R[src][d] is None]
                if miss:
                    out['status'][f'{k}/{key}'] = f'MISSING {miss}'; out['cmp'][key][k] = 'MISSING'; continue
                for d in D9:            # same engine code and settings apart from the prediction file, arms and tag
                    if R['F'][d]['_env'] != R['G'][d]['_env'] or R['F'][d]['_sha'] != R['G'][d]['_sha']:
                        sys.exit(f"REFUSED: {k}/{key}/{d}: refreshed and grid runs differ in env / code: {R['F'][d]['_env']} vs {R['G'][d]['_env']}")
                for src in ('F', 'G'):
                    g = lambda c: np.array([R[src][d][c] for d in D8])
                    sh = g('shares').sum()
                    pm = {BLAB[i]: (round(100.0 * sum(R[src][d]['_b'][i][1] for d in D8) / q, 3) if (q := sum(R[src][d]['_b'][i][0] for d in D8)) > 0 else None)
                          for i in range(NB)}
                    out['arms'][f'{k}/{key}/{src}'] = dict(mm=round(float(g('mtm_stop_usd').mean()), 1), mm_days_pos=int((g('mtm_stop_usd') > 0).sum()),
                                                           plus1=round(float(g('mo1_usd').mean()), 1), settle=round(float(g('gross_usd').mean()), 1),
                                                           shares_day=round(sh / 8.0), plus1_c_sh=round(100.0 * g('mo1_usd').sum() / sh, 3) if sh else None,
                                                           plus1_c_sh_by_minute=pm, mm_1103=round(R[src][BES]['mtm_stop_usd'], 1))
                dd = lambda c: [R['F'][d][c] - R['G'][d][c] for d in D8]
                cps = lambda src, d, i: 100.0 * R[src][d]['_b'][i][1] / R[src][d]['_b'][i][0] if R[src][d]['_b'][i][0] > 0 else np.nan
                bym = {}
                for i in range(NB):
                    x = [cps('F', d, i) - cps('G', d, i) for d in D8]
                    bym[BLAB[i]] = ci(x) if np.all(np.isfinite(x)) else 'NO FILLS ON SOME DAY'
                out['cmp'][key][k] = dict(mm=ci(dd('mtm_stop_usd')), plus1=ci(dd('mo1_usd')), settle=ci(dd('gross_usd')),
                                          shares_day=round(float(np.mean(dd('shares'))), 1),
                                          mm_1103=round(R['F'][BES]['mtm_stop_usd'] - R['G'][BES]['mtm_stop_usd'], 1), plus1_c_sh_by_minute=bym)
            v = out['cmp'][key]
            okj = isinstance(v.get('J'), dict) and v['J']['mm']['above_0']; okx = isinstance(v.get('X'), dict) and v['X']['mm']['above_0']
            v['ADDS_J'] = bool(okj); v['ADDS_X'] = bool(okx); v['ADDS_BOTH'] = bool(okj and okx)
    io.write_json(f'{S4}/report_fresh240.json', out)
    print('per arm, 8 read days, $ / day: MM (days > 0) | +1 s $ | settle | sh / day | +1 s c/sh, by minute 0-60 / 60-120 / 120-180 / 180-240 | 11-03 MM')
    for key, a in out['arms'].items():
        pm = ' / '.join('  -  ' if x is None else f'{x:+.2f}' for x in a['plus1_c_sh_by_minute'].values())
        print(f"{key:18s} MM {a['mm']:+7.1f} ({a['mm_days_pos']}/8) | +1s {a['plus1']:+7.1f} | settle {a['settle']:+7.1f} | {a['shares_day']:6d} sh | "
              f"{a['plus1_c_sh']:+.3f}: {pm} | {a['mm_1103']:+.1f}")
    print('\nrefreshed minus grid, day-clustered mean (t interval, 7 df) days > 0; 11-03 beside')
    for key, v in out['cmp'].items():
        for k in ('J', 'X'):
            c = v.get(k)
            if not isinstance(c, dict):
                print(f'{key:10s} {k}: MISSING'); continue
            bm = '; '.join(f"{b} {x['mean']:+.2f} ({x['lo']:+.2f}..{x['hi']:+.2f})" if isinstance(x, dict) else f'{b} {x}' for b, x in c['plus1_c_sh_by_minute'].items())
            print(f"{key:10s} {k}: MM {c['mm']['mean']:+.1f} ({c['mm']['lo']:+.1f}..{c['mm']['hi']:+.1f}) {c['mm']['days_pos']}/8 11-03 {c['mm_1103']:+.1f} | "
                  f"+1s $ {c['plus1']['mean']:+.1f} ({c['plus1']['lo']:+.1f}..{c['plus1']['hi']:+.1f}) | settle {c['settle']['mean']:+.1f} "
                  f"({c['settle']['lo']:+.1f}..{c['settle']['hi']:+.1f}) | sh/day {c['shares_day']:+.0f} | +1 s c/sh by minute: {bm}")
        print(f"{key:10s} ADDS_J {v['ADDS_J']} ADDS_X {v['ADDS_X']} ADDS_BOTH {v['ADDS_BOTH']}")
    print('statuses:', out['status'] or 'none')
    return 0


if __name__ == '__main__':
    sys.exit(main())
