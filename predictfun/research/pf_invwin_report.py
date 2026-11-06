"""Report of run_invwin.sh: inventory reduction x quoting window on the FIXED live engine (round i, 2026-11-05; declared in STATUS
17:3xZ). Exploratory, IN-SAMPLE (V0 guard fit on the same 8 read days; 11-03 beside). Money = mark@stop $ / day (P&L marked at the
venue mid at the arm's own stop + 100 ms: spread minus adverse moves, no outcome) and settle $ / day (to the outcome = mark + carry),
no rebate. Per arm: 8-day mean, days > 0, 11-03, carry sd per window, mean |inventory| at stop, shares / day, +1 s c/sh, and the
withheld counts of its CAP / RO tokens (an arm whose token never engaged is REFUSED).
Comparisons (declared before the run): day-clustered mean difference over the 8 read days with a t interval (7 df, 2.365) on
mark@stop and on settle; 11-03 difference beside. ADDS = interval of the mark@stop difference above 0, evaluated per timing;
ADDS_BOTH = ADDS at central (J) AND bound (X) timing. A missing run is a status, never a zero.
Usage: python3 -m predictfun.research.pf_invwin_report [ctrl]
"""
import json, os, sys
import numpy as np, pandas as pd
from . import pf_io as io

S4 = f'{io.OUT}/step4'; RUN = f'{S4}/live_tokyoreal_custom_e2_preds_t60k_'
D8 = ['20261023', '20261024', '20261025', '20261026', '20261027', '20261031', '20261101', '20261102']; BES = '20261103'
SIZES = {'s5': ('AV:5:25:0.06', 25), 's20': ('A1:20:100:0', 100)}
PRE = 'MICRO+G4+G2+G6+G5+G3+G1+V0+GS+XNA+XMIN1+XV+'
T7 = 2.365


def extra(name):
    assert name.startswith(PRE) and '/mid@' in name, name
    return name[len(PRE):].split('/mid@')[0]


def ctrl():
    bad = 0
    for d in ('20261024', '20261101'):
        base = f'{RUN}iwctrl/{d}'
        s = json.load(open(f'{base}/summary.json'))['arms']
        names = {extra(n): n for n in s}
        assert set(names) == {'S60', 'S60+CAP1000', 'RO300+S60', 'S60+CAP5'}, names
        ref = s[names['S60']]
        wref = pd.read_parquet(f"{base}/win/{names['S60'].replace('/', '__').replace('+', '_').replace('@', '_sz').replace(':', '-')}.parquet")
        for e in ('S60+CAP1000', 'RO300+S60', 'S60+CAP5'):
            a = s[names[e]]
            keys = ('mtm_stop_usd', 'gross_usd', 'fills', 'shares', 'carry_stop_usd', 'mo1_usd', 'windows')
            diff = {k: (ref[k], a[k]) for k in keys if ref[k] != a[k]}
            w = pd.read_parquet(f"{base}/win/{names[e].replace('/', '__').replace('+', '_').replace('@', '_sz').replace(':', '-')}.parquet")
            wsame = wref.drop(columns=[]).reset_index(drop=True).equals(w.reset_index(drop=True))
            same = not diff and wsame
            ok = (not same) if e == 'S60+CAP5' else same          # CAP5 = positive control: must differ
            bad += int(not ok)
            print(d, e, 'vs S60:', 'EQUAL' if same else f'DIFF {diff} windows_equal={wsame}', '(must differ)' if e == 'S60+CAP5' else '', 'ok' if ok else 'BAD')
    print('CONTROLS', 'PASS' if bad == 0 else f'FAIL ({bad})')
    return 0 if bad == 0 else 1


def load(k, sz):
    """{extra: {day: arm summary}} from the two batches; statuses for missing day-runs."""
    out, status = {}, {}
    for b in 'ab':
        for d in D8 + [BES]:
            f = f'{RUN}iw{k}_{sz}{b}/{d}/summary.json'
            if not os.path.exists(f):
                status[f'{sz}{b}/{d}'] = 'MISSING'; continue
            for n, a in json.load(open(f))['arms'].items():
                assert n.endswith('/mid@' + SIZES[sz][0]), n
                out.setdefault(extra(n), {})[d] = a
    return out, status


def ci(x):
    x = np.asarray(x, float); m = x.mean(); h = T7 * x.std(ddof=1) / np.sqrt(len(x))
    return dict(mean=round(m, 1), lo=round(m - h, 1), hi=round(m + h, 1), excludes_0=bool(m - h > 0 or m + h < 0), above_0=bool(m - h > 0),
                days_pos=int((x > 0).sum()))


def main():
    if sys.argv[1:] == ['ctrl']:
        return ctrl()
    out = dict(note='exploratory, in-sample; fixed engine round i; $ / day without rebate', arms={}, cmp={}, status={})
    for sz, (zname, hi) in SIZES.items():
        W = [(f'S{w}', f'S{w}+CAP{hi}', f'RO{w}+S{w + 60}', f'RO{w}+S{w + 60}+CAP{hi}') for w in (60, 120, 180)]
        INV = [(alt, w[0]) for w in W for alt in w[1:]] + [(f'S240+CAP{hi}', 'S240')]
        WIN = [(f'S{w}', 'S60') for w in (120, 180, 240)] + [(f'S{w}+CAP{hi}', f'S60+CAP{hi}') for w in (120, 180, 240)] \
            + [(f'RO{w}+S{w + 60}+CAP{hi}', f'RO60+S120+CAP{hi}') for w in (120, 180)]
        for k in ('J', 'X'):
            A, st = load(k, sz); out['status'][f'{k}/{sz}'] = st
            for e, by in A.items():
                if not all(d in by for d in D8 + [BES]):
                    out['arms'][f'{k}/{sz}/{e}'] = 'INCOMPLETE'; continue
                eng = {t: sum(by[d]['withheld_decisions'].get(t, 0) for d in D8 + [BES]) for t in ('CAPN', 'RO')}
                need = [t for t, tok in (('CAPN', 'CAP'), ('RO', 'RO')) if any(p.startswith(tok) for p in e.split('+'))]
                if any(eng[t] == 0 for t in need):
                    raise SystemExit(f'REFUSED: {k}/{sz}/{e}: token never engaged {eng}')
                g = lambda key: [by[d][key] for d in D8]
                sh = sum(by[d]['shares'] for d in D8)
                out['arms'][f'{k}/{sz}/{e}'] = dict(
                    mark=round(float(np.mean(g('mtm_stop_usd'))), 1), mark_days_pos=int(sum(x > 0 for x in g('mtm_stop_usd'))),
                    mark_1103=round(by[BES]['mtm_stop_usd'], 1), settle=round(float(np.mean(g('gross_usd'))), 1),
                    settle_days_pos=int(sum(x > 0 for x in g('gross_usd'))), settle_1103=round(by[BES]['gross_usd'], 1),
                    worst_day_settle=round(min(g('gross_usd')), 1), carry_sd_window=round(float(np.mean(g('carry_stop_sd'))), 2),
                    abs_inv_at_stop=round(float(np.mean(g('abs_net_stop_mean'))), 1), shares_day=round(sh / 8.0),
                    mo1_c_sh=round(100.0 * sum(g('mo1_usd')) / sh, 3) if sh else None, withheld=eng)
            for grp, pairs in (('inventory', INV), ('window', WIN)):
                for a, b in pairs:
                    key = f'{grp}/{sz}/{a} - {b}'
                    if a not in A or b not in A or not all(d in A[a] and d in A[b] for d in D8 + [BES]):
                        out['cmp'].setdefault(key, {})[k] = 'MISSING'; continue
                    out['cmp'].setdefault(key, {})[k] = dict(
                        mark=ci([A[a][d]['mtm_stop_usd'] - A[b][d]['mtm_stop_usd'] for d in D8]),
                        settle=ci([A[a][d]['gross_usd'] - A[b][d]['gross_usd'] for d in D8]),
                        mark_1103=round(A[a][BES]['mtm_stop_usd'] - A[b][BES]['mtm_stop_usd'], 1),
                        carry_sd_change=round(float(np.mean([A[a][d]['carry_stop_sd'] - A[b][d]['carry_stop_sd'] for d in D8])), 2),
                        carry_sd_days_lower=int(sum(A[a][d]['carry_stop_sd'] < A[b][d]['carry_stop_sd'] for d in D8)))   # descriptive (added after the run)
    for key, v in out['cmp'].items():
        okj = isinstance(v.get('J'), dict) and v['J']['mark']['above_0']
        okx = isinstance(v.get('X'), dict) and v['X']['mark']['above_0']
        v['ADDS_J'] = bool(okj); v['ADDS_X'] = bool(okx); v['ADDS_BOTH'] = bool(okj and okx)
    io.write_json(f'{S4}/report_invwin.json', out)
    for key, a in out['arms'].items():
        if isinstance(a, dict):
            print(f"{key:34s} mark {a['mark']:+8.1f} ({a['mark_days_pos']}/8) 11-03 {a['mark_1103']:+7.1f} | settle {a['settle']:+8.1f} "
                  f"({a['settle_days_pos']}/8) worst {a['worst_day_settle']:+8.1f} | carry sd {a['carry_sd_window']:5.2f} |inv| {a['abs_inv_at_stop']:5.1f} "
                  f"| {a['shares_day']:6d} sh/day +1s {a['mo1_c_sh']} | {a['withheld']}")
        else:
            print(f'{key:34s} {a}')
    for key, v in out['cmp'].items():
        cells = []
        for k in ('J', 'X'):
            c = v.get(k)
            cells.append(f'{k}: MISSING' if not isinstance(c, dict) else
                         f"{k}: mark {c['mark']['mean']:+.1f} ({c['mark']['lo']:+.1f}..{c['mark']['hi']:+.1f}) {c['mark']['days_pos']}/8 "
                         f"settle {c['settle']['mean']:+.1f} ({c['settle']['lo']:+.1f}..{c['settle']['hi']:+.1f}) 11-03 {c['mark_1103']:+.1f} "
                         f"carry sd {c['carry_sd_change']:+.2f} lower {c['carry_sd_days_lower']}/8")
        print(f"{key:44s} {' | '.join(cells)} | ADDS_BOTH {v['ADDS_BOTH']}")
    print('statuses (missing day-runs):', {k: v for k, v in out['status'].items() if v} or 'none')
    return 0


if __name__ == '__main__':
    sys.exit(main())
