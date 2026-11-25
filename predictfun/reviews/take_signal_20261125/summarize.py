"""TAKE-SIGNAL v1 stage 3: declared tables and readings (contract.md) from takes.parquet. Writes tables/*.csv and readings.json."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
R = H.parents[2]
OUT = R / 'data/predict_fun/research/take_signal_20261125'
T = H / 'tables'
Q = 50
VALS = ['mark_1s', 'mark_3s', 'mark_10s', 'mark_30s', 'rt_1s', 'rt_3s', 'rt_10s', 'end', 'res']
STR = ['edge-0.5', '0.5-1', '1-2', '>=2']
TBL = ['000-030', '030-060', '060-180', '180-240', '240-270', '270-298']
DAYS = ['20261024', '20261025', '20261026', '20261027']


def agg(d, keys):
    """Share-weighted net value per filled share (cents) at Q = 50, plus counts; a value's denominator = shares where it is defined."""
    d = d.assign(sh=np.minimum(Q, d.avail_sh))
    def f(x):
        o = dict(triggers=len(x), filled=int((x.status == 'filled').sum()), markets=x[x.status == 'filled'].mid.nunique(),
                 fill_rate=float((x.status == 'filled').mean()), shares=float(x.sh.sum()), stale_c=float(x.stale_c.mean()))
        for v in VALS:
            ok = x[v].notna() & (x.sh > 0)
            o[f'{v}_c'] = float(100 * (x[v][ok] * x.sh[ok]).sum() / x.sh[ok].sum()) if ok.any() else np.nan
            if v in ('mark_3s', 'end', 'res'):
                o[f'{v}_usd'] = float((x[v][ok] * x.sh[ok]).sum())
        return pd.Series(o)
    return d.groupby(keys, observed=True).apply(f, include_groups=False).reset_index()


def main():
    T.mkdir(exist_ok=True)
    tk = pd.read_parquet(OUT / 'takes.parquet')
    out = {}
    for clk in ('tokyo', 'slow'):
        x = tk[tk.clock == clk]
        agg(x, ['dir']).to_csv(T / f'{clk}_by_dir.csv', index=False)
        agg(x, ['strength']).to_csv(T / f'{clk}_by_strength.csv', index=False)
        agg(x, ['tb']).to_csv(T / f'{clk}_by_time.csv', index=False)
        agg(x, ['tb', 'strength']).to_csv(T / f'{clk}_by_time_strength.csv', index=False)
        agg(x, ['money', 'strength']).to_csv(T / f'{clk}_by_money_strength.csv', index=False)
        agg(x, ['day']).to_csv(T / f'{clk}_by_day.csv', index=False)
        agg(x, ['day', 'strength']).to_csv(T / f'{clk}_by_day_strength.csv', index=False)
        agg(x.assign(spr=np.where(x.known_spread_c <= 1 + 1e-7, '<=1c', '>1c')), ['spr', 'strength']).to_csv(T / f'{clk}_by_spread_strength.csv', index=False)
        agg(x.assign(k=1), ['k']).to_csv(T / f'{clk}_all.csv', index=False)
        a = agg(x.assign(k=1), ['k']).iloc[0]
        dd = agg(x, ['day']).set_index('day')
        s = agg(x, ['strength']).set_index('strength').reindex(STR)
        tb = agg(x, ['tb']).set_index('tb').reindex(TBL)
        rd = {}
        for h in ('mark_1s', 'mark_3s', 'mark_10s'):
            signs = [float(np.sign(dd.loc[d, f'{h}_c'])) for d in DAYS if d in dd.index and dd.loc[d, 'filled'] >= 30]
            rd[h] = dict(pooled_c=float(a[f'{h}_c']), positive=bool(a[f'{h}_c'] > 0), day_signs=signs,
                         positive_on_3_of_4_days=bool(sum(v > 0 for v in signs) >= 3))
        vals = s['mark_3s_c'].to_numpy()
        rd['monotone_mark_3s_across_strength'] = bool(np.all(np.diff(vals) > 0))
        rd['mark_3s_by_strength_c'] = dict(zip(STR, map(float, vals)))
        early = agg(x[x.tb.isin(['000-030', '030-060'])].assign(k=1), ['k']).iloc[0]
        later = agg(x[~x.tb.isin(['000-030', '030-060'])].assign(k=1), ['k']).iloc[0]
        rd['early_vs_later_mark_3s_c'] = dict(early=float(early['mark_3s_c']), later=float(later['mark_3s_c']),
                                              early_filled=int(early.filled), later_filled=int(later.filled))
        rd['fill_rate'] = float(a.fill_rate)
        rd['end_c'] = float(a['end_c'])
        rd['res_c'] = float(a['res_c'])
        # declared selection: Oct 24-25 choose (|r250| >= b, window) with the best mark_3s among cells with >= 30 fills in >= 10 markets
        cand = []
        dev, hold = x[x.day.isin(DAYS[:2])], x[x.day.isin(DAYS[2:])]
        for b, lab in ((0, 'edge'), (0.5, '0.5'), (1, '1'), (2, '2')):
            for w in TBL + ['all']:
                m = (dev.r250.abs() >= b) & ((dev.tb == w) if w != 'all' else True)
                if m.sum() == 0:
                    continue
                g = agg(dev[m].assign(k=1), ['k']).iloc[0]
                if g.filled >= 30 and g.markets >= 10:
                    cand.append(dict(b=lab, b_val=b, window=w, dev_mark_3s_c=float(g['mark_3s_c']), dev_filled=int(g.filled), dev_markets=int(g.markets)))
        cand = pd.DataFrame(cand)
        cand.to_csv(T / f'{clk}_selection_candidates.csv', index=False)
        if len(cand):
            best = cand.sort_values(['dev_mark_3s_c', 'dev_filled'], ascending=[False, False]).iloc[0]
            m = (hold.r250.abs() >= best.b_val) & ((hold.tb == best.window) if best.window != 'all' else True)
            g = agg(hold[m].assign(k=1), ['k']).iloc[0]
            gd = agg(hold[m], ['day']).set_index('day')
            rd['selection'] = dict(chosen=dict(best), holdout_mark_3s_c=float(g['mark_3s_c']), holdout_filled=int(g.filled),
                                   holdout_markets=int(g.markets), holdout_mark_1s_c=float(g['mark_1s_c']), holdout_mark_10s_c=float(g['mark_10s_c']),
                                   holdout_end_c=float(g['end_c']), holdout_day_mark_3s_c={d: float(gd.loc[d, 'mark_3s_c']) for d in gd.index},
                                   holdout_positive=bool(g['mark_3s_c'] > 0))
        else:
            rd['selection'] = None
        out[clk] = rd
    (H / 'readings.json').write_text(json.dumps(out, indent=1, default=float) + '\n')
    print(json.dumps(out, indent=1, default=float))


if __name__ == '__main__':
    main()
