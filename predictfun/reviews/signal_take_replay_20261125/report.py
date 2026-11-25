"""SIGNAL-TAKE-REPLAY v1 report: paired deltas host -> host + SIGNAL_TAKE on the 19 markets; every number computed from the ledgers."""
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import take_replay as TR  # noqa: E402

KEYS = ['net_c', 'completed_c', 'residual_c', 'fees_c', 'matched_shares', 'maker_shares', 'taker_shares', 'mean_abs_inventory', 'peak_inventory',
        'both_grid_fraction']


def load(mid, arm):
    p = (TR.EXIT_OUT if arm in TR.HOSTS else TR.OUT) / f'{mid}_{arm}.json.gz'
    return json.loads(gzip.decompress(p.read_bytes()))


def usd(x):
    return f'{x / 100:+,.2f}'


def md(rows, cols):
    out = ['| ' + ' | '.join(cols) + ' |', '|' + '|'.join(['---'] + ['---:'] * (len(cols) - 1)) + '|']
    return '\n'.join(out + ['| ' + ' | '.join(str(v) for v in r) + ' |' for r in rows])


def main():
    m = json.loads((H / 'manifest.json').read_text())
    day_of = {mid: d for d, mid, _ in m['cases']}
    rows, marks = [], []
    for d, mid, _ in m['cases']:
        for host in TR.HOSTS:
            a, b = load(mid, host), load(mid, f'{host}_TAKE')
            r = dict(day=d, mid=mid, host=host)
            for k in KEYS:
                r[f'{k}_host'] = a['metrics'].get(k, 0.0) or 0.0
                r[f'{k}_take'] = b['metrics'].get(k, 0.0) or 0.0
                r[f'd_{k}'] = r[f'{k}_take'] - r[f'{k}_host']
            r.update(triggers=b['metrics']['triggers'], sent=b['metrics']['takes_sent'], filled=b['metrics']['takes_filled'],
                     trig_status=b['metrics']['trigger_status'], ioc_status=b['metrics']['ioc_status'])
            rows.append(r)
            marks += [dict(x, host=host, mid=mid, day=d) for x in b['take_marks']]
    df = pd.DataFrame(rows)
    mk = pd.DataFrame(marks)
    df.drop(columns=['trig_status', 'ioc_status']).to_csv(H / 'per_market.csv', index=False)
    mk.to_csv(H / 'take_marks.csv', index=False)
    out, read = [], {}
    pooled = []
    for host in TR.HOSTS:
        x = df[df.host == host]
        dd = x.groupby('day').d_net_c.sum()
        absd = x.d_net_c.abs().sort_values(ascending=False)
        lo = x[x.mid != 3789484].d_net_c.sum()
        y = mk[mk.host == host]
        w = y.q
        def wm(col):
            z = y[col].notna()
            return float((y[col][z] * w[z]).sum() / w[z].sum()) if z.any() else float('nan')
        read[host] = dict(pooled_delta_c=float(x.d_net_c.sum()), positive=bool(x.d_net_c.sum() > 0), day_delta_c=dd.to_dict(),
                          days_positive=int((dd > 0).sum()), robust_3_of_4=bool((dd > 0).sum() >= 3), markets_improved=int((x.d_net_c > 0).sum()),
                          top3_abs_share=float(absd.iloc[:3].sum() / absd.sum()) if absd.sum() > 0 else None, leave_out_3789484_c=float(lo),
                          d_completed_c=float(x.d_completed_c.sum()), d_residual_c=float(x.d_residual_c.sum()), d_fees_c=float(x.d_fees_c.sum()),
                          gain_only_in_residual=bool(x.d_net_c.sum() > 0 and x.d_completed_c.sum() <= 0),
                          take_m1s_c=wm('m1s_c'), take_m3s_c=wm('m3s_c'), take_m10s_c=wm('m10s_c'), take_end_c=wm('end_c'),
                          take_m3s_positive=bool(wm('m3s_c') > 0), filled_take_shares=float(w.sum()), triggers=int(x.triggers.sum()),
                          sent=int(x.sent.sum()), filled=int(x.filled.sum()))
        pooled.append([host, usd(x.net_c_host.sum()), usd(x.net_c_take.sum()), usd(x.d_net_c.sum()), usd(x.d_completed_c.sum()), usd(x.d_residual_c.sum()),
                       usd(-x.d_fees_c.sum()), f"{int(x.triggers.sum()):,} / {int(x.sent.sum()):,} / {int(x.filled.sum()):,}",
                       f"{x.mean_abs_inventory_host.mean():.0f} → {x.mean_abs_inventory_take.mean():.0f}",
                       f"{100 * x.both_grid_fraction_host.mean():.1f}% → {100 * x.both_grid_fraction_take.mean():.1f}%"])
    out.append(md(pooled, ['Host', 'Host END net', 'With takes', 'Delta', 'Delta completed turnover', 'Delta remaining inventory', 'Taker fees',
                           'Triggers / sent / filled', 'Mean abs inventory', 'Both-side uptime']))
    dayrows = []
    for d in sorted(df.day.unique()):
        dayrows.append([d] + [usd(df[(df.host == h) & (df.day == d)].d_net_c.sum()) for h in TR.HOSTS] + [int((df.day == d).sum() // 2)])
    out.append(md(dayrows, ['Day'] + [f'{h} delta' for h in TR.HOSTS] + ['Markets']))
    mrows = []
    for h in TR.HOSTS:
        r = read[h]
        mrows.append([h, f"{r['filled_take_shares']:,.0f}", f"{r['take_m1s_c']:+.2f}", f"{r['take_m3s_c']:+.2f}", f"{r['take_m10s_c']:+.2f}", f"{r['take_end_c']:+.2f}"])
    out.append(md(mrows, ['Host', 'Filled take shares', '+1 s', '+3 s', '+10 s', 'END']))
    per = []
    for _, r in df.sort_values(['host', 'day', 'mid']).iterrows():
        per.append([r.host, r.day, r.mid, usd(r.net_c_host), usd(r.net_c_take), usd(r.d_net_c), f'{int(r.filled)}/{int(r.sent)}/{int(r.triggers)}'])
    out.append(md(per, ['Host', 'Day', 'Market', 'Host END net', 'With takes', 'Delta', 'Takes filled/sent/triggers']))
    st = pd.DataFrame([dict(host=r['host'], **r['trig_status']) for r in rows]).groupby('host').sum(numeric_only=True)
    io = pd.DataFrame([dict(host=r['host'], **r['ioc_status']) for r in rows]).groupby('host').sum(numeric_only=True)
    read['trigger_status'] = st.to_dict(orient='index')
    read['ioc_status'] = io.to_dict(orient='index')
    (H / 'readings.json').write_text(json.dumps(read, indent=1, default=float) + '\n')
    (H / 'report_tables.md').write_text('\n\n'.join(out) + '\n')
    print(json.dumps(read, indent=1, default=float))
    print('\n\n'.join(out[:3]))


if __name__ == '__main__':
    main()
