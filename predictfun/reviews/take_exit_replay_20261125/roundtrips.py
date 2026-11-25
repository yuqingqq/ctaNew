"""TAKE-EXIT-REPLAY v1: the contract's round-trip measurements (FIFO take -> close pairs, holding time, value per share) from the saved
ledgers; descriptive, no new replay. Usage: run_light.sh txr_rt -- python3 .../roundtrips.py"""
import collections, gzip, json
from pathlib import Path
import numpy as np, pandas as pd
H = Path(__file__).resolve().parent
OUT = Path('/home/yuqing/ctaNew/data/predict_fun/research/take_exit_replay_20261125')
m = json.loads((H / 'manifest.json').read_text())
rows = []
for d, mid, _ in m['cases']:
    for arm in m['arms']:
        rec = json.loads(gzip.decompress((OUT / f'{mid}_{arm}.json.gz').read_bytes()))
        io = {int(k) for k in rec['iocs']}
        fees = collections.defaultdict(float)
        for x in rec['ioc_results']:
            fees[int(x['gid'])] += x['fee_c']
        lots = collections.deque()
        for t, gid, s, q, p, it in sorted(rec['fills'], key=lambda f: (f[0], f[1])):
            sg = 1 if s == 0 else -1
            kind = 'take' if gid in io else 'passive'
            fee_ps = fees[gid] / q if gid in io and q > 0 else 0.
            while q > 1e-9 and lots and lots[0][0] != sg:
                ls, lq, lp, lt, lk, lf = lots[0]
                x = min(q, lq)
                rows.append(dict(day=d, mid=mid, arm=arm, q=x, entry_kind=lk, exit_kind=kind, hold_ms=t - lt,
                                 gross_c=x * (p - lp) * ls, fees_c=x * (lf + fee_ps)))
                q -= x; lq -= x
                if lq <= 1e-9: lots.popleft()
                else: lots[0] = (ls, lq, lp, lt, lk, lf)
            if q > 1e-9:
                lots.append((sg, q, p, t, kind, fee_ps))
        for ls, lq, lp, lt, lk, lf in lots:
            rows.append(dict(day=d, mid=mid, arm=arm, q=lq, entry_kind=lk, exit_kind='open_at_END', hold_ms=rec['end'] - lt if 'end' in rec else np.nan,
                             gross_c=lq * (rec['mark'] - lp) * ls, fees_c=lq * lf))
df = pd.DataFrame(rows)
df.to_csv(H / 'roundtrips.csv', index=False)
g = df.groupby(['arm', 'exit_kind']).apply(lambda x: pd.Series(dict(shares=x.q.sum(), net_c_per_share=100 * 0 + (x.gross_c.sum() - x.fees_c.sum()) / x.q.sum(),
     gross_c_per_share=x.gross_c.sum() / x.q.sum(), median_hold_s=float(np.average(x.hold_ms, weights=x.q) / 1000) if x.q.sum() > 0 else np.nan,
     hold_p50_s=float(x.hold_ms.median() / 1000), net_usd=(x.gross_c.sum() - x.fees_c.sum()) / 100)), include_groups=False).reset_index()
out = {f'{a}|{e}': {k: float(v) for k, v in r.items()} for (a, e), r in g.set_index(['arm', 'exit_kind']).iterrows()}
(H / 'roundtrips.json').write_text(json.dumps(out, indent=1) + '\n')
pd.set_option('display.width', 200)
print(g.round(2).to_string(index=False))
