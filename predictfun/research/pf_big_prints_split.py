"""Split of pf_big_prints (2026-11-04): an adverse big order (p99) just before the event [t-250, t-25.5) on Binance (perp or
spot) vs on any of the other eight venues only, per event group; and a rough dollar reading of a rule that pulls on any venue's
big adverse order: share of our P60 bad / good fills caught, times their 8-day dollars (pf_anchor_report, real model).
Usage: python3 -m predictfun.research.pf_big_prints_split
"""
import json
import numpy as np, pandas as pd
from . import pf_io as io

A = pd.read_parquet(f'{io.OUT}/step4/big_prints/events.parquet')
BIN = ['Binance perp', 'Binance spot']; OTH = ['Hyperliquid', 'Bybit perp', 'Bybit spot', 'OKX perp', 'OKX spot', 'Coinbase', 'Kraken', 'Deribit']
w = 'just before'
b = np.column_stack([A[f'{v}|{w}|p99'].fillna(0).values for v in BIN]).any(1); o = np.column_stack([A[f'{v}|{w}|p99'].fillna(0).values for v in OTH]).any(1)
G = [('controls', A.kind == 'control'), ('ours: good fills', (A.kind == 'ours') & (A.cls == 'good')), ('ours: bad fills', (A.kind == 'ours') & (A.cls == 'bad')),
     ('market sweeps: good', (A.kind == 'market sweep') & (A.cls == 'good')), ('market sweeps: bad', (A.kind == 'market sweep') & (A.cls == 'bad'))]
out = {}
print(f"adverse big order (p99) {w} (in time for our cancel): Binance only / other venues only / both / any")
for g, m in G:
    m = m.values; r = dict(binance_only=float(np.mean(b[m] & ~o[m])), others_only=float(np.mean(o[m] & ~b[m])), both=float(np.mean(b[m] & o[m])), any=float(np.mean((b | o)[m])))
    out[g] = r; print(f"  {g:22s} {r['binance_only']:6.1%} / {r['others_only']:6.1%} / {r['both']:6.1%} / {r['any']:6.1%}")
rep = json.load(open(f'{io.OUT}/step4/report_anchor.json'))['P60 (first 60 s)|real']
bad_usd, good_usd = rep['bad_usd'], rep['good_usd']
cb, cg = out['ours: bad fills']['any'], out['ours: good fills']['any']
print(f"rough reading for P60 (8 days): pulling on any venue's big adverse order catches {cb:.1%} of bad fills (~${-bad_usd * cb:.0f} of ${-bad_usd:.0f}) "
      f"and {cg:.1%} of good fills (~${good_usd * cg:.0f} of ${good_usd:.0f}): net ~${-bad_usd * cb - good_usd * cg:+.0f}, if the caught fills are average-sized")
io.write_json(f'{io.OUT}/step4/big_prints/split.json', dict(split=out, rough=dict(bad_caught=cb, good_caught=cg, bad_usd=bad_usd, good_usd=good_usd)))
