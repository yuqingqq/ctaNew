"""
iter-037 — New-listing dynamics: characterization + 3 candidate strategies + honest validation.
Data: xs_feats_*.parquet (5m OHLCV, 218 syms). Listing date = first non-NaN close, but the
panel was backfilled to 2023-01-01 for some files, so true new-listing events = first-close
strictly after 2023-02-01 (the 2023-01-01 cluster is a backfill artifact, NOT a listing).
Funding from funding_*.parquet (mostly starts 2025-01-01; usable from-listing only for late-2024/2025).
"""
import pandas as pd, numpy as np, glob, os
warn = np.seterr(all='ignore')
CACHE = '/home/yuqing/ctaNew/data/ml/cache'

# ---------- 1. build listing table ----------
files = sorted(glob.glob(f'{CACHE}/xs_feats_*.parquet'))
listings = {}
for f in files:
    sym = os.path.basename(f).replace('xs_feats_', '').replace('.parquet', '')
    c = pd.read_parquet(f, columns=['close'])['close'].dropna()
    if len(c) == 0:
        continue
    listings[sym] = (c.index[0], len(c))

lt = pd.DataFrame([(s, d, n) for s, (d, n) in listings.items()],
                  columns=['sym', 'list_date', 'n']).sort_values('list_date')
# true new-listing events: first close after 2023-02-01 (exclude the 2023-01-01 backfill cluster)
events = lt[lt['list_date'] >= '2023-02-01'].copy()
events['year'] = events['list_date'].dt.year
print(f'Total syms: {len(lt)} | true new-listing events (>=2023-02): {len(events)}')
print('Cohort by year:', events['year'].value_counts().sort_index().to_dict())

# resample close to hourly for cleaner early-life math
def load_hourly(sym):
    df = pd.read_parquet(f'{CACHE}/xs_feats_{sym}.parquet', columns=['close', 'high', 'low'])
    df = df.dropna(subset=['close'])
    h = df['close'].resample('1h').last().dropna()
    return h

# ---------- 2. early-life characterization ----------
# return over first {1,3,7,14,30}d measured from FIRST hour (open of life)
hor_days = [1, 3, 7, 14, 30]
char_rows = []
hourly = {}
for s in events['sym']:
    h = load_hourly(s)
    if len(h) < 24 * 31:
        continue
    hourly[s] = h
    p0 = h.iloc[0]
    rec = {'sym': s, 'year': events.set_index('sym').loc[s, 'year']}
    for d in hor_days:
        idx = d * 24
        if len(h) > idx:
            rec[f'ret_{d}d'] = h.iloc[idx] / p0 - 1.0
    # realized vol first 7d (hourly), and max-run-up / drawdown first 7d
    first7 = h.iloc[:7 * 24]
    rec['rv_7d'] = first7.pct_change().std() * np.sqrt(24 * 365)
    rec['maxrunup_7d'] = first7.max() / p0 - 1.0
    rec['maxdd_7d'] = first7.min() / p0 - 1.0
    char_rows.append(rec)

char = pd.DataFrame(char_rows)
print(f'\n=== EARLY-LIFE CHARACTERIZATION (n={len(char)} events with >=31d history) ===')
for d in hor_days:
    col = f'ret_{d}d'
    v = char[col].dropna()
    print(f'  ret_{d:>2}d: mean {v.mean():+.3f}  median {v.median():+.3f}  %neg {100*(v<0).mean():4.0f}%  n={len(v)}')
print(f'  rv_7d (ann): median {char["rv_7d"].median():.2f}')
print(f'  maxrunup_7d: median {char["maxrunup_7d"].median():+.3f} | maxdd_7d: median {char["maxdd_7d"].median():+.3f}')

print('\n=== by COHORT YEAR (median ret per horizon) ===')
for yr, g in char.groupby('year'):
    s = f'  {int(yr)} (n={len(g):>2}): '
    for d in hor_days:
        s += f'{d}d {g[f"ret_{d}d"].median():+.3f}  '
    print(s)

char.to_parquet('/home/yuqing/ctaNew/agents_system/research/scripts/iter037_char.parquet')
events.to_parquet('/home/yuqing/ctaNew/agents_system/research/scripts/iter037_events.parquet')
print('\nSaved char + events.')
