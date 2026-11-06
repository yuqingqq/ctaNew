"""Failed Predict settlement txs (receipt status 0) and the 'orphan' WS pending increases.

Failed txs emit no logs, so their orders come from calldata (same layout as pf_chain_calldata: word0 -> taker Order,
word1 -> maker Order[], word2 = taker fill in its maker asset, word3 = taker receive, word4 -> makerFillAmounts[]).
For our live markets: entries = maker orders at their price and the taker at its limit price (filled shares x price).
Orphans in the slice (step-1 pending_orphans: WS increase with no REST trade at that price within +-1 s) are attributed to
(a) a failed tx's entry, (b) a successful tx's taker-at-limit entry (orders.parquet), (c) neither; window: the tx's
block 0..2.5 s after the increase, same market and YES price, value within 0.2%.
Usage: python3 -m predictfun.research.pf_chain_failed 20261023 1400 1415
Writes onchain/<slice>/failed_orders.parquet, failed.json.
"""
import sys, os, json, gzip, time, collections, urllib.request, urllib.error
import numpy as np, pandas as pd
from . import pf_io as io

day, h0, h1 = sys.argv[1], sys.argv[2], sys.argv[3]
DIR = f'{io.OUT}/onchain/{day}_{h0}_{h1}'
FS = json.load(open(f'{DIR}/fetch_summary.json')); T0, T1 = FS['t0'], FS['t1']; RPC = FS['rpc']
TOK = {int(k): tuple(v) for k, v in json.load(open(f'{DIR}/tokens.json')).items()}
E18 = 1e18
blk = {}
with gzip.open(f'{DIR}/blocks.jsonl.gz', 'rt') as f:
    for line in f:
        b = json.loads(line); blk[b['number']] = b['milliTimestamp'] or b['timestamp'] * 1000
failed = {}
with gzip.open(f'{DIR}/receipts.jsonl.gz', 'rt') as f:
    for line in f:
        r = json.loads(line)
        if r['status'] != '0x1':
            failed[r['transactionHash'].lower()] = int(r['blockNumber'], 16)
cache = f'{DIR}/inputs.jsonl.gz'; have = {}
if os.path.exists(cache):
    with gzip.open(cache, 'rt') as f:
        for line in f:
            r = json.loads(line); have[r['hash']] = r['input']
need = [t for t in failed if t not in have]
with gzip.open(cache, 'at') as fo:
    for s in range(0, len(need), 20):
        batch = [dict(jsonrpc='2.0', id=i, method='eth_getTransactionByHash', params=[h]) for i, h in enumerate(need[s:s + 20])]
        for attempt in range(8):
            try:
                req = urllib.request.Request(RPC, data=json.dumps(batch).encode(), headers={'Content-Type': 'application/json', 'User-Agent': 'pf-research/1'})
                res = json.loads(urllib.request.urlopen(req, timeout=60).read())
                if not isinstance(res, list) or any(x.get('result') is None for x in res):
                    raise RuntimeError(str(res)[:200])
                break
            except (urllib.error.URLError, RuntimeError, TimeoutError, ValueError) as e:
                print('[retry]', attempt, str(e)[:120], flush=True); time.sleep(min(30, 2 ** (attempt + 1)))
        else:
            raise SystemExit('RPC failed')
        for x in res:
            h = x['result']['hash'].lower(); have[h] = x['result']['input']
            fo.write(json.dumps(dict(hash=h, input=x['result']['input'])) + '\n')
        time.sleep(0.5)

def order_at(w, i):
    return dict(maker='0x' + w[i + 1][-40:], token=int(w[i + 4], 16), mA=int(w[i + 5], 16) / E18, tA=int(w[i + 6], 16) / E18, side=int(w[i + 10], 16))

rows = []; n_ours = 0; n_other = 0; n_bad = 0; sel = collections.Counter()
for tx, bn in failed.items():
    inp = have[tx]; sel[inp[:10]] += 1
    d = inp[10:]; w = [d[i:i + 64] for i in range(0, len(d), 64)]
    try:
        tk = order_at(w, int(w[0], 16) // 32); mi = int(w[1], 16) // 32; n = int(w[mi], 16)
        makers = [order_at(w, mi + 1 + int(w[mi + 1 + j], 16) // 32) for j in range(n)]
        fi = int(w[4], 16) // 32; fills = [int(w[fi + 1 + j], 16) / E18 for j in range(int(w[fi], 16))]
        t_fill, t_recv = int(w[2], 16) / E18, int(w[3], 16) / E18
    except (ValueError, IndexError):
        n_bad += 1; continue
    if tk['token'] not in TOK:
        n_other += 1; continue
    n_ours += 1
    for role, o, fill in [('taker', tk, None)] + [('maker', m, f) for m, f in zip(makers, fills)]:
        mid, ix = TOK.get(o['token'], (None, None))
        if mid is None:
            continue
        p = o['mA'] / o['tA'] if o['side'] == 0 else o['tA'] / o['mA']
        if role == 'taker':
            sh = t_fill if o['side'] == 1 else t_recv
        else:
            sh = fill if o['side'] == 1 else fill / p
        p_yes = p if ix == 1 else 1 - p
        yes_side = 0 if ((o['side'] == 0) == (ix == 1)) else 1
        rows.append((tx, bn, blk.get(bn), role, mid, yes_side, int(round(p_yes * 100)), sh))
FO = pd.DataFrame(rows, columns=['tx', 'block', 'block_ms', 'role', 'mid', 'yes_side', 'pt', 'shares'])
FO.to_parquet(f'{DIR}/failed_orders.parquet', index=False)

ORD = pd.read_parquet(f'{DIR}/orders.parquet') if os.path.exists(f'{DIR}/orders.parquet') else None
ent = collections.defaultdict(list)
for r in FO.itertuples():
    ent[(r.mid, r.pt)].append((r.block_ms, r.shares * r.pt / 100.0, 'failed tx'))
if ORD is not None:
    for r in ORD[(ORD.role == 'taker') & np.isfinite(ORD.shares)].itertuples():
        if r.limit_tick_yes != r.exec_tick_yes:
            ent[(r.mid, int(r.limit_tick_yes))].append((r.block_ms, r.shares * r.limit_tick_yes / 100.0, 'taker at limit (ok tx)'))
def near(x, y): return abs(x - y) <= max(0.002 * y, 0.005)
OR = pd.read_parquet(f'{io.OUT}/step1/{day}/pending_orphans.parquet')
OR = OR[(OR.t >= T0 * 1000) & (OR.t < T1 * 1000)].copy()
lab = []
for r in OR.itertuples():
    cand = [k for b, v, k in ent.get((r.mid, r.pt), []) if b is not None and r.t - 200 <= b <= r.t + 2500 and near(r.usd, v)]
    lab.append('failed tx' if 'failed tx' in cand else (cand[0] if cand else 'unexplained'))
OR['why'] = lab
tot = float(OR.usd.sum())
out = dict(slice=f'{day} {h0}-{h1}', failed_tx=len(failed), failed_selectors=dict(sel), failed_tx_our_markets=n_ours,
           failed_tx_other_markets=n_other, undecoded=n_bad, failed_orders_our_markets=int(len(FO)),
           failed_entries_usd=round(float((FO.shares * FO.pt / 100).sum()), 2),
           orphans=dict(n=int(len(OR)), usd=round(tot, 2), share_by_cause={k: round(float(OR.usd[OR.why == k].sum() / max(tot, 1e-9)), 4)
                                                                            for k in ('failed tx', 'taker at limit (ok tx)', 'unexplained')}))
io.write_json(f'{DIR}/failed.json', out)
print(json.dumps(out, indent=1))
