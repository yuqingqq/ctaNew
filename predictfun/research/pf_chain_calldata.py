"""Decode signed orders from Predict settlement calldata and attribute WS settlementsPending increases to them.

Example (10-23 14:23:37, tx 0xaa5c6b...): calldata words 0/1 are offsets to the taker Order and the maker Order[];
Order = (salt, maker, signer, taker, tokenId, makerAmount, takerAmount, expiration, nonce, feeRateBps, side, signatureType,
signature). The taker sold 114.334 Up for >= 112.04733 USDT (limit 0.98); it filled at 0.99 and WS pending showed asks
0.98 = 112.0473 next to bids 0.99 = 113.1907 (the maker). Limit price: BUY = makerAmount / takerAmount, SELL = takerAmount /
makerAmount, in the token's own outcome; YES space flips side and price for Down.
Test over the slice: every WS pending increase (live, both sides) is matched to an order entry of a settlement in that
market whose block is 0..2.5 s after the increase: maker entry (maker side, maker price, filled shares x price) or taker
entry (taker side, taker LIMIT price, filled shares x limit).
Usage: python3 -m predictfun.research.pf_chain_calldata 20261023 1400 1415
Writes onchain/<slice>/inputs.jsonl.gz (cache), orders.parquet, calldata.json.
"""
import sys, os, json, gzip, time, collections, urllib.request, urllib.error
import numpy as np, pandas as pd
from . import pf_io as io

day, h0, h1 = sys.argv[1], sys.argv[2], sys.argv[3]
DIR = f'{io.OUT}/onchain/{day}_{h0}_{h1}'
FS = json.load(open(f'{DIR}/fetch_summary.json')); T0, T1 = FS['t0'], FS['t1']; RPC = FS['rpc']
F = pd.read_parquet(f'{DIR}/chain_fills.parquet')
TXS = sorted(set(F.tx))
E18 = 1e18

# ---- fetch calldata (cached) ----
cache = f'{DIR}/inputs.jsonl.gz'; have = {}
if os.path.exists(cache):
    with gzip.open(cache, 'rt') as f:
        for line in f:
            r = json.loads(line); have[r['hash']] = r['input']
need = [t for t in TXS if t not in have]
with gzip.open(cache, 'at') as fo:
    for s in range(0, len(need), 40):
        batch = [dict(jsonrpc='2.0', id=i, method='eth_getTransactionByHash', params=[h]) for i, h in enumerate(need[s:s + 40])]
        for attempt in range(8):
            try:
                req = urllib.request.Request(RPC, data=json.dumps(batch).encode(), headers={'Content-Type': 'application/json', 'User-Agent': 'pf-research/1'})
                res = json.loads(urllib.request.urlopen(req, timeout=60).read())
                if not isinstance(res, list) or any(x.get('result') is None for x in res):
                    raise RuntimeError(str(res)[:200])
                break
            except (urllib.error.URLError, RuntimeError, TimeoutError, ValueError) as e:
                print('[retry]', attempt, str(e)[:120], flush=True); time.sleep(min(30, 2 ** attempt))
        else:
            raise SystemExit('RPC failed')
        for x in res:
            h = x['result']['hash'].lower(); have[h] = x['result']['input']
            fo.write(json.dumps(dict(hash=h, input=x['result']['input'])) + '\n')
        time.sleep(0.15)
print('[inputs]', len(have), 'fetched now', len(need), flush=True)

# ---- decode ----
TOK = {int(k): tuple(v) for k, v in json.load(open(f'{DIR}/tokens.json')).items()}   # written by pf_chain_recon

def order_at(w, i):
    """Order struct starting at word i."""
    return dict(maker='0x' + w[i + 1][-40:], token=int(w[i + 4], 16), makerAmount=int(w[i + 5], 16) / E18,
                takerAmount=int(w[i + 6], 16) / E18, side=int(w[i + 10], 16))

sel = collections.Counter(); bad = 0; O = []
for tx in TXS:
    inp = have[tx]; sel[inp[:10]] += 1
    d = inp[10:]; w = [d[i:i + 64] for i in range(0, len(d), 64)]
    try:
        ti = int(w[0], 16) // 32; mi = int(w[1], 16) // 32
        tk = order_at(w, ti)
        n = int(w[mi], 16); makers = [order_at(w, mi + 1 + int(w[mi + 1 + j], 16) // 32) for j in range(n)]
        if tk['token'] not in TOK or not all(m['token'] in TOK for m in makers):
            raise ValueError('token not ours')
    except (ValueError, IndexError):
        bad += 1; continue
    for role, o in [('taker', tk)] + [('maker', m) for m in makers]:
        mid, ix = TOK[o['token']]
        p = o['makerAmount'] / o['takerAmount'] if o['side'] == 0 else o['takerAmount'] / o['makerAmount']
        yes_buy = (o['side'] == 0) == (ix == 1)
        O.append((tx, role, o['maker'], mid, ix, o['side'], p, (p if ix == 1 else 1 - p), 0 if yes_buy else 1))
O = pd.DataFrame(O, columns=['tx', 'role', 'maker', 'mid', 'outcome', 'side', 'limit', 'limit_yes', 'yes_side'])
# filled shares and execution price per order from the OrderFilled events
Fg = F.groupby(['tx', 'maker', 'is_taker_order'], as_index=False).agg(shares=('shares', 'sum'), usd=('usd', 'sum'), block_ms=('block_ms', 'first'))
Fg['role'] = np.where(Fg.is_taker_order, 'taker', 'maker')
O = O.merge(Fg[['tx', 'maker', 'role', 'shares', 'usd', 'block_ms']], on=['tx', 'maker', 'role'], how='left')
O['exec'] = O.usd / O.shares
O['exec_yes'] = np.where(O.outcome == 1, O['exec'], 1 - O['exec'])
O['limit_tick_yes'] = np.round(O.limit_yes * 100).astype(int)
O['exec_tick_yes'] = np.round(O.exec_yes * 100)
O.to_parquet(f'{DIR}/orders.parquet', index=False)

# ---- WS pending increases in the slice ----
reg = pd.read_parquet(f'{io.OUT}/step1/{day}/registry.parquet', columns=['mid', 'ws', 'we'])
live = {int(m): (int(ws) * 1000, int(we) * 1000) for m, ws, we in zip(reg.mid, reg.ws, reg.we)}
hours = sorted({time.strftime('%Y%m%d_%H', time.gmtime(t)) for t in (T0 - 60, T0, T1 - 1, T1 + 60)})
prev = {}; INC = []
for h in hours:
    for recv, d in io.ws_books(h):
        mid = d.get('marketId'); t = d['updateTimestampMs']; w_ = live.get(mid)
        if w_ is None or not (w_[0] <= t < w_[1]):
            continue
        cur = {}
        for si, key in ((0, 'bids'), (1, 'asks')):
            for p, v in (d.get('settlementsPending') or {}).get(key) or []:
                k = (si, int(round(float(p) * 100))); cur[k] = cur.get(k, 0.0) + float(v)
        old = prev.get(mid)
        if old is not None and T0 * 1000 <= t < T1 * 1000:
            for k, v in cur.items():
                dl = v - old.get(k, 0.0)
                if dl > 1e-9:
                    INC.append((mid, t, k[0], k[1], dl))
        prev[mid] = cur
INC = pd.DataFrame(INC, columns=['mid', 't', 'side', 'pt', 'usd'])

# entries: maker at its price (= execution), taker at its limit; value = filled shares x price
ent = collections.defaultdict(list)     # (mid, side, pt) -> [(block_ms, value, role, limit != exec)]
for r in O.itertuples():
    if not np.isfinite(r.shares):
        continue
    pt = r.limit_tick_yes
    ent[(r.mid, r.yes_side, pt)].append((r.block_ms, r.shares * pt / 100.0, r.role, bool(r.limit_tick_yes != r.exec_tick_yes)))
def near(x, y): return abs(x - y) <= max(0.002 * y, 0.005)
cls = []
for r in INC.itertuples():
    hit = []
    for sd in (r.side, 1 - r.side):
        for b, v, role, diff in ent.get((r.mid, sd, r.pt), []):
            if r.t - 200 <= b <= r.t + 2500 and near(r.usd, v):
                hit.append((role, diff, sd == r.side))
    if not hit:
        cls.append('none')
    else:
        roles = {h[0] for h in hit}
        cls.append('maker' if roles == {'maker'} else ('taker' if roles == {'taker'} else 'both'))
INC['cls'] = cls
tot = INC.usd.sum()
O_or = pd.read_parquet(f'{io.OUT}/step1/{day}/pending_orphans.parquet')
O_or = O_or[(O_or.t >= T0 * 1000) & (O_or.t < T1 * 1000)].merge(INC[['mid', 't', 'side', 'pt', 'cls']].rename(columns={'side': 'bside'}),
                                                                on=['mid', 't', 'bside', 'pt'], how='left')
tk = O[O.role == 'taker']
tk_off = (tk.limit_tick_yes - tk.exec_tick_yes) * np.where(tk.yes_side == 0, 1, -1)   # + = limit more aggressive than the fill
out = dict(slice=f'{day} {h0}-{h1}', txs=len(TXS), selectors=dict(sel), undecoded_tx=bad,
           orders=dict(taker=int((O.role == 'taker').sum()), maker=int((O.role == 'maker').sum())),
           maker_limit_eq_exec=round(float((O[O.role == 'maker'].limit_tick_yes == O[O.role == 'maker'].exec_tick_yes).mean()), 4),
           taker_limit_eq_exec=round(float((tk.limit_tick_yes == tk.exec_tick_yes).mean()), 4),
           taker_limit_minus_exec_ticks_aggressive={str(k): int(v) for k, v in tk_off.clip(-5, 10).value_counts().sort_index().items()},
           ws_increase_usd=round(float(tot), 2),
           ws_increase_share_by_entry={k: round(float(INC.usd[INC.cls == k].sum() / tot), 4) for k in ('maker', 'taker', 'both', 'none')},
           orphans_in_slice_usd=round(float(O_or.usd.sum()), 2),
           orphan_share_by_entry={k: round(float(O_or.usd[O_or.cls == k].sum() / max(O_or.usd.sum(), 1e-9)), 4) for k in ('maker', 'taker', 'both', 'none')})
io.write_json(f'{DIR}/calldata.json', out)
print(json.dumps(out, indent=1))
