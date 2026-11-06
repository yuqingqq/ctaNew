"""Reconcile REST matches with on-chain fills (BNB Chain) for a slice fetched by pf_chain_fetch.

On-chain: CTF exchange OrderFilled(orderHash, maker, taker; makerAssetId, takerAssetId, makerAmountFilled,
takerAmountFilled, fee) per order per match; the taker order's own fill has taker == exchange. Asset 0 = collateral.
REST: /v1/orders/matches records (transactionHash, taker.hash, makers[].hash / amount / price), deduplicated.
Population: our BTC 5m markets (tokens from REST market.outcomes[].onChainId). Core slice = [t0, t1) by block time
for on-chain fills and by executedAt for REST; lookups run against everything fetched / read (slice -60 s .. +180 s,
REST hour files t0-1 h .. t1+1 h) so edges do not create false misses.
Usage: python3 -m predictfun.research.pf_chain_recon 20261023 1400 1415
Writes onchain/<slice>/recon.json, chain_fills.parquet, rest_legs.parquet, tokens.json.
"""
import sys, json, gzip, collections, datetime as dt
import numpy as np, pandas as pd
from . import pf_io as io

day, h0, h1 = sys.argv[1], sys.argv[2], sys.argv[3]
DIR = f'{io.OUT}/onchain/{day}_{h0}_{h1}'
FS = json.load(open(f'{DIR}/fetch_summary.json')); T0, T1 = FS['t0'], FS['t1']
EXCHANGE = '0x8bc070bedab741406f4b1eb65a72bee27894b689'
SIG_FILLED = '0xd0a08e8c493f9c94f29311604c9de1b4e8c8d4c06bd0c789af57f2d65bfec0f6'
SIG_MATCHED = '0x63bf4d16b7fa898ef4c4b2b6d90fd201e9c56313b65638af6088d149d2ce956c'
E18 = 1e18

# ---- REST ----
tok = {}; rest = {}
hours = sorted({dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime('%Y%m%d_%H') for t in np.arange(T0 - 3600, T1 + 3601, 1800)})
reg = pd.read_parquet(f'{io.OUT}/step1/{day}/registry.parquet', columns=['mid', 'ws', 'we'])
mids = set(int(m) for m in reg.mid)
for h in hours:
    for recv, params, path, pl in io.rest_rows(h, '/v1/orders/matches'):
        for x in pl.get('data') or []:
            mk = x.get('market') or {}; mid = mk.get('id')
            if mid not in mids:
                continue
            for o in mk.get('outcomes') or []:
                tok[int(o['onChainId'])] = (mid, o['indexSet'])
            k = (x['transactionHash'].lower(), x['taker']['hash'].lower())
            if k in rest:
                continue
            rest[k] = dict(mid=mid, exec_s=int(io.iso_s(x['executedAt'])), first_recv_ns=recv, taker=x['taker'],
                           amountFilled=int(x['amountFilled']) / E18, makers=x['makers'])
    io.mem_check(h)
RL = []
for (tx, th), r in rest.items():
    for m in r['makers']:
        RL.append((tx, m['hash'].lower(), th, r['mid'], r['exec_s'], int(m['amount']) / E18, int(m['price']) / E18,
                   m['outcome']['indexSet'], m['quoteType'], r['first_recv_ns']))
RL = pd.DataFrame(RL, columns=['tx', 'order', 'taker_order', 'mid', 'exec_s', 'shares', 'price', 'outcome', 'quote', 'first_recv_ns'])
RLg = RL.groupby(['tx', 'order'], as_index=False).agg(mid=('mid', 'first'), exec_s=('exec_s', 'first'), shares=('shares', 'sum'),
                                                     price=('price', 'first'), n=('shares', 'size'))
rest_tx = set(RL.tx)

# ---- chain ----
blk = {}
with gzip.open(f'{DIR}/blocks.jsonl.gz', 'rt') as f:
    for line in f:
        b = json.loads(line); blk[b['number']] = b['milliTimestamp'] or b['timestamp'] * 1000
F = []; MT = []; n_rcpt = n_fail = n_ours_tx = 0; failed_ours = []
with gzip.open(f'{DIR}/receipts.jsonl.gz', 'rt') as f:
    for line in f:
        r = json.loads(line); n_rcpt += 1
        tx = r['transactionHash'].lower(); bn = int(r['blockNumber'], 16); ok = r['status'] == '0x1'
        ours = False
        for lg in r['logs']:
            if lg['address'].lower() != EXCHANGE:
                continue
            d = lg['data'][2:]; w = [int(d[i:i + 64], 16) for i in range(0, len(d), 64)]
            if lg['topics'][0] == SIG_FILLED:
                ma, ta, mf, tf, fee = w[:5]
                t_ = ta if ma == 0 else ma
                if t_ not in tok:
                    continue
                ours = True
                mid, ix = tok[t_]
                buy = ma == 0
                sh = (tf if buy else mf) / E18; usd = (mf if buy else tf) / E18
                F.append((tx, lg['topics'][1].lower(), '0x' + lg['topics'][2][-40:], '0x' + lg['topics'][3][-40:], mid, ix, buy, sh, usd,
                          fee / E18, bn, blk.get(bn), int(lg['logIndex'], 16)))
            elif lg['topics'][0] == SIG_MATCHED:
                ma, ta, mf, tf = w[:4]
                t_ = ta if ma == 0 else ma
                if t_ in tok:
                    MT.append((tx, lg['topics'][1].lower(), tok[t_][0], bn))
        if ours:
            n_ours_tx += 1
        if not ok:
            n_fail += 1
            failed_ours.append(tx)
F = pd.DataFrame(F, columns=['tx', 'order', 'maker', 'taker', 'mid', 'outcome', 'buy', 'shares', 'usd', 'fee', 'block', 'block_ms', 'log_index'])
F['is_taker_order'] = F.taker == EXCHANGE
F['price'] = F.usd / F.shares
io.write_json(f'{DIR}/tokens.json', {str(k): list(v) for k, v in tok.items()})
F.to_parquet(f'{DIR}/chain_fills.parquet', index=False); RL.to_parquet(f'{DIR}/rest_legs.parquet', index=False)
FM = F[~F.is_taker_order].groupby(['tx', 'order'], as_index=False).agg(mid=('mid', 'first'), shares=('shares', 'sum'), usd=('usd', 'sum'),
                                                                     block_ms=('block_ms', 'first'), n=('shares', 'size'))
FT = F[F.is_taker_order]

# ---- core-slice comparisons ----
core_c = FM[(FM.block_ms >= T0 * 1000) & (FM.block_ms < T1 * 1000)]
core_r = RLg[(RLg.exec_s >= T0) & (RLg.exec_s < T1)]
m_c = core_c.merge(RLg[['tx', 'order', 'shares', 'exec_s']], on=['tx', 'order'], how='left', suffixes=('', '_rest'))
m_r = core_r.merge(FM[['tx', 'order', 'shares', 'block_ms']], on=['tx', 'order'], how='left', suffixes=('', '_chain'))
c_only = m_c[m_c.shares_rest.isna()]; r_only = m_r[m_r.shares_chain.isna()]
both = m_c[m_c.shares_rest.notna()]
rel = (both.shares_rest - both.shares).abs() / both.shares
chain_tx_seen = set(F.tx)
fetched_tx = set()
with gzip.open(f'{DIR}/txs.jsonl.gz', 'rt') as f:
    for line in f:
        fetched_tx.add(json.loads(line)['hash'].lower())
# executedAt vs block time (matched legs)
off = both.exec_s * 1000 - both.block_ms
# taker orders: REST record per (tx, taker order) vs chain taker fills
rt = pd.DataFrame([(tx, th, r['mid'], r['exec_s'], r['amountFilled']) for (tx, th), r in rest.items()], columns=['tx', 'order', 'mid', 'exec_s', 'amountFilled'])
FTg = FT.groupby(['tx', 'order'], as_index=False).agg(mid=('mid', 'first'), shares=('shares', 'sum'), block_ms=('block_ms', 'first'))
core_ft = FTg[(FTg.block_ms >= T0 * 1000) & (FTg.block_ms < T1 * 1000)].merge(rt[['tx', 'order', 'amountFilled']], on=['tx', 'order'], how='left')
# WS timing of matched legs: step-1 timed legs carry the WS pending time (t_ms) keyed by (mid, exec_s, ptick, shares)
LT = pd.read_parquet(f'{io.OUT}/step1/{day}/legs_timed.parquet', columns=['mid', 'exec_s', 'ptick', 'shares', 't_ms', 'how'])
LT = LT[LT.how > 0]
both_ws = both.assign(sh_r=both.shares_rest.round(4)).merge(
    LT.assign(sh_r=LT.shares.round(4))[['mid', 'exec_s', 'sh_r', 't_ms']].drop_duplicates(['mid', 'exec_s', 'sh_r']),
    left_on=['mid', 'exec_s', 'sh_r'], right_on=['mid', 'exec_s', 'sh_r'], how='inner')
ws_lag = both_ws.block_ms - both_ws.t_ms

def pct(x, q=(5, 25, 50, 75, 95)):
    return dict(zip([f'p{v}' for v in q], np.percentile(x, q).round(1).tolist())) if len(x) else None
out = dict(
    slice=f'{day} {h0}-{h1} UTC', blocks=FS['n_blocks'], tx_scanned=FS['n_tx_scanned'], predict_tx_fetched=FS['n_tx_kept'],
    receipts=n_rcpt, receipts_failed=n_fail, tx_touching_our_markets=n_ours_tx, our_markets_tokens=len(tok),
    maker_fills=dict(chain_core=int(len(core_c)), chain_core_usd=round(float(core_c.usd.sum()), 2),
                     matched_to_rest=int(len(both)), chain_only=int(len(c_only)), chain_only_usd=round(float(c_only.usd.sum()), 2),
                     chain_only_tx_in_rest=int(c_only.tx.isin(rest_tx).sum()),
                     rest_core=int(len(core_r)), rest_only=int(len(r_only)), rest_only_shares=round(float(r_only.shares.sum()), 2),
                     rest_only_tx_fetched=int(r_only.tx.isin(fetched_tx).sum()), rest_only_tx_has_our_fills=int(r_only.tx.isin(chain_tx_seen).sum()),
                     shares_rel_diff=pct(rel.values, (50, 95, 99, 100)), shares_exact_share=round(float((rel <= 1e-9).mean()), 4) if len(rel) else None),
    taker_orders=dict(chain_core=int(len(core_ft)), in_rest=int(core_ft.amountFilled.notna().sum()),
                      amount_equal=int(((core_ft.amountFilled - core_ft.shares).abs() <= 1e-6 * core_ft.shares.clip(lower=1)).sum())),
    executedAt_minus_block_ms=pct(off.values), executedAt_eq_floor_block_s=round(float((both.exec_s == (both.block_ms // 1000)).mean()), 4),
    executedAt_eq_round_block_s=round(float((both.exec_s == np.round(both.block_ms / 1000)).mean()), 4),
    block_minus_ws_pending_ms=dict(n=int(len(ws_lag)), pct=pct(ws_lag.values)),
    fills_per_tx=pct(F.groupby('tx').size().values, (50, 90, 99, 100)),
)
io.write_json(f'{DIR}/recon.json', out)
print(json.dumps(out, indent=1))
if len(c_only):
    print('chain-only examples:\n', c_only.head(10).to_string())
if len(r_only):
    print('rest-only examples:\n', r_only.head(10).to_string())
