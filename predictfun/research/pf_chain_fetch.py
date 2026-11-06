"""On-chain Predict settlements for a slice of a day (BNB Chain, public RPC, read-only).

Every transaction in the blocks covering [t0 - 60 s, t1 + 180 s] whose `to` is Predict's operator contract or its CTF
exchange is kept, and its receipt fetched. Found from 10-23 14:23:37 (tx 0xaa5c6b...): the settlement tx is sent by one
of several operator EOAs to 0xdcffeb0c..., which calls the exchange 0x8bc070be... (OrderFilled / OrdersMatched);
collateral is BSC-USD 0x55d398..., outcome tokens are ERC-1155 on 0x22da1810....
Writes data/predict_fun/research/onchain/<day>_<HHMM>_<HHMM>/{blocks,txs,receipts}.jsonl.gz.
Usage: python3 -m predictfun.research.pf_chain_fetch 20261023 1400 1415 [rpc_url]
"""
import sys, os, json, gzip, time, datetime as dt, urllib.request, urllib.error
from . import pf_io as io

day, h0, h1 = sys.argv[1], sys.argv[2], sys.argv[3]
RPC = sys.argv[4] if len(sys.argv) > 4 else 'https://bsc-dataseed.bnbchain.org'
OPERATOR = '0xdcffeb0c30263888a48485a664ec9563a54891a1'
EXCHANGE = '0x8bc070bedab741406f4b1eb65a72bee27894b689'
KEEP_TO = {OPERATOR, EXCHANGE}
D0 = dt.datetime.strptime(day, '%Y%m%d').replace(tzinfo=dt.timezone.utc).timestamp()
T0 = D0 + int(h0[:2]) * 3600 + int(h0[2:]) * 60; T1 = D0 + int(h1[:2]) * 3600 + int(h1[2:]) * 60
OUT = f'{io.OUT}/onchain/{day}_{h0}_{h1}'; os.makedirs(OUT, exist_ok=True)
n_req = 0

def rpc(batch):
    global n_req
    body = json.dumps(batch).encode()
    for attempt in range(8):
        try:
            req = urllib.request.Request(RPC, data=body, headers={'Content-Type': 'application/json', 'User-Agent': 'pf-research/1'})
            r = json.loads(urllib.request.urlopen(req, timeout=60).read()); n_req += 1
            if isinstance(r, dict):
                raise RuntimeError(str(r)[:200])
            bad = [x for x in r if 'error' in x or x.get('result') is None]
            if bad:
                raise RuntimeError(f'{len(bad)} bad items, e.g. {str(bad[0])[:200]}')
            time.sleep(0.15)
            return {x['id']: x['result'] for x in r}
        except (urllib.error.URLError, RuntimeError, TimeoutError, ValueError) as e:
            print('[retry]', attempt, str(e)[:160], flush=True); time.sleep(min(30, 2 ** attempt))
    raise SystemExit('RPC failed 8 times')

def header(n):
    b = rpc([dict(jsonrpc='2.0', id=0, method='eth_getBlockByNumber', params=[hex(n), False])])[0]
    return int(b['timestamp'], 16)

def block_at(ts):
    """Largest block with timestamp <= ts (binary search from a 10-23 anchor, 0.45 s/block guess)."""
    a_n, a_t = 129337268, 1792765417
    lo = a_n + int((ts - a_t) / 0.45) - 4000; hi = lo + 8000
    while header(lo) > ts: lo -= 8000
    while header(hi) <= ts: hi += 8000
    while hi - lo > 1:
        m = (lo + hi) // 2
        if header(m) <= ts: lo = m
        else: hi = m
    return lo

b_lo, b_hi = block_at(T0 - 60), block_at(T1 + 180) + 1
print(f'[blocks] {b_lo}..{b_hi} ({b_hi - b_lo + 1}) for {dt.datetime.fromtimestamp(T0 - 60, dt.timezone.utc)} .. +{(T1 - T0 + 240) / 60:.0f} min', flush=True)
fb = gzip.open(f'{OUT}/blocks.jsonl.gz', 'wt'); ft = gzip.open(f'{OUT}/txs.jsonl.gz', 'wt'); fr = gzip.open(f'{OUT}/receipts.jsonl.gz', 'wt')
kept = []; n_tx = 0
for s in range(b_lo, b_hi + 1, 5):
    res = rpc([dict(jsonrpc='2.0', id=i, method='eth_getBlockByNumber', params=[hex(n), True]) for i, n in enumerate(range(s, min(s + 5, b_hi + 1)))])
    for i in sorted(res):
        b = res[i]; num = int(b['number'], 16)
        fb.write(json.dumps(dict(number=num, timestamp=int(b['timestamp'], 16), milliTimestamp=int(b['milliTimestamp'], 16) if b.get('milliTimestamp') else None,
                                 n_tx=len(b['transactions']))) + '\n')
        n_tx += len(b['transactions'])
        for x in b['transactions']:
            if (x.get('to') or '').lower() in KEEP_TO:
                rec = dict(hash=x['hash'], block=num, index=int(x['transactionIndex'], 16), frm=x['from'].lower(), to=x['to'].lower(), nonce=int(x['nonce'], 16))
                ft.write(json.dumps(rec) + '\n'); kept.append(x['hash'])
    if (s - b_lo) % 500 == 0:
        print(f'[scan] block {s} kept {len(kept)} of {n_tx} txs, requests {n_req}', flush=True); io.mem_check('scan')
fb.close(); ft.close()
for s in range(0, len(kept), 40):
    hs = kept[s:s + 40]
    res = rpc([dict(jsonrpc='2.0', id=i, method='eth_getTransactionReceipt', params=[h]) for i, h in enumerate(hs)])
    for i in sorted(res):
        fr.write(json.dumps(res[i]) + '\n')
    if s % 2000 == 0:
        print(f'[receipts] {s + len(hs)}/{len(kept)} requests {n_req}', flush=True)
fr.close()
io.write_json(f'{OUT}/fetch_summary.json', dict(day=day, t0=T0, t1=T1, rpc=RPC, block_lo=b_lo, block_hi=b_hi, n_blocks=b_hi - b_lo + 1,
                                               n_tx_scanned=n_tx, n_tx_kept=len(kept), n_requests=n_req, keep_to=sorted(KEEP_TO)))
print('[done]', dict(blocks=b_hi - b_lo + 1, tx_scanned=n_tx, kept=len(kept), requests=n_req), flush=True)
