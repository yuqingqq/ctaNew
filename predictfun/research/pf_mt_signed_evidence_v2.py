"""SIGNED-ORDER-EVIDENCE-GATE-v2: offline, exact-hash source capability gate.

Contract: reviews/maker_signed_evidence_20261124/declaration.md. No network,
strategy, fit, clock inference or cancellation inference. All quantities used
for identity/accounting are integers in 1e-18 units. Snapshot disappearance
and an incompletely observed fill budget are NOT cancellation observations.
"""
import argparse
import collections
import csv
import gzip
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import time

from Crypto.Hash import keccak

ROOT = Path('/home/yuqing/ctaNew')
HERE = ROOT / 'predictfun/reviews/maker_signed_evidence_20261124/v2'
CACHE = ROOT / 'data/predict_fun/research/onchain/20261023_1400_1415'
OUT = ROOT / 'data/predict_fun/research/maker_signed_evidence_v2_20261124'
EX = '0x8bc070bedab741406f4b1eb65a72bee27894b689'
E18 = 10**18
FIELDS = 'salt maker signer taker tokenId makerAmount takerAmount expiration nonce feeRateBps side signatureType'.split()
TYPE = 'Order(uint256 salt,address maker,address signer,address taker,uint256 tokenId,uint256 makerAmount,uint256 takerAmount,uint256 expiration,uint256 nonce,uint256 feeRateBps,uint8 side,uint8 signatureType)'
TUPLE = '(uint256,address,address,address,uint256,uint256,uint256,uint256,uint256,uint256,uint8,uint8,bytes)'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def kh(b):
    return keccak.new(digest_bits=256, data=b).digest()


def word(n):
    return n.to_bytes(32, 'big')


DOMAIN = kh(kh(b'EIP712Domain(string name,string version,uint256 chainId,address verifyingContract)') +
            kh(b'predict.fun CTF Exchange') + kh(b'1') + word(56) + word(int(EX, 16)))
SELECTOR = kh(f'matchOrders({TUPLE},{TUPLE}[],uint256,uint256[])'.encode())[:4]
FILLED = '0x' + kh(b'OrderFilled(bytes32,address,address,uint256,uint256,uint256,uint256,uint256)').hex()
CANCELLED = '0x' + kh(b'OrderCancelled(bytes32)').hex()


def order_hash(vals):
    need(len(vals) == 12, 'order length')
    need(all(0 <= vals[i] < 2**160 for i in (1, 2, 3)), 'address padding')
    need(vals[10] in (0, 1) and 0 <= vals[11] <= 255, 'side/type')
    need(vals[5] > 0 and vals[6] > 0, 'zero amounts')
    return '0x' + kh(b'\x19\x01' + DOMAIN + kh(kh(TYPE.encode()) + b''.join(map(word, vals)))).hex()


def decode(inp):
    raw = bytes.fromhex(inp.removeprefix('0x'))
    need(raw[:4] in (SELECTOR, bytes.fromhex('92895911')), 'unsupported selector')
    operator_wrapper = raw[:4] == bytes.fromhex('92895911')
    data = raw[4:]
    need(len(data) % 32 == 0 and len(data) >= 128, 'ABI length')

    def uint(at):
        need(at >= 0 and at % 32 == 0 and at + 32 <= len(data), 'ABI offset')
        return int.from_bytes(data[at:at+32], 'big')

    def get_order(at, role):
        vals = [uint(at + i*32) for i in range(12)]
        sig = at + uint(at + 12*32)
        need(sig >= at + 13*32, 'signature offset')
        n_sig = uint(sig)
        need(sig + 32 + n_sig <= len(data), 'signature length')
        return {'hash': order_hash(vals), 'values': vals, 'role': role}

    ta, ma = uint(0), uint(32)
    need(ta >= 128 and ma >= 128, 'head overlap')
    n = uint(ma)
    need(0 < n <= 10000 and ma + 32*(n+1) <= len(data), 'maker count')
    orders = [get_order(ta, 'taker')]
    for j in range(n):
        off = uint(ma + 32*(1+j))
        need(off >= 32*n, 'maker head overlap')
        orders.append(get_order(ma + 32 + off, 'maker'))
    # Validate the companion array, even though accounting uses emitted logs.
    if not operator_wrapper:
        fa = uint(96)
        need(uint(fa) == n and fa + 32*(1+n) <= len(data), 'fill array')
    # Historical operator wrapper: only the first two dynamic order arguments
    # are decoded. No meaning inferred for other arguments. Exact hash/event
    # binding below is mandatory; a pointer guess cannot authorize a row.
    return orders


def rows(path):
    with gzip.open(path, 'rt') as f:
        for line in f:
            yield json.loads(line)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write('\n')


def bind_event(o, lg):
    """Identity binds the full signed tuple, not maker/price coincidence."""
    v = o['values']
    need(order_hash(v) == lg['topics'][1].lower(), 'event hash mismatch')
    need(int(lg['topics'][2], 16) == v[1], 'event maker mismatch')
    d = bytes.fromhex(lg['data'][2:])
    need(len(d) == 160, 'event data length')
    ma, ta, mf, tf, fee = [int.from_bytes(d[i:i+32], 'big') for i in range(0, 160, 32)]
    need((ma, ta) == ((0, v[4]) if v[10] == 0 else (v[4], 0)), 'event assets mismatch')
    need(mf > 0 and tf > 0 and mf <= v[5], 'event fill bounds')
    need((int(lg['topics'][3], 16) == int(EX, 16)) == (o['role'] == 'taker'), 'event role mismatch')
    return mf, tf, fee


def classify(v, making, n_events, n_txs):
    need(0 < making <= v[5], 'cumulative making exceeds signed budget')
    return dict(budget_status='exhausted_by_observed_fills' if making == v[5] else 'observed_budget_incomplete',
                same_hash_multiple_events=n_events > 1, same_hash_multiple_transactions=n_txs > 1,
                cancellation_identified=False, posting_time_identified=False, queue_front_identified=False)


def sources():
    return [CACHE / x for x in ('inputs.jsonl.gz', 'receipts.jsonl.gz', 'blocks.jsonl.gz', 'tokens.json', 'fetch_summary.json')] + [
        ROOT / 'data/predict_fun/research/maker_trace_v1_20261122/econ/identities.json']


def manifest():
    pipeline = [Path(__file__), Path(__file__).with_name('pf_mt_signed_evidence_selftest_v2.py'), HERE / 'check.py', HERE / 'declaration.md', ROOT / 'predictfun/research/pf_mt_signed_evidence_selftest.py', ROOT / 'predictfun/research/pf_mt_signed_evidence.py']
    dump(HERE / 'manifest.json', {'version': 'signed_order_evidence_v2', 'scope': 'cached_20261023_1400_1415_with_saved_padding',
                                'inputs': {str(p.relative_to(ROOT)): sha(p) for p in sources()},
                                'source_sha256': sha(Path(__file__)), 'candidate_count': 0,
                                'pipeline_sha256': {str(p.relative_to(ROOT)): sha(p) for p in pipeline}})


def run():
    start = time.process_time()
    m = json.loads((HERE / 'manifest.json').read_text())
    need(m['source_sha256'] == sha(Path(__file__)), 'source identity')
    for p, h in m['inputs'].items():
        need(sha(ROOT / p) == h, f'input identity {p}')
    # A real freeze must contain this source and manifest, including working bytes.
    for rel, h in m['pipeline_sha256'].items():
        need(sha(ROOT / rel) == h, 'pipeline identity '+rel)
    for p in [ROOT / rel for rel in m['pipeline_sha256']] + [HERE / 'manifest.json']:
        rel = str(p.relative_to(ROOT))
        committed = subprocess.check_output(['git', '-C', str(ROOT), 'show', 'HEAD:' + rel])
        need(committed == p.read_bytes(), f'uncommitted source {rel}')
    need(not OUT.exists(), 'write-once output')
    OUT.mkdir(parents=True)
    tok = json.loads((CACHE / 'tokens.json').read_text())
    ids = json.loads(sources()[-1].read_text())
    ranks = {r['addr'].lower(): int(r['rank']) for r in ids['ex_ante']['top10_shares']}
    blocks = {int(r['number']): int(r.get('milliTimestamp') or r['timestamp']*1000) for r in rows(CACHE / 'blocks.jsonl.gz')}
    inputs = {}
    for r in rows(CACHE / 'inputs.jsonl.gz'):
        tx = r['hash'].lower()
        need(tx not in inputs or inputs[tx] == r['input'], 'conflicting input')
        inputs[tx] = r['input']
    decoded, statuses, selectors = {}, [], collections.Counter()
    for tx, inp in sorted(inputs.items()):
        selectors[inp[:10]] += 1
        try:
            os = decode(inp)
            by_hash = {}
            for o in os:
                need(o['hash'] not in by_hash or by_hash[o['hash']] == o, 'duplicate hash roles')
                by_hash[o['hash']] = o
            decoded[tx] = by_hash
            statuses.append({'tx': tx, 'status': 'decoded'})
        except ValueError as e:
            statuses.append({'tx': tx, 'status': str(e)})
    seen_events, book, event_rows = set(), {}, []
    counts = collections.Counter()
    cancellations = []
    for rec in rows(CACHE / 'receipts.jsonl.gz'):
        tx = rec['transactionHash'].lower()
        if rec['status'] != '0x1':
            counts['failed_receipts'] += 1
            continue
        for lg in rec['logs']:
            if lg['address'].lower() != EX or not lg['topics']:
                continue
            topic = lg['topics'][0].lower()
            if topic == CANCELLED:
                cancellations.append({'tx': tx, 'log_index': int(lg['logIndex'], 16), 'hash': lg['topics'][1].lower(), 'block_ms': blocks[int(rec['blockNumber'], 16)]})
            if topic != FILLED:
                continue
            data = bytes.fromhex(lg['data'][2:])
            need(len(data) == 160, 'event ABI')
            asset0, asset1 = [int.from_bytes(data[i:i+32], 'big') for i in (0, 32)]
            token = asset1 if asset0 == 0 else asset0
            if str(token) not in tok:
                counts['outside_known_BTC_tokens'] += 1
                continue
            key = (tx, int(lg['logIndex'], 16))
            need(key not in seen_events, 'duplicate event')
            seen_events.add(key)
            h = lg['topics'][1].lower()
            if tx not in decoded or h not in decoded[tx]:
                counts['unbound_BTC_events'] += 1
                continue
            o = decoded[tx][h]
            mf, tf, fee = bind_event(o, lg)
            v = o['values']
            b = int(rec['blockNumber'], 16)
            need(b in blocks, 'missing block time')
            ms = blocks[b]
            addr = '0x' + v[1].to_bytes(20, 'big').hex()
            if h not in book:
                book[h] = dict(values=v, maker=addr, roles=set(), txs=set(), making=0, shares=0, n_events=0, times=[])
            r = book[h]
            need(r['values'] == v, 'same hash different content')
            r['roles'].add(o['role']); r['txs'].add(tx); r['making'] += mf
            r['shares'] += tf if v[10] == 0 else mf
            r['n_events'] += 1; r['times'].append(ms)
            event_rows.append(dict(tx=tx, log_index=key[1], order_hash=h, role=o['role'], block_ms=ms,
                                   making_wei=str(mf), taking_wei=str(tf), fee_wei=str(fee)))
            counts['bound_BTC_events'] += 1
        need(time.process_time()-start < 50, 'CPU guard 50 seconds')
        need(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 1000*1024, 'RSS guard 1000 MiB')
    orders = []
    for h, r in sorted(book.items()):
        v = r['values']; mid, ix = tok[str(v[4])]
        orders.append(dict(order_hash=h, signed_values_json=json.dumps(v), maker=r['maker'], top10_volume_rank=ranks.get(r['maker'], ''), market=mid, outcome=ix,
                           role='|'.join(sorted(r['roles'])), side=v[10], original_making_wei=str(v[5]),
                           signed_share_target_wei=str(v[6] if v[10] == 0 else v[5]),
                           observed_making_wei=str(r['making']), unseen_making_budget_upper_wei=str(v[5]-r['making']),
                           observed_shares_wei=str(r['shares']), expiration_s=v[7],
                           first_settlement_ms=min(r['times']), last_settlement_ms=max(r['times']),
                           settlement_span_ms=max(r['times'])-min(r['times']), n_events=r['n_events'], n_transactions=len(r['txs']),
                           **classify(v, r['making'], r['n_events'], len(r['txs']))))
    def csv_write(path, rr):
        if not rr:
            return
        with path.open('x', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rr[0])); w.writeheader(); w.writerows(rr)
    csv_write(OUT / 'orders.csv', orders); csv_write(OUT / 'events.csv', event_rows); csv_write(OUT / 'transactions.csv', statuses)
    for c in cancellations:
        linked = book.get(c['hash'])
        c['maker_from_signed_order'] = linked['maker'] if linked else ''
        c['top10_volume_rank'] = ranks.get(c['maker_from_signed_order'], '')
        c['link_status'] = 'exact_hash' if linked else 'owner_unresolved'
        c['offchain_cancel_time_identified'] = False
    csv_write(OUT / 'cancellations.csv', cancellations)
    groups = {}
    for name, group in [('all', orders), ('maker', [r for r in orders if 'maker' in r['role']]),
                        ('top3_maker', [r for r in orders if r['top10_volume_rank'] in (1, 2, 3) and 'maker' in r['role']])]:
        groups[name] = dict(orders=len(group), repeated_events=sum(r['same_hash_multiple_events'] for r in group),
                           repeated_transactions=sum(r['same_hash_multiple_transactions'] for r in group),
                           budget_status=dict(collections.Counter(r['budget_status'] for r in group)),
                           expiration_values=dict(collections.Counter(str(r['expiration_s']) for r in group)))
    result = dict(source_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD']).decode().strip(),
                  scope=m['scope'], as_of_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), counts=dict(counts),
                  groups=groups, cached_transactions=len(inputs), selectors=dict(selectors),
                  transaction_statuses=dict(collections.Counter(r['status'] for r in statuses)),
                  cached_cancel_events=len(cancellations),
                  cancel_links=dict(collections.Counter(c['link_status'] for c in cancellations)),
                  top3_linked_cancels=sum(c['top10_volume_rank'] in (1,2,3) for c in cancellations),
                  gate_pass=counts['bound_BTC_events'] > 0 and counts['unbound_BTC_events'] == 0,
                  population_limit='one consumed Oct23 cached slice; executed orders only; no unfilled-order census',
                  inference={'original_signed_terms': bool(orders), 'post_time': False, 'offchain_cancel_time': False,
                             'queue_front': False, 'remaining_exact_without_prior_coverage': False, 'profitability': False},
                  output_sha256={p.name: sha(p) for p in sorted(OUT.glob('*.csv'))},
                  cpu_seconds=time.process_time()-start)
    dump(OUT / 'summary.json', result)
    dump(HERE / 'summary.json', result)
    # Examples selected by hash BEFORE looking at sizes/outcomes; all statuses in full output.
    csv_write(HERE / 'examples.csv', [r for r in orders if r['top10_volume_rank'] in (1, 2, 3)][:12])
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['manifest', 'run']); a = ap.parse_args()
    manifest() if a.mode == 'manifest' else run()
