"""Independent raw-receipt membership and integer-accounting check; no decoder import."""
import collections
import copy
import csv
import gzip
import hashlib
import json
from pathlib import Path
from Crypto.Hash import keccak

ROOT = Path('/home/yuqing/ctaNew')
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'data/predict_fun/research/maker_signed_evidence_v3_20261124'
CACHE = ROOT / 'data/predict_fun/research/onchain/20261023_1400_1415'
EX = '0x8bc070bedab741406f4b1eb65a72bee27894b689'
TOPIC = '0xd0a08e8c493f9c94f29311604c9de1b4e8c8d4c06bd0c789af57f2d65bfec0f6'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def verify(orders, events, raw):
    require(len({r['order_hash'] for r in orders}) == len(orders), 'duplicate order')
    ordermap = {r['order_hash']: r for r in orders}
    eventmap = {(r['tx'], int(r['log_index'])): r for r in events}
    require(len(eventmap) == len(events), 'duplicate event')
    require(set(eventmap) == set(raw), 'event membership')
    sums = collections.defaultdict(lambda: [0, 0, set(), []])
    for key, (h, ma, ta, mf, tf, fee, maker, role) in raw.items():
        r = eventmap[key]
        require(r['order_hash'] == h and h in ordermap, 'event order')
        require(r['role'] == role and (int(r['making_wei']), int(r['taking_wei']), int(r['fee_wei'])) == (mf, tf, fee), 'event accounting')
        o = ordermap[h]; v = json.loads(o['signed_values_json'])
        require(o['maker'] == maker and int(maker, 16) == v[1], 'maker')
        require((ma, ta) == ((0, v[4]) if v[10] == 0 else (v[4], 0)), 'assets')
        s = sums[h]; s[0] += mf; s[1] += tf if v[10] == 0 else mf; s[2].add(key[0]); s[3].append(int(r['block_ms']))
    require(set(sums) == set(ordermap), 'order membership')
    for h, o in ordermap.items():
        v = json.loads(o['signed_values_json'])
        k = keccak.new(digest_bits=256)
        k.update(bytes.fromhex('a852566c4e14d00869b6db0220888a9090a13eccdaea03713ff0a3d27bf9767c') + b''.join(x.to_bytes(32, 'big') for x in v))
        hh = keccak.new(digest_bits=256)
        hh.update(b'\x19\x01' + bytes.fromhex('61e5a410200861954c73b7ec87a9054507b7b09d95035434808a719bf33bac8b') + k.digest())
        require('0x'+hh.hexdigest() == h, 'signed hash')
        making, shares, txs, ts = sums[h]
        require(int(o['original_making_wei']) == v[5], 'original amount')
        require(int(o['expiration_s']) == v[7] and int(o['side']) == v[10], 'terms')
        require(int(o['signed_share_target_wei']) == (v[6] if v[10] == 0 else v[5]), 'signed target')
        require(0 < making <= v[5] and int(o['observed_making_wei']) == making and int(o['observed_shares_wei']) == shares, 'order accounting')
        require(int(o['unseen_making_budget_upper_wei']) == v[5]-making, 'remaining upper bound')
        require(o['budget_status'] == ('exhausted_by_observed_fills' if making == v[5] else 'observed_budget_incomplete'), 'budget status')
        require(int(o['n_events']) == len(ts) and int(o['n_transactions']) == len(txs), 'counts')
        require(o['same_hash_multiple_events'] == str(len(ts) > 1) and o['same_hash_multiple_transactions'] == str(len(txs)>1), 'repeat grades')
        require(int(o['first_settlement_ms']) == min(ts) and int(o['last_settlement_ms']) == max(ts) and int(o['settlement_span_ms']) == max(ts)-min(ts), 'span')
        require(all(o[key] == 'False' for key in ('cancellation_identified','posting_time_identified','queue_front_identified')), 'unsupported identification')
    return {'orders': len(orders), 'events': len(events), 'pass': True}


def main():
    orders = list(csv.DictReader((OUT / 'orders.csv').open()))
    events = list(csv.DictReader((OUT / 'events.csv').open()))
    tok = set(json.loads((CACHE / 'tokens.json').read_text()))
    raw = {}
    cancel_topic = '0x'+keccak.new(digest_bits=256, data=b'OrderCancelled(bytes32)').hexdigest()
    raw_cancels = {}
    block_ms = {}
    with gzip.open(CACHE / 'blocks.jsonl.gz', 'rt') as f:
        for line in f:
            b = json.loads(line); block_ms[int(b['number'])] = int(b.get('milliTimestamp') or b['timestamp']*1000)
    with gzip.open(CACHE / 'receipts.jsonl.gz', 'rt') as f:
        for line in f:
            rec = json.loads(line)
            if rec['status'] != '0x1':
                continue
            for lg in rec['logs']:
                if lg['address'].lower() == EX and lg['topics'] and lg['topics'][0].lower() == cancel_topic:
                    ck = (rec['transactionHash'].lower(), int(lg['logIndex'],16))
                    require(ck not in raw_cancels, 'duplicate cancel')
                    raw_cancels[ck] = (lg['topics'][1].lower(), block_ms[int(rec['blockNumber'],16)])
                if lg['address'].lower() != EX or not lg['topics'] or lg['topics'][0].lower() != TOPIC:
                    continue
                d = lg['data'][2:]
                require(len(d) == 5*64, 'event ABI')
                ma, ta, mf, tf, fee = [int(d[i:i+64], 16) for i in range(0, len(d), 64)]
                if str(ta if ma == 0 else ma) not in tok:
                    continue
                key = rec['transactionHash'].lower(), int(lg['logIndex'],16)
                require(key not in raw, 'duplicate raw event')
                raw[key] = (lg['topics'][1].lower(), ma, ta, mf, tf, fee, '0x'+lg['topics'][2][-40:].lower(),
                            'taker' if int(lg['topics'][3],16) == int(EX,16) else 'maker')
                # Compare against the source clock, independently of exported span fields.
                source_time = block_ms[int(rec['blockNumber'],16)]
                # Map created once below for the actual comparison.
                raw[key] += (source_time,)
    time_map = {k: r[-1] for k,r in raw.items()}
    raw = {k:r[:-1] for k,r in raw.items()}
    for r in events:
        require(int(r['block_ms']) == time_map[(r['tx'],int(r['log_index']))], 'clock mismatch')
    result = verify(orders, events, raw)
    cancels = list(csv.DictReader((OUT/'cancellations.csv').open()))
    require(len(cancels) == len(raw_cancels), 'cancel count')
    require({(r['tx'],int(r['log_index'])) for r in cancels} == set(raw_cancels), 'cancel membership')
    om = {r['order_hash']: r for r in orders}
    for c in cancels:
        h, ms = raw_cancels[(c['tx'],int(c['log_index']))]
        require(c['hash'] == h and int(c['block_ms']) == ms, 'cancel source')
        linked = om.get(h)
        require(c['maker_from_signed_order'] == (linked['maker'] if linked else ''), 'cancel owner')
        require(c['link_status'] == ('exact_hash' if linked else 'owner_unresolved'), 'cancel grade')
        require(c['offchain_cancel_time_identified'] == 'False', 'cancel clock grade')
    result['cancel_events'] = len(cancels)
    corruptions = {}
    for label, field, value in [('original', 'original_making_wei', '1'), ('grade', 'cancellation_identified', 'True'),
                                ('span', 'settlement_span_ms', '-1')]:
        bad = copy.deepcopy(orders); bad[0][field] = value
        try:
            verify(bad, events, raw)
        except ValueError as e:
            corruptions[label] = str(e)
        else:
            raise AssertionError('falsifier escaped: '+label)
    try:
        verify(orders, events[1:], raw)
    except ValueError as e:
        corruptions['missing_event'] = str(e)
    else:
        raise AssertionError('missing event escaped')
    summary = json.loads((HERE / 'summary.json').read_text())
    require(summary['gate_pass'] and summary['counts']['bound_BTC_events'] == len(events), 'summary gate')
    for name, h in summary['output_sha256'].items():
        require(hashlib.sha256((OUT/name).read_bytes()).hexdigest() == h, 'output identity')
    require(summary == json.loads((OUT/'summary.json').read_text()), 'receipt copy')
    result['falsifiers'] = corruptions
    with (HERE / 'checks.json').open('x') as f:
        json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
