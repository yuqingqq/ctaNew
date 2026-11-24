"""Positive controls and corruptions for the signed evidence gate; no real data."""
import copy
import json
from pf_mt_signed_evidence import *
REFUSALS = []


def expect_bad(fn, message):
    try:
        fn()
    except ValueError as e:
        assert message in str(e), (message, str(e))
        REFUSALS.append(message)
    else:
        raise AssertionError('accepted bad input: ' + message)


def enc_order(v):
    return b''.join(map(word, v)) + word(13*32) + word(1) + b'\x01' + b'\0'*31


def enc_call(taker, makers):
    t = enc_order(taker)
    bodies = [enc_order(v) for v in makers]
    offsets, off = [], len(makers)*32
    for b in bodies:
        offsets.append(word(off)); off += len(b)
    m = word(len(makers)) + b''.join(offsets+bodies)
    fill = word(len(makers)) + b''.join(word(1) for _ in makers)
    return '0x' + (SELECTOR + word(128) + word(128+len(t)) + word(1) + word(128+len(t)+len(m)) + t + m + fill).hex()


def event(v, role='maker', mf=2*E18, tf=5*E18):
    assets = (0, v[4]) if v[10] == 0 else (v[4], 0)
    return dict(topics=[FILLED, order_hash(v), '0x'+word(v[1]).hex(),
                        '0x'+word(int(EX, 16) if role == 'taker' else 99).hex()],
                data='0x'+b''.join(map(word, (*assets, mf, tf, 0))).hex())


def main():
    v = [123456789, int('11'*20,16), int('11'*20,16), 0, E18, 4*E18, 10*E18, 4102444800, 0, 200, 0, 0]
    assert kh(TYPE.encode()).hex() == 'a852566c4e14d00869b6db0220888a9090a13eccdaea03713ff0a3d27bf9767c'
    assert DOMAIN.hex() == '61e5a410200861954c73b7ec87a9054507b7b09d95035434808a719bf33bac8b'
    assert order_hash(v) == '0xe6a0edf257e9904e5fd4a1b8fdc3dd092e8789c4fa596ff6275097515ee4fd34'
    # The two makers share an address/price; joining on maker alone would merge them.
    v2 = v.copy(); v2[0] += 1
    tk = v.copy(); tk[0] += 2; tk[10] = 1
    inp = enc_call(tk, [v, v2]); got = decode(inp)
    assert [r['values'] for r in got] == [tk, v, v2]
    assert len({r['hash'] for r in got}) == 3
    assert bind_event(got[1], event(v)) == (2*E18, 5*E18, 0)
    assert bind_event(got[0], event(tk, 'taker', 2*E18, E18)) == (2*E18, E18, 0)
    for i in (0, 1, 4, 5, 6, 7, 10):
        bad = copy.deepcopy(got[1]); bad['values'][i] ^= 1
        expect_bad(lambda: bind_event(bad, event(v)), 'hash mismatch')
    bad = event(v); bad['topics'][2] = '0x'+word(88).hex()
    expect_bad(lambda: bind_event(got[1], bad), 'maker mismatch')
    expect_bad(lambda: bind_event(got[1], event(v, 'taker')), 'role mismatch')
    expect_bad(lambda: bind_event(got[1], event(v, mf=5*E18)), 'fill bounds')
    bad = event(v); bad['data'] = '0x'+word(88).hex()+bad['data'][66:]
    expect_bad(lambda: bind_event(got[1], bad), 'assets mismatch')
    expect_bad(lambda: decode('0x00000000'+inp[10:]), 'selector')
    expect_bad(lambda: decode(inp[:-2]), 'length')
    bad = bytearray.fromhex(inp[2:]); bad[4:36] = word(3)
    expect_bad(lambda: decode('0x'+bad.hex()), 'head overlap')
    bad = v.copy(); bad[1] = 2**160
    expect_bad(lambda: order_hash(bad), 'address padding')
    assert classify(v, 2*E18, 1, 1)['budget_status'] == 'observed_budget_incomplete'
    assert not classify(v, 2*E18, 2, 1)['cancellation_identified']
    assert not classify(v, 2*E18, 2, 1)['same_hash_multiple_transactions']
    assert classify(v, 4*E18, 2, 2)['budget_status'] == 'exhausted_by_observed_fills'
    expect_bad(lambda: classify(v, 4*E18+1, 2, 2), 'cumulative making')
    print(json.dumps({'selftest': 'PASS', 'external_hash_vector': True,
                      'known_bad_refusals': len(REFUSALS), 'accounting_and_identification_controls': True}))


if __name__ == '__main__':
    main()
