"""B13d (2026-11-11, diagnosis, descriptive). Diagnosis (descriptive): why do v3's touch lanes stop quoting? Recompute research's touch-lane withhold chain
(pf-strategy desired_ladder at pfmm-ladder-v3-tokyo-6) from the inputs logged in the Rust parity decisions of tokyo-5
(continuous-parity-20261108, 105 windows of 11-07 / 11-08, consumed days), and attribute each transition of a touch lane
from quoting to not quoting; also size-only changes. Reason order as the code: G4 off-touch, G3 gap, adverse (Binance
flag / pull line / no fresh value), then inventory and the rest."""
import gzip, json, math, sys
from collections import Counter, defaultdict
L90 = [0.0492, 0.0623, 0.0789, 0.0917]; L95 = [0.0698, 0.0841, 0.1009, 0.1094]
def minute(rel): return 0 if rel < 60 else (1 if rel < 120 else (2 if rel < 180 else 3))
def reason(d, side, want):
    book = d['book']; bids, asks = book['bids'], book['asks']
    if not bids or not asks: return 'no book'
    b1, a1 = bids[0][0], asks[0][0]; bs, as_ = bids[0][1], asks[0][1]
    if b1 < 0 or a1 < 0 or a1 <= b1: return 'no book'
    micro = (b1 * as_ + a1 * bs) / max(bs + as_, 1e-12); half = (a1 - b1) / 2.0
    qb = math.floor(micro - half + 1e-9); qa = math.ceil(micro + half - 1e-9)
    p = qb if side == 0 else qa
    if side == 0 and not (1 <= qb < a1): return 'micro price invalid'
    if side == 1 and not (b1 < qa <= 99): return 'micro price invalid'
    if (side == 0 and p < b1) or (side == 1 and p > a1): return 'G4 off-touch (micro)'
    touch = b1 if side == 0 else a1
    nxt = (bids[1][0] if len(bids) > 1 else -1) if side == 0 else (asks[1][0] if len(asks) > 1 else -1)
    if p == touch and (nxt < 0 or abs(touch - nxt) >= 2): return 'G3 gap behind the touch'
    v = d['v'][side]
    if d['binance'][side]: return 'G1/G2 Binance guard'
    if v is None: return 'GS no fresh value'
    if v <= L90[minute(d['rel_s'])]: return 'pull line (v <= PM90)'
    rc = d['rung_cur'][side]
    if rc is not None and rc[0] == p: return 'rung at the touch price'
    return 'inventory / other'
lanes_prev = {}; trans = Counter(); back = defaultdict(list); resize = Counter(); n = 0; last_t = {}
pend = {}
with gzip.open(sys.argv[1], 'rt') as fh:
    for line in fh:
        d = json.loads(line); n += 1
        if d.get('held'): continue
        m = d['market']; t = d['t']
        if not (0 <= d['rel_s'] < 60): 
            lanes_prev[m] = d['lanes']; continue
        prev = lanes_prev.get(m)
        for lane in (0, 1):
            cur = d['lanes'][lane]
            key = (m, lane)
            if key in pend and cur[0] is not None:
                back[pend[key][1]].append(t - pend[key][0]); del pend[key]
            if prev is None: continue
            pw = prev[lane]
            if pw[0] is not None and cur[0] is None:
                r = reason(d, lane, cur); trans[r] += 1; pend[key] = (t, r)
            elif pw[0] is not None and cur[0] is not None and pw[0] == cur[0] and abs(pw[1] - cur[1]) > 1e-9:
                resize[f'{pw[1]:g}->{cur[1]:g}'] += 1
        lanes_prev[m] = d['lanes']
tot = sum(trans.values())
print('decisions read', n, '| touch-lane quote -> no-quote transitions in the first minute:', tot)
for r, c in trans.most_common():
    b = sorted(back[r]); k = len(b)
    q = lambda f: (b[int(f * (k - 1))] if k else float('nan'))
    print(f'  {r}: {c} ({100 * c / tot:.1f}%); back quoting: n {k}, within 100 ms {sum(x <= 100 for x in b) / max(k, 1) * 100:.0f}%, p50 {q(0.5):.0f} ms')
print('size-only changes at the same price (first minute):', dict(resize))
