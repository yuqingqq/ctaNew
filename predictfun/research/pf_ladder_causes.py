"""Order-change causes for the ladder engine (USER 2026-11-06: "check if we also replace / cancel / change orders if the prediction does not
change"). Same read-only hook interface as pf_ladder_tracer.py; used only by pf_step4_ladder_causes.py (pf_ladder_causes_build.py).
The engine uses the guard value v of a side only through its bucket against the minute's lines: none (missing / stale, GS), pull
(v <= line 90), small (line 90 < v < line 95), big (v >= line 95). A lane's order changes only when a decision (D event) sets a new
intent. Each intent change is classified against the previous decision of the window, first match wins:
  stop_60s / no_book          the decision returned no quote (G5 stop, or no known frame / no valid book)
  window_start / book_back    no previous live decision in the window (first one, or after a no-book stretch)
  refill_after_<fill|refusal|losscap>   the engine cleared the intent (fill known, post-only / own refusal, loss cap); this re-places
  prediction                  the side's v bucket changed
  binance_guard               the side's G1 / G2 flag changed
  inventory                   known net position changed (G6)
  book_price                  the best bid or ask price changed
  book_size                   only best-level sizes changed (the micro price moved; same prices)
  other
Every cancel / place request is attributed to the cause of its lane's latest intent change (throttle retries carry it). Also: flip-backs
(a lane returns to the order it had before its previous change, within 1 s, possibly via no order): flipback = neither change
prediction-driven, flipback_pred = both, flipback_mixed = one; and the age of each cancelled order (from activation).
Env: PF_CAUSES_OUT (output json, required), PF_CAUSES_LINES (minute-lines json, default step4/minute_lines_t60k.json).
Refuses (SystemExit) if a window's attributed requests differ from its request list; selftest() is the classifier's positive / negative control.
"""
import collections, json, os
from . import pf_io as io

STOP_S = 60.0
LINES = json.load(open(os.environ.get('PF_CAUSES_LINES') or f'{io.OUT}/step4/minute_lines_t60k.json'))['lines']
AGE_EDGES = (0, 100, 250, 500, 1000, 2000, 5000, 1e12)


def vcat(d, s):
    v = d['vv'][s]
    if v is None:
        return 'none'
    i = min(max(int(d['rel'] // 60), 0), 3)
    if v <= LINES['90'][i]:
        return 'pull'
    return 'big' if v >= LINES['95'][i] else 'small'


def has(x):
    return x is not None and x[0] is not None


def change_kind(old, new):
    if not has(old):
        return 'place'
    if not has(new):
        return 'pull'
    return 'reprice' if old[0] != new[0] else 'resize'


def classify(s, rel, prevd, d, old, reset):
    """cause of an intent change on side s (pure function; see the module docstring)."""
    if 'rel' not in d:
        return 'stop_60s' if rel >= STOP_S - 1e-9 else 'no_book'
    if prevd is None or 'rel' not in prevd:
        return 'window_start' if prevd is None else 'book_back'
    if old is None and reset is not None:
        return f'refill_after_{reset}'
    if vcat(d, s) != vcat(prevd, s):
        return 'prediction'
    if d['btc'][s] != prevd['btc'][s]:
        return 'binance_guard'
    if d['knet'] != prevd['knet']:
        return 'inventory'
    if (d['b1'], d['a1']) != (prevd['b1'], prevd['a1']):
        return 'book_price'
    if (d['b1s'], d['a1s']) != (prevd['b1s'], prevd['a1s']):
        return 'book_size'
    return 'other'


def flipback(lp, t, old, new):
    """lp = (t, old, new, pred) of the lane's previous change: this change reverses it (back to the same order, possibly via no order)
    within 1 s."""
    return lp is not None and has(new) and lp[1] == new and lp[2] == old and t - lp[0] <= 1000.0


def selftest():
    base = dict(rel=10.0, vv=(0.2, 0.2), btc=(False, False), knet=0.0, b1=40, a1=41, b1s=100.0, a1s=100.0)
    cases = [(dict(base, vv=(0.0, 0.2)), 'prediction'), (dict(base, vv=(0.06, 0.2)), 'prediction'),
             (dict(base, vv=(0.15, 0.2), b1s=50.0), 'book_size'), (dict(base, b1=39), 'book_price'), (dict(base, btc=(True, False)), 'binance_guard'),
             (dict(base, knet=5.0), 'inventory'), (dict(base, vv=(None, 0.2)), 'prediction'), (dict(base), 'other')]
    for d, want in cases:
        got = classify(0, 10.0, base, d, (40, 25.0), None)
        if got != want:
            raise SystemExit(f'CAUSES SELFTEST FAILED: {d} -> {got}, expected {want}')
    if classify(0, 61.0, base, {}, (40, 5.0), None) != 'stop_60s' or classify(0, 10.0, base, dict(base), None, 'fill') != 'refill_after_fill':
        raise SystemExit('CAUSES SELFTEST FAILED: stop / refill')
    A, B = (40, 5.0), (39, 5.0)                         # flip-back positive / negative controls
    if not (flipback((0.0, A, None, False), 500.0, None, A) and flipback((0.0, A, B, False), 900.0, B, A)):
        raise SystemExit('CAUSES SELFTEST FAILED: a reversal within 1 s not detected')
    if flipback((0.0, A, B, False), 500.0, A, B) or flipback((0.0, A, B, False), 1500.0, B, A) or flipback((0.0, A, None, False), 500.0, None, B):
        raise SystemExit('CAUSES SELFTEST FAILED: a non-reversal read as a flip-back')
    # a v move inside one bucket (0.2 -> 0.15, both big) must NOT read as prediction
    if classify(0, 10.0, base, dict(base, vv=(0.15, 0.2)), (40, 25.0), None) == 'prediction':
        raise SystemExit('CAUSES SELFTEST FAILED: in-bucket move read as prediction')


selftest()


class Tracer:
    def __init__(self):
        self.path = os.environ['PF_CAUSES_OUT']
        self.C = collections.Counter(); self.k = None; self.d = {}

    def set_inputs(self, *a):
        pass

    def win_begin(self, mid, ws, rules, size, F, kf):
        self.mid = int(mid); self.ws = float(ws); self.prevd = None; self.cause = {}; self.reset = {}; self.last = {}
        self.nreq = 0; self.k = None; self.C['windows'] += 1

    def dclear(self):
        self.d = {}

    def dset(self, **kw):
        self.d.update(kw)

    def e_set(self, **kw):
        if 'cancel' in kw and self.k == 'A':
            cur = kw['cancel']['cur']; lane = self.pl[0]
            age = self.t - (cur['t0'] if 't0' in cur else cur['t_act'])   # live order: activation t0; in-flight create: t_act (future -> pending)
            b = next(i for i in range(len(AGE_EDGES) - 1) if age < AGE_EDGES[i + 1]) if age >= 0 else -1
            lab = 'pending' if b < 0 else f'{AGE_EDGES[b]:g}-{AGE_EDGES[b + 1]:g}'
            self.C[f"cancel_age|{'rung' if lane >= 2 else 'touch'}|{self.cause.get(lane, 'unattributed')}|{lab}"] += 1

    def ev(self, t, pr, seq, kind, pl, withheld, live_ctr, intent, requests, fills, knet):
        self.k = kind; self.t = t; self.pl = pl; self.int0 = dict(intent); self.nreq0 = len(requests); self.knet0 = knet

    def post(self, knet, kcash, intent, known, orders, inflight, zombies, TB, cancel_wait, requests, fills, withheld, live_ctr, pend):
        kind = self.k
        if kind is None:
            return
        self.k = None; C = self.C
        changed = [L for L in intent if intent[L] != self.int0.get(L)]
        if kind == 'D':
            C['d_events'] += 1
            d = dict(self.d); d['knet'] = self.knet0; rel = (self.t - self.ws) / 1000.0
            for L in changed:
                s = L % 2; lt = 'rung' if L >= 2 else 'touch'; old, new = self.int0.get(L), intent[L]
                cause = classify(s, rel, self.prevd, d, old, self.reset.get(L))
                ck = change_kind(old, new); self.cause[L] = cause; self.reset[L] = None
                C[f'change|{lt}|{ck}|{cause}'] += 1
                pred = cause == 'prediction'
                lp = self.last.get(L)
                if flipback(lp, self.t, old, new):                # 2026-11-06 fix: the first version tested a repeat, not a reversal
                    fk = 'flipback_pred' if (pred and lp[3]) else ('flipback' if not (pred or lp[3]) else 'flipback_mixed')
                    C[f'{fk}|{lt}|{cause}'] += 1
                if has(old) or has(new):
                    self.last[L] = (self.t, old, new, pred)
            if changed:
                C['d_events_with_change'] += 1
            self.prevd = d if 'rel' in d else ({} if self.prevd is not None else None)
        else:
            for L in changed:
                if intent[L] is None and self.int0.get(L) is not None:
                    self.reset[L] = {'K': 'fill', 'P': 'refusal', 'A': 'losscap'}.get(kind, kind)
                    C[f'intent_reset|{kind}'] += 1
        new_req = requests[self.nreq0:]
        for r in new_req:
            lane = r[3]; lt = 'rung' if lane >= 2 else 'touch'
            C[f"req|{lt}|{r[2]}|{self.cause.get(lane, 'unattributed')}"] += 1
        self.nreq += len(new_req)

    def win_end(self, res, requests, fills):
        if self.nreq != len(requests):
            raise SystemExit(f'CAUSES RECONCILIATION FAILED {self.mid}: attributed {self.nreq} vs {len(requests)} requests')
        self.C['requests'] += len(requests); self.C['fills'] += len(fills)

    def close(self):
        io.write_json(self.path, dict(lines_90=LINES['90'], lines_95=LINES['95'], counts=dict(sorted(self.C.items()))))
