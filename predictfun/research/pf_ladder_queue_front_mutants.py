"""MUTATION CHECK of pf_ladder_queue_front (QUEUE-PRIORITY-SENSITIVITY-v1; 2026-11-22, session aea5bfb9), run by pf_mutant_one.py with
pf_ladder_queue_front_selftest. Required: the control passes; every mutant is caught BY A CHECK; every target occurs exactly once.
Not planted (equivalent on the engine's paths by construction): class_rows' t <= END - 1 s cut where the +1 s mid already exists only before END
(the selftest still plants a synthetic last-second mark and checks the cut)."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / 'pf_ladder_queue_front.py'
SELFTEST = 'pf_ladder_queue_front_selftest'
FAIL = 'QUEUE FRONT SELFTEST FAILED'
MUTANTS = [
    ('the front never applied', "                o['ahead'] = 0.; self.front_set += 1", "                self.front_set += 1"),
    ('the front applied to refused orders', "        if o['new'] and o['rem'] > 0 and not o['refused']:", "        if o['new']:"),
    ('the front applied in CURRENT', "            if self.front:", "            if True:"),
    ('accepted generations not counted', "            self.accepted_new += 1\n", ""),
    ('the swap allowed after an order', "    C.require(not B.E.o and not B.E.fills and B.E.it == 0, 'the engine can only be replaced before any order or trade')",
     "    C.require(True, 'the engine can only be replaced before any order or trade')"),
    ('the swapped engine with another END', "B.E = FrontLadderEngine({'frames': B.E.f, 'trades': B.E.tr}, B.end, front)", "B.E = FrontLadderEngine({'frames': B.E.f, 'trades': B.E.tr}, B.end - 1000., front)"),
    ('the arms swapped', "B = branch(path, ident, arm == 'FRONT_AT_ACTIVATION')", "B = branch(path, ident, arm == 'CURRENT_QUEUE')"),
    ('a near price counted same-price', "cls = 'same_price' if tp == float(p) else", "cls = 'same_price' if abs(tp - float(p)) <= 1. else"),
    ('any other price counted through', "('through' if (tp < float(p) if int(s) == 0 else tp > float(p)) else 'invalid')", "('through' if tp != float(p) else 'invalid')"),
    ('the trade side not checked', "C.require(cls != 'invalid' and int(E.tr[it, 1]) == int(s), 'a fill at a trade the engine cannot fill')", "C.require(cls != 'invalid', 'a fill at a trade the engine cannot fill')"),
    ('three pilot markets per date', "PARTS, PILOT_PER_DAY = ('pilot', 'rest'), 2", "PARTS, PILOT_PER_DAY = ('pilot', 'rest'), 3"),
    ('the last-second cut dropped', "            if t <= end - 1000. and mm[6] is not None:", "            if mm[6] is not None:"),
    ('the positive and negative parts swapped', "r['pos1s'] += max(v, 0.); r['neg1s'] += min(v, 0.)", "r['pos1s'] += min(v, 0.); r['neg1s'] += max(v, 0.)"),
    ('a shared trade needing three fills', "r['shared_trade_fills'] += int(cnt[int(f[5])] > 1)", "r['shared_trade_fills'] += int(cnt[int(f[5])] > 2)"),
    ('the 5 s censoring read from the 1 s mid', "for h, j in ((5, 8), (30, 10)):", "for h, j in ((5, 6), (30, 10)):"),
    ('the END value without the side sign', "r['end'] += SG[s] * q * (mark - p)", "r['end'] += q * (mark - p)"),
    ('the volume term at FRONT\'s per-share', "vol, val = (qb - qa) * pa, qb * (pb - pa)", "vol, val = (qb - qa) * pb, qa * (pb - pa)"),
    ('a negative FRONT END read as non-negative', "reading = ('queue_helps_policy_nonnegative' if econ == 'IMPROVES' and tot[F]['END'] >= 0 else",
     "reading = ('queue_helps_policy_nonnegative' if econ == 'IMPROVES' and tot[F]['END'] >= -1e9 else"),
    ('does-not-rescue without added fills', "else 'front_priority_does_not_rescue' if adds and dT <= 0 else", "else 'front_priority_does_not_rescue' if dT <= 0 else"),
    ('equal filled shares counted as added', "adds = bool(tot[F]['q'] > tot[Cq]['q'])", "adds = bool(tot[F]['q'] >= tot[Cq]['q'])"),
    ('the economic reading from END alone', "econ = 'IMPROVES' if eA == mA == 'POSITIVE_EVERY_DAY' else", "econ = 'IMPROVES' if eA == 'POSITIVE_EVERY_DAY' else"),
    ('the projection without write-once', "    proj_path = Path(proj_path or REPO / RECEIPT / 'logs' / 'projection.json'); C.require(not proj_path.exists(), 'the projection exists (write-once)')",
     "    proj_path = Path(proj_path or REPO / RECEIPT / 'logs' / 'projection.json')"),
    ('the rest without the projection gate', "C.require(part == 'pilot' or (proj_path.exists() and json.loads(proj_path.read_text()).get('run_rest') is True), 'the rest needs the write-once projection with run_rest true')",
     "C.require(True, 'the rest needs the write-once projection with run_rest true')"),
    ('measure without write-once', "C.require(not (md / 'audit.json').exists(), 'the measurement exists (write-once)');", ""),
    ('run_day without write-once', "C.require(not P_['rec'].exists(), f'{day}: the output exists (write-once)');", ""),
    ('run_day without the treatment identity', "    C.require(gm['treatment_sha256'] == treatment_sha(), 'the treatment differs from the frozen one')\n", ""),
]
