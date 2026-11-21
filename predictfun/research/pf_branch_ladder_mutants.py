"""MUTATION CHECK of pf_branch_ladder (the multi-order ladder client; 2026-11-21, session aea5bfb9), run by pf_mutant_one.py with
pf_ladder_rest_selftest (PF_TEACHER_SYN passed through; run(mod) imports the runner over the mutated client). Required: the control passes;
every mutant is caught BY A CHECK; every target occurs exactly once."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / 'pf_branch_ladder.py'
SELFTEST = 'pf_ladder_rest_selftest'
FAIL = 'LADDER SELFTEST FAILED'
MUTANTS = [
    ('the LONG trigger strict', "    if I >= S - TOL:\n        return 'LONG_RESTRICTED'", "    if I > S + TOL:\n        return 'LONG_RESTRICTED'"),
    ('the recovery band strict', "    if abs(I) <= R + TOL:\n        return 'UNRESTRICTED'", "    if abs(I) < R - TOL:\n        return 'UNRESTRICTED'"),
    ('no direct reversal', "    if I <= -S + TOL:\n        return 'SHORT_RESTRICTED'", "    if I <= -S + TOL and old != 'LONG_RESTRICTED':\n        return 'SHORT_RESTRICTED'"),
    ('cancel-pending orders unreserved', "            for gid in self.unclosed[s]:\n                res[s] += self.E.o[gid]['size'] - self.known_filled(gid, g)",
     "            for gid in [x for x in self.unclosed[s] if self.E.o[x]['req'] is None]:\n                res[s] += self.E.o[gid]['size'] - self.known_filled(gid, g)"),
    ('the removal closed at its effect', "ack = o['remove'] + BE.CANCEL_ACK if o['req'] is not None else np.inf", "ack = o['remove'] if o['req'] is not None else np.inf"),
    ('order fills known without the notice delay', "        return sum(q for t, i, _, q, _, _ in self.E.fills if i == gid and t + BE.FILL_NOTICE <= g)",
     "        return sum(q for t, i, _, q, _, _ in self.E.fills if i == gid and t <= g)"),
    ('the position known without the notice delay', "Ik = sum((1. if s == 0 else -1.) * q for t, _, s, q, _, _ in self.E.fills if t + BE.FILL_NOTICE <= g); res = {0: 0., 1: 0.}",
     "Ik = sum((1. if s == 0 else -1.) * q for t, _, s, q, _, _ in self.E.fills if t <= g); res = {0: 0., 1: 0.}"),
    ('the cap check without the quantity', "                if q > room + TOL:", "                if q > room + self.q0 + TOL:"),
    ('duplicate prices allowed', "            if any(o['price'] == price for o in v['orders']):\n                continue\n", ""),
    ('no own-cross rule', "            if self.own_cross and any(", "            if False and any("),
    ('more than two orders per side', "            if v['vacancy'] <= 0:\n                break\n", ""),
    ('the birth rung forgotten', "        return self.add9([side, float(rung), price, float(q), 0., d + BE.PLACE, 1e30, d, 1.])",
     "        return self.add9([side, 0., price, float(q), 0., d + BE.PLACE, 1e30, d, 1.])"),
    ('the primary subreason fixed', "primary=next(f for f in STRUCT_FLAGS if flags[f])", "primary='distance'"),
    ('the mode from the previous tick', "        self.E.advance(g); self.acted = set(); self.notices(g); self.check_position(g); self.structural(g); self.update_mode(g)",
     "        self.E.advance(g); self.acted = set(); self.notices(g); self.check_position(g); self.structural(g); self.update_mode(g - BE.STEP) if g > 0 else None"),
    ('one closure clears the side', "self.closed[gid] = (why, float(g)); self.unclosed[s].remove(gid);", "self.closed[gid] = (why, float(g)); self.unclosed[s] = [];"),
    ('a fixed side order', "    for s in ((B.episode + k) % 2, 1 - (B.episode + k) % 2):\n        B.side_commands(g, s, pol)", "    for s in (0, 1):\n        B.side_commands(g, s, pol)"),
    ('no fallback without a book', "        if not v['book_ok']:\n", "        if False:\n"),
    ('the per-level size without the side multiplier', "min(self.q0 * f, max(0., D - Bs), max(0., room))", "min(self.q0, max(0., D - Bs), max(0., room))"),
    ('no side exposure budget', "max(0., D - Bs), max(0., room))", "max(0., D), max(0., room))"),
    ('the restricted side kept', "        return ('PULL', 'inventory_pull') if self.blocked(v) else ('KEEP', 'keep')", "        return ('KEEP', 'keep')"),
    ('sizes rounded up', "float(math.floor(min(self.q0 * f, max(0., D - Bs), max(0., room)) + TOL))", "float(math.ceil(min(self.q0 * f, max(0., D - Bs), max(0., room)) - TOL))"),
    ('the restricted side may post', "        if self.blocked(v):\n            return 'WAIT', None, 'restricted'", "        if False:\n            return 'WAIT', None, 'restricted'"),
    ('resume without the tape check', "C.require(BE.tape_id(path['frames'], path['trades'], path.get('mark')) == saved['tape_id'],", "C.require(True,"),
    ('the fingerprint without the mode log', "                                      B.mode, B.mode_log, B.struct_log, B.cmd_log], sort_keys=True",
     "                                      B.mode, B.struct_log, B.cmd_log], sort_keys=True"),
]
