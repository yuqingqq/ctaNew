"""MUTATION CHECK of pf_ladder_async_fast (ASYNC-EXECUTION-SENSITIVITY-v1; 2026-11-22, session aea5bfb9), run by pf_mutant_one.py with
pf_ladder_async_fast_selftest. Required: the control passes; every mutant is caught BY A CHECK; every target occurs exactly once.
Not planted (equivalent on the declared tables): the first-request-only guard in ProfileEngine.request (no order is requested twice in the
fixtures; the frozen engine has the same guard) and the END-only economic answer (the synthetic table sets END = +1 s)."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / 'pf_ladder_async_fast.py'
SELFTEST = 'pf_ladder_async_fast_selftest'
FAIL = 'ASYNC FAST SELFTEST FAILED'
ASY = "ASYNC_FAST10=dict(place=10., cancel=10., pending=20., slot_model='desired_levels'))"
MUTANTS = [
    ('the ASYNC activation at 58 ms', ASY, "ASYNC_FAST10=dict(place=58., cancel=10., pending=20., slot_model='desired_levels'))"),
    ('the ASYNC live cancel at 37 ms', ASY, "ASYNC_FAST10=dict(place=10., cancel=37., pending=20., slot_model='desired_levels'))"),
    ('the ASYNC pending-cancel minimum at 95 ms', ASY, "ASYNC_FAST10=dict(place=10., cancel=10., pending=95., slot_model='desired_levels'))"),
    ('the ASYNC slot model frozen', ASY, "ASYNC_FAST10=dict(place=10., cancel=10., pending=20., slot_model='frozen'))"),
    ('the activation from the frozen constant', "return self.add9([side, float(rung), price, float(q), 0., d + self.prof['place'], 1e30, d, 1.])",
     "return self.add9([side, float(rung), price, float(q), 0., d + BE.PLACE, 1e30, d, 1.])"),
    ('the pending race rule dropped', "o = self.o[gid]; arr = max(g + self.prof['cancel'], o['submit'] + self.prof['pending']) if g < o['act'] else g + self.prof['cancel']",
     "o = self.o[gid]; arr = g + self.prof['cancel']"),
    ('retiring orders counted as desired', "return [gid for gid in self.unclosed[s] if self.E.o[gid]['req'] is None]", "return list(self.unclosed[s])"),
    ('the async desired invariant dropped', "C.require(all(len(self.desired(s)) <= self.levels for s in (0, 1)), f'more than {self.levels} desired generations on a side at {g}')",
     "C.require(True, f'more than {self.levels} desired generations on a side at {g}')"),
    ('the async reservation invariant dropped', "C.require(Ik + Bk <= self.cap + LB.TOL and -Ik + Ak <= self.cap + LB.TOL, f'reservation inequality at {g}: I {Ik}, B {Bk}, A {Ak}, cap {self.cap}')",
     "C.require(True, f'reservation inequality at {g}: I {Ik}, B {Bk}, A {Ak}, cap {self.cap}')"),
    ('near END from the frozen constant', "                elif g + self.prof['place'] >= self.end:", "                elif g + BE.PLACE >= self.end:"),
    ('the async settings not recorded', "        if profile != 'BASELINE':\n            self.settings", "        if False:\n            self.settings"),
    ('the baseline settings changed', "        if profile != 'BASELINE':\n            self.settings", "        if True:\n            self.settings"),
    ('run_arm with the front flag inverted', "front, prof = RUN_ARMS[arm]; B = ProfileBranch(path, ident['end_rel'], ident, front, prof)",
     "front, prof = RUN_ARMS[arm]; B = ProfileBranch(path, ident['end_rel'], ident, not front, prof)"),
    ('FRONT_ASYNC without the front treatment', "FRONT_ASYNC_FAST10=(True, 'ASYNC_FAST10'))", "FRONT_ASYNC_FAST10=(False, 'ASYNC_FAST10'))"),
    ('a retiring tick after the request only', "rt = un & (g >= float(req)) if req is not None else np.zeros(nt, bool)", "rt = un & (g > float(req)) if req is not None else np.zeros(nt, bool)"),
    ('every cap refusal counted while retiring', "cap_ret += int(0 <= k < nt and ret[k, int(s)] > 0)", "cap_ret += int(0 <= k < nt)"),
    ('the occupancy not capped at the levels', "desired_occupancy=float(np.minimum(des, levels).sum() / (2 * levels * nt))", "desired_occupancy=float(des.sum() / (2 * levels * nt))"),
    ('the interaction sign', "('FRONT_BASELINE', -1.), ('CURRENT_BASELINE', 1.)))", "('FRONT_BASELINE', -1.), ('CURRENT_BASELINE', -1.)))"),
    ('a zero END not nonnegative', "nonneg = [a for a in MEASURED if tot[a]['END'] >= 0]", "nonneg = [a for a in MEASURED if tot[a]['END'] > 0]"),
    ('run_day without the profile identity', "    C.require(gm['profile_sha256'] == profile_sha() and gm['treatment_sha256'] == QF.treatment_sha(), 'the profile or the treatment differs from the frozen one')\n", ""),
    ('the rest without the projection gate', "C.require(part == 'pilot' or (proj_path.exists() and json.loads(proj_path.read_text()).get('run_rest') is True), 'the rest needs the write-once projection with run_rest true')",
     "C.require(True, 'the rest needs the write-once projection with run_rest true')"),
    ('run_day without write-once', "C.require(not P_['rec'].exists(), f'{day}: the output exists (write-once)');", ""),
    ('the projection without write-once', "    proj_path = Path(proj_path or REPO / RECEIPT / 'logs' / 'projection.json'); C.require(not proj_path.exists(), 'the projection exists (write-once)')",
     "    proj_path = Path(proj_path or REPO / RECEIPT / 'logs' / 'projection.json')"),
    ('measure without write-once', "C.require(not (md / 'audit.json').exists(), 'the measurement exists (write-once)');", ""),
]
