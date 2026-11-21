"""MUTATION CHECK of pf_ladder_rest (LADDER_REST_INVENTORY_v1 runner and measurement; 2026-11-21, session aea5bfb9), run by pf_mutant_one.py with
pf_ladder_rest_selftest (PF_TEACHER_SYN passed through). Required: the control passes; every mutant is caught BY A CHECK; every target occurs
exactly once."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / 'pf_ladder_rest.py'
SELFTEST = 'pf_ladder_rest_selftest'
FAIL = 'LADDER SELFTEST FAILED'
MUTANTS = [
    ('the opening backdated', "        if j >= 0 and 0 <= b[j] < a[j] <= 100:\n            return t, seen, none", "        if True:\n            return t, seen, none"),
    ('markouts past END', "            if tt > end:\n                row += [None, None]; continue", "            if tt > end + 1e9:\n                row += [None, None]; continue"),
    ('the fill mid from a later frame', "j = int(np.searchsorted(frames[:, 0], tt, side='right')) - 1; m_ = AR.mid_at(frames, tt)",
     "j = int(np.searchsorted(frames[:, 0], tt, side='right')) - 1; m_ = AR.mid_at(frames, tt + 50.)"),
    ('the locked boundary inclusive', "if t < e_g),", "if t <= e_g),"),
    ('the carry at the interval end', "carry = 0. if ma is None else pa * (mb - ma)", "carry = 0. if ma is None else pb * (mb - ma)"),
    ('downtime precedence reversed', "    return next((d for d in DOWNTIME if d in flags), 'other')", "    return next((d for d in reversed(DOWNTIME) if d in flags), 'other')"),
    ('the maximum-rank live checkpoint', "if kk not in self.live_s or r < self.live_s[kk]['rank']:", "if kk not in self.live_s or r > self.live_s[kk]['rank']:"),
    ('entries recorded past END', "            if q <= room + LB.TOL and g + PLACE < self.B.end:", "            if q <= room + LB.TOL:"),
    ('the reconstruction mode strict', "        if float(g_) <= t:\n            mode = new", "        if float(g_) < t:\n            mode = new"),
    ('the reconstruction without same-tick posts',
     "created_at_t = {int(c[3]) for c in cmds[:int(sn['cmd_index'])] if int(c[7]) == CMD_ACTIONS.index('POST') and float(c[0]) == t and int(c[3]) >= 0}", "created_at_t = set()"),
    ('one more pilot market', "    pilot = {day: [x['m'] for x in sel if x['day'] == day][:PILOT_PER_DAY] for day in DAYS}",
     "    pilot = {day: [x['m'] for x in sel if x['day'] == day][:PILOT_PER_DAY + 1] for day in DAYS}"),
    ('the maximum-rank market', "            sel.append(dict(day=day, m=int(ms[0]),", "            sel.append(dict(day=day, m=int(ms[-1]),"),
    ('fair_edge stretched', "        fe = np.nan; ph = float(min(5, int((d - ws) // MINUTE) + 1))", "        fe = 0.; ph = float(min(5, int((d - ws) // MINUTE) + 1))"),
    ('the posted size as q0', "    out.update(fp_size=float(pt['q']),", "    out.update(fp_size=100.,"),
    ('values without the ask sign', "def sgn(s):\n    return 1. if int(s) == 0 else -1.", "def sgn(s):\n    return 1."),
    ('the manifest written over', "    path = Path(path or REPO / RECEIPT / 'manifest.json'); C.require(not path.exists(), 'the manifest exists (write-once)')",
     "    path = Path(path or REPO / RECEIPT / 'manifest.json')"),
    ('the measurement written over', "C.require(not (md / 'audit.json').exists(), 'the measurement exists (write-once)'); ", ""),
    ('run_day before the freeze', "        C.require(frozen(), 'the runner / client / selftest / declaration / manifest / launcher are not the committed ones: no market runs before the freeze')",
     "        pass"),
    ('the full run without the projection', "        C.require(pilot or projection_allows(), 'the full run needs the write-once pilot projection with run_rest true')", "        pass"),
    ('the phase from the opening', "ph = float(min(5, int((d - ws) // MINUTE) + 1))", "ph = 1."),
]
