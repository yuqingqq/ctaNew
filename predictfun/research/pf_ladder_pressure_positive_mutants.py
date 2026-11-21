"""MUTATION CHECK of pf_ladder_pressure_positive (POSITIVE-PRESSURE-GUARD-v1; 2026-11-21, session aea5bfb9), run by pf_mutant_one.py with
pf_ladder_pressure_positive_selftest. Required: the control passes; every mutant is caught BY A CHECK; every target occurs exactly once."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / 'pf_ladder_pressure_positive.py'
SELFTEST = 'pf_ladder_pressure_positive_selftest'
FAIL = 'POSITIVE SELFTEST FAILED'
T_SIG = "return bool(self.enabled and st == 'ok' and np.isfinite(x) and float(np.float32(x)) > 0.), x, ag"
MUTANTS = [
    ('a zero tie counted ON', T_SIG, "return bool(self.enabled and st == 'ok' and np.isfinite(x) and float(np.float32(x)) >= 0.), x, ag"),
    ('a non-ok status acts', T_SIG, "return bool(self.enabled and np.isfinite(x) and float(np.float32(x)) > 0.), x, ag"),
    ('the disabled guard acts', T_SIG, "return bool(st == 'ok' and np.isfinite(x) and float(np.float32(x)) > 0.), x, ag"),
    ('the challenger keeps the q5 test', T_SIG, "return bool(self.enabled and st == 'ok' and P.on_value(x, self.edges[int(v['side'])])), x, ag"),
    ('the screen keeps fallback callbacks', "keep[i] = a == ('KEEP', 'keep')", "keep[i] = a[0] == 'KEEP'"),
    ('the screen keeps inventory pulls', "keep[i] = a == ('KEEP', 'keep')", "keep[i] = a[1] in ('keep', 'inventory_pull')"),
    ('the screen without the q5-OFF restriction', "pop = keep & fin & off;", "pop = keep & fin;"),
    ('the mask from x >= 0', "mask = np.array([float(np.float32(v)) > 0. for v in xs], bool)", "mask = np.array([float(np.float32(v)) >= 0. for v in xs], bool)"),
    ('IMPROVES from either contrast', "econ = 'IMPROVES' if eA == mA == 'POSITIVE_EVERY_DAY' else", "econ = 'IMPROVES' if 'POSITIVE_EVERY_DAY' in (eA, mA) else"),
    ('the candidate continues without the screen', "stop_candidate=bool(not (econ == 'IMPROVES' and screen_answer == 'SELECTION_BEYOND_EXPOSURE'))", "stop_candidate=bool(not econ == 'IMPROVES')"),
    ('run_day without the part check', "    C.require((part == 'pilot') == (day == gm['pilot_day']), f'{day} does not belong to the {part} part')\n", ""),
    ('the rest without the projection', "C.require(part == 'pilot' or (proj_path.exists() and json.loads(proj_path.read_text()).get('run_rest') is True), 'the rest needs the write-once whole-round projection with run_rest true')", "None"),
    ('measure without the screen prerequisite', "    C.require((out / 'screen.json').exists(), 'the screen must run before the measurement'); md.mkdir(parents=True, exist_ok=True)", "    md.mkdir(parents=True, exist_ok=True)"),
]
