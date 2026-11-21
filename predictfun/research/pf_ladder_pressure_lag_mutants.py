"""MUTATION CHECK of pf_ladder_pressure_lag (PRESSURE-LEAD-SENSITIVITY-v1; 2026-11-21, session aea5bfb9), run by pf_mutant_one.py with
pf_ladder_pressure_lag_selftest. Required: the control passes; every mutant is caught BY A CHECK; every target occurs exactly once."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / 'pf_ladder_pressure_lag.py'
SELFTEST = 'pf_ladder_pressure_lag_selftest'
FAIL = 'LAG SELFTEST FAILED'
T_K = "k = int(round(float(v['t']) / STEP)) - self.lag; xs = self.cache['x'][s]"
T_LOG = "(float(v['t']) - self.lag * STEP, float(ag) + self.lag * STEP)"
MUTANTS = [
    ('the lag not applied', T_K, "k = int(round(float(v['t']) / STEP)); xs = self.cache['x'][s]"),
    ('the lag doubled', T_K, "k = int(round(float(v['t']) / STEP)) - 2 * self.lag; xs = self.cache['x'][s]"),
    ('a future read', T_K, "k = int(round(float(v['t']) / STEP)) + self.lag; xs = self.cache['x'][s]"),
    ('warm-up reads tick 0', "        if k < 0:\n            return np.nan, 'warmup', np.nan", "        if k < 0:\n            k = 0"),
    ('the source time without the lag', T_LOG, "(float(v['t']), float(ag) + self.lag * STEP)"),
    ('the effective age without the lag', T_LOG, "(float(v['t']) - self.lag * STEP, float(ag))"),
    ('lag_join without the lag', "        j = k - lag\n", "        j = k\n"),
    ('lag_join warm-up reads the last tick', "        if j < 0:\n            warm += 1; vals[i] = np.nan", "        if j < -1:\n            warm += 1; vals[i] = np.nan"),
    ('lag_join ignores a non-ok status', "        elif int(c['status'][s][j]) != 0:\n            nok += 1; vals[i] = np.nan", "        elif False:\n            nok += 1; vals[i] = np.nan"),
    ('the zero-lag mismatch not counted', "mis += int(P.bits(c['x'][s][k]) != P.bits(g[P.FEATURE].iat[i]))", "mis += 0"),
    ('run_selection without the join check', "C.require(st['zero_lag_mismatched'] == 0 and st['off_tick'] == 0, f'the join keys do not reproduce the saved values: {st}')", "None"),
    ('same_price from the rungs', "same_price=float(o2['price']) == float(o['price'])", "same_price=float(o2['rung']) == float(o['rung'])"),
    ('the ahead difference reversed', "ahead_difference=a2 - float(v[1]))", "ahead_difference=float(v[1]) - a2)"),
    ('the queue label not applied', "'q_surrendered_mean': 'q_next_minus_pull_ahead_mean'", "'q_surrendered_mean': 'q_surrendered_mean'"),
    ('the retained gain inverted', "out['END'][d] = (z1 - b0) / (z0 - b0) if z0 != b0 else np.nan", "out['END'][d] = (z0 - b0) / (z1 - b0) if z1 != b0 else np.nan"),
    ('the reading without the selection screen', "    fails += [f'selection: {sel_lag}'] if sel_lag != 'SELECTION_BEYOND_EXPOSURE' else []\n", ""),
    ('the reading without per share', """    fails += [f'mo1s_per_share: {vs["mo1s_per_share"]["all"]["answer"]}'] if vs['mo1s_per_share']['all']['answer'] != 'IMPROVES_EVERY_DAY' else []\n""", ""),
    ('run_day without the parent record check', "    if expect is not None:\n        C.require(C.sha(Pp['rec']) == expect, f'{day}: the pressure day record differs from the manifest')\n", ""),
    ('run_day without the source check', "    C.require(gm['sources'] == sources(), 'a source (runner, parent or builder) differs from the frozen one')\n", ""),
    ('run_day without the write-once check', "P_ = day_paths(day, pilot, out); C.require(not P_['rec'].exists(), f'{day}: the output exists (write-once)');", "P_ = day_paths(day, pilot, out);"),
    ('measure without the selection prerequisite', "    C.require((out / 'selection.json').exists(), 'the selection control must run before the measurement'); md.mkdir(parents=True, exist_ok=True)",
     "    md.mkdir(parents=True, exist_ok=True)"),
]
