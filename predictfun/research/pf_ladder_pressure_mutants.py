"""MUTATION CHECK of pf_ladder_pressure (BINANCE-PRESSURE-GUARD-v1; 2026-11-21, session aea5bfb9), run by pf_mutant_one.py with
pf_ladder_pressure_selftest. Required: the control passes; every mutant is caught BY A CHECK; every target occurs exactly once. Not planted
(equivalent by construction): dropping on_value's x > 0 (pf_ladder_rest.bin_of maps 0 to 'zero' and both last edges are positive, so q5 is
always positive)."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / 'pf_ladder_pressure.py'
SELFTEST = 'pf_ladder_pressure_selftest'
FAIL = 'PRESSURE SELFTEST FAILED'
MUTANTS = [
    ('q4 counted as ON', "M.bin_of(float(x), edges_s) == 'q5')", "M.bin_of(float(x), edges_s) in ('q4', 'q5'))"),
    ("the other side's edges", "on_value(x, self.edges[int(v['side'])])), x, ag", "on_value(x, self.edges[1 - int(v['side'])])), x, ag"),
    ('the disabled guard still active', "return bool(self.enabled and st == 'ok' and on_value(", "return bool(st == 'ok' and on_value("),
    ('a non-ok status acts on its value', "return bool(self.enabled and st == 'ok' and on_value(", "return bool(self.enabled and on_value("),
    ('the guard relabels inventory pulls', "        if a == 'KEEP':\n            on, x, ag = self.signal(v)", "        if a in ('KEEP', 'PULL'):\n            on, x, ag = self.signal(v)"),
    ('the guard relabels restricted waits', "        if a == 'POST':\n            on, x, ag = self.signal(v)", "        if a in ('POST', 'WAIT'):\n            on, x, ag = self.signal(v)"),
    ('the signal read one tick late', "k = int(round(float(v['t']) / STEP)); xs = self.cache['x'][s]", "k = int(round(float(v['t']) / STEP)) + 1; xs = self.cache['x'][s]"),
    ('off-grid steps accepted', "ok = np.abs(tg[j] - d) <= 1e-6", "ok = np.abs(tg[j] - d) <= STEP"),
    ('the wrong market column', "x[k] = np.float32(row[0][FI]); ag[k] = np.float32(row[0][AGE_I])", "x[k] = np.float32(row[0][FI + 1]); ag[k] = np.float32(row[0][AGE_I])"),
    ('parity by value, not bits', "if bits(cache['x'][int(s)][k]) == bits(v):", "if cache['x'][int(s)][k] == np.float32(v):"),
    ('the side sign without the negation', "eq = cache['x'][0][both] == -cache['x'][1][both]", "eq = cache['x'][0][both] == cache['x'][1][both]"),
    ('the future check without the perturbation', "aft = feat_fn(mut, fair2, dep2, ch)[s][j]", "aft = feat_fn(W_, fair_m, depth, ch)[s][j]"),
    ('the control with the wrong sign', "ctl = 0. if on else (-CONTROL_ADVERSE if s == 0 else CONTROL_ADVERSE)", "ctl = 0. if on else (CONTROL_ADVERSE if s == 0 else -CONTROL_ADVERSE)"),
    ('the queue pull and first review swapped', "for lst, dst in ((self.pol.q_first, self.first), (self.pol.q_pull, self.pull)):", "for lst, dst in ((self.pol.q_first, self.pull), (self.pol.q_pull, self.first)):"),
    ('a pressure pull not queued for the observer', "self.q_pull.append((int(o['gid']), float(v['t'])))", "None"),
    ('strata without the side', "json.dumps([d, int(m), int(s), GD.band(r, REM_BANDS)", "json.dumps([d, int(m), 0, GD.band(r, REM_BANDS)"),
    ('strata without the B indicator', "GD.inv_sign(i), b]) for d, m, s, r, a, i, b in", "GD.inv_sign(i), 0]) for d, m, s, r, a, i, b in"),
    ('draws with replacement', "rng.choice(idx[k], size=s, replace=False)", "rng.choice(idx[k], size=s, replace=True)"),
    ('censored states in the statistic', "g['uncensored'] = (~g.censored_1s.astype(bool)) & (g.win_mo1s.notna() | (g.win_n == 0))", "g['uncensored'] = g.win_n >= 0"),
    ('the late term by the cutoff only', "f'{nm}_id_late': float(F[ms & ~el].end.sum())", "f'{nm}_id_late': float(F[ms & ~cut].end.sum())"),
    ('time at the limit strictly above', "high += (t - t_prev) if abs(I_) >= S_LIM - 1e-9 else 0.", "high += (t - t_prev) if abs(I_) > S_LIM else 0."),
    ('lost / added without the price', "key = lambda f: (round(float(f[0]), 6), int(f[2]), round(float(f[4]), 6), round(float(f[3]), 6))",
     "key = lambda f: (round(float(f[0]), 6), int(f[2]), round(float(f[3]), 6))"),
    ('the reducing side reversed', "(e[8] < -1e-9 if s == 0 else e[8] > 1e-9)", "(e[8] > 1e-9 if s == 0 else e[8] < -1e-9)"),
    ('the first-review rate from the last command', "first.setdefault(int(c[3]), c)", "first[int(c[3])] = c"),
    ('better_selection without the selection control', " and sel_answer == 'SELECTION_BEYOND_EXPOSURE')", ")"),
    ('quoting_less without the per-share condition', "less = all(v < 0 for v in dq.values()) and out['mo1s_per_share']['all']['answer'] != 'IMPROVES_EVERY_DAY'", "less = all(v < 0 for v in dq.values())"),
    ('run_day without the source check', "    C.require(gm['sources'] == sources(), 'a source (runner or builder) differs from the frozen one')\n", ""),
    ('run_day without the write-once check', "P = day_paths(day, pilot, out); C.require(not P['rec'].exists(), f'{day}: the output exists (write-once)');", "P = day_paths(day, pilot, out);"),
    ('measure without the selection prerequisite', "    C.require((out / 'selection.json').exists(), 'the selection control must run before the measurement'); md.mkdir(parents=True, exist_ok=True)",
     "    md.mkdir(parents=True, exist_ok=True)"),
]
