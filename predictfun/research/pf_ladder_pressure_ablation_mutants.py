"""MUTATION CHECK of pf_ladder_pressure_ablation (PRESSURE-ACTION-ATTRIBUTION-v1; 2026-11-21, session aea5bfb9), run by pf_mutant_one.py with
pf_ladder_pressure_ablation_selftest. Required: the control passes; every mutant is caught BY A CHECK; every target occurs exactly once."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / 'pf_ladder_pressure_ablation.py'
SELFTEST = 'pf_ladder_pressure_ablation_selftest'
FAIL = 'ABLATION SELFTEST FAILED'
T_C = "        if self.C:\n            return super().live(v, o)"
T_W = "return super().slot(v, r, price) if self.W else self.inner.slot(v, r, price)"
MUTANTS = [
    ('the cancellation gate inverted', T_C, "        if not self.C:\n            return super().live(v, o)"),
    ('the cancellation gate ignored', T_C, "        if True:\n            return super().live(v, o)"),
    ('the withholding gate inverted', T_W, "return super().slot(v, r, price) if not self.W else self.inner.slot(v, r, price)"),
    ('the withholding gate ignored', T_W, "return super().slot(v, r, price)"),
    ('the disabled branch skips the first review', "            self.seen.add(int(o['gid'])); self.q_first.append((int(o['gid']), float(v['t'])))\n        return self.inner.live(v, o)",
     "            self.seen.add(int(o['gid']))\n        return self.inner.live(v, o)"),
    ('the rest takes the pilot', "rest = {d: [int(m) for m in v if int(m) not in {int(x) for x in pilot[d]}] for d, v in complete.items()}",
     "rest = {d: [int(m) for m in v if int(m) in {int(x) for x in pilot[d]}] for d, v in complete.items()}"),
    ('the covering check dropped', "    C.require(all(sorted([int(x) for x in pilot[d]] + rest[d]) == sorted(int(m) for m in complete[d]) for d in complete), 'the parts do not cover the complete markets')\n", ""),
    ('withholding-given-cancellation against the wrong arm', "'withhold_given_cancel': ('FULL_11', 'PULL_ONLY_10')", "'withhold_given_cancel': ('FULL_11', 'WITHHOLD_ONLY_01')"),
    ('the interaction without the baseline term', "- byd[d]['WITHHOLD_ONLY_01'][k] + byd[d]['BASELINE_00'][k]", "- byd[d]['WITHHOLD_ONLY_01'][k] - byd[d]['BASELINE_00'][k]"),
    ('a zero day counted positive', "return 'POSITIVE_EVERY_DAY' if all(v > 0 for v in vals) else", "return 'POSITIVE_EVERY_DAY' if all(v >= 0 for v in vals) else"),
    ('equal markets counted better', "markets_better=int((mE > 1e-9).sum())", "markets_better=int((mE >= 0).sum())"),
    ('the top-2 share', "float(mE.abs().sort_values(ascending=False).head(3).sum() / ab)", "float(mE.abs().sort_values(ascending=False).head(2).sum() / ab)"),
    ('all arms negative from any', "total=bool(all(tot[a]['END'] < 0 for a in ARMS))", "total=bool(any(tot[a]['END'] < 0 for a in ARMS))"),
    ('the per share from the END', "per_share=float(g.all_cut_mo1s.sum()) / q if q > 0 else np.nan", "per_share=float(g.value.sum()) / q if q > 0 else np.nan"),
    ('run_day without the part check', "C.require(part in PARTS, f'{part} is not a declared part')", "None"),
    ('run_day without the source check', "    C.require(gm['sources'] == sources(), 'a source (runner, parent or builder) differs from the frozen one')\n", ""),
    ('the rest without the projection', "C.require(part == 'pilot' or (proj_path.exists() and json.loads(proj_path.read_text()).get('run_rest') is True), 'the rest needs the write-once pilot projection with run_rest true')", "None"),
    ('run_day without the write-once check', "P_ = day_paths(day, part, out); C.require(not P_['rec'].exists(), f'{day}: the output exists (write-once)');", "P_ = day_paths(day, part, out);"),
    ('an empty part loads anyway', "    if not mans:\n        LS.RN.atomic_json(", "    if False:\n        LS.RN.atomic_json("),
    ('measure without the write-once check', "md = out / 'measure'; C.require(not (md / 'audit.json').exists(), 'the measurement exists (write-once)'); md.mkdir(parents=True, exist_ok=True)",
     "md = out / 'measure'; md.mkdir(parents=True, exist_ok=True)"),
]
