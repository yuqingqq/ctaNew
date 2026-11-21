"""MUTATION CHECK of pf_ladder_guard (ASK-RESERVATION-GUARD-v1; 2026-11-21, session aea5bfb9), run by pf_mutant_one.py with pf_ladder_guard_selftest.
Required: the control passes; every mutant is caught BY A CHECK; every target occurs exactly once. Not planted (equivalent by construction):
withholding bid slots too (B >= 200 means two unclosed bids, so no bid vacancy), and valid_q on censored rows (the statistic never reads them)."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / 'pf_ladder_guard.py'
SELFTEST = 'pf_ladder_guard_selftest'
FAIL = 'GUARD SELFTEST FAILED'
MUTANTS = [
    ('the signal strictly above the edge', "return bool(self.enabled and np.isfinite(b) and float(b) >= self.edge)", "return bool(self.enabled and np.isfinite(b) and float(b) > self.edge)"),
    ('the disabled guard still active', "return bool(self.enabled and np.isfinite(b) and float(b) >= self.edge)", "return bool(np.isfinite(b) and float(b) >= self.edge)"),
    ('the signal from the ask reservations', "b = np.float32(v['B'])", "b = np.float32(v['A'])"),
    ('the guard pulls bids too', "if int(v['side']) == 1 and a == 'KEEP' and self.signal(v):", "if a == 'KEEP' and self.signal(v):"),
    ('the guard relabels inventory pulls', "if int(v['side']) == 1 and a == 'KEEP' and self.signal(v):", "if int(v['side']) == 1 and self.signal(v):"),
    ('selected by q4', "== 'q5' for v in g.ctx_bid_reserved_shares.astype(float)]", "== 'q4' for v in g.ctx_bid_reserved_shares.astype(float)]"),
    ('draws with replacement', "rng.choice(idx[k], size=s, replace=False)", "rng.choice(idx[k], size=s, replace=True)"),
    ('strata without the age band', "band(r, REM_BANDS), band(a, AGE_BANDS), inv_sign(i)", "band(r, REM_BANDS), 0, inv_sign(i)"),
    ('strata without the remaining band', "json.dumps([d, int(m), band(r, REM_BANDS),", "json.dumps([d, int(m), 0,"),
    ('censored states in the statistic', "g['uncensored'] = (~g.censored_1s.astype(bool)) & (g.win_mo1s.notna() | (g.win_n == 0))", "g['uncensored'] = g.win_n >= 0"),
    ('the +1 s comparison without the shared cutoff', "cut = F[F.t <= end - 1000.]", "cut = F"),
    ('reducing suppressions with the wrong sign', "reducing_suppressed=sum(1 for z in gl if z[6] > 1e-9)", "reducing_suppressed=sum(1 for z in gl if z[6] < -1e-9)"),
    ('better_selection without the selection control', " and out['selection'] == 'SELECTION_BEYOND_EXPOSURE')", ")"),
    ('quoting_less without the per-share condition', "less = all(v < 0 for v in dq.values()) and out['mo1s_per_share']['all']['answer'] != 'IMPROVES_EVERY_DAY'", "less = all(v < 0 for v in dq.values())"),
    ('no-fill generations counted as censored', "c = int((G_.lifetime_mo1s.isna() & (G_.n_fills > 0)).sum())", "c = int(G_.lifetime_mo1s.isna().sum() + (G_.n_fills == 0).sum())"),
    ('run_day before the freeze', "        C.require(frozen(), 'the guard runner / selftest / declaration / manifest / launcher are not the committed ones')", "        pass"),
]
