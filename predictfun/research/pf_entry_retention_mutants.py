"""MUTATION CHECK of pf_entry_retention (ENTRY-RETENTION-DISCRIMINATION-v1; 2026-11-21, session aea5bfb9), run by pf_mutant_one.py with
pf_entry_retention_selftest. Required: the control passes; every mutant is caught BY A CHECK; every target occurs exactly once."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / 'pf_entry_retention.py'
SELFTEST = 'pf_entry_retention_selftest'
FAIL = 'ENTRY RETENTION SELFTEST FAILED'
MUTANTS = [
    ('a same-review pull marked distinct', "dict(self.stage['first_review'], stage='first_pull', same_as_review=True) if first else", "dict(self.stage['first_review'], stage='first_pull', same_as_review=False) if first else"),
    ('a first pull from a keeping review', "            if a == 'PULL' and 'first_pull' not in self.stage:", "            if 'first_pull' not in self.stage:"),
    ('a distinct pull marked same_as_review', "dict(capture(self, v, dec, 'first_pull', self.seed), same_as_review=False)", "dict(capture(self, v, dec, 'first_pull', self.seed), same_as_review=True)"),
    ('the prefix runs past the pull', "            why = 'pull'; break", "            why = 'pull'"),
    ('the prefix ignores the closure', "        if gid0 in Bx.closed:\n            why = 'closed:' + str(Bx.closed[gid0][0]); break\n", ""),
    ('the prefix ignores the saved fills', " and cut(saved['fill_list']) == cut(jl(Bx.E.fills))", ""),
    ('the prefix ignores the saved requests', "\n                and cut(saved['requests']) == cut(jl([e for e in Bx.E.events if e[1].endswith('_request')])))", ")"),
    ('the prefix ignores the saved decisions', "same = bool(cut(saved['decision_list']) == cut(jl(Bx.decisions)) and", "same = bool(True and"),
    ('callbacks count the forced POST', "calls = len(Bx.decisions) - nd0 - 1 - (len(Bx.fallbacks) - nf0)", "calls = len(Bx.decisions) - nd0 - (len(Bx.fallbacks) - nf0)"),
    ('callbacks count engine fallbacks', "calls = len(Bx.decisions) - nd0 - 1 - (len(Bx.fallbacks) - nf0)", "calls = len(Bx.decisions) - nd0 - 1"),
    ('the probe count unchecked', " and cst['checked'] == -(-len(pol.log) // AUDIT_EVERY)", ""),
    ('the effect ignores the pending clock', "return (g + BE.CANCEL) if g >= o['act'] else max(g + BE.CANCEL, o['submit'] + BE.PENDING)", "return g + BE.CANCEL"),
    ('the equivalence ignores the KEEP fills', " and Lh['status'] == 'ok' and fh == fk)", " and Lh['status'] == 'ok')"),
    ('the reconciliation ignores D_initial', "rec = bool(abs(tp['value'] - float(saved['D_initial'])) <= 1e-6 and ", "rec = bool("),
    ('the reconciliation ignores the pre-effect fills', "\n                   and fp_ == [x for x in fh if float(x[0]) < eff] and float(Bx.E.o[gid0]['remove']) == eff)", " and float(Bx.E.o[gid0]['remove']) == eff)"),
    ('the horizon unchecked', "horizon = bool(float(ctx['end']) == END == float(ap['end'])", "horizon = bool(True or float(ctx['end']) == END == float(ap['end'])"),
    ('the review target from activation', "st[nm]['target'] = stage_target(fh, gid0, s, mark, st[nm]['effect'], ref)", "st[nm]['target'] = stage_target(fh, gid0, s, mark, act, ref)"),
    ('the unavoidable value signed wrong', "    return te['value'] - ts['value']", "    return ts['value'] - te['value']"),
    ('the admission inverted', "admission='would_POST' if dec['action'] == 'POST' else 'would_WAIT'", "admission='would_WAIT' if dec['action'] == 'POST' else 'would_POST'"),
    ('the reproduction ignores V', "rep = bool((dec.get('V') is None or dec['V'] == V) and ", "rep = bool("),
    ('the hidden copy not consulted', "hidden_same=bool(T.same_decision(dec, d2))", "hidden_same=True"),
    ('the flip signed wrong', "m0['V'] > 0 and mo['V'] < 0", "m0['V'] < 0 and mo['V'] > 0"),
    ('the changes over every input', "if group_of(n) == gname and ch[i]", "if ch[i]"),
    ('the eligible rows of every status', " & (S.dec_status == 'model') & S.V.notna()]", " & S.V.notna()]"),
    ('unmatched strata', "rng = np.random.default_rng(seed); grp = [g for _, g in D.groupby(['day', 'side', 'hour'], sort=True)]", "rng = np.random.default_rng(seed); grp = [D]"),
    ('the percentile reversed', "percentile=float((null <= ms + 1e-12).mean())", "percentile=float((null >= ms - 1e-12).mean())"),
    ('never thin', "thin=bool(n_sel < THIN or len(D) - n_sel < THIN)", "thin=False"),
    ('discrimination ignores thin', "disc = {nm: bool(sel[nm]['percentile'] is not None and not sel[nm]['thin'] and", "disc = {nm: bool(sel[nm]['percentile'] is not None and"),
    ('value_fails strict', "cal[nm]['mae_V'] >= cal[nm]['mae_zero']", "cal[nm]['mae_V'] > cal[nm]['mae_zero']"),
    ('the measurement overwritten', "C.require(not (out / 'retention.json').exists(), 'the measurement exists (immutable)'); ", ""),
    ('a partition read without its hash', "C.require(hashlib.sha256(b).hexdigest() == rec['sha256'] and rec['sources'] == sources() and rec['manifest_sha256'] == C.sha(man_path)",
     "C.require(True and rec['sources'] == sources() and rec['manifest_sha256'] == C.sha(man_path)"),
    ('run_part without the exposure manifest', "    C.require(EX.committed_equal(EX.RECEIPT_REL, EXPO / 'manifest.json'), 'the exposure manifest is not the committed one')\n", ""),
]
