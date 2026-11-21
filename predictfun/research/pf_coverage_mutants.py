"""MUTATION CHECK of pf_coverage (PI0-COVERAGE-v1; 2026-11-21, session aea5bfb9), run by pf_mutant_one.py with pf_coverage_selftest (PF_TEACHER_SYN
passed through). Required: the control passes; every mutant is caught BY A CHECK; every target occurs exactly once."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / 'pf_coverage.py'
SELFTEST = 'pf_coverage_selftest'
FAIL = 'COVERAGE SELFTEST FAILED'
MUTANTS = [
    # population
    ('one world too many per hour', "n = min(per_hour, len(r)); pi = n / len(r)", "n = min(per_hour + 1, len(r)); pi = n / len(r)"),
    ('pi_world of one', "n = min(per_hour, len(r)); pi = n / len(r)", "n = min(per_hour, len(r)); pi = 1."),
    ('worlds ranked by their key only', "r = sorted(ws, key=lambda w: (rank('world', LS.wkey(w)), LS.ckey(LS.wkey(w))));", "r = sorted(ws, key=lambda w: LS.ckey(LS.wkey(w)));"),
    ('the pilot by key order', "return sorted([x for x in sel if x['day'] == day], key=lambda x: (x['rank'], LS.ckey(LS.wkey(x))))[:n]",
     "return sorted([x for x in sel if x['day'] == day], key=lambda x: LS.ckey(LS.wkey(x)))[:n]"),
    ('overlapping blocks', "h0 <= int(x['hour']) < h0 + BLOCK_HOURS]", "h0 <= int(x['hour']) <= h0 + BLOCK_HOURS]"),
    ('the manifest written over', "C.require(not path.exists(), 'the manifest exists (write-once)')", "C.require(True, 'the manifest exists (write-once)')"),
    # census
    ('later live decisions as reviews', "    if st == 'live' and not first:\n        return None, 'live_later'\n", ""),
    ('constrained decisions as model', "    if exe is None or exe[3] != dec['action']:\n        return None, 'constrained'\n", ""),
    ('every live decision a first review', "first = bool(v['state'] == 'live' and v['gid'] not in reviewed)", "first = bool(v['state'] == 'live')"),
    ('creations unrecorded', "                created[int(exe[4])] = (int(round(g)), s)\n", "                pass\n"),
    ('the view after the decision', "            v = B.view(g, s); nl, nd = len(pol.log), len(B.decisions)\n            BE.decide(B, g, s, pol)\n",
     "            nl, nd = len(pol.log), len(B.decisions)\n            BE.decide(B, g, s, pol); v = B.view(g, s)\n"),
    # queries
    ('every query of a cell', "n = min(PER_CELL, len(o)); pi = n / len(o)", "n = len(o); pi = n / len(o)"),
    ('the weight without the world sample', "weight=1. / (pw * pi)", "weight=1. / pi"),
    ('the cell hour fixed', "            hour = int(sel_by_key[(r['day'], r['m'], r['t_ms'])]['hour'])\n", "            hour = 0\n"),
    ('queries ranked by their key only', "o = sorted(rs, key=lambda r: (rank('query', key(r)), LS.ckey(key(r))));", "o = sorted(rs, key=lambda r: LS.ckey(key(r)));"),
    # labels
    ('the reroll unchecked', "    same = bool(Lf.get('status') == 'ok' and [L._h(Lf['fills']), L._h(Lf['decisions'])] == list(census_hashes));", "    same = True;"),
    ('the hold ledger unchecked', "checks = dict(ledger_ok=bool(L.reconcile_arm(Lh, ledger_ns(Lh, mark))))", "checks = dict(ledger_ok=True)"),
    ('the POST gid unchecked', "                checks['gid'] = bool(gidf is not None and int(gidf) == int(gid_h))\n", "                checks['gid'] = True\n"),
    ('the factual action unchecked', "            checks['factual'] = bool(d0 is not None and d0[3] == ('POST' if q['cat'] == 'absent_POST' else 'WAIT'))\n",
     "            checks['factual'] = bool(d0 is not None)\n"),
    ('the absent target from the decision', "eff = float(gen_h['act'])\n", "eff = float(q['t'])\n"),
    ('the review target from the activation', "            eff = float(st['effect']) if st.get('effect') is not None else np.nan;", "            eff = float(gen_h['act']);"),
    ('the reconciliation always true', "    return bool(abs(r['q'] - target['q']) <= 1e-9 and abs(r['value'] - target['value']) <= 1e-6 and abs(r['qm'] - target['qm']) <= 1e-6)",
     "    return bool(abs(r['q'] - target['q']) <= 1e-9)"),
    ('fills strictly after the effect', "float(x[0]) >= float(eff)]", "float(x[0]) > float(eff)]"),
    # measurement
    ('fallback decisions eligible', "D['eligible'] = (D.status == 'ok') & D.V.notna() & (D.dec_status == 'model')", "D['eligible'] = (D.status == 'ok') & D.V.notna()"),
    ('qm kept on no fill', "M=(tg.get('qm') if z == 1. else 0.) if tg else np.nan", "M=tg.get('qm') if tg else np.nan"),
    ('the value MSE unweighted', "bias_V=wm(ev, w), mse_V=wm(ev ** 2, w)", "bias_V=wm(ev, w), mse_V=float((ev ** 2).mean())"),
    ('degeneracy unflagged', "degenerate=bool(m.all() or not m.any())", "degenerate=False"),
    ('the POST tie posts', "    return (V > 0) if cat.startswith('absent') else (V >= 0)", "    return (V >= 0) if cat.startswith('absent') else (V >= 0)"),
    ('the KEEP tie pulls', "    return (V > 0) if cat.startswith('absent') else (V >= 0)", "    return (V > 0) if cat.startswith('absent') else (V > 0)"),
    ('the null with replacement', "m[rng.choice(g, k, replace=False)] = True", "m[rng.choice(g, k, replace=True)] = True"),
    ('covered on rows or fills', "if a['rows'] >= MIN_ROWS and a['fills'] >= MIN_FILLS)", "if a['rows'] >= MIN_ROWS or a['fills'] >= MIN_FILLS)"),
    ('concentration strictly above one half', "value_error_concentrated=bool(r['top5_world_err'] is not None and r['top5_world_err'] >= .5)",
     "value_error_concentrated=bool(r['top5_world_err'] is not None and r['top5_world_err'] > .5)"),
    ('POSITIVE without beating all', "'POSITIVE' if (J > 0 and J > a['all']['J'] and J > nu['null_p95'])", "'POSITIVE' if (J > 0 and J > nu['null_p95'])"),
    ('POSITIVE without the null', "'POSITIVE' if (J > 0 and J > a['all']['J'] and J > nu['null_p95'])", "'POSITIVE' if (J > 0 and J > a['all']['J'])"),
    ('movable rows unchecked', "nu['free_strata'] >= MIN_FREE_STRATA and nu['free_rows'] >= MIN_FREE_ROWS", "nu['free_strata'] >= MIN_FREE_STRATA"),
    # guards
    ('run_part before the freeze', "def run_part(k, out=OUT):\n    C.require(frozen(), 'not frozen:", "def run_part(k, out=OUT):\n    C.require(True, 'not frozen:"),
    ('run_part without the projection', "    C.require(pj is not None and pj.get('run_rest') is True, 'the write-once pilot projection does not allow the build')\n", ""),
    ('a pilot written over', "    C.require(not Path(str(rec) + '.json').exists(), f'{day}: the pilot exists (write-once)')\n", ""),
    ('the queue ratio boundary', "    return '0' if r == 0 else '(0,1]' if r <= 1 else '(1,3]' if r <= 3 else '>3'\n\n\ndef obin", "    return '0' if r == 0 else '(0,1]' if r < 1 else '(1,3]' if r <= 3 else '>3'\n\n\ndef obin"),
    ('ordinal 2 merged', "    return 'missing_or_invalid' if not np.isfinite(o) else '1' if o <= 1 else '2' if o <= 2 else '>=3'", "    return 'missing_or_invalid' if not np.isfinite(o) else '1' if o <= 1 else '>=3'"),
    ('the horizon intervals closed left', "        if lo < t_to_end <= hi:\n", "        if lo <= t_to_end < hi:\n"),
    ('the null only on the combined populations', "        nu, J = null(g, mk['pi0_rule']); nulls[name] = nu;", "        if not name.endswith('combined'):\n            continue\n        nu, J = null(g, mk['pi0_rule']); nulls[name] = nu;"),
    ('the date share of the severe losses', "dsmax = float(dsev.max() / dsev.sum()) if dsev.sum() > 0 else None", "dsmax = float(dsum.max() / dsum.sum()) if dsum.sum() > 0 else None"),
    ('one date at more than one half', "value_error_one_date=bool(r.get('max_date_share_err') is not None and r['max_date_share_err'] >= .5)", "value_error_one_date=bool(r.get('max_date_share_err') is not None and r['max_date_share_err'] > .5)"),
    ('the reference without its review stage', "A_.stage.isin(['entry', 'first_review'])]", "A_.stage.isin(['entry'])]"),
    ('the label query counts unrecorded', "tim['label_queries'].append(len(wq));", "tim['label_queries'].append(1);"),
    ('the measurement written over', "C.require(not (out / 'audit.json').exists(), 'the measurement exists (immutable)')", "C.require(True, 'the measurement exists (immutable)')"),
]

if __name__ == '__main__':
    base = SRC.read_text(); bad = [(i, n) for i, (n, o, _) in enumerate(MUTANTS) if base.count(o) != 1]
    print(f'{len(MUTANTS)} mutants; targets not exactly once: {bad}')
