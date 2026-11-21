"""MUTATION CHECK of pf_adv2 (the ADV2 candidate: inputs, reader, extraction, weights, fits, support, controller and policies; review
315979d2's round; 2026-11-21, session aea5bfb9). Each planted error is a textual replacement in a copy of pf_adv2.py (the file on disk is
never touched), executed as the module under its real path in a fresh process by pf_mutant_one.py, which runs pf_adv2_selftest.run on it.
Required: the unchanged copy passes; every mutant is caught BY A CHECK (the selftest's FAIL message; a crash is not a catch); every target
occurs exactly once. Not mutated (equivalent or defence in depth, no valid run distinguishes them): the per-part row count (implied by the
hashed parts list), the reload bit-equality refusal (it fires only on a broken LightGBM save), the count / membership clause of
check_inputs (a changed list also changes its frozen hash).
Per-mutant jobs: pf_mutant_queue.sh with pf_mutant_one.py / pf_mutant_summary.py (MUTANTS, SRC, SELFTEST, FAIL from here)."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / 'pf_adv2.py'
SELFTEST = 'pf_adv2_selftest'
FAIL = 'ADV2 SELFTEST FAILED'
MUTANTS = [
    ('FAIR without fair_edge', "FAIR=list(dict.fromkeys(['fair_edge', 't_to_end'] + list(R2.CTX)))", "FAIR=list(dict.fromkeys(['t_to_end'] + list(R2.CTX)))"),
    ('the input-hash check removed', "        C.require(sha_of(f) == INPUT_SHA[c], f'{c}: the input list hash changed')\n", ""),
    ('another objective', "PARAMS, ROUNDS = dict(BX.PARAMS, objective='regression'), 200", "PARAMS, ROUNDS = dict(BX.PARAMS, objective='huber'), 200"),
    ('the parts-list hash unchecked', "    C.require(hashlib.sha256(LS.ckey(acc['parts']).encode()).hexdigest() == acc['dataset_sha256'], f'{d.name}: the parts list does not hash to the dataset')\n", ""),
    ('the part bytes unchecked', "        C.require(hashlib.sha256(b).hexdigest() == p['sha256'], f\"{d.name} part {p['part']}: its bytes differ from the accepted hash\")\n", ""),
    ('the dataset hash unchecked', "acc.get('accepted') is True and acc['dataset_sha256'] == want_sha", "acc.get('accepted') is True"),
    ('rows of any day accepted', "        C.require(r['day'] in days, f\"a row of {r['day']}, outside the declared days {days}\")\n", ""),
    ('no crossing purge', "        if status == 'ok' and crosses(r['day'], r['t_ms']):", "        if False:"),
    ('the inputs hash unchecked', "x.shape == (107,) and hashlib.sha256(x.tobytes()).hexdigest() == fk['inputs_sha256']", "x.shape == (107,)"),
    ('a result of another head accepted', "res['status'] == 'ok' and res['kind'] == h and int(res['side']) == s", "res['status'] == 'ok' and int(res['side']) == s"),
    ('the weight as the label', "        y, w = float(res['Y']), float(r['w'])", "        y, w = float(r['w']), float(r['w'])"),
    ('the generation from another input', "GEN_I = T.FEATS.index('ctx_generation_ordinal')", "GEN_I = T.FEATS.index('ctx_generation_age_ms')"),
    ('the decision time not stored', "decision_ms=float(k_['decision_ms']), generation", "decision_ms=0., generation"),
    ('raw weights in the fit', "label=np.asarray(y, float), weight=norm(w), feature_name", "label=np.asarray(y, float), weight=np.asarray(w, float), feature_name"),
    ('all 107 columns for every candidate', "    cols = [T.FEATS.index(c) for c in INPUTS[cand]]; lg = lgb()", "    cols = list(range(107)); lg = lgb()"),
    ('a model file overwritten', "            C.require(not p.exists(), f'{p.name}: a model file exists (never overwritten)'); b.save_model(str(p))", "            b.save_model(str(p))"),
    ('an unweighted training mean', "wn = norm(e['w']); ym = float(np.sum(wn * e['y']) / np.sum(wn))", "wn = norm(e['w']); ym = float(np.mean(e['y']))"),
    ('widened training ranges', "            rng[f] = [float(v.min()), float(v.max())] if len(v) else None", "            rng[f] = [float(v.min()) - 1e9, float(v.max()) + 1e9] if len(v) else None"),
    ('the horizon unchecked', "        if hb == 'nan' or hb not in m['horizon_bins']:\n            return 'unsupported_horizon'\n", ""),
    ('a 20-original floor', "MIN_ORIG = float(R2.END), float(L.STEP), 86400000, R2.INV_TOL, 200", "MIN_ORIG = float(R2.END), float(L.STEP), 86400000, R2.INV_TOL, 20"),
    ('the categories unchecked', "        if any(str(cats[d][0]) not in set(m['categories'][d]) for d in R2.DIMS):\n            return 'unsupported_state_category'\n", ""),
    ('POST on a tie', "a = ('POST' if a_ > 0 else 'WAIT') if head == 'post' else ('PULL' if a_ < 0 else 'KEEP')", "a = ('POST' if a_ >= 0 else 'WAIT') if head == 'post' else ('PULL' if a_ < 0 else 'KEEP')"),
    ('PULL on a tie', "a = ('POST' if a_ > 0 else 'WAIT') if head == 'post' else ('PULL' if a_ < 0 else 'KEEP')", "a = ('POST' if a_ > 0 else 'WAIT') if head == 'post' else ('PULL' if a_ <= 0 else 'KEEP')"),
    ('the other side\'s head', "a_ = float(predict(self.H[(s, head)], self.cand,", "a_ = float(predict(self.H[(1 - s, head)], self.cand,"),
    ('the cap unchecked', "        if st == 'absent' and v['room'] < -1e-9:\n            return dict(rec, status='cap_blocked', action='WAIT')\n", ""),
    ('near END unchecked', "        if st == 'absent' and g + KP.PLACE >= B.end:\n            return dict(rec, status='near_end', action='WAIT')\n", ""),
    ('a nonfinite prediction used', "        if not np.isfinite(a_):\n            return dict(rec, status='nonfinite_prediction', action=fb, row=row, A=a_)\n", ""),
    ('the unsupported fallback WAITs', "            return dict(rec, status=ns, action=fb, row=row)", "            return dict(rec, status=ns, action='WAIT', row=row)"),
    ('the review rule keeps', "policy=self.name, status='first_review_pull', action='PULL')", "policy=self.name, status='first_review_pull', action='KEEP')"),
    ('N0 posts', "policy=self.name, status='n0', action='WAIT')", "policy=self.name, status='n0', action='POST')"),
    ('a hybrid routes by the wrong state', "        return (self.entry if v['state'] == 'absent' else self.retain).decide(v, B)", "        return (self.retain if v['state'] == 'absent' else self.entry).decide(v, B)"),
    ('ENTRY_ONLY retains by RICH', "        p = Hybrid(Candidate(heads['RICH'], csup, 'RICH'), Teacher0(M, sup), name)", "        p = Hybrid(Candidate(heads['RICH'], csup, 'RICH'), Candidate(heads['RICH'], csup, 'RICH'), name)"),
    ('an undeclared policy made', "    C.require(name in POLICIES, f'an undeclared policy {name}')\n", ""),
]
