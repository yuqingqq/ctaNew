"""MUTATION CHECK of pf_adv2_entry (ADV2-ENTRY-SMOOTH-v1: the declared delta, the fits, R3's training identity, the policies; 2026-11-21,
session aea5bfb9). Each planted error is a textual replacement in a copy of pf_adv2_entry.py, executed under its real path in a fresh process
by pf_mutant_one.py, which runs pf_adv2_entry_selftest.run on it. Required: the control passes; every mutant is caught BY A CHECK; every target
occurs exactly once."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / 'pf_adv2_entry.py'
SELFTEST = 'pf_adv2_entry_selftest'
FAIL = 'ADV2 ENTRY SELFTEST FAILED'
MUTANTS = [
    ('the old min_data_in_leaf', "DELTA = dict(min_data_in_leaf=1000)", "DELTA = dict(min_data_in_leaf=200)"),
    ('200 rounds', "PARAMS, ROUNDS = dict(A.PARAMS, **DELTA), 50", "PARAMS, ROUNDS = dict(A.PARAMS, **DELTA), 200"),
    ('the delta check removed', "    C.require(diff == {'min_data_in_leaf': (200, 1000)} and ROUNDS == 50 and A.ROUNDS == 200, f'the declared delta changed: {diff}, rounds {ROUNDS}')\n", ""),
    ('the parameter hash unchecked', "    C.require(A.sha_of(PARAMS) == PARAMS_SHA, 'the resolved parameters changed')\n", ""),
    ('unnormalized weights', "weight=A.norm(w), feature_name=list(A.INPUTS['RICH'])", "weight=np.asarray(w, float), feature_name=list(A.INPUTS['RICH'])"),
    ('the columns reversed', "lg = A.lgb(); cols = [T.FEATS.index(c) for c in A.INPUTS['RICH']]", "lg = A.lgb(); cols = [T.FEATS.index(c) for c in A.INPUTS['RICH']][::-1]"),
    ('predictions on the FAIR columns', "    return A.predict(b, 'RICH', X)", "    return A.predict(b, 'FAIR', X)"),
    ('a model overwritten', "        C.require(not p.exists(), f'{p.name}: a model file exists (never overwritten)'); b.save_model(str(p))", "        b.save_model(str(p))"),
    ('the in-sample error unweighted', "insample_wmse=float(np.sum(wn * (p1 - e['y']) ** 2) / np.sum(wn))", "insample_wmse=float(np.mean((p1 - e['y']) ** 2))"),
    ("R3's extraction not checked as committed", "    C.require(committed_equal(receipt / 'fit' / 'extract.json', Path(out) / 'train' / 'extract.json'), \"R3's extraction manifest is not the committed one\")\n", ""),
    ("R3's training files not checked", "    C.require(A.manifest_of(E, ex['statuses'])['heads'] == ex['heads'], \"R3's training files differ from their extraction manifest\")\n", ""),
    ('the KEEP heads trained too', "    return {h: E[h] for h in HEADS}", "    return E"),
    ('SMOOTH entry from the RICH heads', "        ent = A.Candidate(smooth, csup, 'RICH'); ent.name = NAME", "        ent = A.Candidate(rich['RICH'], csup, 'RICH'); ent.name = NAME"),
    ('SMOOTH retention by the candidate', "        p = A.Hybrid(ent, A.Teacher0(M, sup), NAME)", "        p = A.Hybrid(ent, ent, NAME)"),
    ('the SMOOTH name lost', "ent.name = NAME", "ent.name = 'RICH'"),
    ('an undeclared policy accepted', "    C.require(name in POLICIES, f'an undeclared policy {name}')\n", ""),
]
