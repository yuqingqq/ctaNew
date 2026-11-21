"""MUTATION CHECK of pf_adv2_entry_replay (the ADV2-ENTRY-SMOOTH-v1 bounded own-state comparison and reached entry states; 2026-11-21, session
aea5bfb9). Each planted error is a textual replacement in a copy of pf_adv2_entry_replay.py, run by pf_mutant_one.py with
pf_adv2_entry_replay_selftest. Required: the control passes; every mutant is caught BY A CHECK; every target occurs exactly once."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / 'pf_adv2_entry_replay.py'
SELFTEST = 'pf_adv2_entry_replay_selftest'
FAIL = 'ADV2 ENTRY REPLAY SELFTEST FAILED'
MUTANTS = [
    ("R3's domain", "DOMAIN = 'adv2_entry_smooth_v1_reached_20261121'", "DOMAIN = 'adv2_rich_v1_reached_20261121'"),
    ('live states selected', "        if r['state'] != 'absent' or r['status'] != 'model' or r['A'] is None", "        if r['status'] != 'model' or r['A'] is None"),
    ('fallback decisions selected', "if r['state'] != 'absent' or r['status'] != 'model' or r['A'] is None", "if r['state'] != 'absent' or r['A'] is None"),
    ('the inclusion count of the first decision', "n[s] = n.get(s, 0) + 1;", "n[s] = 1;"),
    ('the trace fills hash of the decisions', "fills_sha256=L._h(Lx['fills']), decisions_sha256=L._h(Lx['decisions'])", "fills_sha256=L._h(Lx['decisions']), decisions_sha256=L._h(Lx['decisions'])"),
    ('the trace calls from the binding counter', "teacher=dict(calls=len(pol.log), binding_checked=pol.bind_checked", "teacher=dict(calls=pol.bind_checked + 1, binding_checked=pol.bind_checked"),
    ('the trace without order sizes', "sizes=[float(o['size']) for o in B.E.o]", "sizes=[]"),
    ('r3_match ignores the fills', "\n                and ref.get('fills_sha256') == e['trace']['fills_sha256'])", ")"),
    ('r3_match ignores the value', "and ref.get('value') == e['value']", ""),
    ('the fork check skipped', "        if ctx['fingerprint'] != d['fingerprint']:\n            out.append(dict(r, status='fork_mismatch')); continue\n", ""),
    ('the input comparison skipped', "same_in = bool(fi.get('status') == 'ok' and np.asarray(fi['inputs'], np.float32).tobytes() == np.asarray(rec_in, np.float32).tobytes())", "same_in = True"),
    ('the arm ledgers not attached', "            for n_, v_ in LS.arm_ledgers(tf, res).items():\n                res['arms'][n_].update(v_)\n", "            pass\n"),
    ('the pair probe at another rate', "tf = L.teacher_factory(M, sup, world, every=AUDIT_EVERY); res = L.pair_bound(ctx, tf, 'post')", "tf = L.teacher_factory(M, sup, world, every=7); res = L.pair_bound(ctx, tf, 'post')"),
    ('the RICH prediction from the SMOOTH head', "'RICH': float(A.predict(rich['RICH'][(s, 'post')], 'RICH', xr)[0])", "'RICH': float(N.predict(smooth[(s, 'post')], xr)[0])"),
    ('part identity ignores the sources', " and rec.get('sources') == sources()", ""),
    ('part identity ignores the bytes', " and hashlib.sha256(b).hexdigest() == rec.get('sha256')", ""),
    ('an existing record not checked', "        C.require(part_record_ok(json.loads(rp.read_text()), members, (out / 'parts' / f'part_{k:03d}.jsonl.gz').read_bytes(), fit_sha()),\n                  f'part {k}: an existing record of another identity')\n", ""),
    ('frames without the membership check', "        rows = LS.gz_rows(b); C.require([LS.ckey(LS.wkey(r)) for r in rows] == [LS.ckey(LS.wkey(x)) for x in members], f'part {k}: its rows are not exactly its members')", "        rows = LS.gz_rows(b)"),
    ('the run without the committed fit', "    C.require(N.committed_equal(rec_dir / 'fit' / 'fit.json', N.OUT / 'fit.json'), 'the fit manifest is not committed in HEAD: no outcome before the model hashes')\n", ""),
]
