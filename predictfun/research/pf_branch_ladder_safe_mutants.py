"""MUTATION CHECK of pf_branch_ladder_safe (2026-11-21, session aea5bfb9), run by pf_mutant_one.py with pf_ladder_audit_selftest (run(mod) imports
the audit over the mutated module). Required: the control passes; every mutant is caught BY A CHECK; every target occurs exactly once."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / 'pf_branch_ladder_safe.py'
SELFTEST = 'pf_ladder_audit_selftest'
FAIL = 'AUDIT SELFTEST FAILED'
MUTANTS = [
    ('no object tape / mark check', "    C.require(BE.tape_id(B.E.f, B.E.tr, B.mark) == saved['tape_id'] and BE.canon_mark(B.mark) == BE.canon_mark(saved['mark']), \"the object's own tape / mark differ from the envelope\")",
     "    pass"),
    ('no ident check', "    C.require(B.ident == saved['ident'], \"the object's ident differs from the envelope\")", "    pass"),
    ('executable fields reduced', "    return [B.config, B.q0, B.cap, B.levels, B.inv, B.own_cross, B.end, B.E.end, B.E.client]", "    return [B.config]"),
    ('no next_k check', "    C.require((int(saved['next_k']) == 0 and not started(B)) or (int(saved['next_k']) == B.k + 1 and B.acted == {0, 1}), \"next_k is not the object's next tick\")",
     "    pass"),
    ('a save off the tick boundary', "    C.require((int(next_k) == 0 and not started(B)) or (int(next_k) == B.k + 1 and B.acted == {0, 1}), 'a save that is not at a tick boundary')", "    pass"),
    ('no path tape check', "    C.require(BE.tape_id(path['frames'], path['trades'], path.get('mark')) == saved['tape_id'], \"a path other than the saved state's tape\")", "    pass"),
]
