"""GUARDS THAT EXIST, WORK, AND CANNOT FIRE WHERE IT MATTERS (BE 158/159).

Rule 15 at the level of PLACEMENT rather than content: a guard whose only
caller is a selftest has never proved it can fire in production, and neither
has one whose refusing branch is switched off by the way its callers invoke
it. Measured over 281 modules: **101 guard-shaped functions are off the
production path** -- 80 TEST_ONLY, 20 ORPHAN, 1 PARAM_DISABLED.

READ THIS FIRST -- 09-07, AND WHY THE ANSWER IS NOT THE ALARMING ONE.
`p003_day_status_20260907_v1.json` credits `assert_settlement_day_admissible`
-- "the rule-11 guard has been holding it shut" -- and this sweep finds that
function called ONLY from `selftest`. The obvious reading is that a protected
day was protected by nobody asking. THAT READING IS WRONG, and the artifact
settles it: `p003_de_point_estimate_day_20260907_L250ms__20260908T122617Z.json`
records, on BOTH arms, `status: NOT_VALUED_DAY_NOT_ADMISSIBLE`,
`admissibility.admissible: False`, `class: NOT_ADMISSIBLE`,
`refusal_name: SETTLEMENT_DAY_NOT_ADMISSIBLE`. A day run WAS performed on
09-07 on 2026-09-08 and it was gated by LIVE code:
`de_multiday_gate1_runner.run_day` line 9131 computes
`settlement_admissibility(day, params, fixture=fixture)` and line 9133
branches on it. 09-07 was shielded by a computed verdict that the production
path honoured, not by an unasked question.

WHAT IS TRUE, AND IS A DIFFERENT GUARANTEE. Two things gate this endpoint:
`settlement_admissibility` COMPUTES and never raises (rule 14), and
`assert_settlement_day_admissible` RAISES -- its own docstring calls it "the
entry that DECIDES ... any caller asking to VALUE a day gets a refusal by
name". The COMPUTER is on the path; the REFUSER is not. So a caller reaching
the endpoint THROUGH `run_day` is gated today, and a caller invoking the
settlement estimator DIRECTLY meets nothing, because the backstop written for
exactly that case is reachable only from the selftest. The receipt's own
sentence is the distinction, and it is exact: "The estimator REFUSES if asked
directly (SETTLEMENT_DAY_NOT_ADMISSIBLE); this receipt records that it was not
asked." Not-asked and would-have-refused are different guarantees. Only the
first is evidenced for 09-07, and the second is what survives someone asking
tomorrow through a path that is not `run_day`.
(`settlement_endpoint.admissible_days` is `None` in params v29, which is why
the class is NOT_ADMISSIBLE; that field is the switch that opens the day.)

A HIT IS NOT A DEFECT, AND ONE HIT IS THE OPPOSITE OF ONE. The scalar
`de_head_scoring.score_lgbm_condvalue` appears here as reachable only from its
own falsifier. That is CORRECT AND MUST NOT BE "FIXED": `d09f25c` replaced it
with `score_lgbm_condvalue_batch`, and the scalar survives deliberately as the
EXACT-EQUALITY REFERENCE the falsifier scores against to prove batching is
score-neutral. Delete it and you silently remove the proof that the
optimization changes nothing. `be_daybook_build.placement_latency_of` is the
same shape -- a helper for the 184-check battery, test-only by design.
So 101 is THE SIZE OF THE CLASS, not a defect count; each module needs its
owning seat's judgement.

THE CHEAP LIVE ONE. `verify_run_inputs` (TEST_ONLY here) costs 1.2 ms and
0.91 MB hashed on a day run, and against params v29 it REFUSES TODAY: 2 of 10
cascade modules differ (`de_head_scoring.py` declared 53a406a0ae2a11ff / on
disk 31c368384770351f, `de_phase4_diag_runner.py` declared cb97b94dbd3fc6ca /
on disk 9e2a0977d8dfc3be) -- exactly the files the three optimization commits
touch. Wiring it needs BOTH a params crank and a call-site change; see
`WIRING_VERIFY_RUN_INPUTS` below.
"""

import ast, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent   # the tree this module lives in


#: WHAT MUST CHANGE FOR `verify_run_inputs` TO BE WIRED (BE 158, for DE).
#: Both, not either: the call site alone refuses every day until the params
#: are repointed, and the params alone leave the cascade unchecked.
WIRING_VERIFY_RUN_INPUTS = {
    "call_site": {
        "module": "de_multiday_gate1_runner.py",
        "where": "run_day, inside the existing `if not fixture:` block "
                 "(~line 9026), at S0 and BEFORE the book is loaded",
        "from": "verify_pinned_models(params); "
                "_theta_pins = verify_pinned_thetas(params)",
        "to": "_vr = verify_run_inputs(params); _theta_pins = _vr['thetas']",
        "why_it_restores_the_check": (
            "the cascade loop in `verify_be_module` is gated on "
            "`actual_sha is None`; the day path reaches it only through "
            "`import_be_cascade`, which SUPPLIES the digest, so the loop is "
            "skipped and every real day records n_modules_checked=1, "
            "scope='THE ENTRY POINT ONLY'. `verify_run_inputs` calls it "
            "with no digest, which re-enables the ten-module check."),
        "constraints": [
            "must stay INSIDE `if not fixture:` -- a fixture run opens no "
            "path under data/, and lines 3750-3751 record that these are "
            "deliberately not called there",
            "`_theta_pins` must keep the RESULT object, not a second read "
            "(DE 203 computes the rule-11 field from it)",
            "leave `import_be_cascade`'s own call alone; it legitimately "
            "checks the entry point it imported",
            "digests resolve against Path(__file__).parents[2] -- the "
            "WORKTREE the runner runs from, not the main tree",
        ],
    },
    "params_crank": {
        "file": "declarations/de_multiday_gate1_params_v29.json",
        "what": "be_cascade.modules must be repointed for the 2 modules "
                "that differ, which is a params + design version pair",
        "measured_2026_09_10": {"n_cascade": 10, "n_differing": 2,
                                "cost_ms": 1.2, "bytes_hashed": 913852},
    },
    "sequencing": "wiring before the crank stops every day run; DE holds "
                  "the heavy lock for plan step 2, so the order is the "
                  "coordinator's call",
}

TEST_NAMES = ("falsify", "selftest", "_selftest", "main_selftest")


def _is_test_name(n: str) -> bool:
    return (n in TEST_NAMES or n.startswith("test_") or n.startswith("_fx")
            or n.startswith("fx_") or n.endswith("_selftest")
            or n.endswith("_falsify") or n.startswith("falsify"))


def _raises(node) -> list:
    out = []
    for n in ast.walk(node):
        if isinstance(n, ast.Raise) and n.exc is not None:
            e = n.exc
            nm = None
            if isinstance(e, ast.Call) and isinstance(e.func, ast.Name):
                nm = e.func.id
            elif isinstance(e, ast.Call) and isinstance(e.func, ast.Attribute):
                nm = e.func.attr
            elif isinstance(e, ast.Name):
                nm = e.id
            if nm:
                out.append(nm)
    return out


def _guardish(fn: ast.FunctionDef) -> bool:
    r = _raises(fn)
    if any("Refus" in x or "Error" in x for x in r):
        return True
    return fn.name.split("_")[0] in ("verify", "assert", "require", "check",
                                     "must", "ensure")


def collect(root: Path) -> dict:
    mods = {}
    for p in sorted(root.glob("*.py")):
        try:
            tree = ast.parse(p.read_text(errors="replace"))
        except SyntaxError:
            continue
        mods[p.name] = tree
    guards = {}        # name -> [(module, lineno, params_none)]
    calls = []         # (module, caller_chain, callee, lineno)

    def walk(mod, node, chain):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)):
                nchain = chain + [ch.name]
                if isinstance(node, ast.Module) and _guardish(ch):
                    nones = [a.arg for a, d in zip(
                        ch.args.args[-len(ch.args.defaults):] if ch.args.defaults else [],
                        ch.args.defaults) if isinstance(d, ast.Constant)
                        and d.value is None]
                    nones += [a.arg for a, d in zip(ch.args.kwonlyargs,
                                                    ch.args.kw_defaults)
                              if isinstance(d, ast.Constant) and d.value is None]
                    guards.setdefault(ch.name, []).append(
                        {"module": mod, "line": ch.lineno,
                         "none_params": nones, "node": ch})
                walk(mod, ch, nchain)
            else:
                if isinstance(ch, ast.Call):
                    f = ch.func
                    nm = f.id if isinstance(f, ast.Name) else (
                        f.attr if isinstance(f, ast.Attribute) else None)
                    if nm:
                        calls.append({"module": mod, "chain": list(chain),
                                      "callee": nm, "line": ch.lineno,
                                      "kwargs": [k.arg for k in ch.keywords
                                                 if k.arg],
                                      "nargs": len(ch.args)})
                walk(mod, ch, chain)

    for mod, tree in mods.items():
        walk(mod, tree, [])
    return guards, calls


def analyse():
    guards, calls = collect(ROOT)
    by_callee = {}
    for c in calls:
        by_callee.setdefault(c["callee"], []).append(c)

    A, B, C = [], [], []
    for name, defs in sorted(guards.items()):
        sites = [c for c in by_callee.get(name, [])
                 if not (len(c["chain"]) == 1 and c["chain"][0] == name)]
        prod = [c for c in sites if not any(_is_test_name(x) for x in c["chain"])]
        test = [c for c in sites if any(_is_test_name(x) for x in c["chain"])]
        where = ", ".join(f"{d['module']}:{d['line']}" for d in defs)
        if not sites:
            A.append({"guard": name, "defined": where, "n_calls": 0})
        elif not prod:
            B.append({"guard": name, "defined": where,
                      "n_test_calls": len(test),
                      "test_callers": sorted({f"{c['module']}:{'.'.join(c['chain']) or '<module>'}"
                                              for c in test})[:4]})
        else:
            # class C: a refusal gated on a None-defaulted parameter that
            # every production caller supplies.
            for d in defs:
                for p in d["none_params"]:
                    if not _gates_a_raise(d["node"], p):
                        continue
                    passers = [c for c in prod if p in c["kwargs"]]
                    if passers:
                        C.append({
                            "guard": name, "defined": f"{d['module']}:{d['line']}",
                            "param": p,
                            "n_production_callers": len(prod),
                            "n_that_DISABLE_the_branch": len(passers),
                            "disabled_at": sorted({f"{c['module']}:{'.'.join(c['chain'])}:{c['line']}"
                                                   for c in passers})[:6],
                            "callers": sorted({f"{c['module']}:{'.'.join(c['chain']) or '<module>'}:{c['line']}"
                                               for c in prod})[:6]})
    return A, B, C


def _gates_a_raise(fn, param) -> bool:
    """Is there a `raise` inside an `if ... <param> is None ...` branch?"""
    for n in ast.walk(fn):
        if not isinstance(n, ast.If):
            continue
        names = {x.id for x in ast.walk(n.test) if isinstance(x, ast.Name)}
        if param not in names:
            continue
        neg = any(isinstance(c, ast.Is) for cmp_ in
                  [x for x in ast.walk(n.test) if isinstance(x, ast.Compare)]
                  for c in cmp_.ops)
        if not neg and not any(isinstance(x, ast.UnaryOp)
                               for x in ast.walk(n.test)):
            continue
        # A BARE `if p is None: raise` means "this argument is REQUIRED",
        # and a caller supplying it is correct use, not a disabled guard.
        # The defect shape is the INVERSE: the gated branch does WORK that
        # can refuse, so supplying the parameter SKIPS verification.
        has_loop = any(isinstance(x, (ast.For, ast.While)) for b in n.body
                       for x in ast.walk(b))
        if not (has_loop or len(n.body) > 2):
            continue
        for b in n.body:
            if _raises(b):
                return True
    return False


def falsify() -> int:
    checks = []

    def note(n, ok):
        checks.append((n, bool(ok)))
    A, B, C = analyse()
    names_C = {c["guard"] for c in C}
    # POSITIVE CONTROL: the guard DA 190 found must be in class C.
    names_B = {x["guard"] for x in B}
    # POSITIVE CONTROL 1: DA 190's defect -- the composite guard is test-only.
    note("verify_run_inputs is detected as TEST_ONLY", "verify_run_inputs" in names_B)
    # POSITIVE CONTROL 2: the day path's weakened call is detected.
    note("verify_be_module is detected as PARAM_DISABLED",
         "verify_be_module" in names_C)
    # POSITIVE CONTROL: the sweep finds something at all.
    note("the sweep is not vacuous", (len(A) + len(B) + len(C)) > 0)
    # KNOWN-BAD: a synthetic module with a guard called only from falsify
    # must land in B, and the same guard called from a real function must not.
    import tempfile, textwrap
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "zz_fixture.py"
        p.write_text(textwrap.dedent('''
            class Refused(RuntimeError): pass
            def verify_thing(x):
                if x: raise Refused("no")
            def falsify():
                verify_thing(1)
        '''))
        global ROOT
        keep = ROOT
        ROOT = Path(td)
        a2, b2, c2 = analyse()
        ROOT = keep
        note("a fixture guard called only from falsify lands in TEST_ONLY",
             any(x["guard"] == "verify_thing" for x in b2))
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "zz_fixture.py"
        p.write_text(textwrap.dedent('''
            class Refused(RuntimeError): pass
            def verify_thing(x):
                if x: raise Refused("no")
            def production():
                verify_thing(1)
            def falsify():
                verify_thing(1)
        '''))
        keep = ROOT
        ROOT = Path(td)
        a3, b3, c3 = analyse()
        ROOT = keep
        note("the same guard WITH a production caller is NOT reported",
             not any(x["guard"] == "verify_thing" for x in b3))
    for n, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {n}")
    bad = [n for n, ok in checks if not ok]
    print(json.dumps({"falsifier": "be_offpath_guards", "n": len(checks),
                      "failed": bad}))
    return 1 if bad else 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(falsify())
    A, B, C = analyse()
    print("=== C  PARAM_DISABLED: a refusal switched off by how callers invoke it")
    for x in C:
        print(json.dumps(x))
    print(f"  -> {len(C)}")
    print()
    print("=== B  TEST_ONLY: every call site is inside a test function")
    for x in B:
        print(json.dumps(x))
    print(f"  -> {len(B)}")
    print()
    print("=== A  ORPHAN: no call site anywhere")
    for x in A:
        print(json.dumps(x))
    print(f"  -> {len(A)}")
