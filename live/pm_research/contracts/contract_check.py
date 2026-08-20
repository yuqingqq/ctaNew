#!/usr/bin/env python3
"""Structural contract diff over contracts.yaml (replaces contract_inventory.py).

The v8 checker was proven unsound (review M8-3): it scanned markdown prose,
ignored git failures, could not parse generics, used unqualified field names,
and — decisively — reported ZERO changes when
`Known[Uncertain[JointCompetitionState]]` was narrowed to `CompetitionState`,
the exact regression class it existed to catch.

This one compares OWNER-QUALIFIED FIELDS WITH THEIR TYPES, so narrowing,
moving, renaming and deleting are all detected. Subprocess failures are fatal.
Intentional removals go in removals_allowlist.yaml with a reason.

  python3 contract_check.py <base-ref> [<ref>|WORKTREE]
  python3 contract_check.py --selftest
"""
import subprocess, sys, yaml, os

REL = "live/pm_research/contracts/contracts.yaml"
ALLOW = os.path.join(os.path.dirname(__file__), "removals_allowlist.yaml")

def load(ref):
    if ref in ("WORKTREE", "", None):
        return yaml.safe_load(open(REL))
    r = subprocess.run(["git", "show", f"{ref}:{REL}"], capture_output=True, text=True)
    if r.returncode != 0:                      # FATAL — v8 silently passed here
        sys.exit(f"FATAL: cannot read {REL} at ref {ref!r}: {r.stderr.strip()}")
    return yaml.safe_load(r.stdout)

def flatten(doc):
    """owner-qualified name -> type string. Types are compared, not just names."""
    out = {}
    for tname, t in (doc.get("types") or {}).items():
        out[f"type:{tname}"] = t.get("kind", "record")
        for g in t.get("generic", []) or []:
            out[f"type:{tname}<{g}>"] = "generic-param"
        for f, ty in (t.get("fields") or {}).items():
            out[f"field:{tname}.{f}"] = str(ty)
        for m, sig in (t.get("methods") or {}).items():
            out[f"method:{tname}.{m}"] = str(sig)
        for v in t.get("variants", []) or []:
            out[f"variant:{tname}|{v}"] = "variant"
        for b in t.get("builtin_ids", []) or []:
            out[f"plugin:{tname}.{b}"] = "builtin"
        if "protocol" in t:
            for k, sig in t["protocol"].items():
                out[f"proto:{tname}.{k}"] = str(sig)
    for mname, m in (doc.get("modules") or {}).items():
        out[f"module:{mname}"] = "module"
        for k in ("produces", "consumes", "requires"):
            if k in m:
                out[f"module:{mname}.{k}"] = str(m[k])
    for p, ports in (doc.get("ports") or {}).items():
        out[f"ports:{p}"] = str(sorted(ports))
    for r in doc.get("rules", []) or []:
        out[f"rule:{r}"] = "rule"
    return out

def diff(a, b):
    removed = {k: a[k] for k in a if k not in b}
    changed = {k: (a[k], b[k]) for k in a if k in b and a[k] != b[k]}
    added = {k: b[k] for k in b if k not in a}
    return removed, changed, added

def allowlist():
    if not os.path.exists(ALLOW):
        return {}
    return yaml.safe_load(open(ALLOW)) or {}

def selftest():
    """The regressions this tool must catch — including v8's exact blind spot."""
    base = {"types": {"DecisionProblem": {"fields": {
        "competition": "Known[Uncertain[JointCompetitionState]] | Unavailable",
        "belief": "Known[BeliefProcess]"}}}}
    cases = {
        "narrowing (v6/v8 blind spot)": {"types": {"DecisionProblem": {"fields": {
            "competition": "CompetitionState", "belief": "Known[BeliefProcess]"}}}},
        "deletion": {"types": {"DecisionProblem": {"fields": {"belief": "Known[BeliefProcess]"}}}},
        "rename": {"types": {"DecisionProblem": {"fields": {
            "competition_state": "Known[Uncertain[JointCompetitionState]] | Unavailable",
            "belief": "Known[BeliefProcess]"}}}},
        "move to another owner": {"types": {"DecisionProblem": {"fields": {"belief": "Known[BeliefProcess]"}},
            "Other": {"fields": {"competition": "Known[Uncertain[JointCompetitionState]] | Unavailable"}}}},
    }
    ok = True
    for name, mutated in cases.items():
        rem, chg, _ = diff(flatten(base), flatten(mutated))
        caught = bool(rem or chg)
        print(f"  {'PASS' if caught else 'FAIL'}  detects {name}")
        ok &= caught
    unchanged = diff(flatten(base), flatten(base))
    same_ok = not (unchanged[0] or unchanged[1] or unchanged[2])
    print(f"  {'PASS' if same_ok else 'FAIL'}  no false positive on identical input")
    return 0 if (ok and same_ok) else 1

if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    if len(sys.argv) < 2:
        sys.exit("usage: contract_check.py <base-ref> [<ref>|WORKTREE] | --selftest")
    a = flatten(load(sys.argv[1]))
    b = flatten(load(sys.argv[2] if len(sys.argv) > 2 else "WORKTREE"))
    removed, changed, added = diff(a, b)
    allowed = allowlist()
    fatal = False
    print(f"REMOVED ({len(removed)}):")
    for k, v in sorted(removed.items()):
        why = allowed.get(k)
        print(f"  {k}: {v}" + (f"   [allowed: {why}]" if why else "   *** UNEXPLAINED ***"))
        fatal |= not why
    print(f"TYPE-CHANGED ({len(changed)}):")
    for k, (o, n) in sorted(changed.items()):
        why = allowed.get(k)
        print(f"  {k}: {o!r} -> {n!r}" + (f"   [allowed: {why}]" if why else "   *** UNEXPLAINED ***"))
        fatal |= not why
    print(f"ADDED ({len(added)}):")
    for k in sorted(added):
        print(f"  {k}")
    sys.exit(1 if fatal else 0)
