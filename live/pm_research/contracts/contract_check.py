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
    out["meta:version_floor"] = "monotonic"   # bumps are fine; regression caught below
    out["_version"] = doc.get("version")
    for p in (doc.get("prelude") or {}).get("primitives", []):
        out[f"prelude:primitive:{p}"] = "primitive"
    for e in (doc.get("prelude") or {}).get("external", []):
        out[f"prelude:external:{e}"] = "external"
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
        if "registry" in t:                                 # M9-2: registry deletion
            out[f"registry:{tname}"] = str(t["registry"])
        if "validation" in t:                               # M9-2: n-ary -> pairwise
            out[f"validation:{tname}"] = str(t["validation"])
        if t.get("notes"):                                  # M9-2: normative notes
            out[f"note:{tname}"] = " ".join(str(t["notes"]).split())
    for mname, m in (doc.get("modules") or {}).items():
        out[f"module:{mname}"] = "module"
        for k in ("produces", "consumes", "requires", "ports"):
            if k in m:
                v = m[k]
                if isinstance(v, dict):
                    norm = sorted(f"{k2}={v2}" for k2, v2 in v.items())
                else:
                    norm = sorted(map(str, v if isinstance(v, list) else [v]))
                out[f"module:{mname}.{k}"] = str(norm)   # order-insensitive (v9 audit)
    for p, ports in (doc.get("ports") or {}).items():
        out[f"ports:{p}"] = str(sorted(ports))
    rules = doc.get("rules") or {}
    if isinstance(rules, list):
        for r in rules:
            out[f"rule:{r}"] = "rule"
    else:
        for r, body in rules.items():                       # M9-2: enforcement bodies
            out[f"rule:{r}"] = " ".join(str(body.get("body", "")).split())
            for c in body.get("checks", []) or []:
                out[f"rule:{r}.check:{c.split(':')[0]}"] = c
    return out

def diff(a, b):
    removed = {k: a[k] for k in a if k not in b}
    changed = {k: (a[k], b[k]) for k in a if k in b and a[k] != b[k]}
    added = {k: b[k] for k in b if k not in a}
    return removed, changed, added

def invariants(doc):
    """Target-side checks that a backward diff cannot express (M9-2)."""
    errs = []
    types = set((doc.get("types") or {}).keys())
    pre = (doc.get("prelude") or {})
    known = types | set(pre.get("primitives", [])) | set(pre.get("external", []))
    import re as _re
    for tname, t in (doc.get("types") or {}).items():
        for fname, ty in (t.get("fields") or {}).items():
            for ref in _re.findall(r'[A-Z][A-Za-z0-9_]*', str(ty)):
                if ref not in known and not ref.isupper():
                    errs.append(f"unresolved reference {tname}.{fname} -> {ref}")
    produced = set()
    for m in (doc.get("modules") or {}).values():
        p = m.get("produces")
        produced |= set(p if isinstance(p, list) else [p] if p else [])
    for mname, m in (doc.get("modules") or {}).items():
        for c in (m.get("consumes") or []):
            base = str(c).split("[")[0].split(".")[0]
            if base in types and base not in {str(x).split("[")[0] for x in produced} \
               and base not in ("DecisionProblem", "Decision", "HealthEvent",
                                "HeartbeatRegistration", "HeartbeatPulse",
                                "CouplingGraph"):   # supplied by SP-Strategy config
                errs.append(f"{mname} consumes {base} with no declared producer")
    return errs

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
    va, vb = a.pop("_version", 0), b.pop("_version", 0)
    removed, changed, added = diff(a, b)
    if (vb or 0) < (va or 0):
        print(f"*** VERSION REGRESSION: {va} -> {vb}")
        removed["meta:version"] = f"{va} -> {vb}"
    inv = invariants(load(sys.argv[2] if len(sys.argv) > 2 else "WORKTREE"))
    if inv:
        print(f"INVARIANT FAILURES ({len(inv)}):")
        for e in inv:
            print(f"  *** {e}")
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
    sys.exit(1 if (fatal or inv) else 0)
