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
MIG_REL = "live/pm_research/contracts/migrations.yaml"

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
    def refs(x):
        return [r for r in _re.findall(r'[A-Z][A-Za-z0-9_]*', str(x)) if not r.isupper()]
    def check(where, x):
        for r in refs(x):
            if r not in known:
                errs.append(f"unresolved reference {where} -> {r}")
    for tname, t in (doc.get("types") or {}).items():
        for fname, ty in (t.get("fields") or {}).items():
            check(f"{tname}.{fname}", ty)
        for mname, sig in (t.get("methods") or {}).items():      # M10-2
            check(f"{tname}.{mname}()", sig)
        for pname, sig in (t.get("protocol") or {}).items():     # M10-2
            check(f"{tname}::{pname}", sig)
        for v in (t.get("variants") or []):                      # M10-2
            # a variant is Tag(payload) — the Tag is a constructor, not a type.
            inner = _re.search(r'\((.*)\)\s*$', str(v))
            if not inner:
                continue
            for part in inner.group(1).split(","):
                ty = part.split(":", 1)[1] if ":" in part else part
                check(f"{tname}|{str(v).split('(')[0]}", ty)
        if "registry" in t and str(t["registry"]) not in known:  # M10-2
            errs.append(f"unresolved registry {tname} -> {t['registry']}")
    # duplicate local/external declaration (violates R-SSOT)
    pre_ext = set((doc.get("prelude") or {}).get("external", []))
    for dup in sorted(set((doc.get("types") or {}).keys()) & pre_ext):
        errs.append(f"duplicate declaration (local type AND prelude.external): {dup}")
    # module outputs/inputs must resolve
    for mname, m in (doc.get("modules") or {}).items():          # M10-2
        for k in ("produces", "consumes", "requires"):
            v = m.get(k)
            if v:
                mods = set((doc.get("modules") or {}).keys())
                for item in (v if isinstance(v, list) else [v]):
                    if str(item) in mods:
                        continue                       # a module reference, not a type
                    check(f"module {mname}.{k}", item)
    # declared rules must have a validator id or checks
    for rname, body in (doc.get("rules") or {}).items():
        if isinstance(body, dict) and not (body.get("body") or body.get("checks")):
            errs.append(f"rule {rname} has no body or checks")
    produced = set()
    for m in (doc.get("modules") or {}).values():
        p = m.get("produces")
        produced |= set(p if isinstance(p, list) else [p] if p else [])
    for mname, m in (doc.get("modules") or {}).items():
        for c in (m.get("consumes") or []):
            base = str(c).split("[")[0].split(".")[0]
            cfg = set(doc.get("config_supplied") or [])   # declared, not hardcoded
            if base in types and base not in {str(x).split("[")[0] for x in produced} \
               and base not in cfg and not base.startswith(("SP-", "DA-", "BE-", "DE-", "EV-", "OP-")):
                errs.append(f"{mname} consumes {base} with no declared producer")
    return errs

class StrictLoader(yaml.SafeLoader):                     # M11-2: duplicate keys are errors
    pass
def _no_dupes(loader, node, deep=False):
    seen = set()
    for k, _ in node.value:
        key = loader.construct_object(k, deep=deep)
        if key in seen:
            raise yaml.YAMLError(f"duplicate YAML key: {key!r}")
        seen.add(key)
    return yaml.SafeLoader.construct_mapping(loader, node, deep)
StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_dupes)

def migrations(ref):
    """Load migration records FROM THE TARGET REF (M11-1), not the worktree."""
    if ref in ("WORKTREE", "", None):
        if not os.path.exists(MIG_REL):
            return []
        raw = open(MIG_REL).read()
    else:
        r = subprocess.run(["git", "show", f"{ref}:{MIG_REL}"], capture_output=True, text=True)
        if r.returncode != 0:
            return []
        raw = r.stdout
    recs = (yaml.load(raw, Loader=StrictLoader) or {}).get("migrations", [])
    seen = {}
    for m in recs:                                        # M11-1: conflicting records fail
        sig = (m.get("from_version"), m.get("to_version"), m.get("operation"), m.get("key"))
        if sig in seen:
            sys.exit(f"FATAL: duplicate migration record for {sig}")
        seen[sig] = m
    return recs

def authorises(recs, op, key, old, new, va, vb):
    """A record must match operation, key, EXACT old/new and the version step."""
    for m in recs:
        if (m.get("operation") == op and m.get("key") == key
                and str(m.get("old")) == str(old)
                and (op == "remove" or str(m.get("new")) == str(new))
                and m.get("from_version") == va and m.get("to_version") == vb):
            return m.get("reason", "")
    return None

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
    inv_cases = {
        "undefined protocol return type": {"prelude": {"external": []},
            "types": {"X": {"protocol": {"f": "() -> Nope"}}}},
        "undefined module output": {"prelude": {"external": []},
            "modules": {"M": {"produces": ["Nope"]}}},
        "undefined variant payload": {"prelude": {"external": []},
            "types": {"X": {"variants": ["A(Nope)"]}}},
        "duplicate local/external type": {"prelude": {"external": ["Dup"]},
            "types": {"Dup": {"fields": {}}}},
        "rule without body": {"rules": {"R-X": {}}},
    }
    for name, doc in inv_cases.items():
        caught = bool(invariants(doc))
        print(f"  {'PASS' if caught else 'FAIL'}  invariant detects {name}")
        ok &= caught
    rec = [{"from_version": 1, "to_version": 2, "operation": "change",
            "key": "field:X.y", "old": "A", "new": "B", "reason": "t"}]
    ab = authorises(rec, "change", "field:X.y", "A", "B", 1, 2) is not None
    ac = authorises(rec, "change", "field:X.y", "A", "C", 1, 2) is None
    ver = authorises(rec, "change", "field:X.y", "A", "B", 2, 3) is None
    print(f"  {'PASS' if ab else 'FAIL'}  migration authorises its exact A->B")
    print(f"  {'PASS' if ac else 'FAIL'}  migration REJECTS A->C at an allowed path")
    print(f"  {'PASS' if ver else 'FAIL'}  migration REJECTS a different version step")
    ok &= ab and ac and ver
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
    a.pop("meta:version", None); b.pop("meta:version", None)   # internal, not a contract
    removed, changed, added = diff(a, b)
    if (vb or 0) < (va or 0):
        print(f"*** VERSION REGRESSION: {va} -> {vb}")
        removed["meta:version"] = f"{va} -> {vb}"
    inv = invariants(load(sys.argv[2] if len(sys.argv) > 2 else "WORKTREE"))
    if inv:
        print(f"INVARIANT FAILURES ({len(inv)}):")
        for e in inv:
            print(f"  *** {e}")
    recs = migrations(sys.argv[2] if len(sys.argv) > 2 else "WORKTREE")
    fatal = False
    print(f"REMOVED ({len(removed)}):")
    for k, v in sorted(removed.items()):
        why = authorises(recs, "remove", k, v, None, va, vb)
        print(f"  {k}: {v}" + (f"   [allowed: {why}]" if why else "   *** UNEXPLAINED ***"))
        fatal |= not why
    print(f"TYPE-CHANGED ({len(changed)}):")
    for k, (o, n) in sorted(changed.items()):
        why = authorises(recs, "change", k, o, n, va, vb)
        print(f"  {k}: {o!r} -> {n!r}" + (f"   [allowed: {why}]" if why else "   *** UNEXPLAINED ***"))
        fatal |= not why
    # M11-1: a record is stale only within ITS OWN declared version step
    unused = [m for m in recs
              if m.get("from_version") == va and m.get("to_version") == vb
              and not authorises([m], m["operation"], m["key"], m.get("old"),
                                 m.get("new"), va, vb) is None
              and m["key"] not in removed and m["key"] not in changed]
    if unused:
        print(f"UNUSED MIGRATIONS FOR {va}->{vb} ({len(unused)}):")
        for m in unused:
            print(f"  *** {m['key']}")
        fatal = True
    print(f"ADDED ({len(added)}):")
    for k in sorted(added):
        print(f"  {k}")
    sys.exit(1 if (fatal or inv) else 0)
