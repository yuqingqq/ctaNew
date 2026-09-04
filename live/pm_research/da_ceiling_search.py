"""FINDING (Q-DA-246): refuted 'no value ceiling has ever been computed in either programme' -- AST search over 189 files / 3,421 functions, as-of 2026-09-04T13:54:40Z, found state_gate_v1.bound_over_bins and adverse_move_fast's oracle_upper_bound_cents_per_decision.

DA: is there ANY value ceiling in either programme? Searched by SHAPE.

A negative existence claim carries an as-of and a searched surface or it is
not a claim. The reviewer searched vocabulary and a `v < 0` shape. This is a
THIRD method and deliberately not either: an AST pass for the structural
signature of a ceiling -- a reduction whose terms are FILTERED OR CLIPPED ON A
SIGN. Every ceiling of the form "sum the losses" has it; no amount of renaming
hides it.
"""
import ast, sys, json
from pathlib import Path

ROOTS = [Path("live/pm_research"), Path("live/mm_research")]
REDUCERS = {"sum", "fsum", "nansum", "total", "accumulate"}

def zeroish(node):
    return (isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool)
            and node.value == 0)

def has_sign_test(node):
    """Does this subtree compare something to zero, or clip at zero?"""
    for n in ast.walk(node):
        if isinstance(n, ast.Compare):
            if zeroish(n.left) or any(zeroish(c) for c in n.comparators):
                if any(isinstance(o, (ast.Lt, ast.Gt, ast.LtE, ast.GtE))
                       for o in n.ops):
                    return True
        if isinstance(n, ast.Call):
            f = getattr(n.func, "id", None)
            if f in ("min", "max") and any(zeroish(a) for a in n.args):
                return True
    return False

hits, n_files, n_funcs = [], 0, 0
for root in ROOTS:
    for f in sorted(root.glob("*.py")):
        n_files += 1
        try:
            tree = ast.parse(f.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            hits.append({"file": str(f), "line": 0, "shape": "UNPARSEABLE"})
            continue
        src = f.read_text(encoding="utf-8", errors="replace").splitlines()
        for n in ast.walk(tree):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                n_funcs += 1
            # SHAPE 1: a reduction over a comprehension with a sign filter
            if isinstance(n, ast.Call):
                fn = getattr(n.func, "id", None) or getattr(n.func, "attr",
                                                            None)
                if fn in REDUCERS and n.args:
                    a = n.args[0]
                    if isinstance(a, (ast.GeneratorExp, ast.ListComp)):
                        conds = [c for g in a.generators for c in g.ifs]
                        if any(has_sign_test(c) for c in conds) or \
                                has_sign_test(a.elt):
                            hits.append({
                                "file": str(f), "line": n.lineno,
                                "shape": "REDUCE_OVER_SIGN_FILTERED_TERMS",
                                "src": src[n.lineno - 1].strip()[:140]})
            # SHAPE 2: `if <sign test>:` whose body accumulates with +=
            if isinstance(n, ast.If) and has_sign_test(n.test):
                for b in ast.walk(n):
                    if isinstance(b, ast.AugAssign) and isinstance(b.op,
                                                                   ast.Add):
                        tgt = getattr(b.target, "id", None) or getattr(
                            b.target, "attr", None) or ""
                        # a COUNTER increments by 1; a ceiling adds a VALUE
                        if not (isinstance(b.value, ast.Constant)
                                and b.value.value == 1):
                            hits.append({
                                "file": str(f), "line": b.lineno,
                                "shape": "SIGN_GUARDED_VALUE_ACCUMULATION",
                                "src": src[b.lineno - 1].strip()[:140]})
                        break
print(json.dumps({
    "as_of_utc": sys.argv[1] if len(sys.argv) > 1 else None,
    "surfaces": [str(r) for r in ROOTS],
    "n_python_files_searched": n_files,
    "n_functions_visited": n_funcs,
    "n_hits": len(hits),
    "hits": hits,
    "method": ("AST: a reduction (sum/fsum) whose comprehension is filtered "
               "on a comparison to zero or clipped with min/max(0, .), OR an "
               "`if <sign test>` guarding a `+=` of a VALUE (a += 1 counter "
               "is excluded, since counting negatives is not summing them)"),
}, indent=1))
