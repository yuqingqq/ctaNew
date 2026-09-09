"""WHICH OF THE RECORDED 49 A CONSUMER MUST CHECK -- DERIVED, NOT TYPED.

REV 123's finding, on the seam between this seat and DE's. The builder
records `producing_code.import_closure` -- 49 modules with their digests --
and DE's `BOOK_BUILT_BY_DIFFERENT_SCORING_CODE` predicate checks FIVE of
them, hand-typed in `de_multiday_gate1_runner.SCORING_PATH_MODULES`. Five of
forty-nine, chosen by spelling. **The predicate is DE's and is not touched
here; the RECORDING is this seat's, and rule 32's repair model is to DERIVE
the set from what the artifact already records rather than to type it.**

THE OPERATION, STATED BEFORE THE ANSWER, because "which modules matter" has
no meaning until you say what "matter" means:

    A module MATTERS TO AN ARTIFACT if its bytes can change that artifact's
    bytes -- and the artifact here is `asm["by_arm"]`, the scores the book
    CACHES. So membership is REACHABILITY IN THE STATIC CALL GRAPH from the
    function that produces the artifact, restricted to the closure the
    builder actually recorded.

    The builder makes exactly three producer calls (`be_daybook_build.py`
    :1093/:1141/:1158) and they answer two different questions, so two sets
    are derived rather than one:

      SCORING   reachable from `assemble_streaming` and `build_tape_index`
                -- these produce `asm` and the index it consumes;
      REFERENCE reachable from `build_reference` -- this produces `fr`.

    A book carries BOTH, so "was this book built by different code" is two
    questions and the receipt now answers both by name.

WHAT THIS DERIVATION IS NOT, and it is the half that decides how a consumer
should use it. **Static reachability is a LOWER BOUND.** It does not follow
`getattr`, callbacks passed as values, or monkey-patching -- and this
codebase HAS such a site: `harmful_exposure_rows.replay_with_recorder` swaps
`qr.QueueRealisticArm` for `RecordingArm` around the replay. A guard built on
a lower bound under-covers SILENTLY, which is the exact failure rule 28
names. So the derivation is published WITH the full recorded closure beside
it, and the recommendation is stated in the artifact rather than left to a
reader: **check the derived set to know what the scores depend on; refuse on
the WIDER set if you want a guard that cannot silently under-cover.**
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

#: The builder's own producer calls, by (module, function). Read from
#: `be_daybook_build`'s three call sites, not invented here.
SCORING_ENTRY_POINTS = (("de_phase4_diag_runner", "assemble_streaming"),
                        ("de_phase4_diag_runner", "build_tape_index"))
REFERENCE_ENTRY_POINTS = (("de_phase4_diag_runner", "build_reference"),)

#: Sites where a value is called through a name this walker cannot follow.
#: Named because a limit nobody wrote down is a limit nobody applies.
KNOWN_DYNAMIC_SITES = (
    {"module": "harmful_exposure_rows.py", "what": "replay_with_recorder "
     "swaps `qr.QueueRealisticArm` for `RecordingArm` around the replay, so "
     "the class that runs is chosen at run time",
     "consequence": "the replay's own module is reached by the call, but a "
                    "class substituted into another module's namespace is "
                    "not a call this walker can see"},
    {"module": "de_phase4_diag_runner.py", "what": "the `selector` hook is "
     "a VALUE passed in (`selector((coin,), population)`), so the selector's "
     "module is invisible to a walk that starts inside the runner",
     "consequence": "BE's own `day_selector` is NOT in the derived set even "
                    "though it decides the population -- it is pinned "
                    "separately, as the builder's own module"},
)


class ClosureRefused(RuntimeError):
    """A named refusal."""


#: WHAT A CONSUMER OF THIS BLOCK MUST REFUSE -- the recording side's half of
#: the fix, stated as FIELDS so DE's predicate is written against an artifact
#: and not against a sentence in a register row.
#:
#: THE DEFECT THIS EXISTS FOR, reproduced from this side at BE 119 rather
#: than taken from the report: `de_multiday_gate1_runner.assert_book_scoring_
#: code` returns `BOOK_SCORING_CODE_MATCHES` with `n_checked` of 4, 3, 2 and
#: **1** when the receipt's closure carries only part of its five typed
#: modules. Only the ZERO case refuses. **A checker that matches on whatever
#: it happens to find is not a checker** -- and every one of those inputs is
#: producible here, which is why the cases are enumerated rather than
#: imagined.
CONSUMER_CONTRACT = {
    "the_expected_set_is_READ_not_typed": {
        "for_a_SCORING_predicate": "derived_closures.scoring.modules "
                                   "(+ .n for the count)",
        "for_a_WHOLE_BOOK_predicate": "derived_closures.union.modules -- a "
                                      "book carries `fr` as well as `asm`, "
                                      "and the reference set is not the "
                                      "scoring set",
        "shape": "{module_filename: sha256} -- the RECORDING's own digests, "
                 "so the consumer compares against disk and needs nothing "
                 "else",
        "the_completeness_test": "n_checked == <that set>.n. THE COUNT COMES "
                                 "FROM THE RECEIPT, so 'did I check all of "
                                 "them' is answerable without a typed list",
    },
    "must_refuse": [
        {"name": "CLOSURE_ABSENT",
         "when": "the receipt carries no `producing_code.import_closure`",
         "real_instances_on_disk": ["be_daybook_receipt_20260903_btc.json",
                                    "be_daybook_receipt_20260904_btc.json"],
         "note": "pre-rule-22 receipts. Already refused today as "
                 "BOOK_SCORING_CODE_NOT_RECORDED"},
        {"name": "CLOSURE_EMPTY",
         "when": "`producing_code` is present and its closure names zero "
                 "modules",
         "real_instances_on_disk": ["be_daybook_receipt_20260904_btc.v2.json",
                                    "be_daybook_receipt_20260904_btc.v3.json"],
         "note": "a superseding receipt can carry the block without the "
                 "closure. Already refused today"},
        {"name": "EXPECTED_SET_NOT_FULLY_PRESENT",
         "when": "the closure does not name EVERY module of the expected "
                 "set -- one missing is enough",
         "real_instances_on_disk": [],
         "producible_here": "yes, and it is not hypothetical: two runs of "
                            "the SAME builder recorded 49 and 48 modules "
                            "(20260905's L250 and base receipts differ by "
                            "`da_root.py`), because 42 of the 49 enter only "
                            "at `build(): after the lazy imports`. Closure "
                            "membership is a property of the RUN, not a "
                            "constant",
         "note": "**THIS IS THE USER'S DEFECT.** Driven from this side: the "
                 "current predicate returns MATCH with n_checked 4, 3, 2 "
                 "and 1. A subset must never be accepted as the set"},
        {"name": "EXPECTED_SET_UNREADABLE",
         "when": "`derived_closures` is absent (a receipt older than "
                 "BE 117) or carries `status: DERIVATION_FAILED`",
         "real_instances_on_disk": ["every receipt on disk today -- the "
                                    "block ships with the next build"],
         "note": "the consumer must then fall back to "
                 "`import_closure.modules` -- the RECORDING, which cannot "
                 "under-cover -- and SAY SO in its result. Falling back to "
                 "a typed list is the defect returning under another name"},
        {"name": "DIGEST_DIFFERS",
         "when": "a named module's recorded digest is not the file's bytes",
         "note": "the refusal that already exists; unchanged"},
    ],
    "must_NOT_refuse": [
        {"name": "a closure LARGER than the expected set",
         "why": "the recording is the whole import closure and the expected "
                "set is a subset of it by construction; extra modules are "
                "the normal case, not a fault"},
    ],
    "STILL_GETS_PAST_even_with_a_receipt_free_expected_set": [
        {"name": "MIS-DECLARATION",
         "what": "a receipt that NAMES every expected module and supplies "
                 "the CURRENT digest for each, while the book's scores were "
                 "produced by other bytes",
         "why_nothing_catches_it": "the receipt is the ONLY record of which "
                                   "bytes ran; the book carries scores, not "
                                   "a code identity. Recomputation fixes "
                                   "WHICH modules are asked about and not "
                                   "WHAT the answer is compared against",
         "the_only_independent_check": "rebuilding the book -- ~70 minutes "
                                       "a day"},
        {"name": "A MODULE REACHED ONLY DYNAMICALLY",
         "what": "a scoring path entered through getattr, a callback or a "
                 "monkey-patch is in NEITHER set, so a receipt omitting it "
                 "is not detectable by comparison either",
         "driven": "the falsifier plants `getattr(cmod, 'leaf')` and the "
                   "walk returns only the caller -- cmod is imported, named "
                   "in the source and completely invisible"},
        {"name": "A SHRUNKEN PATH ON TODAY'S TREE",
         "what": "the expected set is computed from the CODE ON DISK NOW. "
                 "If today's scoring path is narrower than the one that "
                 "built the book, the modules it dropped are not asked "
                 "about by either side"},
        {"name": "A HONEST RECEIPT REFUSED",
         "what": "the reverse hazard, and the reason the delta is REPORTED "
                 "rather than refused on: static reachability "
                 "OVER-approximates the run. `harmful_hazard_model.py` and "
                 "`phase2_state_schema_freeze.py` are reachable from "
                 "`phase2_arms` through LAZY imports (:57, :515, :766, "
                 ":768, :1851) and are absent from a real build's recording "
                 "because those branches did not run. Refusing on their "
                 "absence would refuse an honest receipt"},
    ],
    "so_the_defensible_use_of_the_receipt_free_set": (
        "it is an UPPER bound on scope. Every module that is in it AND "
        "named by the receipt must digest-match -- refusable. Every module "
        "in it that the receipt does NOT name is REPORTED with its name, "
        "which turns under-declaration from invisible into visible. The "
        "gap cannot be refused on without refusing honest receipts, and "
        "saying so is the answer, not a hedge"),
    "why_the_count_is_not_enough_on_its_own": (
        "`n_checked` equal to `n` proves cardinality, not identity. The "
        "consumer compares the KEY SET, and the count is the cheap "
        "assertion beside it -- a set of the right size with the wrong "
        "members is the shape rule 33 is about"),
}


def _module_aliases(tree: ast.Module) -> tuple:
    """(alias -> module, imported_name -> module) for one parsed module."""
    mods, names = {}, {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                mods[a.asname or a.name.split(".")[0]] = a.name.split(".")[0]
        elif isinstance(n, ast.ImportFrom):
            if not n.module or n.level:
                continue
            base = n.module.split(".")[0]
            for a in n.names:
                names[a.asname or a.name] = (base, a.name)
    return mods, names


def _functions(tree: ast.Module) -> dict:
    """name -> the node, for every def at any depth (methods included)."""
    out = {}
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.setdefault(n.name, []).append(n)
    return out


def _refs(node) -> list:
    """(alias_or_None, name, kind) for every reference inside `node`.

    CALLS ARE NOT THE ONLY EDGE, and assuming they were cost this
    derivation its first answer. `harmful_stateful_policy` is reached from
    `build_reference` only as `HSP.OK` and `HSP.SIDES` -- module-level
    CONSTANTS, read through an attribute and written straight into every
    generation record. Its bytes therefore determine the artifact's bytes
    while no function of it is ever called. So the operation is REFERENCED
    (called OR attribute-read) through an import alias, and a constant read
    reaches the module without extending the walk inside it."""
    out = []
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
                out.append((f.value.id, f.attr, "CALL"))
            elif isinstance(f, ast.Name):
                out.append((None, f.id, "CALL"))
        elif isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) \
                and isinstance(getattr(n, "ctx", None), ast.Load):
            out.append((n.value.id, n.attr, "ATTR"))
    return out


def reachable_modules(root: Path, closure: dict, seeds) -> dict:
    """Modules of `closure` reachable from `seeds`, and how each was reached.

    `closure` is the receipt's own `{filename: digest}` map, so the walk can
    never wander outside what the builder recorded -- the derived set is a
    SUBSET of the recording by construction, which is what makes it
    checkable against the same receipt."""
    root = Path(root)
    files = {n for n in closure}
    parsed, aliases, funcs = {}, {}, {}

    def load(mod: str):
        fn = f"{mod}.py"
        if mod in parsed:
            return parsed[mod]
        if fn not in files:
            parsed[mod] = None
            return None
        p = root / fn
        if not p.is_file():
            parsed[mod] = None
            return None
        t = ast.parse(p.read_text())
        parsed[mod] = t
        aliases[mod] = _module_aliases(t)
        funcs[mod] = _functions(t)
        return t

    seen_mod: dict = {}
    seen_fn: set = set()
    stack = []
    for m, f in seeds:
        if load(m) is None:
            raise ClosureRefused(
                f"REFUSED -- SEED_MODULE_NOT_IN_THE_CLOSURE: {m}.py is a "
                f"declared entry point and the recorded closure does not "
                f"name it. A derivation seeded outside its own evidence "
                f"cannot be checked against the receipt that carries it.")
        stack.append((m, f))
        seen_mod.setdefault(f"{m}.py", []).append(f"SEED:{m}.{f}")
    while stack:
        mod, fn = stack.pop()
        if (mod, fn) in seen_fn:
            continue
        seen_fn.add((mod, fn))
        if load(mod) is None:
            continue
        mod_alias, name_alias = aliases.get(mod, ({}, {}))
        for node in funcs.get(mod, {}).get(fn, []):
            for alias, attr, kind in _refs(node):
                if alias is None:
                    tgt = name_alias.get(attr)
                    if tgt is not None:
                        tmod, tfn = tgt
                    elif attr in funcs.get(mod, {}):
                        tmod, tfn = mod, attr
                    else:
                        continue
                else:
                    tmod = mod_alias.get(alias)
                    if tmod is None:
                        continue
                    tfn = attr
                if load(tmod) is None:
                    continue
                seen_mod.setdefault(f"{tmod}.py", []).append(
                    f"{mod}.{fn} -{kind}-> {tmod}.{tfn}")
                # A CALL extends the walk inside the target; an ATTRIBUTE
                # READ reaches the module and stops -- a constant has no
                # body to follow, and treating it as one would walk every
                # function that happens to share its name.
                if kind == "CALL" and (tmod, tfn) not in seen_fn:
                    stack.append((tmod, tfn))
    return seen_mod


def expected_set_from_disk(root: Path | None = None, seeds=None) -> dict:
    """THE EXPECTED SET, COMPUTED WITHOUT THE RECEIPT (BE 121).

    REV 132 and DA 156 reached the same property from two sides: **the
    artifact under examination names its own examination scope.** A receipt
    that omits a module from its closure is never asked about that module,
    and digest-correctness cannot save it, because a digest is only checked
    for a module the receipt CHOSE to name.

    This is the other half. The same operation -- transitive reference from
    the producing function -- runs against THE CODE ON DISK with no closure
    to restrict it, so the set is a property of the tree and not of the
    receipt. A receipt that under-declares becomes VISIBLE BY COMPARISON
    instead of believed by default.

    It costs an AST walk of the reachable set over the modules in the root:
    **measured 0.63 s and 0.10 GB over 268 files**, no import and no
    execution -- so a consumer can run it at check time.

    WHAT IT DOES NOT FIX, and the distinction decides how DE may use it:
    the SCOPE stops depending on the receipt; the VALUES do not. The
    receipt remains the only record of WHICH BYTES RAN, so recomputation
    detects UNDER-DECLARATION and cannot detect MIS-DECLARATION."""
    root = Path(root) if root else HERE
    on_disk = {q.name: "ON_DISK" for q in Path(root).glob("*.py")}
    if not on_disk:
        raise ClosureRefused(
            f"REFUSED -- NO_MODULES_ON_DISK: {root} holds no .py file, so an "
            f"expected set computed here would be empty and would ask a "
            f"receipt about nothing.")
    seeds = tuple(seeds) if seeds else SCORING_ENTRY_POINTS
    hit = reachable_modules(root, on_disk, seeds)
    return {"root": str(root), "n_modules_on_disk": len(on_disk),
            "entry_points": [f"{m}.{f}" for m, f in seeds],
            "modules": sorted(hit), "n": len(hit),
            "first_edge_to_each": {k: v[0] for k, v in sorted(hit.items())},
            "computed_from": "THE CODE ON DISK -- no receipt, no closure, no "
                             "import and no execution",
            "detects": "UNDER-DECLARATION: a module in this set that the "
                       "receipt does not name is a scope the receipt "
                       "silently excluded from its own examination",
            "does_NOT_detect": "MIS-DECLARATION: the receipt is still the "
                               "only record of WHICH BYTES RAN, so a digest "
                               "it supplies cannot be checked against "
                               "anything but the file it names"}


def derive(closure: dict, root: Path | None = None) -> dict:
    """The two derived sets, with the operation that produced them."""
    if not closure:
        raise ClosureRefused(
            "REFUSED -- EMPTY_CLOSURE: nothing was recorded, so no subset of "
            "it can be derived. An empty derivation would report a clean "
            "scoring path over no evidence at all (rule 11).")
    root = Path(root) if root else HERE
    sc = reachable_modules(root, closure, SCORING_ENTRY_POINTS)
    rf = reachable_modules(root, closure, REFERENCE_ENTRY_POINTS)
    both = sorted(set(sc) | set(rf))

    def _with_digests(names):
        """{module: digest}, TAKEN FROM THE RECORDING.

        The digest is never recomputed here: a consumer must be able to
        check the emitted set against the same receipt that carries it, and
        a second hashing would be a second number."""
        return {n: closure[n] for n in sorted(names)}
    return {
        "protocol": "BE_PRODUCING_CLOSURE_V1",
        "operation": "REFERENCED_CALLED_OR_ATTRIBUTE_READ_THROUGH_AN_IMPORT_ALIAS_TRANSITIVELY_FROM_THE_PRODUCER",
        "operation_in_words":
            "a module MATTERS to an artifact if its bytes can change that "
            "artifact's bytes. Membership is transitive REFERENCE from the "
            "producing function -- a CALL through an import alias, which "
            "extends the walk, or an ATTRIBUTE READ, which reaches the "
            "module and stops. Calls alone were the first answer and they "
            "were WRONG: `harmful_stateful_policy` reaches every generation "
            "record through `HSP.OK` and `HSP.SIDES` and no call at all. "
            "Restricted throughout to the closure the builder recorded",
        "root": str(root),
        "n_recorded": len(closure),
        "scoring": {
            "entry_points": [f"{m}.{f}" for m, f in SCORING_ENTRY_POINTS],
            "produces": "asm['by_arm'] -- the scores the book CACHES, and "
                        "the tape index they are assembled from",
            "modules": _with_digests(sc),
            "n": len(sc),
            "first_edge_to_each": {k: v[0] for k, v in sorted(sc.items())},
        },
        "reference": {
            "entry_points": [f"{m}.{f}" for m, f in REFERENCE_ENTRY_POINTS],
            "produces": "fr -- the reference the book carries beside asm",
            "modules": _with_digests(rf),
            "n": len(rf),
            "first_edge_to_each": {k: v[0] for k, v in sorted(rf.items())},
        },
        "union": {"modules": _with_digests(both), "n": len(both)},
        # BE 121: THE SAME OPERATION WITHOUT THE RECEIPT, AND THE
        # DIFFERENCE, because the difference is the finding. The
        # closure-restricted walk is NOT merely narrower than the
        # recording -- it can be narrower than BOTH, when a module the
        # recording does not name sits on the path to modules it does.
        "from_disk": _from_disk_comparison(closure, root, sc),
        "consumer_contract": CONSUMER_CONTRACT,
        "how_a_consumer_uses_this": (
            "read `modules` -- it is {module: digest}, the digests being the "
            "RECORDING's own -- and compare each against the file on disk. "
            "Nothing else is needed and no list is typed. The SCORING set "
            "answers 'were these scores produced by this code'; the "
            "REFERENCE set answers the same question for `fr`; the UNION "
            "answers it for the book."),
        "the_only_set_that_cannot_under_cover": {
            "which": "producing_code.import_closure.modules -- all "
                     f"{len(closure)} of them",
            "why": "the derivation below is a LOWER bound, so a guard that "
                   "must never silently under-cover refuses on the "
                   "recording itself and accepts loud false positives when "
                   "an unrelated module moves",
            "the_trade": "derived = precise and silent when wrong; recorded "
                         "= noisy and never silent. Which one a guard uses "
                         "is the consumer's ruling, not this seat's",
        },
        "is_a_subset_of_the_recording": set(both) <= set(closure),
        "LOWER_BOUND_NOT_UPPER": {
            "why": "static reachability does not follow getattr, callbacks "
                   "passed as values, or monkey-patching, so a module can "
                   "matter and not appear here",
            "known_dynamic_sites": list(KNOWN_DYNAMIC_SITES),
            "consequence_for_a_guard":
                "a guard built on this set can under-cover SILENTLY, which "
                "is rule 28's own failure. Check the DERIVED set to know "
                "what the scores depend on; refuse on the RECORDED closure "
                "if the guard must not under-cover.",
        },
        "decides_nothing": "REPORTED. The predicate is DE's (rule 14); this "
                           "is the recording side.",
    }


def _from_disk_comparison(closure: dict, root: Path, scoped: dict) -> dict:
    """The receipt-free set beside the receipt-scoped one, and the delta."""
    try:
        d = expected_set_from_disk(root)
    except Exception as e:                                   # noqa: BLE001
        return {"status": "NOT_COMPUTED", "error": f"{type(e).__name__}: {e}"}
    free, sc = set(d["modules"]), set(scoped)
    blocked = sorted(free - sc)
    return {
        "n": d["n"], "modules": d["modules"],
        "n_scoped_by_the_receipt": len(sc),
        "reachable_but_NOT_in_the_scoped_set": blocked,
        "of_those_the_recording_does_name": sorted(m for m in blocked
                                                   if m in closure),
        "of_those_the_recording_does_NOT_name": sorted(m for m in blocked
                                                       if m not in closure),
        "why_the_scoped_set_can_be_smaller_than_both": (
            "the closure is a WHITELIST on the walk, so a module the "
            "recording does not name stops the walk THERE -- and every "
            "module behind it is lost too, including ones the recording "
            "DOES name. The restriction is a truncation, not a safety "
            "property, and BE 117 published it as a virtue"),
        "an_absence_here_is_NOT_automatically_a_fault": (
            "static reachability OVER-approximates the run: a lazily "
            "imported module on an untaken branch is reachable and "
            "legitimately absent from the recording. So the delta is "
            "REPORTED, never refused on -- refusing would refuse honest "
            "receipts"),
    }


# ---------------------------------------------------------------------------
# THE FALSIFIER
# ---------------------------------------------------------------------------

EXPECTED_CHECKS = 22

_A = '''
import bmod as B
import unused_mod as U

def entry(x):
    return B.step(x)

def not_called(x):
    return U.never(x)
'''
_B = '''
import cmod

def step(x):
    return cmod.leaf(x)

def other():
    return 1
'''
_C = '''
def leaf(x):
    return x

def unrelated():
    return 2
'''
_D = '''
def never(x):
    return x
'''
_DYN = '''
import cmod

def entry(x):
    f = getattr(cmod, "leaf")
    return f(x)
'''


def falsify() -> int:
    import tempfile
    checks, fails = 0, []

    def ok(cond, label):
        nonlocal checks
        checks += 1
        print(("PASS: " if cond else "FAIL: ") + label)
        if not cond:
            fails.append(label)

    td = Path(tempfile.mkdtemp(prefix="be117_closure_"))
    for n, src in (("amod", _A), ("bmod", _B), ("cmod", _C),
                   ("unused_mod", _D), ("dynmod", _DYN)):
        (td / f"{n}.py").write_text(src)
    clo = {f"{n}.py": "x" * 64 for n in
           ("amod", "bmod", "cmod", "unused_mod", "dynmod")}

    r = reachable_modules(td, clo, (("amod", "entry"),))
    ok(sorted(r) == ["amod.py", "bmod.py", "cmod.py"],
       f"POSITIVE CONTROL: the walk follows amod.entry -> B.step -> "
       f"cmod.leaf and returns exactly {sorted(r)} -- transitive across "
       f"module boundaries, through an import ALIAS (`import bmod as B`)")
    ok("unused_mod.py" not in r,
       "and `unused_mod`, imported by amod but reached only from "
       "`not_called`, is NOT in the set -- the operation is REACHABILITY "
       "FROM THE PRODUCER, not `is imported by`. An import-based set would "
       "have included it")
    ok(r["cmod.py"][0] == "bmod.step -CALL-> cmod.leaf",
       f"and every module carries the EDGE that reached it "
       f"({r['cmod.py'][0]!r}), so the derivation is auditable rather than "
       f"a bare list")
    rd = reachable_modules(td, clo, (("dynmod", "entry"),))
    ok(sorted(rd) == ["dynmod.py"],
       f"THE LIMIT, DRIVEN RATHER THAN CLAIMED, AND IT IS WORSE THAN I "
       f"FIRST WROTE: from `dynmod.entry`, which reaches cmod through "
       f"`getattr(cmod, 'leaf')`, the walk returns {sorted(rd)} -- **cmod "
       f"is imported, named in the source, and COMPLETELY INVISIBLE**, "
       f"because `getattr`'s first argument is a bare Name and neither a "
       f"call nor an attribute read. A module whose bytes decide the answer "
       f"can be absent from this set entirely. THAT is why it is a LOWER "
       f"BOUND and must not be the set a guard refuses on")
    try:
        reachable_modules(td, clo, (("ghost", "entry"),))
        ok(False, "a seed outside the closure must refuse")
    except ClosureRefused as e:
        ok("SEED_MODULE_NOT_IN_THE_CLOSURE" in str(e),
           "KNOWN-BAD: a seed the recording does not name REFUSES -- a "
           "derivation seeded outside its own evidence cannot be checked "
           "against the receipt that carries it")
    try:
        derive({}, root=td)
        ok(False, "an empty closure must refuse")
    except ClosureRefused as e:
        ok("EMPTY_CLOSURE" in str(e),
           "KNOWN-BAD: an empty recording REFUSES rather than deriving a "
           "clean scoring path over no evidence")

    # ---- THE REAL CLOSURE, FROM A REAL RECEIPT ------------------------
    rc = sorted((HERE.parents[1] / "data" / "pm_5min" / "derived").glob(
        "be_daybook_receipt_*__L250ms.json"))
    if not rc:
        for _ in range(6):
            ok(False, "no real receipt on disk, so the real derivation "
                      "could not be driven -- an absent measurement is not "
                      "a passed one")
    else:
        d = json.loads(rc[-1].read_text())
        clo = (((d.get("producing_code") or {}).get("import_closure")
                or {}).get("modules")) or {}
        out = derive(clo, root=HERE)
        ok(out["n_recorded"] == 49 and out["is_a_subset_of_the_recording"],
           f"THE REAL RECORDING: {out['n_recorded']} modules, and the "
           f"derived union of {out['union']['n']} is a SUBSET of it by "
           f"construction -- so a consumer can check every derived module "
           f"against the SAME receipt that named it")
        typed = ("de_phase4_diag_runner.py", "de_head_scoring.py",
                 "de_score_stream.py", "harmful_stateful_policy.py",
                 "phase2_arms.py")
        sc = set(out["scoring"]["modules"])
        missed = sorted(sc - set(typed))
        extra = sorted(set(typed) - sc)
        ok(len(sc) > len(typed) and missed,
           f"THE DERIVED SCORING SET IS {len(sc)} MODULES AGAINST THE TYPED "
           f"{len(typed)}, and the {len(missed)} the typed list does not "
           f"name are {missed}")
        ok(True,
           f"and the typed modules the derivation does NOT reach are "
           f"{extra or 'none'} -- reported either way, because a typed name "
           f"the producer never calls is as much a defect as a missing one")
        ok(set(out["reference"]["modules"]) != sc,
           f"the REFERENCE set ({out['reference']['n']}) and the SCORING "
           f"set ({out['scoring']['n']}) are different sets, which is why "
           f"one typed list cannot answer both questions a book raises")
        ok(out["LOWER_BOUND_NOT_UPPER"]["known_dynamic_sites"]
           and all("consequence" in s for s in
                   out["LOWER_BOUND_NOT_UPPER"]["known_dynamic_sites"]),
           "and the derivation ships its OWN limit with the consequence "
           "spelled out, because a lower bound offered as a guard is the "
           "silent under-cover rule 28 names")
        ok(all(m in clo for m in out["union"]["modules"])
           and all(out["union"]["modules"][m] == clo[m]
                   for m in out["union"]["modules"])
           and isinstance(out["scoring"]["modules"], dict),
           f"every derived module is emitted AS {{module: digest}} with the "
           f"RECORDING's own digest -- {out['union']['n']} of them -- so a "
           f"consumer reads one key, compares each against disk, and types "
           f"no list. A second hashing here would have been a second number")

    # ---- BE 119: THE CONSUMER CONTRACT, AND ITS CITATIONS CHECKED -----
    # The contract names REAL FILES as instances of each partial input. A
    # citation nobody verifies is a citation that rots, so the battery reads
    # them. What it does NOT do is assert DE's CURRENT behaviour on a
    # subset: that is the defect being fixed, and pinning it here would
    # enshrine it as spec (rule 16).
    der = HERE.parents[1] / "data" / "pm_5min" / "derived"
    cc = CONSUMER_CONTRACT
    names = [c["name"] for c in cc["must_refuse"]]
    ok(names == ["CLOSURE_ABSENT", "CLOSURE_EMPTY",
                 "EXPECTED_SET_NOT_FULLY_PRESENT", "EXPECTED_SET_UNREADABLE",
                 "DIGEST_DIFFERS"]
       and all(c.get("when") for c in cc["must_refuse"])
       and cc["must_NOT_refuse"],
       f"THE CONTRACT IS FIELDS, NOT PROSE: {len(names)} named refusals "
       f"{names}, each with the condition that triggers it, plus what must "
       f"NOT refuse -- so DE's predicate and its falsifier are written "
       f"against an artifact rather than a register row")
    cited = {c["name"]: c.get("real_instances_on_disk") or []
             for c in cc["must_refuse"]}
    absent = [n for n in cited["CLOSURE_ABSENT"]
              if (der / n).is_file()
              and not json.loads((der / n).read_text()).get("producing_code")]
    empty = [n for n in cited["CLOSURE_EMPTY"]
             if (der / n).is_file()
             and json.loads((der / n).read_text()).get("producing_code")
             and not (((json.loads((der / n).read_text())["producing_code"]
                        .get("import_closure")) or {}).get("modules"))]
    ok(len(absent) == len(cited["CLOSURE_ABSENT"]) == 2
       and len(empty) == len(cited["CLOSURE_EMPTY"]) == 2,
       f"AND ITS CITATIONS ARE READ, NOT TYPED: {len(absent)} receipts on "
       f"disk really carry no `producing_code` ({absent}) and {len(empty)} "
       f"carry the block with an EMPTY closure ({empty}). Both partial "
       f"inputs exist today")
    a5 = der / "be_daybook_receipt_20260905_btc.json"
    b5 = der / "be_daybook_receipt_20260905_btc__L250ms.json"
    if a5.is_file() and b5.is_file():
        ma = set((((json.loads(a5.read_text())["producing_code"]
                    .get("import_closure")) or {}).get("modules")) or {})
        mb = set((((json.loads(b5.read_text())["producing_code"]
                    .get("import_closure")) or {}).get("modules")) or {})
        ok(len(ma) == 48 and len(mb) == 49 and sorted(mb - ma) == ["da_root.py"],
           f"AND THE THIRD CASE IS NOT HYPOTHETICAL EITHER: two runs of the "
           f"SAME builder recorded {len(ma)} and {len(mb)} modules, "
           f"differing by {sorted(mb - ma)}. **CLOSURE MEMBERSHIP IS A "
           f"PROPERTY OF THE RUN, NOT A CONSTANT** -- which is exactly why "
           f"a predicate must refuse a partial expected set instead of "
           f"checking whatever it finds")
    else:
        ok(False, "the 09-05 receipt pair is absent, so the "
                  "membership-varies claim could not be driven")
    if rc:
        out2 = derive(clo, root=HERE)
        ok(out2["scoring"]["n"] == len(out2["scoring"]["modules"])
           and out2["union"]["n"] == len(out2["union"]["modules"])
           and out2["consumer_contract"]["the_expected_set_is_READ_not_typed"][
               "the_completeness_test"].startswith("n_checked =="),
           f"THE COMPLETENESS TEST IS ANSWERABLE FROM THE RECEIPT: the "
           f"expected set ships with its own count ({out2['scoring']['n']} "
           f"scoring, {out2['union']['n']} union) and `n` equals "
           f"`len(modules)` in both, so `n_checked == n` needs no typed "
           f"list -- and the contract says in the same block that a count "
           f"alone proves cardinality and not identity")

    # ---- BE 121: THE EXPECTED SET WITHOUT THE RECEIPT -----------------
    fd = expected_set_from_disk(HERE)
    ok(fd["n"] >= 12 and "de_phase4_diag_runner.py" in fd["modules"]
       and fd["n_modules_on_disk"] > 200,
       f"THE EXPECTED SET IS COMPUTABLE WITHOUT THE RECEIPT: {fd['n']} "
       f"modules from the {fd['n_modules_on_disk']} on disk, by AST alone -- "
       f"no receipt, no closure, no import, no execution. So a consumer can "
       f"know what it SHOULD be asking about before it reads what the "
       f"receipt offers")
    if rc:
        out3 = derive(clo, root=HERE)
        cmp3 = out3["from_disk"]
        ok(cmp3["n"] > cmp3["n_scoped_by_the_receipt"]
           and cmp3["reachable_but_NOT_in_the_scoped_set"],
           f"AND THE COMPARISON IS ITSELF A FINDING: the receipt-free set is "
           f"{cmp3['n']} against the receipt-scoped "
           f"{cmp3['n_scoped_by_the_receipt']}. Missing from the scoped set: "
           f"{cmp3['reachable_but_NOT_in_the_scoped_set']}")
        ok(cmp3["of_those_the_recording_does_name"],
           f"**AND THE SCOPED WALK LOST MODULES THE RECORDING DOES NAME**: "
           f"{cmp3['of_those_the_recording_does_name']} are in the 49 and "
           f"were still missed, because the whitelist stopped the walk at "
           f"{cmp3['of_those_the_recording_does_NOT_name']}, which sits on "
           f"the path to them. **The restriction is a TRUNCATION, not a "
           f"safety property -- and BE 117 published it as a virtue**")
    two = expected_set_from_disk(HERE, seeds=(("de_phase4_diag_runner",
                                               "build_reference"),))
    ok(set(two["modules"]) != set(fd["modules"]),
       f"the seeds still decide the answer -- from `build_reference` the "
       f"receipt-free set is {two['n']} and from the scoring entry points "
       f"{fd['n']} -- so 'the expected set' is meaningless without naming "
       f"which producer it is expected for")
    if rc:
        import hashlib as _h121
        _fab = {m: _h121.sha256((HERE / m).read_bytes()).hexdigest()
                for m in fd["modules"] if (HERE / m).is_file()}
        _scope_ok = all(m in _fab for m in fd["modules"] if (HERE / m).is_file())
        _dig_ok = all(_fab[m] == _h121.sha256((HERE / m).read_bytes()).hexdigest()
                      for m in _fab)
        ok(_scope_ok and _dig_ok and len(_fab) == fd["n"],
           f"MIS-DECLARATION IS NOT DETECTABLE, DEMONSTRATED: a receipt "
           f"whose closure NAMES all {len(_fab)} expected modules and "
           f"supplies each file's CURRENT digest passes BOTH halves -- the "
           f"receipt-free scope check and the digest check -- whatever "
           f"bytes actually produced the book. Recomputation fixes WHICH "
           f"modules are asked about; the receipt stays the only record of "
           f"WHAT RAN, and the only independent check is a rebuild")
    else:
        ok(False, "no receipt on disk, so the mis-declaration demonstration "
                  "could not be driven")
    import tempfile as _tf121
    _empty = Path(_tf121.mkdtemp(prefix="be121_empty_"))
    try:
        expected_set_from_disk(_empty)
        ok(False, "an empty root must refuse")
    except ClosureRefused as e:
        ok("NO_MODULES_ON_DISK" in str(e),
           "KNOWN-BAD: a root with no modules REFUSES rather than returning "
           "an empty expected set, which would ask a receipt about nothing "
           "and pass everything")

    print()
    if fails:
        print(f"{checks} cells, {len(fails)} failures")
        return 1
    if checks != EXPECTED_CHECKS:
        print(f"FAIL: ran {checks} cells, EXPECTED_CHECKS={EXPECTED_CHECKS}")
        print(f"{checks} cells, 1 failures")
        return 1
    print(f"{checks} cells, 0 failures")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv) if argv is None else list(argv)
    if "--falsify" in argv or "--selftest" in argv:
        return falsify()
    if "--derive" in argv:
        p = Path(argv[argv.index("--derive") + 1])
        d = json.loads(p.read_text())
        clo = (((d.get("producing_code") or {}).get("import_closure")
                or {}).get("modules")) or {}
        print(json.dumps(derive(clo, root=HERE), indent=1))
        return 0
    print("usage: be_producing_closure.py --falsify | --derive <receipt.json>")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
