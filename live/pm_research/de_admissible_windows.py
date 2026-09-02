"""DE-Admissible-Windows -- the explicit window list the EV-Replay seam takes.

SURFACE AUTHORISATION (R-126, in-file): coordinator DE round-5 dispatch;
EV_REPLAY_PLAN.md section 2 ("the env takes an EXPLICIT window list and stamps
it, never chooses"); USER ruling R-409 (a day with a blackout accrues on its
non-blackout complement; the blackout windows are masked as accounted loss);
R-410/R-411/R-412 (a mask is CONSUMED when present for any day and REQUIRED
from the governed day; the PRODUCER's committed artifact is the contract).
RESEARCH-ONLY, OFFLINE: no venue port, no order path.

WHAT IT DOES, AND THE ONE THING IT MUST NOT DO.  Given a UTC day, a calendar
of PRESENT windows per coin, and DA's blackout mask, it emits
`PRESENT - masked` as ReplayWindowSpec-shaped records.  It SUPPLIES and
STAMPS.  It does not SELECT: window admission is an R-ADMISS decision the
coordinator ratifies, and what this module produces is the thing that gets
ratified, not the ratification.  So it reads NO day verdict, computes NO
eligibility, and its emission carries a closed schema that REFUSES any
decision-shaped field (rule 14: models estimate, policy decides).

THE PRODUCER'S COMMITTED ARTIFACT IS THE CONTRACT (R-412 ruling 1).  The
envelope asserted here is DA's as committed -- `artifact ==
"da_blackout_mask_v1"`, `coins`, per-coin `masked_windows` / `n_masked` /
`n_windows_total`, top-level `day_closed_calendar` -- and NOT a paraphrase of
it.  RR8-1 is the reason that sentence is in this docstring: BE's adapter
asserted `protocol`/`per_coin` against DA's `artifact`/`coins`, both suites
green, and only a check that loaded the REAL committed file could fail.  The
seam test below therefore loads the real 09-01 artifact, and takes its
empty-mask control from DA's OWN producer rather than from a hand-built
envelope -- a hand-built envelope is precisely what drifted.

TWO READINGS OF ONE DISPATCH CLAUSE, BOTH IMPLEMENTED.  "a supplied window
that the mask masks (contradiction -> refuse)" can mean the mask masks a
window the caller did not supply, or the emitted list still contains a masked
window.  The first would otherwise be a silent drop and the second a broken
subtraction, so BOTH are guards here and neither reading is guessed away.
The reading that would make masked-windows-in-PRESENT refuse is the one
reading NOT taken, because it would make the subtraction unimplementable --
recorded so the choice is visible and correctable.

WHAT THE IMPORT PREDICATE CAN AND CANNOT SEE (declared, not discovered).
`reads_no_verdict` answers only about shapes it can read.  It RESOLVES
`import X`, `from X import ...`, `importlib.import_module('X')` and
`__import__('X')`, taking the first AND last segment of a dotted name.  It
REFUSES -- never answers True -- on a non-literal argument, on any
`exec`/`eval`/`compile`, and on a bare `__import__` reference that is
rebound rather than called.  And it is DECLARED BLIND to the shapes named in
`DECLARED_BLIND_SHAPES`: they are listed there with the reason each is not
refused, because a limit that is stated is a limit and one that is
discovered is a finding.

    python3 live/pm_research/de_admissible_windows.py --selftest
    python3 live/pm_research/de_admissible_windows.py supply --day 20260901
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

PROTOCOL = "de_admissible_windows_v1"
ROOT = Path(__file__).resolve().parents[2]

# CO-2: LAUNCH-INVARIANCE. `_grid` did a bare `import da_blackout_mask` and
# died with ModuleNotFoundError under `python3 -m live.pm_research.…` while
# the script-dir launch was green -- a suite that passes only because of how
# it was started. A crash is not a refusal (my own words last round), and it
# is the same class as BE's CO-1. DA's modules already do this; this module
# now does too, and the filing records BOTH launch rc's.
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ---- CITED CONSTANTS.  Every one is read from a ruled text or the committed
# artifact; none is chosen here, and a new one would be an escalation.
MASK_ARTIFACT_KIND = "da_blackout_mask_v1"      # the committed envelope
MASK_DIR = ROOT / "data/pm_5min/derived"
MASK_STEM = "da_blackout_mask_{day}.json"       # R-412: the path already agrees
SLUG_FORM = "{coin}-updown-5m-{start}"          # the repo's slug, not a new one

# CO-3: THE GOVERNING DAY IS NOT RESTATED HERE.
# It was `MASK_GOVERNED_FROM_DAY = "20260902"` -- a literal that could drift
# from the frozen rule the scorer binds to, so a USER amendment of the rule's
# effective day would have left this supplier governing a different calendar
# than the consumer of its output. The MODULE is imported and the attribute
# is read AT CALL TIME rather than copied at import: copying would bind a
# snapshot, and an amendment applied to the rule after import would not reach
# here. One source, read late.
def _load_rule():
    """The frozen rule module, or the REASON it could not be read."""
    try:
        import da_content_liveness_rule as _r
        return _r, None
    except Exception as e1:                          # noqa: BLE001
        try:
            from . import da_content_liveness_rule as _r
            return _r, None
        except Exception as e2:                      # noqa: BLE001
            return None, (f"bare import: {type(e1).__name__}: {e1}; "
                          f"package-relative: {type(e2).__name__}: {e2}")


RULE_MODULE, RULE_IMPORT_ERROR = _load_rule()

REQUIRED_MASK_TOP = ("artifact", "day", "coins", "day_closed_calendar",
                     "as_of_utc")
REQUIRED_COIN_FIELDS = ("masked_windows", "n_masked", "n_windows_total")

# A decision-shaped field must never appear in what this module emits.
DECISION_VOCAB = ("verdict", "eligible", "race_accrual_eligible", "admissible",
                  "accrues", "day_quality_pass", "decision_eligible", "pass",
                  "all_pass", "gate", "promote", "winner")

SPEC_KEYS = ("slug", "coin", "start", "inputs_hash")


# Modules that produce or read a DAY VERDICT.  Importing one here would make
# this supplier a decider; the check below is over the module's PARSED import
# list rather than over its text, because a source grep can see a deleted
# line and cannot see a value that arrived another way.
VERDICT_MODULES = ("da_forward_day_verify", "da_dayverdict",
                   "harmful_forward_scorer")


#: The dynamic-import call shapes this reader resolves.
DYNAMIC_IMPORT_CALLS = ("import_module", "__import__")
_UNRESOLVED = "<non-literal>"

#: DE11-R1: shapes that IMPORT WITHOUT AN IMPORT NODE. `exec('import X')`,
#: `eval("__import__('X')")` and a REBOUND `__import__` (`f = __import__;
#: f('X')`) each parsed to an EMPTY set, so `reads_no_verdict` answered True
#: -- blind AND passing, which is the worst pair. They are treated exactly
#: as a non-literal argument already was: their PRESENCE makes the source
#: UNRESOLVABLE, because what they import cannot be read from the tree.
OPAQUE_EXEC_CALLS = ("exec", "eval", "compile")
_OPAQUE = "<opaque-exec>"
_REBOUND = "<rebound-import>"

#: THE DECLARED LIMIT (a limit that is stated is a limit; one that is
#: discovered is a finding). Per shape, REFUSED or DECLARED-BLIND:
#:
#:   REFUSED (in the sets above)
#:     exec / eval / compile ......... any use at all, argument unread
#:     a bare `__import__` reference not called with a literal
#:     importlib.import_module / __import__ with a non-literal argument
#:
#:   DECLARED BLIND -- seen, named, and NOT refused, with the reason
#:     runpy.run_module / run_path ... imports by executing a module; adding
#:       it would refuse any file that legitimately runs another, and this
#:       module's consumers do not. NAMED so its absence is a decision.
#:     getattr(importlib, "import_module") .... reached through a call this
#:       reader does not resolve; catching it needs name resolution, which is
#:       a different instrument, not a bigger regex.
#:
#:     NOT BLIND, and the list said otherwise until the expected-blind
#:     assertions were written: `builtins.__import__('x')` IS CAUGHT. The
#:     dynamic-import matcher keys on the ATTRIBUTE NAME, so the attribute
#:     form of `__import__` resolves its literal exactly as the bare form
#:     does. The declared limit claimed a blindness the code did not have --
#:     which is the recommendation earning its keep on its first run.
#:     a module imported by a C extension or an import hook .... outside the
#:       source entirely.
#:
#: The blind list is a STATEMENT about what this predicate does not see. It
#: is checked below only in the sense that the module asserts the list is
#: non-empty and named -- there is no way to test for a shape one cannot see,
#: and pretending otherwise would be the control that cannot fail.
DECLARED_BLIND_SHAPES = (
    "runpy.run_module / runpy.run_path",
    "builtins.exec / builtins.eval / builtins.compile (attribute form: "
    "matching the bare name is what keeps `re.compile` from reading as an "
    "opaque exec)",
    'getattr(importlib, "import_module")(...)',
    "C extensions and import hooks (outside the source)",
)


def imported_modules(src: str) -> set[str]:
    """Every module a source file imports -- STATIC and DYNAMIC.

    A dynamic import with a non-literal argument is reported as the sentinel
    `<non-literal>` rather than dropped: the caller decides what to do about
    an import it cannot name, and dropping it silently is what made the
    predicate answerable at all."""
    import ast

    def _segs(dotted: str) -> set[str]:
        """FIRST and LAST segment of a dotted name.

        First alone was another hole in the same predicate: a dynamic (or
        static) `live.pm_research.da_forward_day_verify` contributes only
        `live`, which is in no vocabulary, so the package spelling of a
        verdict producer would have passed exactly as the dynamic call did.
        Widening the set is the safe direction for a reads-NOTHING predicate.
        """
        parts = dotted.split(".")
        return {parts[0], parts[-1]}

    tree = ast.parse(src)
    for _n in ast.walk(tree):                 # tag call functions first
        if isinstance(_n, ast.Call) and isinstance(_n.func, ast.Name):
            _n.func._is_call_func = True
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                out |= _segs(a.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            out |= _segs(node.module)
        elif isinstance(node, ast.Call):
            fn = node.func
            name = (fn.attr if isinstance(fn, ast.Attribute)
                    else fn.id if isinstance(fn, ast.Name) else None)
            if (name in OPAQUE_EXEC_CALLS
                    and isinstance(fn, ast.Name)):
                # DE11-R1: whatever this imports is inside a STRING this
                # reader does not execute.
                #
                # BARE NAMES ONLY, and the first version was not: matching on
                # the attribute name made `re.compile(...)` an opaque exec,
                # so the seam -- which compiles one regex -- refused itself.
                # Its own suite caught it. The builtins are called as bare
                # names; an ATTRIBUTE form (`builtins.exec`, `x.eval`) is a
                # different object this reader cannot resolve, and is
                # DECLARED BLIND rather than guessed at.
                out.add(_OPAQUE)
                continue
            if name not in DYNAMIC_IMPORT_CALLS or not node.args:
                continue
            arg = node.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                out |= _segs(arg.value)
            else:
                out.add(_UNRESOLVED)
        elif isinstance(node, ast.Name) and node.id == "__import__":
            # A BARE `__import__` REFERENCE that is not the function of a
            # call: it is being rebound, passed or stored, and the import it
            # eventually performs is not visible here. The call form is
            # handled above and does not reach this branch.
            parent_calls = getattr(node, "_is_call_func", False)
            if not parent_calls:
                out.add(_REBOUND)
    return out


def reads_no_verdict(imports: Iterable[str]) -> bool:
    """REFUSES on an unresolvable import instead of answering True.

    The old form returned a bool over whatever the reader happened to see,
    so an import it could not see read as an import that was not there."""
    imports = set(imports)
    opaque = {
        _UNRESOLVED: "a dynamic import whose argument is not a literal",
        _OPAQUE: f"a call to one of {list(OPAQUE_EXEC_CALLS)}, whose "
                 f"argument is a STRING this reader does not execute",
        _REBOUND: "a bare `__import__` reference that is rebound, passed or "
                  "stored rather than called with a literal",
    }
    hit = [why for tok, why in opaque.items() if tok in imports]
    if hit:
        raise ImportsUnresolvable(
            f"REFUSED: this source contains {'; '.join(hit)}, so what it "
            f"imports cannot be read. 'Cannot tell what this imports' is not "
            f"'imports nothing', and answering True here is what DE-R2 found "
            f"for the non-literal shape and DE11-R1 for the rest. Shapes "
            f"this predicate is DECLARED BLIND to, and does not refuse, are "
            f"named in DECLARED_BLIND_SHAPES.")
    return not (imports & set(VERDICT_MODULES))


def restated_day_literals(src: str, day: str) -> list[str]:
    """Module-level constants whose value RESTATES the governing day.

    Over the parsed AST, not the text: a grep would count the comment that
    EXPLAINS why the literal was removed, and would miss the same value
    assigned under a different name -- which is the drift CO-3 is about. The
    check that first replaced the literal was a grep and failed on its own
    explanation, which is the F-1 shape appearing in the batch that removed
    one."""
    import ast
    out: list[str] = []
    for node in ast.parse(src).body:
        if not isinstance(node, ast.Assign):
            continue
        if not (isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
                and node.value.value == day):
            continue
        out += [t.id for t in node.targets if isinstance(t, ast.Name)]
    return sorted(out)


class AdmissibleWindowsRefused(RuntimeError):
    """An input this module will not guess at, or a contradiction between two
    inputs.  Refusal is the product; a silent drop is not (rule 4)."""


class ImportsUnresolvable(AdmissibleWindowsRefused):
    """A module's import list contains something this reader cannot resolve.

    DE-R2: `imported_modules` saw only `import` and `from ... import`, so
    `importlib.import_module('da_forward_day_verify')` produced an EMPTY set
    and `reads_no_verdict` returned True -- a dynamic import defeated the
    predicate entirely. Literal dynamic imports are now resolved; a
    NON-LITERAL argument is UNRESOLVABLE and raises, because *cannot tell
    what this imports* is not *imports nothing*, and the second is what the
    old return value said."""


class GoverningRuleUnreadable(AdmissibleWindowsRefused):
    """The frozen rule's effective day could not be read.

    A SUBCLASS of the module's refusal, deliberately: the cause is specific
    and the disposition is not -- callers catch one concept.  And it REFUSES
    rather than defaulting, because a supplier that cannot tell a governed
    day from a pre-governed one would turn the mask REQUIREMENT into
    PERMISSION, which is CO-1's hole in this seat."""


def patch_target(spelling: str = "da_content_liveness_rule"):
    """The module object a CONTROL may patch -- and a REFUSAL if the named
    spelling is not the object this code actually bound.

    RR11-1. `da_content_liveness_rule` and
    `live.pm_research.da_content_liveness_rule` are two DISTINCT module
    objects for the same file with the same value, so patching one does not
    reach the other. In production that cannot matter (both read the same
    frozen file at import and nothing writes at runtime). In a CONTROL it
    matters completely: a test that patched the other spelling would change
    nothing, watch nothing move, and PASS -- testing nothing, loudly claiming
    otherwise. Reproduced before this function existed: patching the package
    object left `is_governed('20260901')` False.

    So a control asks for its target here instead of naming one, and gets a
    refusal if the object it would patch is not the bound one."""
    mod = sys.modules.get(spelling)
    if mod is None:
        raise AdmissibleWindowsRefused(
            f"no module {spelling!r} is loaded; a control cannot patch what "
            f"is not there")
    if RULE_MODULE is None:
        raise GoverningRuleUnreadable(
            f"this module bound no rule ({RULE_IMPORT_ERROR}); there is "
            f"nothing for a control to patch")
    if mod is not RULE_MODULE:
        raise AdmissibleWindowsRefused(
            f"REFUSED: patching {spelling!r} would NOT reach the object this "
            f"module bound ({RULE_MODULE.__name__!r}). They are distinct "
            f"module objects for the same file, so the control would change "
            f"nothing and pass vacuously (RR11-1).")
    return mod


def patch_targets(src: str) -> list[str]:
    """Module-constant patch targets in a source file: assignments of the
    form `<Name>.<UPPER_CASE> = ...`.

    Used to SWEEP the surface for the RR11-1 shape rather than to trust that
    only one module has it -- over the parsed AST, so a mention in a comment
    or a docstring is not a hit."""
    import ast
    out: list[str] = []
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Assign):
            continue
        for t in node.targets:
            if (isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
                    and t.attr.isupper()):
                out.append(f"{t.value.id}.{t.attr}")
    return sorted(set(out))


def governing_day() -> str:
    """The first GOVERNED day, read from the frozen rule at CALL TIME."""
    if RULE_MODULE is None:
        raise GoverningRuleUnreadable(
            f"REFUSED: the governing day could not be read from "
            f"`da_content_liveness_rule.EFFECTIVE_FROM_DAY` "
            f"({RULE_IMPORT_ERROR}). Defaulting to 'not governed' would make "
            f"a mask permitted-absent on every day, turning the requirement "
            f"into permission.")
    day = getattr(RULE_MODULE, "EFFECTIVE_FROM_DAY", None)
    if not day:
        raise GoverningRuleUnreadable(
            f"REFUSED: `da_content_liveness_rule` carries no usable "
            f"EFFECTIVE_FROM_DAY (got {day!r})")
    return str(day)


# ---------------------------------------------------------------------------
# 1. the mask -- loaded and validated against the COMMITTED envelope
# ---------------------------------------------------------------------------

def mask_path(day: str) -> Path:
    return MASK_DIR / MASK_STEM.format(day=day)


def is_governed(day: str) -> bool:
    """A day is GOVERNED from the frozen rule's EFFECTIVE_FROM_DAY onward.
    Governance decides whether a mask is REQUIRED; PRESENCE decides whether
    one is consumed (R-411's in-band amendment of R-410).

    REFUSES if the rule is unreadable -- it never answers False on ignorance,
    because False here is the permissive answer."""
    return str(day) >= governing_day()


def validate_mask(mask: Any, day: str) -> None:
    """REFUSE anything that is not the producer's committed envelope."""
    if not isinstance(mask, dict):
        raise AdmissibleWindowsRefused("mask is not an object")
    missing = [k for k in REQUIRED_MASK_TOP if k not in mask]
    if missing:
        raise AdmissibleWindowsRefused(
            f"mask is MISSING top-level {missing}. The PRODUCER's committed "
            f"artifact is the contract (R-412 ruling 1): the envelope is "
            f"`artifact`/`coins`/`day_closed_calendar`, not `protocol`/"
            f"`per_coin` -- asserting a paraphrase is RR8-1.")
    if mask["artifact"] != MASK_ARTIFACT_KIND:
        raise AdmissibleWindowsRefused(
            f"mask declares artifact {mask['artifact']!r}, which does not "
            f"identify as {MASK_ARTIFACT_KIND!r}")
    if str(mask["day"]) != str(day):
        raise AdmissibleWindowsRefused(
            f"mask is for day {mask['day']!r}, not {day!r}. A mask read for "
            f"the wrong day would mask real windows and unmask masked ones.")
    if mask["day_closed_calendar"] is not True:
        raise AdmissibleWindowsRefused(
            f"mask carries day_closed_calendar={mask['day_closed_calendar']!r}. "
            f"A partial mask lists only the windows that exist SO FAR, so "
            f"supplying off one supplies the complement of a day that has not "
            f"finished (the artifact's own consumer_note says exactly this).")
    coins = mask["coins"]
    if not isinstance(coins, dict) or not coins:
        raise AdmissibleWindowsRefused("mask.coins must be a non-empty object")
    for coin, body in coins.items():
        miss = [k for k in REQUIRED_COIN_FIELDS if k not in body]
        if miss:
            raise AdmissibleWindowsRefused(
                f"mask.coins[{coin!r}] is MISSING {miss}")
        if not isinstance(body["masked_windows"], list):
            raise AdmissibleWindowsRefused(
                f"mask.coins[{coin!r}].masked_windows is not a list")
        if len(body["masked_windows"]) != body["n_masked"]:
            raise AdmissibleWindowsRefused(
                f"mask.coins[{coin!r}]: n_masked={body['n_masked']} but "
                f"{len(body['masked_windows'])} windows are listed -- the "
                f"producer's own count and list disagree")


def load_mask(day: str, path: Path | None = None) -> dict:
    p = path or mask_path(day)
    if not p.exists():
        raise AdmissibleWindowsRefused(f"no mask artifact at {p}")
    mask = json.loads(p.read_text())
    validate_mask(mask, day)
    return mask


def mask_identity(mask: dict | None) -> dict:
    """What `inputs_hash` must cover: the mask's IDENTITY, not its prose.

    Two runs whose masks differ in which windows they mask must not produce
    the same stamp, and two runs under the same mask must -- so the identity
    is (artifact, day, as_of, per-coin masked starts) and nothing else."""
    if mask is None:
        return {"artifact": None, "day": None, "as_of_utc": None,
                "masked": {}, "basis": "NO_MASK"}
    return {
        "artifact": mask["artifact"],
        "day": str(mask["day"]),
        "as_of_utc": mask["as_of_utc"],
        "masked": {c: sorted(int(w) for w in b["masked_windows"])
                   for c, b in sorted(mask["coins"].items())},
        "basis": "MASK",
    }


def _sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True,
                                     separators=(",", ":")).encode()
                          ).hexdigest()


# ---------------------------------------------------------------------------
# 2. the guards -- named, enumerated, and individually auditable
# ---------------------------------------------------------------------------
# Each guard is a NAMED predicate that raises.  They are a list rather than a
# run of inline `if`s so the mutation audit can disable exactly one and prove
# the corresponding known-bad stops firing -- a guard nobody can switch off is
# a guard nobody has shown to be load-bearing (R-346's blank-each-refusal).

def _g_mask_required(ctx) -> None:
    if ctx["mask"] is None and is_governed(ctx["day"]):
        raise AdmissibleWindowsRefused(
            f"day {ctx['day']} is GOVERNED (>= {governing_day()}, read from "
            f"the frozen rule) and "
            f"no mask artifact was supplied; expected it at "
            f"{mask_path(ctx['day'])}. Absence must mean 'the producer did "
            f"not run', never 'nothing was thin' (R-412 ruling 2), so "
            f"accrual on the complement cannot silently become no accrual.")


def _g_coin_present_in_mask(ctx) -> None:
    if ctx["mask"] is None:
        return
    absent = sorted(c for c in ctx["present"] if c not in ctx["mask"]["coins"])
    if absent:
        raise AdmissibleWindowsRefused(
            f"coin(s) {absent} are PRESENT but carry no entry in the mask's "
            f"`coins`. An absent coin is not a coin with nothing masked -- "
            f"absence is not a pass, and treating it as one would supply a "
            f"coin's whole day unmasked on the producer's silence.")


def _g_masked_subset_of_present(ctx) -> None:
    if ctx["mask"] is None:
        return
    for coin, starts in sorted(ctx["present"].items()):
        body = ctx["mask"]["coins"].get(coin)
        if body is None:
            # NOT this guard's job: a coin absent from the mask is
            # `coin_present_in_mask`'s refusal. Skipping keeps each guard to
            # exactly one job -- the first version indexed blindly and, with
            # the other guard disabled, raised a KeyError instead of
            # refusing. A crash is not a refusal, and the mutation audit is
            # what surfaced it.
            continue
        have = set(starts)
        masked = {int(w) for w in body["masked_windows"]}
        orphan = sorted(masked - have)
        if orphan:
            raise AdmissibleWindowsRefused(
                f"{coin}: the mask masks {len(orphan)} window(s) that the "
                f"supplied calendar does not contain (first {orphan[:3]}). "
                f"The two inputs disagree about what EXISTS; dropping them "
                f"silently is the rule-4 failure.")


def _g_no_masked_window_emitted(ctx) -> None:
    """Post-condition on this module's OWN output."""
    if ctx["mask"] is None:
        return
    for coin, specs in sorted(ctx["emitted"].items()):
        body = ctx["mask"]["coins"].get(coin)
        if body is None:
            continue
        masked = {int(w) for w in body["masked_windows"]}
        leaked = sorted(s["start"] for s in specs if s["start"] in masked)
        if leaked:
            raise AdmissibleWindowsRefused(
                f"{coin}: {len(leaked)} MASKED window(s) survived the "
                f"subtraction (first {leaked[:3]}) -- the emission "
                f"contradicts the mask it was built from")


def _g_no_decision_field(ctx) -> None:
    """Nothing decision-shaped may leave here (rule 14)."""
    hits = [k for k in _walk_keys(ctx["emission"]) if k in DECISION_VOCAB]
    if hits:
        raise AdmissibleWindowsRefused(
            f"the emission carries decision-shaped field(s) {sorted(set(hits))}. "
            f"This module SUPPLIES and STAMPS; admission is an R-ADMISS "
            f"decision the coordinator ratifies, and a supplier that shipped "
            f"a verdict would be making it.")


def _walk_keys(obj: Any) -> Iterable[str]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k)
            yield from _walk_keys(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _walk_keys(v)


GUARDS: tuple[tuple[str, Callable], ...] = (
    ("mask_required_on_governed_day", _g_mask_required),
    ("coin_present_in_mask", _g_coin_present_in_mask),
    ("masked_subset_of_present", _g_masked_subset_of_present),
    ("no_masked_window_emitted", _g_no_masked_window_emitted),
    ("no_decision_field", _g_no_decision_field),
)
GUARD_NAMES = tuple(n for n, _ in GUARDS)


# ---------------------------------------------------------------------------
# 3. supply
# ---------------------------------------------------------------------------

def window_spec(coin: str, start: int, ident_hash: str) -> dict:
    """A ReplayWindowSpec-shaped record (contracts v25 `ReplayWindowSpec`:
    slug + inputs_hash), carrying coin and start so a consumer never has to
    re-parse the slug to get them back."""
    return {"slug": SLUG_FORM.format(coin=coin, start=start),
            "coin": coin, "start": int(start),
            "inputs_hash": _sha({"mask_identity": ident_hash, "coin": coin,
                                 "start": int(start)})}


def supply(day: str, present: dict[str, Sequence[int]],
           mask: dict | None = None, *, mask_path_override: Path | None = None,
           load_if_present: bool = True,
           skip_guard: str | None = None) -> dict:
    """PRESENT - masked, per coin, as stamped window specs.

    `mask` may be passed directly (the seam test does, from the producer);
    otherwise it is loaded from the committed path when one exists.  A day at
    or after the governed day REFUSES without one; before it, a mask is
    consumed when present and permitted absent -- and the emission says
    WHICH, because "no mask" and "empty mask" are different facts."""
    if not isinstance(present, dict) or not present:
        raise AdmissibleWindowsRefused(
            "present must be a non-empty {coin: [window starts]} calendar; "
            "this module never derives it -- deriving the calendar is "
            "selecting, and it supplies")
    for coin, starts in present.items():
        if not isinstance(starts, (list, tuple)):
            raise AdmissibleWindowsRefused(f"present[{coin!r}] is not a list")
        if len(set(starts)) != len(starts):
            raise AdmissibleWindowsRefused(
                f"present[{coin!r}] carries duplicate window starts")
    if mask is None and load_if_present:
        p = mask_path_override or mask_path(day)
        if p.exists():
            mask = load_mask(day, p)
    if mask is not None:
        validate_mask(mask, day)

    ident = mask_identity(mask)
    ident_hash = _sha(ident)
    emitted: dict[str, list] = {}
    counts: dict[str, dict] = {}
    ctx = {"day": str(day), "present": {c: list(s) for c, s in present.items()},
           "mask": mask, "emitted": emitted, "emission": None}

    for name, guard in GUARDS:
        if name == skip_guard or guard is _g_no_masked_window_emitted \
                or guard is _g_no_decision_field:
            continue                      # post-conditions run after emission
        guard(ctx)

    for coin, starts in sorted(ctx["present"].items()):
        masked = set()
        if mask is not None and coin in mask["coins"]:
            masked = {int(w) for w in mask["coins"][coin]["masked_windows"]}
        keep = [s for s in sorted(int(x) for x in starts) if s not in masked]
        emitted[coin] = [window_spec(coin, s, ident_hash) for s in keep]
        counts[coin] = {"n_present": len(starts), "n_masked_applied":
                        len(masked & set(int(x) for x in starts)),
                        "n_supplied": len(keep)}

    emission = {
        "protocol": PROTOCOL,
        "day": str(day),
        "governed": is_governed(day),
        "mask_consumed": mask is not None,
        "mask_requirement_basis": (
            f"GOVERNED from {governing_day()} (R-410/R-412, read from "
            f"da_content_liveness_rule): a mask is REQUIRED"
            if is_governed(day) else
            f"PRE-GOVERNED (< {governing_day()}): a mask is CONSUMED when "
            f"present and permitted absent (R-411)"),
        "mask_identity": ident,
        "mask_identity_hash": ident_hash,
        "counts": counts,
        "n_supplied_total": sum(v["n_supplied"] for v in counts.values()),
        "windows": emitted,
        "supplies_not_selects":
            "window admission is an R-ADMISS decision the coordinator "
            "ratifies; this emission is what gets ratified, not the "
            "ratification",
    }
    ctx["emission"] = emission
    for name, guard in GUARDS:
        if name == skip_guard:
            continue
        if guard in (_g_no_masked_window_emitted, _g_no_decision_field):
            guard(ctx)
    return emission


# ---------------------------------------------------------------------------
# 4. selftest -- red-first, on the producer's REAL rows
# ---------------------------------------------------------------------------
REAL_DAY = "20260901"
EMPTY_DAY = "20260827"
EXPECTED_CHECKS = 69


def _grid(day: str) -> list[int]:
    """The day's 5-minute grid, from DA's OWN day bounds and window length --
    not a convention invented here."""
    import da_blackout_mask as M
    lo, _hi = M.day_bounds(day)
    return [lo + M.WINDOW_S * i for i in range(M.WINDOWS_PER_DAY)]


def selftest() -> int:
    n = [0]

    def ok(cond, label):
        if not cond:
            raise SystemExit(f"[de_admissible_windows] FAIL: {label}")
        n[0] += 1
        print(f"  PASS  {label}")

    def refuses(fn, label, needle=None):
        try:
            fn()
        except AdmissibleWindowsRefused as exc:
            if needle and needle not in str(exc):
                raise SystemExit(
                    f"[de_admissible_windows] FAIL: {label} -- refused, but "
                    f"not for the stated reason ({exc})")
            n[0] += 1
            print(f"  PASS  {label}")
            return
        raise SystemExit(f"[de_admissible_windows] FAIL (no refusal): {label}")

    # ---- the REAL committed artifact ----------------------------------
    real_path = mask_path(REAL_DAY)
    ok(real_path.exists(),
       f"the producer's committed artifact is on disk at {real_path.name} -- "
       f"the seam test reads it, not a fixture (RR8-1's only kind of check)")
    real = load_mask(REAL_DAY)
    ok(real["artifact"] == MASK_ARTIFACT_KIND and "coins" in real
       and real["day_closed_calendar"] is True,
       f"and it loads under the COMMITTED envelope: artifact "
       f"{real['artifact']!r}, `coins`, day_closed_calendar true")

    grid = _grid(REAL_DAY)
    present = {c: list(grid) for c in real["coins"]}
    em = supply(REAL_DAY, present, real)

    # numbers READ from the artifact, never literalled
    per_coin_ok = all(
        len(em["windows"][c]) == real["coins"][c]["n_windows_total"]
        - real["coins"][c]["n_masked"] for c in real["coins"])
    ok(per_coin_ok,
       "PER COIN, against the artifact's OWN numbers: len(window list) == "
       "n_windows_total - n_masked for all "
       f"{len(real['coins'])} coins "
       f"({ {c: len(em['windows'][c]) for c in sorted(real['coins'])} })")
    ok(em["n_supplied_total"]
       == sum(v["n_windows_total"] - v["n_masked"]
              for v in real["coins"].values())
       and em["n_supplied_total"]
       == len(grid) * len(real["coins"]) - real["total_masked_windows"],
       f"and the total reconciles two independent ways: "
       f"{em['n_supplied_total']} supplied = 288x{len(real['coins'])} - "
       f"{real['total_masked_windows']} masked")
    ok(all(s["start"] not in set(real["coins"][c]["masked_windows"])
           for c, specs in em["windows"].items() for s in specs),
       "no masked window survives the subtraction, checked over every "
       "emitted record")
    ok(em["mask_consumed"] is True and em["governed"] is False
       and "PRE-GOVERNED" in em["mask_requirement_basis"],
       "09-01 is PRE-GOVERNED and its mask is CONSUMED ANYWAY -- presence "
       "consumes, governance requires (R-411's in-band amendment), and the "
       "emission says which")

    # ---- the stamp ----------------------------------------------------
    spec = em["windows"]["btc"][0]
    ok(set(spec) == set(SPEC_KEYS) and spec["slug"].startswith("btc-updown-5m-")
       and len(spec["inputs_hash"]) == 64,
       f"records are ReplayWindowSpec-shaped and stamped: {spec['slug']}")
    ok(em["mask_identity"]["masked"]["btc"]
       == sorted(real["coins"]["btc"]["masked_windows"])
       and em["mask_identity"]["as_of_utc"] == real["as_of_utc"],
       "the stamp's identity covers the artifact name, day, as_of and the "
       "per-coin masked starts -- the mask's identity, not its prose")
    moved = json.loads(json.dumps(real))
    moved["coins"]["btc"]["masked_windows"] = \
        moved["coins"]["btc"]["masked_windows"][:-1]
    moved["coins"]["btc"]["n_masked"] -= 1
    em2 = supply(REAL_DAY, present, moved)
    ok(em2["mask_identity_hash"] != em["mask_identity_hash"]
       and em2["windows"]["btc"][0]["inputs_hash"]
       != em["windows"]["btc"][0]["inputs_hash"],
       "KNOWN-BAD ON THE STAMP: unmasking ONE window changes the identity "
       "hash AND every window's inputs_hash -- a stamp that did not move "
       "would let two different masks produce the same provenance")
    ok(supply(REAL_DAY, present, real)["mask_identity_hash"]
       == em["mask_identity_hash"],
       "POSITIVE CONTROL: the same mask stamps identically, so the hash is "
       "sensitive to content and not to the run")

    # ---- the empty mask, from DA's OWN producer ------------------------
    import da_blackout_mask as M
    empty = M.build_mask(EMPTY_DAY)
    validate_mask(empty, EMPTY_DAY)          # the contract, not the key set
    extra = sorted(set(empty) - set(real))
    ok(empty["total_masked_windows"] == 0
       and empty["artifact"] == MASK_ARTIFACT_KIND
       and not set(REQUIRED_MASK_TOP) - set(empty),
       f"the empty-mask control is the PRODUCER's own emission for "
       f"{EMPTY_DAY} (0 masked) and satisfies the envelope THIS module "
       f"depends on -- not a hand-built one, which is exactly what drifted "
       f"in RR8-1")
    ok(True if not extra else all(k not in REQUIRED_MASK_TOP for k in extra),
       f"AND THE CONTRACT IS A REQUIRED SUBSET, NOT AN EQUAL SET: the live "
       f"producer emits {extra or 'no'} field(s) the committed 09-01 file "
       f"does not, and that is ADDITIVE GROWTH, not drift. The first version "
       f"of this control asserted `sorted(empty) == sorted(real)` and went "
       f"red the moment DA legitimately added a `producer` provenance block "
       f"-- a consumer that breaks on a producer's additive change is making "
       f"someone else's improvement look like its own regression.")
    grid_e = _grid(EMPTY_DAY)
    em_e = supply(EMPTY_DAY, {c: list(grid_e) for c in empty["coins"]}, empty)
    ok(all(len(em_e["windows"][c]) == len(grid_e) for c in empty["coins"])
       and em_e["mask_consumed"] is True,
       f"an EMPTY mask supplies the FULL list: "
       f"{ {c: len(v) for c, v in sorted(em_e['windows'].items())} }")

    # ---- refusals, each red-first, each with its positive control ------
    refuses(lambda: supply("20260902", {"btc": [1]}, None, load_if_present=False),
            "KNOWN-BAD: a GOVERNED day with no mask REFUSES and names the "
            "expected path",
            needle="da_blackout_mask_20260902.json")
    ok(supply("20260901", {"btc": [1]}, None,
              load_if_present=False)["mask_consumed"] is False,
       "POSITIVE CONTROL: the same call on a PRE-GOVERNED day is permitted, "
       "and the emission records mask_consumed false rather than implying an "
       "empty mask -- 'no mask' and 'empty mask' are different facts")
    partial = json.loads(json.dumps(real))
    partial["day_closed_calendar"] = False
    refuses(lambda: supply(REAL_DAY, present, partial),
            "KNOWN-BAD: day_closed_calendar false REFUSES -- a partial mask "
            "lists only the windows that exist so far",
            needle="day_closed_calendar")
    drift = json.loads(json.dumps(real))
    drift["protocol"] = drift.pop("artifact")
    drift["per_coin"] = drift.pop("coins")
    refuses(lambda: supply(REAL_DAY, present, drift),
            "KNOWN-BAD (RR8-1 in this seat): the `protocol`/`per_coin` "
            "envelope REFUSES -- the producer's committed artifact is the "
            "contract, and asserting a paraphrase is the defect",
            needle="MISSING top-level")
    wrongday = json.loads(json.dumps(real))
    wrongday["day"] = "20260831"
    refuses(lambda: supply(REAL_DAY, present, wrongday),
            "KNOWN-BAD: a mask for another day REFUSES -- it would mask real "
            "windows and unmask masked ones")
    miscount = json.loads(json.dumps(real))
    miscount["coins"]["eth"]["n_masked"] += 1
    refuses(lambda: supply(REAL_DAY, present, miscount),
            "KNOWN-BAD: the producer's own count and list disagreeing REFUSES")
    refuses(lambda: supply(REAL_DAY, dict(present, newcoin=list(grid)), real),
            "KNOWN-BAD: a coin PRESENT but absent from `coins` REFUSES -- "
            "absence is not a coin with nothing masked",
            needle="absence is not a pass")
    short = {c: [s for s in grid
                 if s != real["coins"][c]["masked_windows"][0]]
             for c in real["coins"]}
    refuses(lambda: supply(REAL_DAY, short, real),
            "KNOWN-BAD: the mask masking a window the calendar does not "
            "contain REFUSES -- the two inputs disagree about what EXISTS, "
            "and dropping it silently is the rule-4 failure",
            needle="disagree about what EXISTS")
    refuses(lambda: supply(REAL_DAY, {}, real),
            "KNOWN-BAD: an empty calendar REFUSES -- this module never "
            "derives one, because deriving it would be selecting")
    refuses(lambda: supply(REAL_DAY, {"btc": [1, 1]}, real),
            "KNOWN-BAD: duplicate window starts REFUSE")

    # ---- it supplies, it does not select -------------------------------
    ok(not any(k in DECISION_VOCAB for k in _walk_keys(em)),
       "the emission carries NO decision-shaped field -- checked over every "
       "key at every depth, not asserted")
    imps = imported_modules(Path(__file__).read_text())
    ok(reads_no_verdict(imps),
       f"and the module IMPORTS no verdict producer -- checked over its "
       f"parsed import list {sorted(imps)}, not by grepping its own text for "
       f"a word (a source grep can see a deleted line but not a value that "
       f"arrived another way, which is the F-1 shape)")
    ok(not reads_no_verdict(imps | {VERDICT_MODULES[0]}),
       f"KNOWN-BAD: adding {VERDICT_MODULES[0]!r} to that same import list "
       f"trips the predicate, so it is a check and not a constant")
    # ---- DE-R2: a dynamic import used to defeat the predicate ----------
    dyn = ("import importlib\n"
           "m = importlib.import_module('da_forward_day_verify')\n")
    ok("da_forward_day_verify" in imported_modules(dyn)
       and not reads_no_verdict(imported_modules(dyn)),
       "DE-R2 CLOSED: `importlib.import_module('da_forward_day_verify')` is "
       "RESOLVED and trips the predicate -- it used to produce an empty set "
       "and answer True, so a dynamic import defeated the check entirely")
    ok(not reads_no_verdict(imported_modules(
        "__import__('da_forward_day_verify')\n")),
       "and `__import__` with a literal is resolved the same way")
    refuses(lambda: reads_no_verdict(imported_modules(
                "import importlib\nm = importlib.import_module(name)\n")),
            "KNOWN-BAD: a NON-LITERAL dynamic import REFUSES rather than "
            "returning True -- 'cannot tell what this imports' is not "
            "'imports nothing', and the second is what the old form said",
            needle="is not 'imports nothing'")
    ok(_UNRESOLVED in imported_modules(
        "import importlib\nm = importlib.import_module(x)\n"),
       "the unresolvable import is REPORTED as a sentinel rather than "
       "dropped, so the caller decides what to do about what it cannot name")
    ok(not reads_no_verdict(imported_modules(
        "import live.pm_research.da_forward_day_verify\n")),
       "AND A HOLE OF MY OWN, found while fixing this one: a DOTTED name "
       "contributed only its FIRST segment, so the package spelling of a "
       "verdict producer resolved to `live` and passed. First AND last "
       "segment are taken now -- widening is the safe direction for a "
       "reads-NOTHING predicate")
    # ---- DE11-R1: three shapes that imported with no import node --------
    _X = "da_forward_day_verify"
    for _name, _src in (
            ("exec", f"exec('import {_X}')"),
            ("eval", f"eval(\"__import__('{_X}')\")"),
            ("rebound __import__", f"f = __import__\nf('{_X}')")):
        refuses(lambda src=_src: reads_no_verdict(imported_modules(src)),
                f"DE11-R1 ({_name}): REFUSES -- it used to parse to an EMPTY "
                f"set and answer True, which is blind AND passing, the worst "
                f"pair. The shape is NAMED in the message")
    ok(_OPAQUE in imported_modules("exec('import x')")
       and _REBOUND in imported_modules("f = __import__\n")
       and _UNRESOLVED in imported_modules(
           "import importlib\nimportlib.import_module(v)\n"),
       "each unreadable shape is reported as its OWN sentinel, so the "
       "refusal can say WHICH one it saw rather than 'something'")
    ok(reads_no_verdict(imported_modules("__import__('json')\n")) is True,
       "POSITIVE CONTROL: `__import__` CALLED with a literal still resolves "
       "normally -- the rebound branch does not swallow the call form")
    ok(not reads_no_verdict(imported_modules(f"__import__('{_X}')\n")),
       "and that same call form still CATCHES a verdict producer")
    ok(not reads_no_verdict(imported_modules(
        f"from importlib import import_module\nimport_module('{_X}')\n")),
       "the `from importlib import import_module` form is caught")
    ok(not reads_no_verdict(imported_modules(
        f"import importlib as il\nil.import_module('{_X}')\n")),
       "and so is the aliased-module form -- the call is matched on the "
       "attribute name, not on the module alias")
    # ---- THE DECLARED LIMIT, TESTED FOR ITS OBSERVABLE CONSEQUENCE ------
    # A shape one cannot see cannot be tested for directly -- but what CAN
    # be asserted is that the predicate still behaves as DECLARED on it:
    # the set does not grow (it did not start catching the shape) and no
    # exception is raised (it did not start refusing it). Either change
    # breaks the assertion, so a declared limit that silently stops being
    # true is noticed in BOTH directions. Same construction as the audit's
    # `refuses_on_the_control: false`, pointed the other way.
    for _label, _src, _want in (
            ("runpy.run_path", "import runpy\nrunpy.run_path('x')\n",
             {"runpy"}),
            ("runpy.run_module", "import runpy\nrunpy.run_module('x')\n",
             {"runpy"}),
            ("builtins.exec (attribute form)",
             "import builtins\nbuiltins.exec('import x')\n", {"builtins"}),

            ("getattr-reached import_module",
             "import importlib\ngetattr(importlib, 'import_module')('x')\n",
             {"importlib"})):
        _got = imported_modules(_src)
        ok(_got == _want,
           f"EXPECTED-BLIND ({_label}): the predicate still sees exactly "
           f"{sorted(_want)} and neither catches nor refuses the shape "
           f"({sorted(_got)}) -- the declared limit holds, and this "
           f"assertion fails if a later change starts doing either")
    # AND ONE SHAPE THE LIST GOT WRONG, found by these very assertions.
    ok(imported_modules(
        "import builtins\nbuiltins.__import__('x')\n") == {"builtins", "x"},
       "NOT BLIND AFTER ALL: `builtins.__import__('x')` IS CAUGHT -- the "
       "dynamic-import matcher keys on the ATTRIBUTE NAME, so the attribute "
       "form resolves its literal exactly as the bare form does. The "
       "declared list claimed a blindness the code did not have, and the "
       "expected-blind assertions found it on their FIRST RUN")
    ok(not reads_no_verdict(imported_modules(
        "import builtins\nbuiltins.__import__('da_forward_day_verify')\n")),
       "so a verdict producer imported that way is CAUGHT, not passed")
    ok(reads_no_verdict(imported_modules(
        "import importlib\n"
        "getattr(importlib, 'import_module')('da_forward_day_verify')\n"))
       is True,
       "AND THE CONSEQUENCE OF A REAL BLIND SHAPE, STATED PLAINLY: through "
       "the getattr form a verdict producer WOULD pass. That is what "
       "'declared blind' means, and writing it as a check is the difference "
       "between a limit that is stated and one that is discovered")

    ok(len(DECLARED_BLIND_SHAPES) >= 4
       and any("runpy" in x for x in DECLARED_BLIND_SHAPES)
       and any("builtins" in x for x in DECLARED_BLIND_SHAPES),
       f"THE LIMIT IS DECLARED, not discovered: {len(DECLARED_BLIND_SHAPES)} "
       f"shapes this predicate does NOT see are named with their reasons "
       f"(runpy, builtins.__import__, getattr-reached import_module, C "
       f"extensions). There is no way to TEST for a shape one cannot see -- "
       f"asserting otherwise would be the control that cannot fail -- so "
       f"the assertion is that the list exists and names them")

    ok("da_content_liveness_rule" in imps,
       "and this module's OWN dynamic import (the RR11-1 demonstration) "
       "resolves to `da_content_liveness_rule` -- the frozen liveness rule, "
       "not a verdict producer, which is what the predicate needed to be "
       "able to say")
    leaky = dict(em, all_pass=True)
    refuses(lambda: _g_no_decision_field({"emission": leaky}),
            "KNOWN-BAD: a decision-shaped field in the emission REFUSES, so "
            "the check above is a filter and not a blanket")

    # ---- CO-3: the governing day is BOUND, not restated ----------------
    ok(RULE_MODULE is not None and governing_day()
       == RULE_MODULE.EFFECTIVE_FROM_DAY,
       f"the governing day is READ from the frozen rule "
       f"(da_content_liveness_rule.EFFECTIVE_FROM_DAY = "
       f"{governing_day()!r}), not restated as a literal here")
    _src = Path(__file__).read_text()
    ok(restated_day_literals(_src, governing_day()) == [],
       "and NO module-level constant here restates that day -- checked over "
       "the parsed AST, so it counts neither the comment explaining the "
       "removal nor a rename, and one source means a USER amendment of the "
       "rule cannot leave this supplier on a different calendar than the "
       "scorer")
    ok(restated_day_literals(f'X = {governing_day()!r}\n', governing_day())
       == ["X"],
       "KNOWN-BAD: the same predicate NAMES a restated literal under any "
       "constant name, so it is a check and not a constant")
    # THE CONTROL THE DISPATCH ASKED FOR: patch the rule, and the supplier
    # must FOLLOW. It follows because the attribute is read at CALL time; a
    # value copied at import would have pinned a snapshot and this control
    # would fail, which is exactly why it is not copied.
    # RR11-1: ASK FOR THE TARGET, never name one. `patch_target` refuses if
    # the object is not the one this module bound, so a control that reached
    # for the other spelling fails loudly instead of testing nothing.
    _tgt = patch_target()
    ok(_tgt is RULE_MODULE and _tgt is sys.modules[RULE_MODULE.__name__],
       f"RR11-1: the control's patch target IS the object under test "
       f"({RULE_MODULE.__name__!r}), asserted before anything is patched")
    _orig = _tgt.EFFECTIVE_FROM_DAY
    try:
        _tgt.EFFECTIVE_FROM_DAY = "20260901"
        ok(is_governed("20260901") and governing_day() == "20260901",
           "KNOWN-BAD/FOLLOW-THE-RULE: with the rule patched to 20260901, "
           "09-01 becomes GOVERNED here immediately -- the supplier follows "
           "the rule rather than a snapshot of it")
        refuses(lambda: supply("20260901", {"btc": [1]}, None,
                               load_if_present=False),
                "and the consequence propagates: 09-01 with no mask now "
                "REFUSES, where a moment ago it was permitted")
        _tgt.EFFECTIVE_FROM_DAY = "20261231"
        ok(not is_governed("20260902"),
           "POSITIVE CONTROL, the other direction: pushed to 20261231, "
           "09-02 stops being governed -- the binding tracks both ways")
        _tgt.EFFECTIVE_FROM_DAY = None
        try:
            is_governed("20260902")
            unreadable_refuses = False
        except GoverningRuleUnreadable:
            unreadable_refuses = True
        ok(unreadable_refuses,
           "KNOWN-BAD: an unreadable governing day REFUSES rather than "
           "answering False -- False is the PERMISSIVE answer here, and "
           "defaulting to it would turn the mask REQUIREMENT into "
           "PERMISSION (CO-1's hole, in this seat)")
    finally:
        _tgt.EFFECTIVE_FROM_DAY = _orig
    ok(governing_day() == _orig and _orig == "20260902",
       f"and the rule is restored: governing_day() back to {_orig!r}")

    # ---- RR11-1, red-first: the OTHER spelling is refused ---------------
    # THE REPO ROOT GOES ON sys.path FOR THE DURATION OF THIS DEMONSTRATION
    # AND COMES OFF AGAIN. Without it the package spelling is unimportable
    # under the script-dir launch, the demonstration silently does not run,
    # and the two launchers report DIFFERENT CHECK COUNTS -- which is CO-2's
    # class reappearing in the check accounting. Inserted and removed so the
    # dual-module-identity hazard (IMPORT_LAYOUT section 3) is not left
    # standing for anything else.
    import importlib
    _pkg = None
    _added = str(ROOT) not in sys.path
    if _added:
        sys.path.insert(0, str(ROOT))
    try:
        _pkg = importlib.import_module("live.pm_research."
                                       "da_content_liveness_rule")
    except Exception:                                # noqa: BLE001
        pass
    finally:
        if _added and str(ROOT) in sys.path:
            sys.path.remove(str(ROOT))
    if _pkg is not None and _pkg is not RULE_MODULE:
        _po = _pkg.EFFECTIVE_FROM_DAY
        try:
            _pkg.EFFECTIVE_FROM_DAY = "20260901"
            ok(not is_governed("20260901"),
               "KNOWN-BAD REPRODUCED: patching the PACKAGE spelling moves "
               "NOTHING here -- a control written that way would watch a "
               "value it did not change and PASS, testing nothing (RR11-1)")
        finally:
            _pkg.EFFECTIVE_FROM_DAY = _po
        refuses(lambda: patch_target("live.pm_research."
                                     "da_content_liveness_rule"),
                "AND IT NOW FAILS LOUDLY: patch_target REFUSES that spelling "
                "because it is not the object this module bound",
                needle="would NOT reach the object")
    else:
        ok(False,
           "the package spelling could not be loaded as a distinct object, "
           "so the RR11-1 demonstration did not run -- reported as a FAILURE "
           "rather than skipped, because a control that quietly does not run "
           "is the vacuity this whole finding is about")
        ok(False, "(second half of the same demonstration, likewise not run)")
    refuses(lambda: patch_target("no_such_rule_module"),
            "KNOWN-BAD: a control cannot patch a module that is not loaded")
    ok(patch_target() is RULE_MODULE,
       "POSITIVE CONTROL: the correct spelling is ADMITTED and returns the "
       "bound object")

    # ---- the SWEEP: all five launch-invariant modules -------------------
    here = Path(__file__).resolve().parent
    swept = {}
    for _f in ("de_admissible_windows.py", "de_lane4_real_parity.py",
               "rule_policy_v1.py", "ev_replay_seam.py", "de_actionspace.py"):
        swept[_f] = patch_targets((here / _f).read_text())
    ok(len(swept) == 5 and swept["de_admissible_windows.py"]
       == ["_pkg.EFFECTIVE_FROM_DAY", "_tgt.EFFECTIVE_FROM_DAY"],
       f"SWEEP of all five launch-invariant modules: the only module-constant "
       f"patch targets on the whole surface are this module's TWO, and the "
       f"sweep found the second one immediately -- `_tgt`, the handle "
       f"`patch_target()` guards, and `_pkg`, the DELIBERATE wrong-spelling "
       f"demonstration whose entire point is that it reaches nothing "
       f"({swept['de_admissible_windows.py']}). Neither names a module "
       f"directly, which is what RR11-1 asks for.")
    ok(all(not v for f, v in swept.items()
           if f != "de_admissible_windows.py"),
       f"and the other four patch no module constant at all, so RR11-1's "
       f"shape exists in exactly one place: "
       f"{ {f: v for f, v in swept.items() if f != 'de_admissible_windows.py'} }")
    ok(patch_targets("import m\nm.CONST = 1\n") == ["m.CONST"]
       and patch_targets("# m.CONST = 1\nx = 2\n") == [],
       "KNOWN-BAD/POSITIVE CONTROL on the sweep itself: it NAMES a real "
       "patch and IGNORES one that is only mentioned in a comment, because "
       "it reads the AST rather than the text")

    # ---- mutation audit: every guard blanked, one at a time -------------
    audit = mutation_audit(real, present, grid)
    ok(audit["all_load_bearing"],
       f"MUTATION AUDIT: each of the {audit['n_input_guards']} INPUT guards "
       f"was disabled in turn and its known-bad STOPPED refusing -- every one "
       f"is load-bearing, none is decoration ({audit['survivors']} survivors)")
    ok(all(v["kind"] == "POST_CONDITION"
           for k, v in audit["per_guard"].items()
           if k in ("no_masked_window_emitted", "no_decision_field"))
       and audit["n_post_conditions"] == 2,
       "and the two POST-CONDITIONS are reported as post-conditions, not "
       "counted as killed mutants: they check this module's own emission and "
       "no input can make them fire, so a mutation 'kill' there would be a "
       "control that cannot fail wearing a kill's clothes")
    ok(audit["n_guards"] == len(GUARDS) and set(audit["per_guard"])
       == set(GUARD_NAMES),
       f"and the audit covers every declared guard by name: "
       f"{sorted(audit['per_guard'])}")

    ok(n[0] + 1 == EXPECTED_CHECKS,
       f"check count asserted at run time: {n[0] + 1} == {EXPECTED_CHECKS}")
    print(f"[de_admissible_windows] selftest OK -- {n[0]} checks")
    return 0


def mutation_audit(real: dict, present: dict, grid: list) -> dict:
    """Blank each INPUT guard in turn and confirm its known-bad stops firing.

    A guard that refuses whether or not it runs is being done by something
    else, and counting it would inflate the boundary (R-346: five of nineteen
    refusals were invisible to a green suite; one was dead code).

    THE FIRST VERSION OF THIS HARNESS WAS BROKEN and said so loudly enough to
    be caught: its "live" run passed `skip_guard` too, so every guard was
    measured with itself already disabled and three read as survivors. A
    broken mutation harness and a perfectly-covered suite produce identical
    output (R-347), which is why the live and disabled runs are now visibly
    different calls.

    POST-CONDITIONS ARE REPORTED SEPARATELY, NOT COUNTED AS MUTANTS. Two
    guards check this module's OWN emission and cannot be made to fire by any
    input -- only by a broken subtraction. Driving them at the unit is their
    control (the selftest does), and calling that a killed mutant would be a
    control that cannot fail wearing a kill's clothes."""
    def _short_calendar():
        return {c: [s for s in grid
                    if s != real["coins"][c]["masked_windows"][0]]
                for c in real["coins"]}

    # (live call, disabled call) per INPUT guard -- visibly different calls
    cases = {
        "mask_required_on_governed_day": (
            lambda: supply("20260902", {"btc": [1]}, None,
                           load_if_present=False),
            lambda: supply("20260902", {"btc": [1]}, None,
                           load_if_present=False,
                           skip_guard="mask_required_on_governed_day")),
        "coin_present_in_mask": (
            lambda: supply(REAL_DAY, dict(present, newcoin=list(grid)), real),
            lambda: supply(REAL_DAY, dict(present, newcoin=list(grid)), real,
                           skip_guard="coin_present_in_mask")),
        "masked_subset_of_present": (
            lambda: supply(REAL_DAY, _short_calendar(), real),
            lambda: supply(REAL_DAY, _short_calendar(), real,
                           skip_guard="masked_subset_of_present")),
    }
    per_guard: dict[str, dict] = {}
    for name, (live, disabled) in cases.items():
        try:
            live()
            refused_live = False
        except AdmissibleWindowsRefused:
            refused_live = True
        try:
            disabled()
            refused_disabled = False
        except AdmissibleWindowsRefused:
            refused_disabled = True
        per_guard[name] = {"kind": "INPUT",
                           "refuses_when_live": refused_live,
                           "refuses_when_disabled": refused_disabled,
                           "load_bearing": refused_live and not refused_disabled}

    # the two post-conditions, driven at the unit, reported as what they are
    for name, drive in (
            ("no_masked_window_emitted",
             lambda: _g_no_masked_window_emitted({
                 "mask": real,
                 "emitted": {"btc": [window_spec(
                     "btc", real["coins"]["btc"]["masked_windows"][0], "x")]}})),
            ("no_decision_field",
             lambda: _g_no_decision_field({"emission": {"all_pass": True}}))):
        try:
            drive()
            fires = False
        except AdmissibleWindowsRefused:
            fires = True
        per_guard[name] = {"kind": "POST_CONDITION",
                           "fires_on_its_known_bad": fires,
                           "load_bearing": fires}

    survivors = sorted(k for k, v in per_guard.items()
                       if not v["load_bearing"])
    return {"n_guards": len(per_guard),
            "n_input_guards": len(cases),
            "n_post_conditions": len(per_guard) - len(cases),
            "per_guard": per_guard, "survivors": survivors,
            "all_load_bearing": not survivors}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("cmd", nargs="?", choices=["supply"])
    ap.add_argument("--day", default=REAL_DAY)
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd == "supply":
        if selftest() != 0:
            return 1
        mask = load_mask(a.day)
        em = supply(a.day, {c: list(_grid(a.day)) for c in mask["coins"]}, mask)
        print(json.dumps({k: v for k, v in em.items() if k != "windows"},
                         indent=2, sort_keys=True))
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
