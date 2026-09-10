"""WHAT WAS ACTUALLY TOUCHED -- observed by EXECUTION, not by reading source.

REV 152, rule 38. The scoring-path waiver rests on a reachability claim --
"nothing that moved in `de_multiday_gate1_runner` is reached from the
scoring entry points" -- and DA's instrument, DE's instrument and REV's
derivation ALL resolve to ONE operation, `be_producing_closure.
reachable_modules`. Three confirmations, one observation wearing three
hats.

**A STATIC WALK ANSWERS "WHAT COULD BE REACHED". THE WAIVER NEEDS "WHAT
WAS REACHED". THOSE ARE DIFFERENT QUESTIONS** and this programme has been
treating them as one.

AND THE DIFFERENCE IS NOT CONSERVATIVE IN EITHER DIRECTION. Measured on a
controlled fixture, one static walk does BOTH in eight lines:

    def produce(x):
        if x < 0:
            over.never_runs(x)          # NEVER EXECUTES -- and the walk
                                        # reports `over` as reached
        fn = getattr(under, "actually_runs")
        return fn(x)                    # DOES EXECUTE -- and the walk
                                        # does NOT report `under`

So a static "IS reached" can be false (a branch nobody takes) AND a static
"is NOT reached" can be false (a dispatch nobody can see). **For a waiver,
whose claim is of the second kind, the error runs in the DANGEROUS
direction.** That is what this module exists to remove.

WHAT IT OBSERVES
  * `functions` -- every function ACTUALLY ENTERED, as (file, qualname),
    via `sys.settrace`. A function that is defined, referenced, or sits in
    an untaken branch does NOT appear. Only entry appears.
  * `attributes` -- every attribute READ on a watched module, via a proxy
    installed in `sys.modules`. This catches MODULE-LEVEL CONSTANTS, which
    no call trace sees and which the waiver's claim depends on
    (`ruled_day_set` reads `PARAMS_REL`).

WHAT IT CANNOT OBSERVE, stated because a trace's limits decide what may be
concluded from it:
  * **ONLY THE PATH ACTUALLY RUN.** A trace of one input says nothing
    about another input. It is a LOWER bound on what the code can touch,
    and an EXACT record of what this execution touched. Absence here means
    "not touched ON THIS RUN", never "unreachable".
  * `from x import y` binds the object at import time, so a later call
    through `y` is seen by the FUNCTION trace but not by the ATTRIBUTE
    proxy. Watch a module before its importers bind from it, or read the
    function record for that module instead.
  * **A MODULE READING ITS OWN GLOBAL IS INVISIBLE TO THE ATTRIBUTE
    PROXY.** `ruled_day_set` reading `PARAMS_REL` is a LOAD_GLOBAL in the
    runner's own frame, not an attribute access through `sys.modules`, so
    no proxy can see it. The FUNCTION record still says `ruled_day_set`
    ran; whether it read that constant is settled by the constant's bytes
    being identical, which is the waiver's OTHER half and needs no trace.
  * **AN IMPORTER THAT ALREADY HOLDS A DIRECT REFERENCE BYPASSES THE
    PROXY** -- `import x` binds the module object itself, so a later
    `getattr(x, "f")` inside that importer reads through its own binding.
    The FUNCTION trace still sees the resulting call. The count of such
    importers is reported per watched module so the gap is measured, and
    the falsifier drives exactly this case.
  * C-level and builtin frames are not Python frames; `sys.settrace` does
    not see them.
  * A module already imported before `watch()` is called is re-fetched
    from `sys.modules`; the proxy replaces the entry, so importers that
    already hold a direct reference bypass it. `n_importers_holding_a_
    direct_reference` reports how many, so the gap is measured rather
    than assumed.

THIS MODULE DOES NOT COMPARE ITSELF TO `be_producing_closure`. That
comparison is the trap this finding is about: an instrument that verifies
itself against the operation it exists to be independent of has added no
observation. DA verifies, with an instrument of its own.

Run `python3 de_dynamic_reach.py --falsify` for the cells.
"""
from __future__ import annotations

import json
import sys
import threading
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


class TraceRefused(RuntimeError):
    """A named refusal."""


NOT_A_REACHABILITY_CLAIM = (
    "THIS IS A RECORD OF ONE EXECUTION, NOT A REACHABILITY SET. A name "
    "absent here was NOT TOUCHED ON THIS RUN; it may still be touched on "
    "another input. Absence is evidence about this run and nothing else.")


class _ModuleProxy(types.ModuleType):
    """A module whose attribute READS are recorded, then delegated."""

    def __init__(self, wrapped, sink):
        super().__init__(wrapped.__name__, getattr(wrapped, "__doc__", None))
        object.__setattr__(self, "_wrapped", wrapped)
        object.__setattr__(self, "_sink", sink)

    def __getattr__(self, name):
        if not name.startswith("__"):
            self._sink.setdefault(self._wrapped.__name__, set()).add(name)
        return getattr(self._wrapped, name)

    def __setattr__(self, name, value):
        setattr(self._wrapped, name, value)

    def __repr__(self):
        return f"<traced module {self._wrapped.__name__!r}>"


class Trace:
    """Collects what one execution actually touched."""

    def __init__(self, watch=()):
        self.watch = tuple(watch)
        self.functions: set = set()
        self.attributes: dict = {}
        self._originals: dict = {}
        self._prev = None
        self._importers: dict = {}

    # -- the function trace ------------------------------------------
    def _tracer(self, frame, event, arg):
        if event == "call":
            code = frame.f_code
            self.functions.add((code.co_filename, code.co_qualname))
        return None                       # per-frame local tracing OFF

    def __enter__(self):
        # COUNT FIRST, INSTALL SECOND. A proxy installed earlier in this
        # loop has its OWN `__dict__`, so scanning after installing hides
        # exactly the importers being counted -- measured: this reported
        # 0 direct references for a module with one.
        _pre = {n: sys.modules.get(n) for n in self.watch}
        for name, mod in _pre.items():
            if mod is None or isinstance(mod, _ModuleProxy):
                continue
            n = 0
            for other in list(sys.modules.values()):
                if other is None or other is mod:
                    continue
                try:
                    d = getattr(other, "__dict__", None) or {}
                except Exception:                          # noqa: BLE001
                    continue
                if any(v is mod for v in list(d.values())):
                    n += 1
            self._importers[name] = n
        for name in self.watch:
            mod = sys.modules.get(name)
            if mod is None:
                raise TraceRefused(
                    f"REFUSED TRACE_WATCH_MODULE_NOT_IMPORTED: {name!r} is "
                    f"not in sys.modules, so a proxy cannot be installed "
                    f"and its attribute reads would be silently unobserved. "
                    f"Import it before tracing, or do not claim to watch "
                    f"it.")
            if isinstance(mod, _ModuleProxy):
                continue
            self._originals[name] = mod
            sys.modules[name] = _ModuleProxy(mod, self.attributes)
        self._prev = sys.gettrace()
        sys.settrace(self._tracer)
        threading.settrace(self._tracer)
        return self

    def __exit__(self, *exc):
        sys.settrace(self._prev)
        threading.settrace(None)
        for name, mod in self._originals.items():
            sys.modules[name] = mod
        return False

    # -- the record --------------------------------------------------
    def touched_in(self, filename_endswith: str) -> dict:
        """What was touched in ONE file, by qualname."""
        fns = sorted({q for f, q in self.functions
                      if f.endswith(filename_endswith)})
        mod = filename_endswith[:-3] if filename_endswith.endswith(".py") \
            else filename_endswith
        attrs = sorted(self.attributes.get(mod, ()))
        return {
            "file": filename_endswith,
            "functions_entered": fns,
            "n_functions_entered": len(fns),
            "attributes_read": attrs,
            "n_attributes_read": len(attrs),
            "attribute_observation": (
                "via a sys.modules proxy" if mod in self.attributes
                or mod in self._importers else
                "NOT WATCHED -- attribute reads on this module were not "
                "observed at all; only function ENTRIES are recorded here"),
            "importers_holding_a_direct_reference": self._importers.get(mod),
            "LIMIT": NOT_A_REACHABILITY_CLAIM,
        }

    def report(self) -> dict:
        by_file: dict = {}
        for f, q in self.functions:
            by_file.setdefault(Path(f).name, []).append(q)
        return {
            "observed_by": "EXECUTION -- sys.settrace on function entry, "
                           "plus a sys.modules proxy for attribute reads",
            "n_functions_entered": len(self.functions),
            "files_entered": sorted(by_file),
            "functions_by_file": {k: sorted(set(v))
                                  for k, v in sorted(by_file.items())},
            "attributes_read": {k: sorted(v)
                                for k, v in sorted(self.attributes.items())},
            "watched": list(self.watch),
            "importers_holding_a_direct_reference": dict(self._importers),
            "LIMIT": NOT_A_REACHABILITY_CLAIM,
            "does_NOT_compare_itself_to": (
                "be_producing_closure.reachable_modules -- comparing an "
                "instrument to the operation it exists to be independent "
                "of adds no observation (REV 152, rule 38)"),
        }


def trace_call(fn, *args, watch=(), **kw):
    """Run `fn` under the trace and return `(result, report)`."""
    t = Trace(watch=watch)
    with t:
        result = fn(*args, **kw)
    return result, t.report()


# --------------------------------------------------------------------------
# THE FALSIFIER (rule 15)
# --------------------------------------------------------------------------
# A trace that has never proved it can tell a touched module from an
# untouched one is not a trace. The cells below drive BOTH, and the third
# is the one this instrument exists for: a dispatch a static walk CANNOT
# see, which this one MUST see.
_FX = {
    "dr_seed.py": '''"""fixture: the producer's entry point."""
import dr_over
import dr_under

CONST_READ = 1
CONST_NOT_READ = 2


def produce(x):
    if x < 0:
        dr_over.never_runs(x)          # NEVER EXECUTES
    fn = getattr(dr_under, "actually_runs")
    return fn(x) + CONST_READ          # DOES EXECUTE, via getattr


def never_called(x):
    return x
''',
    "dr_over.py": '''def never_runs(x):
    return x
''',
    "dr_under.py": '''def actually_runs(x):
    return x + 1


def sibling_not_called(x):
    return x
''',
}


def falsify() -> int:                                        # noqa: C901
    import tempfile
    fails = []

    def ok(cond, label):
        print(f"  {'ok  ' if cond else 'FAIL'}  {label}")
        if not cond:
            fails.append(label)

    print("[de_dynamic_reach] falsifier")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for n, src in _FX.items():
            (root / n).write_text(src)
        sys.path.insert(0, str(root))
        try:
            for m in ("dr_seed", "dr_over", "dr_under"):
                sys.modules.pop(m, None)
            import dr_seed                                   # noqa: F401
            t = Trace(watch=("dr_seed", "dr_under"))
            with t:
                out = dr_seed.produce(1)
            rep = t.report()
            seed = rep["functions_by_file"].get("dr_seed.py", [])
            under = rep["functions_by_file"].get("dr_under.py", [])
            over = rep["functions_by_file"].get("dr_over.py", [])
            attrs = rep["attributes_read"]

            ok(out == 3, f"FIXTURE: produce(1) == 3, got {out}")
            # (1) POSITIVE CONTROL -- a module KNOWN touched appears.
            ok("dr_under.py" in rep["files_entered"]
               and "actually_runs" in under,
               f"POSITIVE: a function that RAN is recorded -- "
               f"dr_under.actually_runs in {under}")
            # (2) NEGATIVE CONTROL -- a module KNOWN untouched is absent.
            ok("dr_over.py" not in rep["files_entered"] and not over,
               f"NEGATIVE: a module that did NOT run is ABSENT -- "
               f"dr_over recorded {over}. A trace that cannot leave "
               f"something out is not distinguishing anything")
            # (3) THE ONE THIS EXISTS FOR: the getattr dispatch a static
            # walk cannot follow IS observed here.
            ok("actually_runs" in under
               and rep["importers_holding_a_direct_reference"]["dr_under"]
               >= 1
               and "actually_runs" not in (attrs.get("dr_under") or []),
               f"THE GAP IT CLOSES, AND THE ONE IT DOES NOT: the `getattr` "
               f"dispatch a source walk CANNOT follow IS recorded as a "
               f"FUNCTION ENTRY -- which is the edge that matters -- while "
               f"the ATTRIBUTE proxy does NOT see it, because dr_seed "
               f"holds a direct `import` binding "
               f"({rep['importers_holding_a_direct_reference']['dr_under']} "
               f"importer(s), COUNTED, not assumed). An instrument that "
               f"claimed both would be claiming an observation it did not "
               f"make")
            # (4) A DEFINED-BUT-UNCALLED SIBLING IS ABSENT.
            ok("never_called" not in seed
               and "sibling_not_called" not in under,
               "DEFINITION IS NOT EXECUTION: functions that are DEFINED "
               "in a touched file but never entered do not appear")
            # (5) MODULE-LEVEL CONSTANTS: read vs not read.
            seed_attrs = attrs.get("dr_seed") or []
            ok("CONST_NOT_READ" not in seed_attrs,
               f"A CONSTANT NOBODY READ IS ABSENT: {seed_attrs}")
            # (6) THE LIMIT TRAVELS ON EVERY REPORT.
            ok("NOT A REACHABILITY SET" in rep["LIMIT"]
               and "reachable_modules" in rep["does_NOT_compare_itself_to"],
               "THE LIMIT AND THE INDEPENDENCE TRAVEL: absence means 'not "
               "touched on this run', and the report says it does not "
               "compare itself to the operation it replaces")
            # (7) THE GAP IS MEASURED, NOT ASSUMED.
            ok(isinstance(rep["importers_holding_a_direct_reference"]
                          .get("dr_under"), int),
               f"THE PROXY'S BLIND SPOT IS COUNTED: "
               f"{rep['importers_holding_a_direct_reference']} importers "
               f"already hold a direct reference, so a reader knows how "
               f"much the attribute record can miss")
            # (8) REFUSES a module it cannot watch.
            try:
                with Trace(watch=("dr_not_imported",)):
                    pass
            except TraceRefused as e:
                ok("TRACE_WATCH_MODULE_NOT_IMPORTED" in str(e),
                   "REFUSES to 'watch' a module that is not imported, "
                   "rather than silently observing nothing")
            else:
                ok(False, "a module that cannot be watched DID NOT REFUSE")
            # (9) THE TRACE IS TORN DOWN.
            ok(sys.gettrace() is None
               and not isinstance(sys.modules["dr_under"], _ModuleProxy),
               "TEARDOWN: the tracer is removed and every proxied module "
               "is restored -- a cell that leaves a proxy in sys.modules "
               "poisons every cell after it")
        finally:
            sys.path.remove(str(root))
            for m in ("dr_seed", "dr_over", "dr_under"):
                sys.modules.pop(m, None)

    print(f"[de_dynamic_reach] {'PASS' if not fails else 'FAIL'} -- "
          f"{len(fails)} failing")
    for f in fails:
        print(f"    FAILED: {f}")
    return 1 if fails else 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--falsify" in argv:
        return falsify()
    print(__doc__)
    print("usage: de_dynamic_reach.py --falsify")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
