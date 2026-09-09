"""THE DAY'S COLLECTOR ERA, RESOLVED FROM THE DAY.

WHY THIS MODULE EXISTS (BE 113, gate item 1). Both of this seat's day-scoped
selectors named their era like this:

    era = HER._era_or_refuse(fi, None, "<caller>")

and `_era_or_refuse` with `era=None` returns `fi.ERA`, a MODULE-LEVEL
LITERAL pinned to `clob_v3_1`. It is day-independent: it answers the same
thing for a day it was never given. The literal's own era closed
**2026-08-30T05:30:01Z**, and every September day lies entirely inside
**clob_v4_1** (2026-08-31T22:00:02Z -> now). The two gap tables are
DISJOINT -- measured, 1,143 slugs against 728, ZERO in common -- so
`fi.gaps_by_slug("clob_v3_1")` knows nothing about any September window and
`gaps.get(slug, [])` returned `[]` for every one of them.

WHAT THAT COST, on 09-03, measured rather than argued: **160 of the day's
247 supplied windows carry gaps under the day's own era, totalling 2,294.7 s
of tape, and ALL 160 were handed to `build_reference` as `[]`** -- assembled
as if continuous. DA 141 measured that 60.3 % of that day's settled money
sits in those windows. Every book receipt on disk records
`selection.era: clob_v3_1`; all twelve of them, on September days.

THE SHAPE OF THE FIX, and why it is a new file rather than an edit to the
era module. `harmful_exposure_rows.py` and `flow_intensity.py` are BOTH in
`fit_manifest.json`'s `fit_code_files`, so `de_phase4_diag_runner.
pin_statuses` watches them: a CALLED function whose bytes move and which
nobody declared additive is **BLOCKING**, and `verify_called_code()` refuses
the run. Editing `_era_or_refuse` would therefore stop every day run --
R-835's class of blocker -- and it would also falsify that function's own
DECLARED_ADDITIVE reason, which says in as many words that it "names the era
from `fi.ERA`". So the era module keeps its exact bytes and its declaration
stays true: the day is resolved HERE, and the resolved era is PASSED to
`_era_or_refuse`, whose `None` branch the day path no longer takes.

A DEFAULT THAT ANSWERS FOR A DAY IT WAS NEVER GIVEN IS NOT A DEFAULT, IT IS
A GUESS. This module has no default. Every outcome it cannot resolve is a
REFUSAL BY NAME with the counts that produced it:

  NO_ERA_SPANS_IN_THE_LEDGER   the collector ledger yields no start/stop
                               spans at all -- an empty era table would
                               otherwise make every window "in no era"
  NO_WINDOWS                   an empty population resolves nothing, and
                               "the era of no windows" is not a fact
  WINDOW_START_UNPARSEABLE     a slug carrying no epoch cannot be placed
  WINDOWS_IN_NO_ERA            windows the collector was not running for
                               (real: 08-31 has 201 of 288)
  DAY_STRADDLES_ERA_BOUNDARY   more than one era across the day, so no
                               single `gaps_by_slug(era)` describes it
                               (real: 08-30 is 66 v3_1 / 221 v4 / 1 none)
  WINDOW_IN_MORE_THAN_ONE_ERA  overlapping spans of different versions

THE PREDICATE IS `flow_intensity.covered_slugs`'s OWN: a window lies in an
era iff `a <= ws and ws + WINDOW_S <= b`. It is not re-invented here -- a
second definition of "in the era" is a second era.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


class EraUnresolved(RuntimeError):
    """A named refusal. There is no default to fall back to."""


def era_spans(fi) -> list:
    """`fi._eras()` in SECONDS, with the version, sorted.

    `fi` is a parameter and not an import so the falsifier can drive every
    refusal against a stand-in -- the same injection point BE48 §B.2 added
    to `day_slugs` after finding a refusal with zero driven coverage."""
    return sorted((a / 1e9, b / 1e9, v) for a, b, v in fi._eras())


def resolve(fi, day: str, slugs) -> dict:
    """THE DAY'S ERA, or a refusal by name. Returns the evidence (rule 10).

    `slugs` is the population the era will FILTER GAPS FOR -- the caller's
    own `want` -- and not the day's raw window list, because an era resolved
    over one population and applied to another is a third thing."""
    slugs = sorted(slugs)
    spans = era_spans(fi)
    if not spans:
        raise EraUnresolved(
            f"REFUSED -- NO_ERA_SPANS_IN_THE_LEDGER: {getattr(fi, 'GAPS', '?')}"
            f" yields no collector_start/collector_stop pair, so every window "
            f"of {day} would resolve to no era and the day would look "
            f"uncollected. An absent ledger is not an empty era.")
    if not slugs:
        raise EraUnresolved(
            f"REFUSED -- NO_WINDOWS: {day} supplied no window to resolve an "
            f"era from. 'The era of no windows' is not a fact, and a default "
            f"here would be a guess about a day nobody looked at.")
    w = float(getattr(fi, "WINDOW_S", 300.0))
    by_era: dict = {}
    no_era: list = []
    multi: list = []
    for s in slugs:
        try:
            ws = int(str(s).rsplit("-", 1)[1])
        except (IndexError, ValueError):
            raise EraUnresolved(
                f"REFUSED -- WINDOW_START_UNPARSEABLE: slug {s!r} of {day} "
                f"carries no trailing epoch, so it cannot be placed in any "
                f"era span.") from None
        hit = sorted({v for a, b, v in spans if a <= ws and ws + w <= b})
        if not hit:
            no_era.append(s)
        elif len(hit) > 1:
            multi.append((s, hit))
        else:
            by_era.setdefault(hit[0], []).append(s)
    counts = {k: len(v) for k, v in sorted(by_era.items())}
    if multi:
        raise EraUnresolved(
            f"REFUSED -- WINDOW_IN_MORE_THAN_ONE_ERA: {len(multi)} window(s) "
            f"of {day} lie inside spans of DIFFERENT collector versions, "
            f"first {multi[0][0]} in {multi[0][1]}. Overlapping eras make "
            f"`gaps_by_slug(era)` ambiguous for that window.")
    # ORDER MATTERS AND BOTH FACTS TRAVEL. A day can carry BOTH faults --
    # 08-30 is 66 clob_v3_1, 221 clob_v4 AND one window in no span -- so
    # whichever name is raised, the other count is in the message. The
    # STRADDLE is named first because it is the more fundamental fact: a
    # hole leaves some windows undescribed, while a straddle means no
    # single `gaps_by_slug(era)` describes the day at all.
    if len(counts) > 1:
        raise EraUnresolved(
            f"REFUSED -- DAY_STRADDLES_ERA_BOUNDARY: {day} spans "
            f"{len(counts)} collector eras {counts} (and {len(no_era)} "
            f"window(s) in no span at all). A selector applies ONE "
            f"`gaps_by_slug(era)` to every window, so a straddling day has "
            f"no single era to be built under -- naming one would status "
            f"part of the day with another era's gap table.")
    if no_era:
        raise EraUnresolved(
            f"REFUSED -- WINDOWS_IN_NO_ERA: {len(no_era)} of {len(slugs)} "
            f"windows of {day} lie inside NO collector span (first "
            f"{no_era[0]}); the rest resolve {counts}. The collector was not "
            f"running for the whole of those windows, so no era's gap table "
            f"describes them and the day is refused rather than built under "
            f"whichever era happens to cover the majority.")
    era = next(iter(counts))
    g = getattr(fi, "GAPS", None)
    return {
        "protocol": "BE_ERA_FOR_DAY_V1",
        "day": day,
        "era": era,
        "resolved_from": "the day's OWN supplied windows against the "
                         "collector ledger's start/stop spans",
        "predicate": "a window lies in an era iff a <= ws and ws + WINDOW_S "
                     "<= b -- flow_intensity.covered_slugs's own predicate, "
                     "not a second definition",
        "window_s": w,
        "n_windows": len(slugs),
        "n_windows_in_the_era": counts[era],
        "n_windows_by_era": counts,
        "n_era_spans_in_the_ledger": len(spans),
        # A LIVE ERA HAS NO END, AND `float("inf")` IS NOT JSON. `_eras()`
        # gives the open span `inf`, and `json.dumps` writes the literal
        # `Infinity`, which every strict reader rejects -- and a receipt is
        # exactly what automated readers resolve (rule 13). The open end is
        # None with a field that says so, rather than a token that parses
        # in Python and nowhere else.
        "era_span_bounds_s": [[a, (None if b == float("inf") else b)]
                              for a, b, v in spans if v == era],
        "era_span_open_ended": any(b == float("inf")
                                   for a, b, v in spans if v == era),
        "ledger": (None if g is None else
                   {"path": str(g),
                    "bytes": (Path(g).stat().st_size
                              if Path(g).exists() else None)}),
        # THE DEFAULT THAT WAS NOT USED, RECORDED. On every September day
        # these two differ, and that difference IS the defect -- so a
        # receipt carrying both says what the book would have been built
        # under instead of leaving a reader to know it.
        "module_default_NOT_used": {
            "fi.ERA": getattr(fi, "ERA", None),
            "agrees_with_the_resolved_era": getattr(fi, "ERA", None) == era,
            "why_it_is_recorded":
                "`harmful_exposure_rows._era_or_refuse(fi, None, ...)` "
                "returns this literal day-independently. Where it differs "
                "from the resolved era, a book built through the default "
                "carried NO gaps for this day.",
        },
        "decides_nothing": "REPORTED. The caller passes this era to "
                           "`_era_or_refuse` and to `gaps_by_slug` (rule 14).",
    }


def era_for_day(fi, day: str, slugs) -> str:
    """The era alone, for a caller that wants no evidence block."""
    return resolve(fi, day, slugs)["era"]


# ---------------------------------------------------------------------------
# THE FALSIFIER
# ---------------------------------------------------------------------------

class _FakeFi:
    """A stand-in `fi`, so every refusal is reachable (BE48 §B.2)."""
    WINDOW_S = 300.0
    ERA = "clob_v3_1"
    GAPS = None

    def __init__(self, spans):
        self._spans = list(spans)

    def _eras(self):
        return [(a * 1e9, b * 1e9, v) for a, b, v in self._spans]


EXPECTED_CHECKS = 18


def falsify() -> int:
    checks, fails = 0, []

    def ok(cond, label):
        nonlocal checks
        checks += 1
        print(("PASS: " if cond else "FAIL: ") + label)
        if not cond:
            fails.append(label)

    def refuses(fn, needle, label):
        nonlocal checks
        checks += 1
        try:
            fn()
            print("FAIL: " + label + " (did not refuse)")
            fails.append(label)
            return
        except EraUnresolved as e:
            hit = needle in str(e)
            print(("PASS: " if hit else "FAIL: ") + label)
            if not hit:
                fails.append(f"{label} (wrong name: {e})")

    # ---- 1. SYNTHETIC: every refusal, by name -------------------------
    #: two clean eras, and a hole between 2000 and 3000
    F = _FakeFi([(0.0, 1000.0, "eraA"), (2000.0, 3000.0, "eraB")])
    inA = ["x-updown-5m-100", "x-updown-5m-400"]
    inB = ["x-updown-5m-2100"]
    hole = ["x-updown-5m-1500"]
    refuses(lambda: resolve(_FakeFi([]), "d", inA),
            "NO_ERA_SPANS_IN_THE_LEDGER",
            "KNOWN-BAD: an empty era table REFUSES -- otherwise every window "
            "resolves to no era and a collected day reads as uncollected")
    refuses(lambda: resolve(F, "d", []), "NO_WINDOWS",
            "KNOWN-BAD: an empty population REFUSES -- 'the era of no "
            "windows' is not a fact")
    refuses(lambda: resolve(F, "d", ["not-a-slug"]),
            "WINDOW_START_UNPARSEABLE",
            "KNOWN-BAD: a slug with no trailing epoch REFUSES by name")
    refuses(lambda: resolve(F, "d", inA + hole), "WINDOWS_IN_NO_ERA",
            "KNOWN-BAD: a window the collector was not running for REFUSES, "
            "rather than the day being built under whichever era covers the "
            "majority")
    refuses(lambda: resolve(F, "d", inA + inB), "DAY_STRADDLES_ERA_BOUNDARY",
            "KNOWN-BAD: a day spanning two eras REFUSES -- one selector "
            "applies ONE gaps_by_slug(era) to every window")
    refuses(lambda: resolve(_FakeFi([(0.0, 1000.0, "eraA"),
                                     (0.0, 1000.0, "eraB")]), "d", inA),
            "WINDOW_IN_MORE_THAN_ONE_ERA",
            "KNOWN-BAD: overlapping spans of different versions REFUSE -- "
            "`gaps_by_slug(era)` would be ambiguous for that window")
    r = resolve(F, "d", inA)
    ok(r["era"] == "eraA" and r["n_windows"] == r["n_windows_in_the_era"] == 2
       and r["module_default_NOT_used"]["fi.ERA"] == "clob_v3_1"
       and r["module_default_NOT_used"]["agrees_with_the_resolved_era"]
       is False,
       f"POSITIVE CONTROL: a clean population RESOLVES ({r['era']}, "
       f"{r['n_windows_in_the_era']}/{r['n_windows']} windows) and RECORDS "
       f"the module default it did not use "
       f"({r['module_default_NOT_used']['fi.ERA']}) -- a guard shown only to "
       f"refuse has proved half of itself (rule 16)")
    ok(resolve(F, "d", ["x-updown-5m-700"])["era"] == "eraA",
       "and the predicate is the INCLUSIVE one flow_intensity uses: a window "
       "starting at 700 with WINDOW_S 300 ends exactly at the span's end and "
       "is IN the era")
    refuses(lambda: resolve(F, "d", ["x-updown-5m-701"]), "WINDOWS_IN_NO_ERA",
            "while one starting a second later runs past the end and is NOT "
            "-- the boundary is driven, not assumed")

    def _strict(o):
        """json round-trip that REFUSES the non-standard constants."""
        def boom(x):
            raise ValueError(f"non-standard JSON constant {x!r}")
        return json.loads(json.dumps(o, allow_nan=False), parse_constant=boom)

    ok(_strict(resolve(F, "d", inA))["era"] == "eraA",
       "and the evidence block survives a STRICT json round-trip "
       "(allow_nan=False, parse_constant raising) -- a live era's span end "
       "is `inf` in `_eras()` and `Infinity` is not JSON, so the open end "
       "is None with `era_span_open_ended` beside it")
    _openF = _FakeFi([(0.0, float("inf"), "eraA")])
    _ro = resolve(_openF, "d", inA)
    ok(_ro["era_span_bounds_s"] == [[0.0, None]]
       and _ro["era_span_open_ended"] is True
       and _strict(_ro)["era"] == "eraA",
       f"POSITIVE CONTROL on the open span itself: an era with no end "
       f"reports {_ro['era_span_bounds_s']} and open_ended="
       f"{_ro['era_span_open_ended']}, and still round-trips strictly -- "
       f"which is the case every September day is actually in")

    # ---- 2. THE REAL DAYS, BOTH DIRECTIONS ---------------------------
    real = {"ran": False}
    try:
        import flow_intensity as fi
        import be_daybook_build as B
        real["ran"] = True
    except Exception as e:                                   # noqa: BLE001
        for _ in range(7):
            ok(False, f"the real-day cells could not run: "
                      f"{type(e).__name__}: {e}")
    if real["ran"]:
        s3 = B.day_slugs("20260903", "btc")
        r3 = resolve(fi, "20260903", s3)
        ok(r3["era"] == "clob_v4_1" and r3["n_windows"] == 247
           and r3["n_windows_in_the_era"] == 247
           and r3["module_default_NOT_used"]["fi.ERA"] == "clob_v3_1",
           f"09-03 RESOLVES {r3['era']} over {r3['n_windows_in_the_era']}/"
           f"{r3['n_windows']} supplied windows, against the module default "
           f"{r3['module_default_NOT_used']['fi.ERA']} that every book on "
           f"disk was built under")
        g_def = fi.gaps_by_slug(r3["module_default_NOT_used"]["fi.ERA"])
        g_day = fi.gaps_by_slug(r3["era"])
        n_def = sum(1 for s in s3 if g_def.get(s))
        n_day = sum(1 for s in s3 if g_day.get(s))
        t_day = sum(b - a for s in s3 for a, b in g_day.get(s, ()))
        ok(n_def == 0 and n_day == 160 and abs(t_day - 2294.7) < 0.1,
           f"AND THIS IS WHAT THE DEFAULT COST, DRIVEN: under the module "
           f"default {n_def} of {len(s3)} windows carry gaps; under the day's "
           f"own era {n_day} do, totalling {t_day:.1f} s of tape. Those 160 "
           f"were handed to `build_reference` as [] and assembled as if "
           f"continuous")
        ok(len(set(g_def) & set(g_day)) == 0
           and len(g_def) == 1143 and len(g_day) >= 728,
           f"the two gap tables are DISJOINT -- {len(g_def)} slugs against "
           f"{len(g_day)}, {len(set(g_def) & set(g_day))} in common -- which "
           f"is why the default could not be partly right")
        for day, needle, why in (
                ("20260826", "WINDOWS_IN_NO_ERA",
                 "windows the collector was not running for the whole of"),
                ("20260830", "DAY_STRADDLES_ERA_BOUNDARY",
                 "it is 66 clob_v3_1 against 221 clob_v4, with 1 window in "
                 "no span besides -- and BOTH counts are in the message")):
            try:
                sl = B.day_slugs(day, "btc")
            except Exception:                                # noqa: BLE001
                ok(False, f"{day} could not be supplied, so its refusal "
                          f"could not be driven")
                continue
            refuses(lambda sl=sl, day=day: resolve(fi, day, sl), needle,
                    f"REAL KNOWN-BAD: {day} REFUSES as {needle} -- {why}. "
                    f"The module default answers `clob_v3_1` for it without "
                    f"looking")
        # ---- THE CENSUS: every collected day through the predicate ----
        # A refusal that fires on everything is not a predicate, and one
        # that fires on nothing is not a control. This runs the whole day
        # population and reports both sides.
        cen = {}
        for d in fi.DAYS:
            try:
                sl = B.day_slugs(d, "btc")
            except Exception:                                # noqa: BLE001
                continue
            try:
                cen[d] = resolve(fi, d, sl)["era"]
            except EraUnresolved as e:
                cen[d] = str(e).split(":")[0].replace("REFUSED -- ", "")
        sept = {d: v for d, v in cen.items() if d >= "20260901"}
        aug_ok = {d: v for d, v in cen.items()
                  if d < "20260901" and v.startswith("clob_")}
        ok(len(sept) >= 8 and set(sept.values()) == {"clob_v4_1"}
           and set(aug_ok.values()) == {"clob_v3_1"},
           f"THE CENSUS OVER EVERY COLLECTED DAY: all {len(sept)} September "
           f"days resolve clob_v4_1, and every August day that resolves at "
           f"all resolves clob_v3_1 ({len(aug_ok)} of them). **THE MODULE "
           f"DEFAULT WAS RIGHT FOR THE DAYS IT WAS WRITTEN FOR AND WRONG FOR "
           f"EVERY DAY IN THE QUEUE** -- which is exactly how a literal "
           f"survives review and then goes stale under a running collector")
        ok(sorted(d for d, v in cen.items() if not v.startswith("clob_"))
           == ["20260819", "20260820", "20260821", "20260826", "20260830",
               "20260831"],
           f"and it refuses SIX days of {len(cen)}, each a real collector "
           f"outage or era transition -- a predicate that refused everything "
           f"would pass this file's other cells just as well")

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
    if "--resolve" in argv:
        import flow_intensity as fi
        import be_daybook_build as B
        day = argv[argv.index("--resolve") + 1]
        print(json.dumps(resolve(fi, day, B.day_slugs(day, "btc")), indent=1))
        return 0
    print("usage: be_era_for_day.py --falsify | --resolve <YYYYMMDD>")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
