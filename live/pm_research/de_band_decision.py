"""THE DECISION ARTIFACT: one document, every number attributed.

This is the state a person decides on before spending fourteen nights.
It is built here rather than assembled from four seats' filings, and it
carries one discipline the assembling would have lost:

  EVERY NUMBER CARRIES THE POPULATION AND THE CRITERION THAT PRODUCED
  IT, and that is ENFORCED -- `assert_attributed` walks the document and
  refuses any bare numeric leaf.

The reason is specific rather than general. Tonight's largest error was a
pass rate quoted WITHOUT its criterion: 10 of 11 is true of raw-tape
window-file presence and false of BE's own population gate, which scores
7 of 11 over the same days. It survived three retellings. In a document
someone decides on, that error would be permanent.

Nothing here recommends. Several levers are amendments to a user-authored
plan and none of them is this seat's to choose; the one factual property
worth placing beside them -- that improving the input is the only lever
which does not trade the test's strength for its feasibility -- is stated
as a property, not as advice.

Usage:  de_band_decision.py --falsify
        de_band_decision.py --emit [--out PATH]
"""
from __future__ import annotations

CALL_SITE = {
    "kind": "ARTIFACT_MEDIATED",
    "by": "the decision is read from the artifact this emits",
    "artifact": "band_decision.json",
    "gates": "its own emit -- assert_attributed refuses a bare number, so an unattributed rate cannot reach the artifact a person decides from",
}

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import de_band_hazard as HZ                            # noqa: E402

PROTOCOL = "P003_DE_BAND_DECISION_V1"
UNATTRIBUTED = "NUMBER_WITHOUT_ITS_POPULATION_AND_CRITERION"

#: The criteria, named once. A rate is meaningless without one of these
#: beside it, and two of them disagree on the same eleven days.
C_GATE = ("BE gate: population(day) interior-missing windows <= 1, btc "
          "(be_build_preflight.check_day window-supply row)")
C_TAPE = ("raw-tape 5-minute window FILES present, btc AND eth "
          "(data/pm_5min/raw/<day>/)")
C_UNVERIFIED = ("UNVERIFIED -- reported by another seat; its definition "
                "was not found in any landed artifact (COORDINATION.md, "
                "HANDOFF.md, RESULTS.md, reviews/)")
C_BINOMIAL = ("exact binomial P(X >= 10 | n = 14, p), no simulation")
C_LADDER = ("the frozen de_fair_value_predictive ladder and MIN_NONZERO")

P_11 = "09-01..09-11, 11 consecutive UTC days"
P_8 = "09-04..09-11, 8 consecutive UTC days"
P_BAND = "a hypothetical 14-day band, 10 evaluable required"


#: WHERE THE ETH MEASUREMENT WILL LAND. Read by identity, newest version
#: first; absent means PROJECTED, never a quiet default.
ETH_DECL_GLOB = "be_eth_build_cost_v*.json"
ETH_RATE_KEYS = ("eth_day_success_rate", "eth_pass_rate",
                 "eth_build_success_rate")
ETH_SECONDS_KEYS = ("eth_book_stage_seconds", "eth_build_seconds",
                    "eth_stage_s", "eth_wall_s")
#: The two scenarios the coordinator has been pricing. They are
#: ASSUMPTIONS, and nothing measured stands behind them.
PROJECTED_ETH_RATES = (0.95, 0.90)
UNLABELLED_ETH = "ETH_DEPENDENT_FIGURE_IS_UNLABELLED"
COLLAPSED_RATIO = "ETH_COST_COLLAPSED_INTO_ONE_RATIO"
BARE_BOOLEAN = "A_VERDICT_BOOLEAN_WITHOUT_ITS_BASIS"
#: A headroom smaller than this fraction of the limit is NOISE, not a
#: margin. Declared before the measurement was read, not chosen around it.
MARGIN_IS_NOISE_BELOW = 0.10
#: Verdict keys must carry a basis. A bare boolean in one of these
#: positions is a two-valued field standing in for a state that needs
#: three -- comfortably, marginally under a soft cap, no.
VERDICT_KEY = re.compile(
    r"^(fits|can_|could_|is_safe|safe_to)|(_passes|_clears|_is_ok)$")
COST_AS_RATE = "A_BUILD_COST_IS_NOT_A_SUCCESS_RATE"
PARTIAL = "PARTIALLY_MEASURED"
OWED = "OWED"
ETH_STAGE_DECL = "de_eth_stage_costs_v*.json"
#: THE THREE QUANTITIES, kept apart on purpose. The finding is that they
#: do NOT travel together -- size scales at ~0.66 on both stages, time at
#: 0.27 then 0.45 -- so a single "eth is X% of btc" figure would erase
#: the finding it summarises.
QUANTITIES = ("wall_s", "peak_rss_bytes", "output_bytes")
PROJECTED = "PROJECTED"
MEASURED = "MEASURED"


class DecisionRefused(ValueError):
    """The artifact cannot be emitted as declared."""


def M(value, *, population: str, criterion: str, source: str,
      as_of: str = None, note: str = None, provenance: str = None) -> dict:
    """ONE NUMBER, WITH THE TWO THINGS THAT GIVE IT MEANING."""
    out = {"value": value, "population": population,
           "criterion": criterion, "source": source}
    if provenance:
        out["provenance"] = provenance
    if as_of:
        out["as_of"] = as_of
    if note:
        out["note"] = note
    return out


def V(value: bool, *, basis: str, n_observations, limit_kind: str = None,
      source: str = "") -> dict:
    """A VERDICT, WITH THE BASIS THAT MAKES IT READABLE.

    `fits: true` is a bare boolean: it cannot say whether it fits
    comfortably or by less than the noise, nor whether the limit it fits
    under KILLS or merely throttles. A later reader finds the true and
    plans a night on it.
    """
    return {"verdict": bool(value), "basis": basis,
            "n_observations": n_observations,
            "limit_kind": limit_kind, "source": source,
            "A_BOOLEAN_WITHOUT_THIS_IS_NOT_READABLE": True}


def _is_verdict(node) -> bool:
    return (isinstance(node, dict)
            and {"verdict", "basis", "n_observations"} <= set(node))


def assert_verdicts_carry_their_basis(doc, path: str = "$") -> int:
    """EVERY VERDICT BOOLEAN CARRIES ITS BASIS, or this refuses.

    The attribution guard checked that NUMBERS carry their population and
    criterion and let a bare `fits: true` through -- the same damage in a
    different type (DE 401).
    """
    n = 0
    if _is_verdict(doc):
        return 1
    if isinstance(doc, dict):
        for k, v in doc.items():
            if isinstance(v, bool) and VERDICT_KEY.search(k):
                raise DecisionRefused(
                    f"REFUSED {BARE_BOOLEAN}: {path}.{k} = {v} is a "
                    f"verdict with no basis. Two values cannot express "
                    f"comfortably / marginally-under-a-soft-limit / no, "
                    f"and a reader who finds the true will plan on it.")
            n += assert_verdicts_carry_their_basis(v, f"{path}.{k}")
        return n
    if isinstance(doc, (list, tuple)):
        for i, v in enumerate(doc):
            n += assert_verdicts_carry_their_basis(v, f"{path}[{i}]")
    return n


def _is_measure(node) -> bool:
    return (isinstance(node, dict)
            and {"value", "population", "criterion"} <= set(node))


def assert_attributed(doc, path: str = "$") -> int:
    """EVERY NUMERIC LEAF SITS INSIDE A MEASURE, or this refuses.

    Booleans are not measurements and are allowed; a number is not.
    """
    n = 0
    if _is_measure(doc):
        return 1
    if isinstance(doc, dict):
        for k, v in doc.items():
            n += assert_attributed(v, f"{path}.{k}")
        return n
    if isinstance(doc, (list, tuple)):
        for i, v in enumerate(doc):
            n += assert_attributed(v, f"{path}[{i}]")
        return n
    if isinstance(doc, bool) or doc is None or isinstance(doc, str):
        return 0
    if isinstance(doc, (int, float)):
        raise DecisionRefused(
            f"REFUSED {UNATTRIBUTED}: {path} = {doc!r} is a bare number. "
            f"A rate without its criterion is the error that survived "
            f"three retellings tonight -- 10 of 11 is true of tape "
            f"window-file presence and false of BE's population gate on "
            f"the same eleven days.")
    return n


#: THE AMENDMENT DECLARATIONS this artifact expects, by lever. Naming
#: them here is what gives the guard something to date: an amendment is
#: dated by the COMMIT that lands its file, so the file must be named
#: before it can be dated.
AMENDMENT_FILES = {
    "iii_longer_band":
        "live/pm_research/declarations/de_amendment_longer_band_v1.json",
    "iv_fewer_required_days":
        "live/pm_research/declarations/de_amendment_fewer_days_v1.json",
    "v_start_after_a_clean_run":
        "live/pm_research/declarations/de_amendment_clean_start_v1.json",
}


#: Modules that MENTION tier2 but do not read it (a docstring naming the
#: rule is not a read). Named explicitly so the coverage figure is not
#: quietly flattered, and so the list itself is auditable.
TIER2_PROSE_ONLY = ("de_fair_value_plumbing_run.py",)


def eth_measurement(decl_dir: Path = None) -> dict:
    """THE ETH COST AS A PARAMETER, NOT A LITERAL.

    Read from BE's measurement when it lands; PROJECTED until then, and
    the difference is carried on every figure that depends on it. A
    projected number sitting unlabelled among measured ones is the worst
    place for it.
    """
    d = Path(decl_dir or (HERE / "declarations"))
    got = {}
    for f in sorted(d.glob(ETH_DECL_GLOB), reverse=True):
        try:
            doc = json.loads(f.read_text())
        except Exception:                                   # noqa: BLE001
            continue
        flat = json.dumps(doc)
        for keys, name in ((ETH_RATE_KEYS, "rate"),
                           (ETH_SECONDS_KEYS, "seconds")):
            for k in keys:
                if f'"{k}"' in flat:
                    node, stack = None, [doc]
                    while stack:
                        cur = stack.pop()
                        if isinstance(cur, dict):
                            if k in cur and isinstance(cur[k], (int, float)):
                                node = cur[k]
                                break
                            stack.extend(v for v in cur.values()
                                         if isinstance(v, (dict, list)))
                        elif isinstance(cur, list):
                            stack.extend(cur)
                    if node is not None:
                        got[name] = {"value": float(node),
                                     "declared_by": f.name, "key": k}
                        break
        if got:
            break
    if got:
        return {"provenance": MEASURED, "measured": got,
                "scenarios": [got["rate"]["value"]] if "rate" in got
                else list(PROJECTED_ETH_RATES),
                "why": "read from BE's landed measurement"}
    return {"provenance": PROJECTED, "measured": None,
            "scenarios": list(PROJECTED_ETH_RATES),
            "expected_at": str((Path(decl_dir or (HERE / "declarations"))
                                / ETH_DECL_GLOB)),
            "why": "no ETH build-cost measurement has landed; these are "
                   "ASSUMPTIONS and nothing measured stands behind them",
            "what_is_measured_about_eth_today":
                "its INPUTS are at parity with btc and its data quality "
                "is better (DA's eth_input_audit); the BUILD cost and the "
                "build success rate are not measured"}


def eth_stage_costs(decl_dir: Path = None) -> dict:
    """THE BUILD COST, PER STAGE, PER QUANTITY -- three series, never one.

    A stage may be measured on one quantity and not another, and a run
    may have two stages measured and one still running. Both partial
    states are representable, because both are true right now.
    """
    d = Path(decl_dir or (HERE / "declarations"))
    docs = sorted(d.glob(ETH_STAGE_DECL), reverse=True)
    if not docs:
        return {"provenance": PROJECTED, "stages": {},
                "why": "no per-stage cost declaration has landed"}
    doc = json.loads(docs[0].read_text())
    src = docs[0].name
    stages = {}
    for stage, legs in (doc.get("stages") or {}).items():
        row = {}
        if legs.get("peak_rss_sum_bytes") is not None:
            row.setdefault("peak_rss_bytes", {})["sum"] = M(
                legs["peak_rss_sum_bytes"],
                population=f"{stage} stage, both coins",
                criterion="peak RSS SUM as reported; the per-leg split "
                          "was not supplied and is not derived here",
                source=src, provenance=MEASURED)
        for q in QUANTITIES:
            e = (legs.get("eth") or {}).get(q)
            b = (legs.get("btc") or {}).get(q)
            prov = (MEASURED if e is not None else
                    OWED if q == "peak_rss_bytes" else PROJECTED)
            cell = {"eth": M(e, population=f"eth {stage} stage",
                             criterion=f"measured wall/bytes, {src}",
                             source=src, provenance=prov) if e is not None
                    else {"value": None, "provenance": prov,
                          "population": f"eth {stage} stage",
                          "criterion": "not supplied"},
                    "btc": M(b, population=f"btc {stage} stage",
                             criterion=f"comparator, as_of "
                                       f"{doc.get('btc_comparators_as_of')}",
                             source=src, provenance=prov)
                    if b is not None else
                    {"value": None, "provenance": prov,
                     "population": f"btc {stage} stage",
                     "criterion": "not supplied"}}
            if e is not None and b:
                cell["ratio_eth_over_btc"] = M(
                    e / b, population=f"{stage} stage",
                    criterion=f"eth/btc on {q} ONLY -- APPROXIMATE, the "
                              f"btc comparator is as_of "
                              f"{doc.get('btc_comparators_as_of')}",
                    source=src, provenance=prov)
            if q in row and isinstance(row[q], dict):
                cell.update(row[q])
            row[q] = cell
        stages[stage] = row
    # A REFUSED RUN IS NOT A MEASURED STAGE. Its wall-clock and peak are
    # observations OF the refusal, and counting them would report a cost
    # for a stage that produced nothing.
    refused = set()
    outcome = doc.get("book_stage_outcome") or {}
    if outcome.get("status") == "REFUSED":
        refused.add(outcome.get("stage", "book"))
    measured = [st for st, r in stages.items()
                if r["output_bytes"]["eth"].get("value") is not None
                and st not in refused]
    prov = (MEASURED if len(measured) == len(stages) else
            PARTIAL if measured else PROJECTED)
    return {
        "provenance": prov,
        "stages_measured": measured,
        "stages_not_measured": [st for st in stages if st not in measured],
        "stages": stages,
        "THE_QUANTITIES_DO_NOT_TRAVEL_TOGETHER":
            doc.get("quantities_do_not_travel_together"),
        "archive_volume_ratio_reported": M(
            doc.get("archive_volume_ratio_reported"),
            population="archive volume", criterion="REPORTED by the "
            "coordinator, not recomputed by this seat", source=src,
            provenance=MEASURED),
        "MECHANISM": (lambda m: dict(
            m, reported_peak_ratio=M(
                m.get("reported_peak_ratio"),
                population="peak RSS, eth vs btc as reported",
                criterion="REPORTED by the coordinator; the stage it "
                          "belongs to is not stated, and the supplied "
                          "fragment pair 2.09/4.63 GiB gives 0.451, so "
                          "0.97 cannot be the fragment stage",
                source=src, provenance=MEASURED))
            if isinstance(m, dict) else m)(doc.get("MECHANISM")),
        "THE_CONCLUSION_HOLDS_WITHOUT_THE_BOOK_STAGE":
            doc.get("THE_CONCLUSION_HOLDS_WITHOUT_THE_BOOK_STAGE"),
        "book_stage_outcome": (lambda b: dict(
            {k: v for k, v in b.items()
             if not isinstance(v, (int, float)) or isinstance(v, bool)},
            rc=M(b.get("rc"), population="the eth book run on 09-05",
                 criterion="process exit code of a run that REFUSED",
                 source=src, provenance=MEASURED),
            wall_s=M(b.get("wall_s"),
                     population="the eth book run on 09-05",
                     criterion="wall-clock of a REFUSED run -- not the "
                               "stage's cost", source=src,
                     provenance=MEASURED),
            peak_rss_bytes=M(b.get("peak_rss_bytes"),
                             population="the eth book run on 09-05",
                             criterion="peak RSS of a REFUSED run -- not "
                                       "the stage's cost", source=src,
                             provenance=MEASURED))
            if isinstance(b, dict) else b)(doc.get("book_stage_outcome")),
        "COMPARATORS_MAY_BE_STALE": doc.get("COMPARATORS_MAY_BE_STALE"),
        "btc_comparators_as_of": doc.get("btc_comparators_as_of"),
        "peak_rss_is_owed": doc.get("peak_rss_is_owed"),
        "DAY_IS_CONSUMED_AND_THIS_IS_NOT_A_FINDING_ABOUT_IT":
            doc.get("DAY_IS_CONSUMED_AND_THIS_IS_NOT_A_FINDING_ABOUT_IT"),
        "declared_by": src,
    }


def night_budget(costs: dict) -> dict:
    """WALL-CLOCK ONLY, summed per coin, with the unmeasured stage named."""
    out, total = {}, {}
    for coin in ("eth", "btc"):
        secs, missing = 0.0, []
        ok_stages = set(costs.get("stages_measured") or [])
        for stage, row in costs.get("stages", {}).items():
            v = row["wall_s"][coin].get("value")
            # A stage that REFUSED contributes no cost, even though its
            # run has a wall-clock: the stage did not happen.
            if v is None or stage not in ok_stages:
                missing.append(stage)
            else:
                secs += float(v)
        out[coin] = {"measured_stage_seconds": M(
            secs, population=f"{coin} day build",
            criterion="sum of MEASURED stage wall-clock only",
            source=costs.get("declared_by", "?"),
            provenance=MEASURED if not missing else PARTIAL),
            "stages_missing": missing}
        total[coin] = secs
    serial = total["eth"] + total["btc"]
    missing = sorted({st for o in out.values() for st in o["stages_missing"]})
    return {"per_coin": out,
            "BUDGET_IS_A_RANGE_NOT_A_POINT": {
                "lower_bound_seconds": M(
                    serial, population="one night, both coins",
                    criterion="SERIAL sum of MEASURED stages -- a LOWER "
                              "BOUND, because the unmeasured stage adds "
                              "to it and cannot subtract",
                    source=costs.get("declared_by", "?"),
                    provenance=PARTIAL),
                "upper_bound_seconds": None,
                "unmeasured_stages": missing,
                "why_no_upper_bound":
                    "the book stage is not measured, so no upper bound "
                    "exists that is not an estimate; naming the hole "
                    "beats filling it with a guess",
                "and_it_is_the_largest_stage":
                    "the night budget has its hole in the stage that "
                    "historically costs the most, so the lower bound is "
                    "not close to the answer"},
            "SERIAL_BY_MEMORY_NOT_ONLY_BY_LOCK":
                "if the tape peaks cannot share the slice, stages cannot "
                "be interleaved to shorten a night: the nightly cost is "
                "the SUM of the coins, not the MAX",
            "serial_seconds_measured_stages_only": M(
                serial, population="one night, both coins",
                criterion="SERIAL sum of measured stages only; the "
                          "unmeasured stage is not estimated",
                source=costs.get("declared_by", "?"),
                provenance=PARTIAL if any(
                    o["stages_missing"] for o in out.values()) else MEASURED),
            "IS_NOT_A_TOTAL": "stages that have not finished are absent, "
                              "not zero and not projected"}


GIB = 1024 ** 3


def _three_state(total: int, high: int) -> str:
    """THREE STATES, because two cannot say what this one needs to."""
    if not high:
        return "UNRESOLVED"
    if total > high:
        return "OVER_THE_SOFT_LIMIT"
    frac = (high - total) / high
    if frac >= MARGIN_IS_NOISE_BELOW:
        return f"COMFORTABLY_UNDER_BY_{frac * 100:.1f}_PERCENT"
    return (f"FITS_SOFT_LIMIT_BY_{frac * 100:.1f}_PERCENT_ON_SINGLE_"
            f"OBSERVATIONS")


def memory_threshold() -> dict:
    """THE THRESHOLD, READ FROM THE SYSTEM, not from a remembered
    constant. "The 12 GB rule" is a phrase; systemd holds a number."""
    import subprocess
    out = subprocess.run(
        ["systemctl", "--user", "show", "research.slice",
         "-p", "MemoryHigh", "-p", "MemoryMax"],
        capture_output=True, text=True).stdout
    got = {}
    for line in out.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            try:
                got[k] = int(v)
            except ValueError:
                got[k] = None
    crit = "read live from systemd, not a remembered constant"
    pop = "research.slice"
    return {"MemoryHigh_bytes": M(got.get("MemoryHigh"), population=pop,
                                  criterion=crit, source="systemctl",
                                  provenance=MEASURED),
            "MemoryMax_bytes": M(got.get("MemoryMax"), population=pop,
                                 criterion=crit, source="systemctl",
                                 provenance=MEASURED),
            "MemoryHigh_GiB": M((got["MemoryHigh"] / GIB
                                 if got.get("MemoryHigh") else None),
                                population=pop, criterion=crit,
                                source="systemctl", provenance=MEASURED),
            "MemoryMax_GiB": M((got["MemoryMax"] / GIB
                                if got.get("MemoryMax") else None),
                               population=pop, criterion=crit,
                               source="systemctl", provenance=MEASURED),
            "read_from": "systemctl --user show research.slice",
            "MemoryHigh_is_a_throttle_MemoryMax_is_the_kill": True}


def overlap_question(costs: dict) -> dict:
    """CAN THE TWO COINS SHARE A NIGHT? Decided by PEAK RSS against the
    threshold the system actually holds -- and the UNIT decides it."""
    thr = memory_threshold()
    stages = costs.get("stages", {})
    rows = {}
    for stage, row in stages.items():
        cell = row.get("peak_rss_bytes", {})
        e = cell.get("eth", {}).get("value")
        b = cell.get("btc", {}).get("value")
        total = cell.get("sum", {}).get("value")
        if total is None and e is not None and b is not None:
            total = e + b
        if total is None:
            rows[stage] = {"verdict": "UNRESOLVED",
                           "why": "no peak supplied for this stage"}
            continue
        high = (thr.get("MemoryHigh_bytes") or {}).get("value")
        rows[stage] = {
            "both_coins_peak_sum_bytes": M(
                total, population=f"{stage} stage, both coins",
                criterion="sum of peak RSS; two units in one slice share "
                          "its threshold", source=costs.get("declared_by"),
                provenance=MEASURED),
            "fits_under_MemoryHigh": V(
                (total <= high) if high else False,
                basis=("MemoryHigh is a SOFT limit: it throttles through "
                       "reclaim, it does not kill. The hard kill is "
                       "MemoryMax at 14 GiB, and each unit is separately "
                       "capped at 8 GiB"),
                n_observations="1 day per coin -- single observations, so "
                               "the interval on this headroom is wider "
                               "than the headroom",
                limit_kind="SOFT_THROTTLE",
                source="systemctl x the declaration"),
            "headroom_bytes": M((high - total) if high else None,
                                population=f"{stage}, both coins",
                                criterion="MemoryHigh minus the peak sum",
                                source="systemctl x the declaration",
                                provenance=MEASURED),
            "headroom_fraction_of_threshold": M(
                ((high - total) / high if high else None),
                population=f"{stage}, both coins",
                criterion="headroom as a fraction of MemoryHigh",
                source="systemctl x the declaration",
                provenance=MEASURED),
            "verdict": _three_state(total, high),
            "RECOMMENDATION": (
                "DO NOT PLAN TO OVERLAP" if high and total <= high
                and (high - total) / high < MARGIN_IS_NOISE_BELOW
                else "DO NOT PLAN TO OVERLAP" if high and total > high
                else "no objection on memory" if high else "UNRESOLVED"),
            "why_the_recommendation": (
                "the headroom is inside measurement noise and the limit "
                "it fits under THROTTLES rather than refuses -- NOT "
                "because it exceeds the cap"
                if high and total <= high
                and (high - total) / high < MARGIN_IS_NOISE_BELOW else
                "it exceeds the soft limit" if high and total > high else
                "headroom is comfortable" if high else
                "no peak supplied"),
            "THE_FAILURE_MODE_IS_THE_ARGUMENT": (
                "a throttled night looks like a SLOW night rather than a "
                "broken one. For a fourteen-night band where a lost day "
                "costs a band day, silent slowdown is worse than a clean "
                "refusal: a refusal we would notice, a throttle we would "
                "attribute to load and re-plan around"),
        }
    # THE UNIT DECIDES IT, so both readings are shown rather than one
    # chosen silently.
    tape = rows.get("tape", {})
    tape_total = (tape.get("both_coins_peak_sum_bytes") or {}).get("value")
    unit = None
    if tape_total is not None:
        unit = {
            "the_phrase": "the 12 GB rule",
            "as_12_GiB_which_is_what_systemd_holds": {
                "threshold_bytes": M(12 * GIB, population="the rule",
                                     criterion="12 GiB, binary",
                                     source="systemd holds this exact "
                                            "value", provenance=MEASURED),
                "fits": tape_total <= 12 * GIB,
                "headroom_GiB": M((12 * GIB - tape_total) / GIB,
                                  population="tape, both coins",
                                  criterion="12 GiB minus the peak sum",
                                  source="arithmetic",
                                  provenance=MEASURED)},
            "as_12_GB_decimal": {
                "threshold_bytes": M(12_000_000_000, population="the rule",
                                     criterion="12 GB, decimal",
                                     source="the phrase taken literally",
                                     provenance=PROJECTED),
                "fits": tape_total <= 12_000_000_000,
                "headroom_GiB": M((12_000_000_000 - tape_total) / GIB,
                                  population="tape, both coins",
                                  criterion="12 GB minus the peak sum",
                                  source="arithmetic",
                                  provenance=MEASURED)},
            "THE_ANSWER_FLIPS_ON_THE_UNIT": (
                (tape_total <= 12 * GIB)
                != (tape_total <= 12_000_000_000)),
            "what_the_system_holds": thr.get("MemoryHigh_GiB"),
            "and_MemoryMax_is": thr.get("MemoryMax_GiB"),
            "and_the_margin_either_way":
                "under the GiB reading the pair fits by 0.7% of the "
                "threshold, which is inside measurement noise and is not "
                "a margin to plan a night on; MemoryHigh THROTTLES rather "
                "than kills, and MemoryMax is the kill",
        }
    resolved = all(r.get("verdict") != "UNRESOLVED" for r in rows.values())
    return {"threshold": thr, "per_stage": rows,
            "UNIT_AMBIGUITY": unit,
            "resolved": resolved,
            "verdict": ("SEE_PER_STAGE" if resolved else "UNRESOLVED"),
            "why": "the overlap question is decided by PEAK RSS against "
                   "the slice threshold; peaks that are absent leave the "
                   "stage UNRESOLVED rather than assumed serial"}


def assert_no_collapsed_ratio(block: dict) -> int:
    """NO SINGLE ETH:BTC FIGURE. The whole finding is that the three
    quantities disagree, and a collapsed ratio would erase it."""
    flat = json.dumps(block)
    for bad in ("eth_fraction_of_btc", "eth_vs_btc_ratio",
                "overall_ratio", "eth_is_x_percent"):
        if bad in flat:
            raise DecisionRefused(
                f"REFUSED {COLLAPSED_RATIO}: the block carries {bad!r}. "
                f"Size scales at ~0.66 on both stages and time at 0.27 "
                f"then 0.45 -- one figure for three quantities erases the "
                f"finding it claims to summarise.")
    n = 0
    for stage, row in block.get("stages", {}).items():
        for q, cell in row.items():
            if "ratio_eth_over_btc" in cell:
                crit = cell["ratio_eth_over_btc"]["criterion"]
                if q not in crit:
                    raise DecisionRefused(
                        f"REFUSED {COLLAPSED_RATIO}: a ratio in {stage} "
                        f"does not name the quantity it is a ratio OF.")
                n += 1
    return n


def eth_dependent_block(as_of: str, decl_dir: Path = None) -> dict:
    """EVERY ETH-DEPENDENT FIGURE, EACH CARRYING ITS PROVENANCE."""
    eth = eth_measurement(decl_dir)
    prov = eth["provenance"]
    pair = HZ.forward_rate_pair()
    src = "de_band_hazard x the ETH parameter"
    rows = []
    for base_name, base in (("planning_rate", pair["planning_rate"]["p"]),
                            ("optimistic_bound",
                             pair["optimistic_bound"]["p"])):
        for e in eth["scenarios"]:
            joint = base * e
            ok = HZ.p_at_least(joint)
            pop = (P_11 if base_name == "planning_rate" else P_8)
            rows.append({
                "base_rate": base_name,
                "eth_rate": M(e, population="eth day builds",
                              criterion=(C_GATE if prov == MEASURED
                                         else "ASSUMPTION -- no ETH build "
                                              "measurement has landed"),
                              source=src, provenance=prov),
                "joint_rate": M(joint, population=f"{pop} x eth",
                                criterion=C_BINOMIAL, source=src,
                                provenance=prov),
                "expected_evaluable": M(HZ.BAND_DAYS * joint,
                                        population=P_BAND,
                                        criterion=C_BINOMIAL, source=src,
                                        provenance=prov),
                "P_NO_VERDICT": M(1 - ok, population=P_BAND,
                                  criterion=C_BINOMIAL, source=src,
                                  provenance=prov)})
    param = {k: v for k, v in eth.items()
             if k not in ("scenarios", "measured")}
    param["scenarios"] = [
        M(e, population="eth day builds",
          criterion=(C_GATE if prov == MEASURED
                     else "ASSUMPTION -- no ETH build measurement has "
                          "landed"),
          source=src, provenance=prov) for e in eth["scenarios"]]
    if eth.get("measured"):
        param["measured"] = {
            k: M(v["value"], population="eth day builds",
                 criterion=f"{C_GATE} -- key {v['key']}",
                 source=v["declared_by"], provenance=MEASURED)
            for k, v in eth["measured"].items()}
    costs = eth_stage_costs(decl_dir)
    return {"ETH_COST_PROVENANCE": prov,
            "TWO_SEPARATE_ETH_PARAMETERS": {
                "day_success_rate": {
                    "provenance": prov,
                    "used_for": "the joint rate, the minimum viable daily "
                                "rate and P(fewer than 10)"},
                "build_cost": {
                    "provenance": costs.get("provenance"),
                    "used_for": "the night budget and the overlap "
                                "question"},
                "A_BUILD_COST_IS_NOT_A_SUCCESS_RATE":
                    "measuring what a build COSTS says nothing about how "
                    "often it SUCCEEDS, so a measured cost does not move "
                    "any rate figure on this page"},
            "BUILD_COST": costs,
            "NIGHT_BUDGET": night_budget(costs),
            "CAN_THE_COINS_SHARE_A_NIGHT": overlap_question(costs),
            "eth_parameter": param,
            "rows": rows,
            "recomputes_when_the_measurement_lands": True,
            "IF_PROJECTED_NOTHING_HERE_IS_MEASURED":
                prov == PROJECTED}


def assert_eth_labelled(block: dict) -> int:
    """NO ETH-DEPENDENT FIGURE TRAVELS WITHOUT ITS PROVENANCE."""
    n = 0
    for row in block.get("rows", []):
        for k, v in row.items():
            if _is_measure(v):
                if not v.get("provenance"):
                    raise DecisionRefused(
                        f"REFUSED {UNLABELLED_ETH}: {k} in the "
                        f"{row.get('base_rate')} row carries no "
                        f"provenance. A PROJECTED number sitting "
                        f"unlabelled among measured ones is the worst "
                        f"place for it.")
                n += 1
    return n


def eth_provenance_transition(out_dir: Path, current, as_of: str) -> dict:
    """PROJECTED -> MEASURED IS AN INSTRUMENT CORRECTION, NOT A STATE
    CHANGE, and a reader must be able to tell. The ledger beside the
    artifact is what makes the transition visible at all."""
    # TWO SERIES, because the two eth parameters move independently: the
    # build COST can become measured while the success RATE stays
    # projected, and a single ledger line would hide that.
    cur = current if isinstance(current, dict) else {"rate": current}
    led = Path(out_dir) / "eth_cost_provenance_ledger.jsonl"
    prior = {}
    if led.is_file():
        for line in led.read_text().splitlines():
            try:
                rec = json.loads(line)
            except Exception:                               # noqa: BLE001
                continue
            for k in cur:
                if rec.get(k) is not None:
                    prior[k] = rec[k]
                elif rec.get("provenance") is not None and k == "rate":
                    prior[k] = rec["provenance"]
    led.parent.mkdir(parents=True, exist_ok=True)
    with led.open("a") as fh:
        fh.write(json.dumps(dict(cur, as_of=as_of)) + "\n")
    moved = {k: {"prior": prior.get(k), "current": v,
                 "changed": prior.get(k) is not None and prior[k] != v}
             for k, v in cur.items()}
    changed = any(m["changed"] for m in moved.values())
    return {"series": moved, "changed": changed,
            "prior": prior.get("rate"), "current": cur.get("rate"),
            "CHANGED_FROM_PROJECTED_TO_MEASURED": any(
                m["changed"] and m["prior"] == PROJECTED
                and m["current"] in (MEASURED, PARTIAL)
                for m in moved.values()),
            "this_is_an_INSTRUMENT_CORRECTION_not_a_state_change":
                "the world did not change when the measurement landed; "
                "what changed is what we know, and a reader must be able "
                "to tell those apart (DA's rule tonight)",
            "ledger": str(led)}


def rule34a_prerequisite(root: Path = None) -> dict:
    """RULE 34a's FENCE, AS A BLOCKING PREREQUISITE WITH A COMPUTED STATE.

    The coordinator ruled the fence is due against THE BAND, not the
    morning. So the gap is declared now and its state is RECOMPUTED on
    every emit: when the call sites land, this flips to met without
    anyone editing a sentence.

    The three properties are DA's, from their fence audit. Property 1 is
    already met by da_rule34a_fence (DA 307); what is open is property 2.
    """
    root = Path(root or HERE)
    touch, guarded = [], []
    for f in sorted(root.glob("*.py")):
        src = f.read_text()
        if "tier2" not in src:
            continue
        touch.append(f.name)
        if ("da_rule34a_fence" in src or "is_protected" in src
                or "assert_not_protected" in src):
            guarded.append(f.name)
    reads = [f for f in touch if f not in TIER2_PROSE_ONLY]
    unguarded = [f for f in reads if f not in guarded]
    fence_exists = (root / "da_rule34a_fence.py").is_file()
    return {
        "prerequisite": "RULE 34a MUST BE ENFORCED BEFORE THE BAND OPENS",
        "due_against": "the band, not the morning -- the rule exists to "
                       "protect the validation population, and the band "
                       "cannot start until the effect floor is set and "
                       "two-coin production is demonstrated",
        "BLOCKING": True,
        "met": bool(fence_exists and not unguarded),
        "properties": {
            "1_protected_set_derived_from_an_artifact": {
                "met": fence_exists,
                "by": "da_rule34a_fence parses the floor from the "
                      "procedure documents (DA 307)"},
            "2_a_call_site_on_every_tier2_read_path": {
                "met": not unguarded,
                "modules_touching_tier2": touch,
                "prose_mention_only": list(TIER2_PROSE_ONLY),
                "read_paths": reads,
                "guarded": guarded,
                "UNGUARDED": unguarded},
            "3_falsifier_refuses_a_protected_read_AND_admits_an_"
            "unprotected_one": {
                "met": fence_exists,
                "by": "da_rule34a_fence.is_protected / "
                      "assert_not_protected, both driven"},
        },
        "state": ("THE FENCE EXISTS AND IS SOUND; IT IS UNWIRED ON THE "
                  "READ PATHS" if fence_exists and unguarded else
                  "ENFORCED" if fence_exists else "NO FENCE EXISTS"),
        "recomputed_on_every_emit": True,
        "why_not_built_tonight":
            "a fence built hastily to guard a population that does not "
            "yet exist is how a guard acquires the defects this "
            "programme spent the night removing",
    }


def amendment_status() -> dict:
    """THE CALL SITE (REVIEW 264 fix 7).

    The guard is consulted HERE, in the document a person decides on, for
    every lever that must be declared before the clock -- so its verdict
    is a field of the decision rather than a function nobody calls. Today
    every one of them refuses, because no validation window is declared:
    there is no clock to be before.
    """
    out = {}
    for lever, path in AMENDMENT_FILES.items():
        try:
            got = HZ.amendment_is_admissible(lever, amendment_path=path)
            out[lever] = {"admissible": True,
                          "declared_utc": got["declared_utc"],
                          "clock_start_utc": got["clock_start_utc"],
                          "clock_read_from": got["clock_read_from"],
                          "expected_declaration": path}
        except HZ.BandHazardRefused as exc:
            head = str(exc).split(":")[0].replace("REFUSED ", "").strip()
            out[lever] = {"admissible": False, "refusal": head,
                          "expected_declaration": path,
                          "detail": str(exc)[:200]}
    return {"consulted": "de_band_hazard.amendment_is_admissible",
            "when": "every time this artifact is emitted",
            "per_lever": out,
            "why_this_is_here":
                "REVIEW 264: a guard with cells and no call site "
                "constrains nothing; its verdict now travels in the "
                "document the decision is made from"}


def decision_artifact(as_of: str) -> dict:
    pair = HZ.forward_rate_pair()
    via = HZ.viability_table()
    lv = HZ.levers()
    floor = HZ.g_floor()
    src = "de_band_hazard (this seat, measured 2026-09-12)"

    def rate(block, population):
        return {
            "pass_rate": M(block["p"], population=population,
                           criterion=C_GATE, source=src),
            "days_passing": M(block["n_pass"], population=population,
                              criterion=C_GATE, source=src),
            "days_in_window": M(block["n_days"], population=population,
                                criterion=C_GATE, source=src),
            "expected_evaluable_in_14": M(
                block["expected_evaluable"], population=P_BAND,
                criterion=C_BINOMIAL, source=src),
            "P_at_least_10": M(block["P_at_least_10"], population=P_BAND,
                               criterion=C_BINOMIAL, source=src),
            "P_NO_VERDICT": M(block["P_NO_VERDICT"], population=P_BAND,
                              criterion=C_BINOMIAL, source=src),
            "failing_days": block["failing_days"],
        }

    return {
        "protocol": PROTOCOL,
        "as_of": as_of,
        "what_this_is": "the state a person decides on before committing "
                        "fourteen nights; nothing here recommends",
        "EVERY_NUMBER_CARRIES_ITS_POPULATION_AND_CRITERION": True,
        "why_that_discipline":
            "a pass rate quoted without its criterion survived three "
            "retellings tonight: 10 of 11 is true of raw-tape window-file "
            "presence and false of BE's population gate, which scores 7 "
            "of 11 over the same days",

        "THE_TWO_RATES": {
            "FORWARD_RATE_IS_UNRESOLVED": True,
            "planning_rate": rate(pair["planning_rate"], P_11),
            "optimistic_bound": rate(pair["optimistic_bound"], P_8),
            "the_same_seven_days_pass_in_both": True,
            "factor_between_failure_probabilities": M(
                pair["factor_between_their_failure_probabilities"],
                population=P_BAND, criterion=C_BINOMIAL, source=src,
                note="like for like at the SAME criterion; pairing the "
                     "tape criterion against the gate criterion gives "
                     "4.08x instead, which is how a factor gets "
                     "misquoted"),
            "for_contrast_the_OTHER_criterion": {
                "tape_window_files_11_days": M(
                    9 / 11, population=P_11, criterion=C_TAPE, source=src,
                    note="09-03 is short by one window on both coins"),
                "reported_by_another_seat": M(
                    10 / 11, population=P_11, criterion=C_UNVERIFIED,
                    source="REV, via the coordinator",
                    note="reproduces the tape criterion at a threshold of "
                         "zero missing interior windows; its definition "
                         "was not found in a landed artifact"),
            },
            "THE_REGIME_QUESTION": {
                "open_days": ["20260901", "20260902", "20260903"],
                "population_windows": {
                    d: M(HZ.DAYS[d]["pop_windows"], population=f"{d}, btc",
                         criterion=C_GATE, source=src)
                    for d in ("20260901", "20260902", "20260903")},
                "their_tape_was_complete": True,
                "settled_by": "classify 09-11: the LAST of the old "
                              "failures, or the FIRST of a new one",
                "exclusion_status": "NO CONDITION LICENSES AN EXCLUSION",
                "REV_searched_and_found_none":
                    "instrument, era and supply searched; REV named what "
                    "it did NOT read -- the mask producer source, "
                    "da_content_liveness_rule's implementation, per-window "
                    "content measurements, host metrics -- and the "
                    "direction its bias would push",
                "DA_holds_those_residuals_and_is_looking": True,
                "therefore":
                    "unless DA names a since-changed condition "
                    "independently of these days' outcomes, the planning "
                    "rate is the number and the optimistic bound is a "
                    "bound",
            },
        },

        "IS_IT_VIABLE": {
            "requirement": {
                "band_days": M(HZ.BAND_DAYS, population=P_BAND,
                               criterion="§8 as declared", source=src),
                "evaluable_needed": M(HZ.NEED_EVALUABLE,
                                      population=P_BAND,
                                      criterion="§8 as declared",
                                      source=src),
                "extension": "FORBIDDEN by §8",
            },
            "minimum_daily_joint_rate": [
                {"confidence": M(r["confidence"], population=P_BAND,
                                 criterion=C_BINOMIAL, source=src),
                 "required_rate": M(r["required_daily_joint_rate"],
                                    population=P_BAND,
                                    criterion=C_BINOMIAL, source=src),
                 "gap_from_the_planning_rate": M(
                     r["gap_from_the_planning_rate"], population=P_11,
                     criterion=f"{C_BINOMIAL} vs {C_GATE}", source=src),
                 "planning_rate_clears_it": r["planning_rate_clears_it"],
                 "gap_from_the_optimistic_bound": M(
                     r["gap_from_the_optimistic_bound"], population=P_8,
                     criterion=f"{C_BINOMIAL} vs {C_GATE}", source=src),
                 "optimistic_bound_clears_it":
                     r["optimistic_bound_clears_it"]}
                for r in via["rows"]],
            "reading":
                "the planning rate clears none of the three "
                "confidences; the optimistic bound clears all three, so "
                "the viability question IS the regime question",
        },

        "THE_FLOOR_UNDER_k": {
            "lowest_G_with_any_test": M(
                floor["lowest_G_with_any_test"], population=P_BAND,
                criterion=C_LADDER, source=src),
            "below_it": "the exact sign test returns INSUFFICIENT_"
                        "EVIDENCE and there is no p to correct",
            "at_G_9_and_G_8": "exactly ONE rung passes, so the candidate "
                              "must be positive on EVERY day",
        },

        "THE_LEVERS": {
            "i_accept_the_risk": {
                "cost_at_the_planning_rate": M(
                    lv["i_accept_the_risk"]["cost"]["at_the_planning_rate"],
                    population=P_BAND, criterion=C_BINOMIAL, source=src),
                "cost_at_the_optimistic_bound": M(
                    lv["i_accept_the_risk"]["cost"][
                        "at_the_optimistic_bound"],
                    population=P_BAND, criterion=C_BINOMIAL, source=src),
                "weakens": "nothing -- the test stays as declared",
                "is_an_amendment_the_user_alone_may_make": False},
            "ii_improve_the_input": {
                "required_rate_at_0_90": M(
                    lv["ii_improve_the_input"][
                        "required_rate_at_this_confidence"],
                    population=P_BAND, criterion=C_BINOMIAL, source=src),
                "gap_from_the_planning_rate": M(
                    lv["ii_improve_the_input"]["gap_from_the_planning_rate"],
                    population=P_11, criterion=f"{C_BINOMIAL} vs {C_GATE}",
                    source=src),
                "cost": "unknown until the mask-collapse cause is named; "
                        "may be unavailable",
                "weakens": "nothing -- it changes the INPUT, not the test",
                "is_an_amendment_the_user_alone_may_make": False,
                "A_FACTUAL_PROPERTY_NOT_A_RECOMMENDATION":
                    "this is the only lever that does not trade the "
                    "test's strength for its feasibility"},
            "iii_longer_band": {
                "days_required_at_the_planning_rate": M(
                    lv["iii_longer_band"]["n_required_at_the_planning_rate"],
                    population=P_11, criterion=C_BINOMIAL, source=src),
                "days_required_at_the_optimistic_bound": M(
                    lv["iii_longer_band"][
                        "n_required_at_the_optimistic_bound"],
                    population=P_8, criterion=C_BINOMIAL, source=src),
                "cost": "calendar time; every added day is a day the "
                        "candidates are not yet judged",
                "weakens": "nothing statistically -- k is unchanged",
                "is_an_amendment_the_user_alone_may_make": True,
                "must_be_declared_BEFORE_the_clock": True},
            "iv_fewer_required_days": {
                "largest_k_at_the_planning_rate": M(
                    lv["iv_fewer_required_days"][
                        "max_k_at_the_planning_rate"],
                    population=P_BAND, criterion=C_BINOMIAL, source=src),
                "smallest_k_the_test_can_compute": M(
                    lv["iv_fewer_required_days"][
                        "lowest_k_the_TEST_can_compute"],
                    population=P_BAND, criterion=C_LADDER, source=src),
                "AVAILABLE_AT_THE_PLANNING_RATE": lv[
                    "iv_fewer_required_days"][
                    "AVAILABLE_AT_THE_PLANNING_RATE"],
                "cost": "the test's resolution: at G=10 two rungs pass, "
                        "at G=9 and G=8 exactly one",
                "weakens": "the test itself, and the multiplicity "
                           "arithmetic with it -- Holm's threshold does "
                           "not move, so a smaller G spends the same "
                           "alpha on a coarser ladder",
                "is_an_amendment_the_user_alone_may_make": True,
                "must_be_declared_BEFORE_the_clock": True},
            "v_start_after_a_clean_run": {
                "cost": "calendar time, and the clean run consumes days "
                        "that cannot later be in the band",
                "weakens": "nothing in the test; it buys the RATE by "
                           "choosing when to start",
                "is_an_amendment_the_user_alone_may_make": False,
                "must_be_declared_BEFORE_the_clock": True,
                "caution": "the start condition must be a declared "
                           "PREDICATE, or 'it looked clean' becomes the "
                           "selection"},
        },

        "AMENDMENT_ADMISSIBILITY_NOW": amendment_status(),
        "BLOCKING_PREREQUISITES": {
            "rule_34a_fence": rule34a_prerequisite()},
        "ETH_DEPENDENT_FIGURES": eth_dependent_block(as_of),

        "THE_TRAP": {
            "which_levers": ["iii_longer_band", "iv_fewer_required_days"],
            "why": "both are what a disappointed operator reaches for "
                   "AFTER a shortfall, when the shortfall itself is the "
                   "information being used",
            "therefore": "if either is to be available it must be "
                         "DECLARED NOW, while the outcome is unknown",
            "enforced_by": "de_band_hazard.amendment_is_admissible, which "
                           "REFUSES a declaration with no timestamps, one "
                           "timestamped after the clock start, and any "
                           "consideration once the outcome is known",
            "this_is_a_field_not_advice": True},
    }


def emit(path=None, as_of: str = None) -> dict:
    import datetime as dt
    as_of = as_of or dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    doc = decision_artifact(as_of)
    n = assert_attributed(doc)
    assert_eth_labelled(doc["ETH_DEPENDENT_FIGURES"])
    assert_no_collapsed_ratio(doc["ETH_DEPENDENT_FIGURES"]["BUILD_COST"])
    assert_verdicts_carry_their_basis(doc)
    if path:
        doc["ETH_DEPENDENT_FIGURES"]["provenance_transition"] = (
            eth_provenance_transition(
                Path(path).parent,
                {"rate": doc["ETH_DEPENDENT_FIGURES"][
                    "ETH_COST_PROVENANCE"],
                 "cost": doc["ETH_DEPENDENT_FIGURES"]["BUILD_COST"][
                     "provenance"]},
                as_of))
    doc["n_attributed_measures"] = M(
        n, population="this artifact", criterion="assert_attributed walk",
        source="de_band_decision")
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(doc, indent=2, default=str))
    return doc


def falsify() -> int:
    n = ok = 0

    def ck(name, cond, note=""):
        nonlocal n, ok
        n += 1
        ok += bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
              + (f"  {note}" if note else ""))

    print("== every number carries its population and criterion ==")
    doc = decision_artifact("2026-09-12T00:00:00Z")
    count = assert_attributed(doc)
    ck("the artifact passes the attribution walk",
       count > 25, f"{count} attributed measures")
    ck("  and a PLANTED bare number is caught -- positive control",
       _planted_bare_number_refuses())
    ck("  while a boolean is not a measurement and is allowed",
       assert_attributed({"flag": True, "why": "text"}) == 0)

    print("== the two rates, as a pair, with the criterion attached ==")
    tr = doc["THE_TWO_RATES"]
    ck("both rates carry the SAME criterion, so they are comparable",
       tr["planning_rate"]["pass_rate"]["criterion"]
       == tr["optimistic_bound"]["pass_rate"]["criterion"] == C_GATE)
    ck("  and they name DIFFERENT populations",
       tr["planning_rate"]["pass_rate"]["population"] != tr[
           "optimistic_bound"]["pass_rate"]["population"])
    ck("the OTHER criterion is carried beside them, not instead",
       tr["for_contrast_the_OTHER_criterion"]["tape_window_files_11_days"][
           "criterion"] == C_TAPE)
    ck("  and the unverified outside number is MARKED unverified",
       "UNVERIFIED" in tr["for_contrast_the_OTHER_criterion"][
           "reported_by_another_seat"]["criterion"])
    ck("the regime question is named and UNRESOLVED",
       tr["FORWARD_RATE_IS_UNRESOLVED"] is True
       and tr["THE_REGIME_QUESTION"]["exclusion_status"].startswith("NO"))
    ck("  and REV's search and what it did NOT read are recorded",
       bool(tr["THE_REGIME_QUESTION"].get("REV_searched_and_found_none",
                                          "").strip())
       and "DA_holds_those_residuals_and_is_looking"
       in tr["THE_REGIME_QUESTION"],
       "the FIELDS are present; their wording is free to change")

    print("== THE CALL SITE (REVIEW 264 fix 7) ==")
    st = doc["AMENDMENT_ADMISSIBILITY_NOW"]
    ck("the guard is consulted from ANOTHER module, in the emitted "
       "artifact",
       st["consulted"] == "de_band_hazard.amendment_is_admissible"
       and set(st["per_lever"]) == set(AMENDMENT_FILES))
    ck("  and today every amendment lever REFUSES -- no clock exists yet",
       all(not v["admissible"] for v in st["per_lever"].values()),
       str(sorted({v["refusal"] for v in st["per_lever"].values()})))
    ck("  each naming the declaration file that would date it",
       all(v["expected_declaration"].endswith(".json")
           for v in st["per_lever"].values()))

    print("== the ETH cost is a PARAMETER, and it is labelled ==")
    eb = doc["ETH_DEPENDENT_FIGURES"]
    ck("today it reads PROJECTED -- no ETH build measurement has landed",
       eb["ETH_COST_PROVENANCE"] == PROJECTED
       and eb["IF_PROJECTED_NOTHING_HERE_IS_MEASURED"] is True,
       eb["eth_parameter"]["why"][:70])
    ck("  and it names what IS measured about eth today, so the gap is "
       "specific",
       bool(eb["eth_parameter"].get("what_is_measured_about_eth_today",
                                    "").strip())
       and eb["eth_parameter"]["provenance"] == PROJECTED,
       "the field is present and the provenance agrees with it")
    ck("every ETH-dependent figure carries its provenance",
       assert_eth_labelled(eb) >= 8,
       f"{assert_eth_labelled(eb)} labelled figures")
    stripped = json.loads(json.dumps(eb))
    stripped["rows"][0]["joint_rate"].pop("provenance")
    try:
        assert_eth_labelled(stripped)
        ck("  and a figure with the label REMOVED refuses -- positive "
           "control", False)
    except DecisionRefused as exc:
        ck("  and a figure with the label REMOVED refuses -- positive "
           "control", UNLABELLED_ETH in str(exc))
    rates = {r["eth_rate"]["value"] for r in eb["rows"]}
    ck("the projected rates are marked ASSUMPTION in the criterion, not "
       "in prose",
       all("ASSUMPTION" in r["eth_rate"]["criterion"] for r in eb["rows"]),
       str(sorted(rates)))
    import tempfile
    td = Path(tempfile.mkdtemp(prefix="de_eth_"))
    (td / "declarations").mkdir()
    (td / "declarations" / "be_eth_build_cost_v1.json").write_text(
        json.dumps({"eth_day_success_rate": 0.97,
                    "eth_book_stage_seconds": 1400}))
    got = eth_measurement(td / "declarations")
    ck("WHEN THE MEASUREMENT LANDS it is read, not re-derived by hand",
       got["provenance"] == MEASURED
       and got["measured"]["rate"]["value"] == 0.97
       and got["measured"]["seconds"]["value"] == 1400.0,
       f"rate {got['measured']['rate']['value']} from "
       f"{got['measured']['rate']['declared_by']}")
    blk = eth_dependent_block("2026-09-12T00:00:00Z", td / "declarations")
    ck("  and every figure recomputes against it, now marked MEASURED",
       blk["ETH_COST_PROVENANCE"] == MEASURED
       and all(r["eth_rate"]["value"] == 0.97 for r in blk["rows"])
       and all(r["joint_rate"]["provenance"] == MEASURED
               for r in blk["rows"]),
       f"{len(blk['rows'])} rows recomputed")
    t1 = eth_provenance_transition(td, {"rate": PROJECTED,
                                        "cost": PROJECTED}, "t1")
    t2 = eth_provenance_transition(td, {"rate": PROJECTED,
                                        "cost": PARTIAL}, "t2")
    ck("the transition PROJECTED -> MEASURED is RECORDED as an "
       "instrument correction",
       t1["changed"] is False
       and t2["CHANGED_FROM_PROJECTED_TO_MEASURED"] is True
       and t2["series"]["cost"]["changed"] is True
       and t2["series"]["rate"]["changed"] is False
       and any("INSTRUMENT_CORRECTION" in k for k in t2),
       "the COST moved and the RATE did not -- two series, one ledger")

    print("== the BUILD COST, ingested per stage, three series ==")
    bc = eb["BUILD_COST"]
    ck("the state is PARTIALLY measured -- two stages in, one running",
       bc["provenance"] == PARTIAL
       and set(bc["stages_measured"]) == {"fragment", "tape"}
       and bc["stages_not_measured"] == ["book"],
       f"measured {bc['stages_measured']}, open "
       f"{bc['stages_not_measured']}")
    ck("  so a partial state is REPRESENTABLE, not rounded to one of the "
       "two ends",
       bc["provenance"] not in (MEASURED, PROJECTED))
    ratios = {(st, q): row[q]["ratio_eth_over_btc"]["value"]
              for st, row in bc["stages"].items() for q in row
              if "ratio_eth_over_btc" in row[q]}
    ck("THE FINDING SURVIVES INGESTION: time and size ratios DISAGREE",
       abs(ratios[("fragment", "wall_s")] - 0.273) < 0.01
       and abs(ratios[("tape", "wall_s")] - 0.446) < 0.01
       and abs(ratios[("fragment", "output_bytes")] - 0.668) < 0.01
       and abs(ratios[("tape", "output_bytes")] - 0.661) < 0.01,
       f"time {ratios[('fragment','wall_s')]:.3f}/"
       f"{ratios[('tape','wall_s')]:.3f} vs bytes "
       f"{ratios[('fragment','output_bytes')]:.3f}/"
       f"{ratios[('tape','output_bytes')]:.3f}")
    ck("  and every ratio NAMES the quantity it is a ratio OF",
       assert_no_collapsed_ratio(bc) >= 4,
       f"{assert_no_collapsed_ratio(bc)} per-quantity ratios")
    planted = json.loads(json.dumps(bc))
    planted["eth_fraction_of_btc"] = 0.66
    try:
        assert_no_collapsed_ratio(planted)
        ck("  and a single collapsed figure REFUSES -- positive control",
           False)
    except DecisionRefused as exc:
        ck("  and a single collapsed figure REFUSES -- positive control",
           COLLAPSED_RATIO in str(exc))
    ck("the stale-comparator caveat travels with every ratio",
       all("APPROXIMATE" in row[q]["ratio_eth_over_btc"]["criterion"]
           for st, row in bc["stages"].items() for q in row
           if "ratio_eth_over_btc" in row[q]),
       bc["btc_comparators_as_of"])
    ck("a stage with NO peak stays UNRESOLVED rather than assumed serial",
       eb["CAN_THE_COINS_SHARE_A_NIGHT"]["resolved"] is False
       and eb["CAN_THE_COINS_SHARE_A_NIGHT"]["per_stage"]["book"][
           "verdict"] == "UNRESOLVED",
       "the book stage has no peak, so it is not answered")
    nb = eb["NIGHT_BUDGET"]
    ck("the night budget sums MEASURED stages only and names what is "
       "missing",
       nb["per_coin"]["eth"]["stages_missing"] == ["book"]
       and nb["serial_seconds_measured_stages_only"]["provenance"]
       == PARTIAL,
       f"eth {nb['per_coin']['eth']['measured_stage_seconds']['value']:.0f}s"
       f" + btc "
       f"{nb['per_coin']['btc']['measured_stage_seconds']['value']:.0f}s")
    ck("A BUILD COST IS NOT A SUCCESS RATE: the rate stays PROJECTED "
       "while the cost is measured",
       eb["TWO_SEPARATE_ETH_PARAMETERS"]["day_success_rate"]["provenance"]
       == PROJECTED
       and eb["TWO_SEPARATE_ETH_PARAMETERS"]["build_cost"]["provenance"]
       == PARTIAL,
       "measuring what a build costs says nothing about how often it "
       "succeeds")
    ck("nothing here can read as an eth FINDING about the consumed day",
       bool(bc["DAY_IS_CONSUMED_AND_THIS_IS_NOT_A_FINDING_ABOUT_IT"]))

    print("== peaks, the mechanism, and the unit that decides it ==")
    ov = eb["CAN_THE_COINS_SHARE_A_NIGHT"]
    ck("the threshold is READ FROM THE SYSTEM, not remembered",
       ov["threshold"]["MemoryHigh_GiB"]["value"] == 12.0
       and ov["threshold"]["MemoryMax_GiB"]["value"] == 14.0,
       "research.slice MemoryHigh 12 GiB, MemoryMax 14 GiB")
    u = ov["UNIT_AMBIGUITY"]
    ck("THE ANSWER FLIPS ON THE UNIT and both readings are shown",
       u["THE_ANSWER_FLIPS_ON_THE_UNIT"] is True
       and u["as_12_GiB_which_is_what_systemd_holds"]["fits"] is True
       and u["as_12_GB_decimal"]["fits"] is False,
       "11.92 GiB = 12.799 GB: under 12 GiB, over 12 GB")
    ck("  and the margin under the favourable reading is reported, not "
       "hidden",
       0 < u["as_12_GiB_which_is_what_systemd_holds"]["headroom_GiB"][
           "value"] < 0.1,
       f"{u['as_12_GiB_which_is_what_systemd_holds']['headroom_GiB']['value']:.3f} "
       f"GiB, 0.7% of the threshold")
    ck("THE VERDICT IS THREE-STATE, and the tape stage names its margin "
       "and its n in the verdict itself",
       ov["per_stage"]["tape"]["verdict"].startswith("FITS_SOFT_LIMIT_BY")
       and "SINGLE_OBSERVATIONS" in ov["per_stage"]["tape"]["verdict"]
       and ov["per_stage"]["fragment"]["verdict"].startswith(
           "COMFORTABLY_UNDER"),
       ov["per_stage"]["tape"]["verdict"])
    ck("  and the RECOMMENDATION is DO NOT PLAN TO OVERLAP for the "
       "margin, NOT for exceeding the cap",
       ov["per_stage"]["tape"]["RECOMMENDATION"] == "DO NOT PLAN TO "
                                                    "OVERLAP"
       and "NOT" in ov["per_stage"]["tape"]["why_the_recommendation"],
       "headroom-within-noise and soft-limit throttling")
    ck("  and the failure mode is stated: a throttled night looks SLOW, "
       "not broken",
       bool(ov["per_stage"]["tape"]["THE_FAILURE_MODE_IS_THE_ARGUMENT"]))
    fits = ov["per_stage"]["tape"]["fits_under_MemoryHigh"]
    ck("the boolean carries its BASIS, its n and the limit KIND",
       fits["limit_kind"] == "SOFT_THROTTLE"
       and "single observations" in fits["n_observations"]
       and "does not kill" in fits["basis"],
       "soft throttle, n=1 per coin")
    try:
        assert_verdicts_carry_their_basis({"memory": {"fits_cap": True}})
        ck("  and a BARE verdict boolean refuses -- the shape the "
           "attribution guard let through", False)
    except DecisionRefused as exc:
        ck("  and a BARE verdict boolean refuses -- the shape the "
           "attribution guard let through", BARE_BOOLEAN in str(exc))
    ck("  while a non-verdict flag is still allowed",
       assert_verdicts_carry_their_basis(
           {"computed_not_printed": True, "n": 1}) == 0)
    ck("the fragment stage resolves and the book stage does NOT",
       ov["per_stage"]["fragment"]["verdict"] != "UNRESOLVED"
       and ov["per_stage"]["book"]["verdict"] == "UNRESOLVED",
       "a stage with no peak is unresolved, not assumed")
    mech = bc["MECHANISM"]
    ck("the MECHANISM travels beside the numbers and is FALSIFIABLE",
       bool(mech.get("claim")) and bool(mech.get("FALSIFIER")),
       "a coin or day with a different window count must break it")
    ck("the conclusion is stated as holding WITHOUT the book stage",
       bool(bc["THE_CONCLUSION_HOLDS_WITHOUT_THE_BOOK_STAGE"]))
    ck("  and the refused book run is NOT counted as the stage's cost",
       bc["book_stage_outcome"]["status"] == "REFUSED"
       and bc["book_stage_outcome"]["book_written"] is False
       and "book" in bc["stages_not_measured"],
       "rc=1, no book written")
    nbr = nb["BUDGET_IS_A_RANGE_NOT_A_POINT"]
    ck("the night budget is a RANGE with the hole named, not a point",
       nbr["upper_bound_seconds"] is None
       and nbr["unmeasured_stages"] == ["book"]
       and nbr["lower_bound_seconds"]["provenance"] == PARTIAL,
       f"lower bound {nbr['lower_bound_seconds']['value']:.0f}s, book "
       f"unmeasured")
    ck("  and the sum-not-max consequence is carried",
       bool(nb["SERIAL_BY_MEMORY_NOT_ONLY_BY_LOCK"]))

    print("== rule 34a's fence as a BLOCKING prerequisite ==")
    pr = doc["BLOCKING_PREREQUISITES"]["rule_34a_fence"]
    ck("it is declared BLOCKING and its state is COMPUTED, not written",
       pr["BLOCKING"] is True and pr["recomputed_on_every_emit"] is True)
    ck("property 1 is MET -- the set is artifact-derived (DA 307)",
       pr["properties"]["1_protected_set_derived_from_an_artifact"]["met"]
       is True)
    p2 = pr["properties"]["2_a_call_site_on_every_tier2_read_path"]
    ck("property 2 is OPEN and NAMES the unguarded read paths",
       p2["met"] is False and len(p2["UNGUARDED"]) >= 1,
       str(p2["UNGUARDED"]))
    ck("  and a PROSE mention is not counted as a read path",
       set(p2["prose_mention_only"]) <= set(p2["modules_touching_tier2"])
       and not set(p2["prose_mention_only"]) & set(p2["read_paths"]),
       "a docstring naming the rule is not a read")
    ck("the prerequisite is NOT met while a read path is unguarded",
       pr["met"] is False, pr["state"])
    v = doc["IS_IT_VIABLE"]["minimum_daily_joint_rate"]
    ck("three confidences, each with both gaps",
       len(v) == 3 and all("gap_from_the_planning_rate" in r for r in v))
    ck("  the planning rate clears none of them",
       not any(r["planning_rate_clears_it"] for r in v))
    ck("lever (iv)'s unavailability is a FIELD, not a sentence",
       doc["THE_LEVERS"]["iv_fewer_required_days"][
           "AVAILABLE_AT_THE_PLANNING_RATE"] is False)
    ck("the amendments are marked as the user's alone",
       doc["THE_LEVERS"]["iii_longer_band"][
           "is_an_amendment_the_user_alone_may_make"]
       and doc["THE_LEVERS"]["iv_fewer_required_days"][
           "is_an_amendment_the_user_alone_may_make"]
       and not doc["THE_LEVERS"]["ii_improve_the_input"][
           "is_an_amendment_the_user_alone_may_make"])
    ck("  and (ii)'s property is stated as a PROPERTY, not advice",
       "NOT_A_RECOMMENDATION" in json.dumps(
           doc["THE_LEVERS"]["ii_improve_the_input"]))
    ck("the trap travels in the artifact and names its enforcement",
       doc["THE_TRAP"]["this_is_a_field_not_advice"] is True
       and "REFUSES" in doc["THE_TRAP"]["enforced_by"])

    print(f"\n{ok}/{n} cells pass")
    return 0 if ok == n else 1


def _planted_bare_number_refuses() -> bool:
    doc = decision_artifact("2026-09-12T00:00:00Z")
    doc["THE_TWO_RATES"]["a_helpful_summary_rate"] = 0.75
    try:
        assert_attributed(doc)
        return False
    except DecisionRefused as exc:
        return UNATTRIBUTED in str(exc)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--falsify" in argv:
        return falsify()
    if "--emit" in argv:
        out = argv[argv.index("--out") + 1] if "--out" in argv else None
        doc = emit(out)
        print(json.dumps(doc, indent=2, default=str))
        return 0
    print(json.dumps({"protocol": PROTOCOL}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
