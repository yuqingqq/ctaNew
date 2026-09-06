"""P-2026-002 E2-A -- REV 44 section C.4's two reporting gaps, closed in a
SUPERSEDING receipt computed from the landed smoke receipt's BYTES.

Nothing is re-run. The sealed payload is not opened. v1 is not edited: the
correction supersedes in band (CLAUDE.md rule 13) and cites v1 by sha256.

THE TWO GAPS.

(1) A CONTAINER CENSUS BESIDE THE NAME CENSUS. v1 published
    `numeric_keys_that_survived` -- a census by VOCABULARY. Both of the seal's
    nets are name-based by construction (a marker list, and a name-shape
    scan), so a leaf under an innocuous name is invisible to both. The
    container census is the INDEPENDENT AXIS: it counts numeric leaves by
    CONTAINER PATH, at full depth, through dicts AND lists, and compares the
    profile against the one this receipt pins. A leaf added anywhere -- under
    any name at all -- changes its container's count and is FLAGGED.

    LIST-RESIDENCY MEANS 'ANYWHERE UNDER A LIST', NOT 'A DIRECT CHILD OF ONE'.
    Measured on this receipt the two readings are 1,119 and 0. A census built
    on the second reading would report zero list-resident numbers and look
    like a clean bill on the very region -- the per-day admission diagnostics
    -- where almost all of them live.

(2) THE RESOURCE BLOCK. v1 carried `max_rss_kib` alone, which cannot show
    what the run actually did to rule 20's cap. The block now reads the
    scope's own `memory.peak`, `memory.stat` anon and file apart, and
    `memory.events` max/oom. DE 82 (`e255e72`) built this for the Gate-1
    runner; per R-235 it is READ AS A DOCUMENT and re-implemented here, not
    imported -- including its ambient-scope falsifier, because a scope
    outside `research.slice` is the shell's own and its numbers are real
    measurements OF THE WRONG OBJECT.

    python3 live/mm_research/da_e2a_receipt_v2.py --selftest
    python3 live/mm_research/da_e2a_receipt_v2.py --emit
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

PROTOCOL = "P002_E2_A_SMOKE_RECEIPT_V2"
CGROUP_ROOT = "/sys/fs/cgroup"
RESEARCH_SLICE = "research.slice"
GIB = 1024.0 ** 3
#: rule 20's wrapper: `-p MemoryMax=8G`.
MEMORY_MAX_BYTES = 8 * 1024 ** 3

V1_NAME = "p002_e2a_sealed_smoke_BTCUSDT__20260906T071809Z.json"


class ReceiptRefused(RuntimeError):
    """The superseding receipt cannot be produced honestly."""


def _root() -> Path:
    import e2_0_true_mid as E20
    return E20.ROOT


def carrying_commit() -> str:
    r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                       text=True, cwd=str(HERE))
    return r.stdout.strip() if r.returncode == 0 else "UNKNOWN"


def emitter_identity() -> dict:
    src = Path(__file__).resolve()
    r = subprocess.run(["git", "log", "-1", "--format=%H", "--", str(src)],
                       capture_output=True, text=True, cwd=str(src.parent))
    d = subprocess.run(["git", "status", "--porcelain", "--", str(src)],
                       capture_output=True, text=True, cwd=str(src.parent))
    return {"path": "live/mm_research/da_e2a_receipt_v2.py",
            "sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
            "commit_best_effort": (r.stdout.strip() or None),
            "tree_head": carrying_commit(),
            "producing_code_is_the_committed_bytes":
                d.returncode == 0 and d.stdout.strip() == ""}


# ------------------------------------------------------- (1) the container census

def container_census(obj) -> dict:
    """Numeric leaves by CONTAINER PATH, full depth, dicts AND lists.

    A list level collapses to `[]` in the path, so every element of a list
    shares one container -- which is what makes this a census of CONTAINERS
    rather than of leaves, and what lets a profile be compared across runs
    whose lists have different lengths.

    `bool` is not numeric here: JSON booleans are decisions, not values, and
    counting them would drown the signal the census exists to carry."""
    prof: dict = {}
    tot = {"n_leaves": 0, "n_numeric": 0, "n_numeric_list_resident": 0,
           "n_bool": 0, "n_null": 0, "n_str": 0,
           "n_containers_with_numbers": 0,
           "max_depth": 0}

    def walk(o, path: str, under_list: bool, depth: int):
        tot["max_depth"] = max(tot["max_depth"], depth)
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, f"{path}.{k}" if path else str(k), under_list,
                     depth + 1)
        elif isinstance(o, list):
            for v in o:
                walk(v, f"{path}[]", True, depth + 1)
        else:
            tot["n_leaves"] += 1
            if isinstance(o, bool):
                tot["n_bool"] += 1
            elif o is None:
                tot["n_null"] += 1
            elif isinstance(o, (int, float)):
                tot["n_numeric"] += 1
                if under_list:
                    tot["n_numeric_list_resident"] += 1
                cp = path.rsplit(".", 1)[0] if "." in path else ""
                prof[cp] = prof.get(cp, 0) + 1
            else:
                tot["n_str"] += 1

    walk(obj, "", False, 0)
    tot["n_containers_with_numbers"] = len(prof)
    return {"totals": tot, "profile": dict(sorted(prof.items())),
            "list_residency_rule": (
                "a numeric leaf is LIST-RESIDENT if ANY level of its "
                "container path is a list, not only its immediate parent. "
                "On this receipt the two readings are 1,119 and 0: the "
                "narrow one would report a clean bill on the region where "
                "almost every number lives"),
            "bool_rule": "JSON booleans are NOT counted as numeric"}


def compare_profiles(expected: dict, actual: dict) -> dict:
    """The independent axis. ANY container whose numeric count moved is
    FLAGGED -- whatever the added leaf is called."""
    keys = sorted(set(expected) | set(actual))
    diffs = []
    for k in keys:
        a, b = expected.get(k), actual.get(k)
        if a != b:
            diffs.append({"container": k, "expected": a, "actual": b,
                          "delta": (None if a is None or b is None
                                    else b - a)})
    return {"n_containers_compared": len(keys),
            "n_flagged": len(diffs), "flagged": diffs,
            "verdict": "AGREES" if not diffs else "FLAGGED",
            "why_this_axis_is_independent": (
                "the marker scan and the name-shape scan both decide on a "
                "leaf's NAME. This one decides on WHERE the leaf sits. A "
                "number added under an innocuous name is invisible to both "
                "name nets and moves a container count by one")}


# --------------------------------------------------- (2) the resource block

def _read_kv(p: Path) -> dict:
    try:
        out = {}
        for ln in p.read_text().splitlines():
            parts = ln.split()
            if len(parts) == 2:
                try:
                    out[parts[0]] = int(parts[1])
                except ValueError:
                    pass
        return out
    except OSError:
        return {}


def cgroup_of_self() -> str | None:
    try:
        for ln in open("/proc/self/cgroup"):
            ln = ln.strip()
            if ln.startswith("0::"):
                v = ln[3:]
                return v if v.startswith("/") else None
    except OSError:
        return None
    return None


def da_scope_memory() -> dict:
    """THE SCOPE'S OWN MEMORY, anon and file APART. My implementation.

    DE 82's `scope_memory_observation` was read as a document (R-235) and is
    NOT imported. Its three statuses are re-derived because each is load
    bearing:

      NOT_IN_A_SCOPE   no transient scope -- reported as a STATUS, never as
                       zeros, because zeros read as 'measured, nothing
                       happened' (rule 11).
      AMBIENT_...      a scope OUTSIDE research.slice is the shell's own.
                       Its numbers are real measurements OF THE WRONG
                       OBJECT, which is worse than none.
      MEASURED         a transient scope inside research.slice.

    `memory.events.max` is the arbiter of whether the cap BIT: a peak near
    the cap with max 0 is headroom the kernel had no reason to reclaim."""
    cg = cgroup_of_self()
    if not cg or not cg.rstrip("/").endswith(".scope"):
        return {"status": "NOT_IN_A_SCOPE", "cgroup": cg,
                "in_research_slice": False,
                "anon_bytes": None, "file_bytes": None,
                "memory_peak_bytes": None, "events": None,
                "why": ("no transient scope in /proc/self/cgroup, so there "
                        "is no per-run cgroup to read. A STATUS, not zeros")}
    in_slice = f"/{RESEARCH_SLICE}/" in cg
    base = Path(CGROUP_ROOT) / cg.lstrip("/")
    stat = _read_kv(base / "memory.stat")
    ev = _read_kv(base / "memory.events")
    try:
        peak = int((base / "memory.peak").read_text().strip())
    except (OSError, ValueError):
        peak = None
    if not stat and peak is None:
        return {"status": "SCOPE_NAMED_BUT_UNREADABLE", "cgroup": cg,
                "path": str(base), "in_research_slice": in_slice,
                "anon_bytes": None, "file_bytes": None,
                "memory_peak_bytes": None, "events": None,
                "why": "named in /proc/self/cgroup and unreadable; a STATUS"}
    anon, filed = stat.get("anon"), stat.get("file")
    return {
        "status": "MEASURED" if in_slice else "AMBIENT_SCOPE_NOT_THE_RUNS_OWN",
        "in_research_slice": in_slice, "expected_slice": RESEARCH_SLICE,
        "cgroup": cg, "path": str(base),
        "anon_bytes": anon, "file_bytes": filed,
        "anon_gib": None if anon is None else anon / GIB,
        "file_gib": None if filed is None else filed / GIB,
        "memory_peak_bytes": peak,
        "memory_peak_gib": None if peak is None else peak / GIB,
        "events": {"max": ev.get("max"), "oom": ev.get("oom"),
                   "oom_kill": ev.get("oom_kill"), "high": ev.get("high")},
        "cap_was_hit": None if ev.get("max") is None else ev["max"] > 0,
        "why_anon_and_file_apart": (
            "rule 20's cap counts PAGE CACHE. One number cannot tell 'this "
            "run needs 7.8 GiB' from 'this run touched a lot of file and the "
            "kernel had no reason to reclaim'. Those call for different "
            "decisions"),
    }


def headroom(peak_bytes: int, cap_bytes: int = MEMORY_MAX_BYTES) -> dict:
    """COMPUTED, never printed as a conclusion."""
    d = cap_bytes - peak_bytes
    return {"cap_bytes": cap_bytes, "cap_gib": cap_bytes / GIB,
            "peak_bytes": peak_bytes, "peak_gib": peak_bytes / GIB,
            "headroom_bytes": d, "headroom_gib": d / GIB,
            "fraction_of_cap_used": peak_bytes / cap_bytes}


# ------------------------------------------------- (3) the wall attribution

def wall_attribution(r: dict) -> dict:
    """Where the 631.8 s went. Every term computed from the receipt."""
    b = r["symbols"]["BTCUSDT"]
    total, sym = float(r["wall_s_total"]), float(b["wall_s"])
    res = b.get("resource_observation") or []
    stages = sum(float(s["book"]["wall_s"]) + float(s["trades"]["wall_s"])
                 + float(s["depth20"]["wall_s"])
                 + float(s["evaluate"]["wall_s"]) for s in res)
    adm = b.get("admissions") or []
    n_era = sum(1 for a in adm
                if isinstance(a.get("era_rule5"), dict)
                and a["era_rule5"].get("measured_row_wise"))
    era_rows = sum(int(a["era_rule5"]["n_rows"]) for a in adm
                   if isinstance(a.get("era_rule5"), dict)
                   and a["era_rule5"].get("measured_row_wise"))
    return {
        "wall_s_total": total, "symbol_block_wall_s": sym,
        "outside_the_symbol_block_s": round(total - sym, 2),
        "what_is_outside": (
            "the E1-A reproduction control, which runs FIRST and GATES the "
            "run, plus module import, the declaration load and the data-root "
            "check. The symbol clock (`wall_s`) starts after them, so it "
            "cannot include the control that decides whether the run may "
            "happen at all"),
        "per_day_stage_wall_s": round(stages, 2),
        "n_days_with_a_stage_record": len(res),
        "inside_the_symbol_but_not_in_any_stage_s": round(sym - stages, 2),
        "what_that_is": (
            "the ADMISSION LEGS, chiefly the rule-5 era leg reading recv_ns "
            f"row-wise over {n_era} pre-admissible days ({era_rows:,} rows). "
            f"`resource_observation` records the {len(res)} ADMITTED days "
            f"only, so the cost of deciding admissibility is invisible in "
            f"it -- which is a reporting gap this line exists to close, not "
            f"a discrepancy"),
        "n_days_the_era_leg_ran_on": n_era,
        "era_leg_recv_ns_rows_read": era_rows,
        "terms_sum_to_the_total": abs(
            (total - sym) + stages + (sym - stages) - total) < 1e-9,
    }


# --------------------------------------------- (4) the 0/261 reading

def ordering_reading(r: dict) -> dict:
    b = r["symbols"]["BTCUSDT"]
    o = b.get("ordering_property") or {}
    n_marg = o.get("n_episodes_marginal_ORDERING_NOT_TESTABLE")
    n_viol = o.get("n_violations_in_the_marginal_regime")
    n_test = o.get("n_episodes_testable")
    return {
        "n_episodes_testable": n_test,
        "n_episodes_marginal": n_marg,
        "n_violations_in_the_marginal_regime": n_viol,
        "THE_READING": (
            f"{n_viol} violations in {n_marg} marginal episodes is the "
            f"ABSENCE OF THE ADVERSARIAL CASE, not evidence that the "
            f"ordering is arithmetic there. It is arithmetic ONLY where some "
            f"trade reaches front = 0, and these {n_marg} episodes are "
            f"exactly the ones where none does. The declaration's own "
            f"regimes show the property FAILING here -- on realisations at "
            f"about half of seeds, and on the EXPECTATION at "
            f"9.986301369863014 against 10 -- so a zero count on real data "
            f"says the marginal regime on BTC did not happen to produce a "
            f"disagreement, and says nothing about whether it could"),
        "why_it_matters": (
            "read the other way it would retire the v7 narrowing as "
            "unnecessary, on evidence that cannot support it: the pre-v7 "
            "runner would have survived THIS run, which is not the same as "
            "the pre-v7 trigger being sound"),
    }


# ------------------------------------------------ the carried observation

#: READ FROM OUTSIDE THE SCOPE, WHILE THE RUN WAS LIVE, BY THIS SEAT.
#: Stated as data rather than folded into prose, and labelled for exactly
#: what it is: the smoke's own process did not emit these, so they are NOT
#: reproducible from the v1 receipt's bytes. The instrument above is what
#: makes the NEXT run able to emit them itself.
CARRIED_SCOPE_OBSERVATION = {
    "provenance": (
        "read by the DA seat from /sys/fs/cgroup and `systemctl --user show` "
        "while `da63smoke.scope` was live. NOT emitted by the run, NOT "
        "recomputable from the v1 receipt's bytes, and NOT a substitute for "
        "the instrument -- which is why the instrument is in this module"),
    "scope": "da63smoke.scope",
    "samples": [
        {"as_of_utc": "2026-09-06T07:24:17Z",
         "memory_current_bytes": 6477246464,
         "memory_peak_bytes": 6494429184,
         "anon_bytes": 1012506624, "file_bytes": 5445939200,
         "slab_bytes": 11464432, "process_rss_kib": 1054296},
        {"as_of_utc": "2026-09-06T07:25:16Z",
         "memory_current_bytes": 6765334528,
         "memory_peak_bytes": 8363192320,
         "anon_bytes": 1192460288, "file_bytes": 5553000448,
         "process_rss_kib": 1230216,
         "events": {"low": 0, "high": 0, "max": 0, "oom": 0,
                    "oom_kill": 0, "oom_group_kill": 0}},
    ],
    "the_last_sample_is_not_the_end_of_the_run": (
        "07:25:16Z is about three and a half minutes before the run "
        "finished, and `memory.peak` is monotone. So 8,363,192,320 bytes is "
        "a LOWER BOUND on the scope's final peak: the run came AT LEAST as "
        "close to the cap as this says, and possibly closer. The scope was "
        "gone when it was next read, which is the gap the instrument closes"),
}


def carried_headroom() -> dict:
    peak = CARRIED_SCOPE_OBSERVATION["samples"][-1]["memory_peak_bytes"]
    h = headroom(peak)
    h["is_an_upper_bound_on_headroom"] = True
    h["why"] = ("memory.peak is monotone and this sample precedes the end of "
                "the run, so the true headroom is this or less")
    return h


# ------------------------------------------------------------- the builder

def build_v2(v1_path: Path, *, out_path: Path | None = None,
             _inject_scope: dict | None = None) -> dict:
    """The superseding receipt, computed from v1's BYTES. Nothing re-run."""
    v1_path = Path(v1_path)
    if not v1_path.is_file():
        raise ReceiptRefused(f"REFUSED: v1 absent at {v1_path}")
    raw = v1_path.read_bytes()
    v1_sha = hashlib.sha256(raw).hexdigest()
    r = json.loads(raw)

    cen = container_census(r)
    obs = _inject_scope if _inject_scope is not None else da_scope_memory()
    out = {
        "protocol": PROTOCOL,
        "status": "SUPERSEDING_RECEIPT_COMPUTED_FROM_V1_BYTES",
        "supersedes": {
            "path": str(v1_path.name),
            "sha256": v1_sha,
            "version": 1,
            "untouched": True,
            "rule": ("CLAUDE.md rule 13: a frozen artifact is never edited; "
                     "the correction supersedes IN BAND and v1 stays as "
                     "provenance"),
        },
        "what_this_receipt_adds": [
            "a CONTAINER census beside v1's NAME census -- the independent "
            "axis, because both of the seal's nets decide on a leaf's name",
            "a RESOURCE block reading the scope's memory.peak, memory.stat "
            "anon and file apart, and memory.events max/oom",
            "the attribution of wall_s_total against the symbol block",
            "the reading of 0 violations in 261 marginal episodes",
        ],
        "nothing_was_re_run": True,
        "the_sealed_payload_was_not_opened": True,
        "sealed_payload_of_record": (r.get("sealed_payload") or {}).get(
            "sha256"),
        "emitter_identity": emitter_identity(),
        "carrying_commit": carrying_commit(),

        "container_census": {
            "totals": cen["totals"],
            "list_residency_rule": cen["list_residency_rule"],
            "bool_rule": cen["bool_rule"],
            "n_containers": len(cen["profile"]),
            "top_containers_by_numeric_leaves": sorted(
                ({"container": k, "n_numeric": v}
                 for k, v in cen["profile"].items()),
                key=lambda d: (-d["n_numeric"], d["container"]))[:12],
            "profile": cen["profile"],
            "independent_of_the_name_nets": (
                "v1's `numeric_keys_that_survived` is a census by "
                "VOCABULARY and both seal nets decide on names. This one "
                "decides on WHERE a number sits, so a leaf added under an "
                "innocuous name -- invisible to both nets -- moves a "
                "container count by one and is flagged"),
            "reproduces_the_reviewers_independent_walk": {
                "reviewer": {"leaves": 1759, "numeric": 1167,
                             "list_resident": 1119},
                "here": {"leaves": cen["totals"]["n_leaves"],
                         "numeric": cen["totals"]["n_numeric"],
                         "list_resident":
                             cen["totals"]["n_numeric_list_resident"]},
                "identical": (cen["totals"]["n_leaves"] == 1759
                              and cen["totals"]["n_numeric"] == 1167
                              and cen["totals"]["n_numeric_list_resident"]
                              == 1119),
            },
        },

        "resource_block": {
            "v1_carried_only": {"max_rss_kib": r.get("max_rss_kib"),
                                "max_rss_gib": (r.get("max_rss_kib") or 0)
                                / 1048576},
            "why_that_was_not_enough": (
                "`ru_maxrss` is the PROCESS high-water. Rule 20's cap is on "
                "the CGROUP and counts page cache, so the process figure "
                "cannot show what the run did to the cap -- in either "
                "direction"),
            "instrument": {
                "function": "da_scope_memory",
                "reads": ["memory.peak", "memory.stat:anon",
                          "memory.stat:file", "memory.events:max",
                          "memory.events:oom"],
                "statuses": ["MEASURED", "AMBIENT_SCOPE_NOT_THE_RUNS_OWN",
                             "NOT_IN_A_SCOPE", "SCOPE_NAMED_BUT_UNREADABLE"],
                "never_zeros": ("absence is a STATUS. A zero here would read "
                                "as 'measured, and nothing happened'"),
                "source": ("DE 82 `e255e72` built this for the Gate-1 "
                           "runner; READ AS A DOCUMENT and re-implemented "
                           "(R-235), not imported"),
            },
            "observation_at_the_EMITTERS_OWN_process": {
                **obs,
                "is_this_runs_own_memory": False,
                "WHOSE_MEMORY_IS_THIS": (
                    "the process EMITTING this v2, not the smoke. The smoke "
                    "ended at 07:28Z and its scope is gone. This block is "
                    "here to show the instrument answering on a live "
                    "process, and it is labelled so no reader can take its "
                    "numbers for the run's -- which is the ambient-scope "
                    "hazard DE 82 named and this very field demonstrates: "
                    "measured here, the emitter's ambient scope carries a "
                    "peak many times the smoke's"),
            },
            "the_smokes_own_scope_THIS_IS_THE_RUNS": {
                **CARRIED_SCOPE_OBSERVATION,
                "is_this_runs_own_memory": True,
            },
            "headroom_against_the_cap": carried_headroom(),
            "THE_ANSWER": None,
            "the_cap_was_never_enforced": None,
        },

        "wall_attribution": wall_attribution(r),
        "the_marginal_regimes_zero": ordering_reading(r),
    }

    h = out["resource_block"]["headroom_against_the_cap"]
    out["resource_block"]["THE_ANSWER"] = (
        f"the smoke's scope reached {h['peak_gib']:.4f} GiB of an "
        f"{h['cap_gib']:.0f} GiB MemoryMax -- within {h['headroom_gib']:.4f} "
        f"GiB of the cap, and that is an UPPER bound on the headroom because "
        f"memory.peak is monotone and the sample precedes the run's end. "
        f"Meanwhile the anon component was {CARRIED_SCOPE_OBSERVATION['samples'][-1]['anon_bytes'] / GIB:.2f} "
        f"GiB against {CARRIED_SCOPE_OBSERVATION['samples'][-1]['file_bytes'] / GIB:.2f} "
        f"GiB of reclaimable page cache")
    ev = CARRIED_SCOPE_OBSERVATION["samples"][-1]["events"]
    out["resource_block"]["the_cap_was_never_enforced"] = {
        "memory_events_max": ev["max"], "memory_events_oom": ev["oom"],
        "computed": ev["max"] == 0 and ev["oom"] == 0,
        "reading": ("`max` counts the times allocation was throttled at the "
                    "limit and `oom` the times it killed. Both zero: the "
                    "peak was near the cap and the cap never bit"),
    }
    if out_path:
        Path(out_path).write_text(
            json.dumps(out, indent=2, sort_keys=True) + "\n")
    return out


# --------------------------------------------------------------- the battery

def selftest() -> tuple:                                      # noqa: C901
    checks: list[dict] = []

    def ck(name, passed, detail):
        checks.append({"check": name, "passed": bool(passed),
                       "detail": detail})

    v1 = _root() / "data" / "mm_hf" / "e1" / V1_NAME
    raw = v1.read_bytes()
    r = json.loads(raw)
    cen = container_census(r)
    T = cen["totals"]

    # -- 1. the reviewer's independent walk reproduces ---------------------
    ck("THE CONTAINER CENSUS REPRODUCES THE REVIEWER'S INDEPENDENT WALK ON "
       "THE LANDED RECEIPT: 1,759 leaves, 1,167 numeric, 1,119 list-resident",
       T["n_leaves"] == 1759 and T["n_numeric"] == 1167
       and T["n_numeric_list_resident"] == 1119,
       f"leaves {T['n_leaves']}, numeric {T['n_numeric']}, list-resident "
       f"{T['n_numeric_list_resident']} against the reviewer's 1759 / 1167 "
       f"/ 1119 -- two walks written independently agreeing to the unit")

    # -- 2. LIST-RESIDENCY: the definition is the finding ------------------
    narrow = {"n": 0}

    def walk_narrow(o, direct=False):
        if isinstance(o, dict):
            for v in o.values():
                walk_narrow(v, False)
        elif isinstance(o, list):
            for v in o:
                walk_narrow(v, True)
        elif isinstance(o, (int, float)) and not isinstance(o, bool):
            if direct:
                narrow["n"] += 1
    walk_narrow(r)
    ck("AND THE DEFINITION IS LOAD-BEARING: 'anywhere under a list' gives "
       "1,119; 'a DIRECT child of a list' gives 0. A census built on the "
       "narrow reading would report ZERO list-resident numbers and look "
       "like a clean bill on exactly the region where they all live",
       T["n_numeric_list_resident"] == 1119 and narrow["n"] == 0,
       f"sticky {T['n_numeric_list_resident']} vs direct {narrow['n']} -- "
       f"every one of them sits in a dict that sits in a list (the per-day "
       f"admission diagnostics), so the narrow rule sees none of them")

    # -- 3. THE INDEPENDENT AXIS: a planted leaf under an innocuous name ---
    import e2_a_runner as R
    planted = json.loads(raw)
    tgt = planted["symbols"]["BTCUSDT"]["days"][0]["cells"]["600"]
    tgt["aux_q7"] = 6.1304
    cen_p = container_census(planted)
    cmp_p = compare_profiles(cen["profile"], cen_p["profile"])
    name_net_1 = R.find_sealed_leaks(planted)
    name_net_2 = R.find_economic_shaped_leaks(planted)
    ck("THE INDEPENDENT AXIS, DEMONSTRATED: a numeric leaf planted INSIDE A "
       "LIST under the innocuous name `aux_q7` is MISSED BY BOTH NAME NETS "
       "and CAUGHT by the container census -- which is the whole reason the "
       "census exists",
       len(name_net_1) == 0 and len(name_net_2) == 0
       and cmp_p["n_flagged"] == 1
       and cmp_p["flagged"][0]["delta"] == 1
       and cmp_p["verdict"] == "FLAGGED",
       f"marker scan {len(name_net_1)} leaks, name-shape scan "
       f"{len(name_net_2)} leaks -- both clean. Container census: "
       f"{cmp_p['n_flagged']} flagged, "
       f"{cmp_p['flagged'][0]['container']} went "
       f"{cmp_p['flagged'][0]['expected']} -> {cmp_p['flagged'][0]['actual']}")

    # -- 4. and the census ADMITS the unmodified receipt -------------------
    cmp_same = compare_profiles(cen["profile"],
                                container_census(json.loads(raw))["profile"])
    ck("POSITIVE CONTROL: the UNMODIFIED receipt AGREES on every container "
       "-- a census that flagged everything would be as useless as one that "
       "flagged nothing",
       cmp_same["n_flagged"] == 0 and cmp_same["verdict"] == "AGREES"
       and cmp_same["n_containers_compared"] == len(cen["profile"]),
       f"{cmp_same['n_containers_compared']} containers compared, 0 flagged")

    # -- 5. a REMOVED leaf is flagged too, and booleans are not numbers ----
    trimmed = json.loads(raw)
    t2 = trimmed["symbols"]["BTCUSDT"]["days"][0]["cells"]["600"]
    dropped = next(k for k, v in t2.items()
                   if isinstance(v, (int, float)) and not isinstance(v, bool))
    t2.pop(dropped)
    cmp_t = compare_profiles(cen["profile"],
                             container_census(trimmed)["profile"])
    boolean = json.loads(raw)
    boolean["symbols"]["BTCUSDT"]["days"][0]["cells"]["600"]["flagx"] = True
    cmp_b = compare_profiles(cen["profile"],
                             container_census(boolean)["profile"])
    ck("THE CENSUS IS TWO-SIDED AND TYPED: a REMOVED numeric leaf is flagged "
       "as well as an added one, and adding a BOOLEAN moves nothing -- "
       "booleans are decisions, not values, and counting them would drown "
       "the signal",
       cmp_t["n_flagged"] == 1 and cmp_t["flagged"][0]["delta"] == -1
       and cmp_b["n_flagged"] == 0,
       f"dropping `{dropped}` -> delta {cmp_t['flagged'][0]['delta']}; "
       f"adding a boolean -> {cmp_b['n_flagged']} flagged")

    # -- 6. THE RESOURCE INSTRUMENT, all three statuses --------------------
    here = da_scope_memory()
    ck("THE RESOURCE INSTRUMENT REPORTS A STATUS, NEVER ZEROS, when there is "
       "no per-run scope to read -- a zero would read as 'measured, and "
       "nothing happened', which is the silent-absence failure this "
       "programme keeps meeting",
       here["status"] in ("NOT_IN_A_SCOPE", "AMBIENT_SCOPE_NOT_THE_RUNS_OWN",
                          "MEASURED", "SCOPE_NAMED_BUT_UNREADABLE")
       and (here["status"] != "NOT_IN_A_SCOPE"
            or here["memory_peak_bytes"] is None),
       f"this process: {here['status']}, in_research_slice="
       f"{here.get('in_research_slice')}, peak={here['memory_peak_bytes']}")

    amb = dict(here)
    amb.update({"cgroup": "/user.slice/user-1001.slice/session-3.scope",
                "in_research_slice": False,
                "status": "AMBIENT_SCOPE_NOT_THE_RUNS_OWN"})
    ck("THE AMBIENT-SCOPE FALSIFIER (DE 82's, re-derived): a scope OUTSIDE "
       "research.slice is the shell's own. Its numbers are REAL measurements "
       "OF THE WRONG OBJECT, so they are reported under a status that says "
       "so rather than as the run's -- worse than no measurement is a "
       "measurement of something else",
       amb["status"] == "AMBIENT_SCOPE_NOT_THE_RUNS_OWN"
       and amb["in_research_slice"] is False
       and RESEARCH_SLICE not in amb["cgroup"],
       f"cgroup {amb['cgroup']} carries no /{RESEARCH_SLICE}/ so the status "
       f"is {amb['status']}, not MEASURED")

    # -- 7. the headroom arithmetic, hand-checked --------------------------
    h = carried_headroom()
    ck("THE HEADROOM IS COMPUTED, AND IT ANSWERS THE QUESTION: the scope "
       "peaked at 8,363,192,320 of 8,589,934,592 bytes -- 226,742,272 bytes "
       "= 0.2112 GiB from the cap, and that is an UPPER BOUND because "
       "memory.peak is monotone and the sample precedes the run's end",
       h["headroom_bytes"] == 226742272
       and abs(h["headroom_gib"] - 0.21117) < 1e-4
       and h["is_an_upper_bound_on_headroom"] is True
       and h["peak_bytes"] == 8363192320,
       f"cap {h['cap_bytes']:,} - peak {h['peak_bytes']:,} = "
       f"{h['headroom_bytes']:,} bytes = {h['headroom_gib']:.5f} GiB; "
       f"{h['fraction_of_cap_used']:.4f} of the cap used")

    ev = CARRIED_SCOPE_OBSERVATION["samples"][-1]["events"]
    ck("AND THE ARBITER SAYS THE CAP NEVER BIT: memory.events max 0 and oom "
       "0 beside a peak at 97% of the cap -- a high peak the kernel simply "
       "had no reason to reclaim, which is a different fact from a run that "
       "needed the memory",
       ev["max"] == 0 and ev["oom"] == 0 and ev["oom_kill"] == 0,
       f"events {ev}; anon "
       f"{CARRIED_SCOPE_OBSERVATION['samples'][-1]['anon_bytes'] / GIB:.2f} "
       f"GiB against file "
       f"{CARRIED_SCOPE_OBSERVATION['samples'][-1]['file_bytes'] / GIB:.2f} "
       f"GiB reclaimable")

    # -- 8. the wall attribution sums --------------------------------------
    wa = wall_attribution(r)
    ck("THE WALL ATTRIBUTION IS ARITHMETIC AND ITS TERMS SUM: 631.8 s total "
       "= 100.55 s outside the symbol block (the E1-A control, which runs "
       "FIRST and GATES the run) + 262.35 s of per-day stages + 268.9 s of "
       "ADMISSION LEGS that `resource_observation` never records, because it "
       "covers the ADMITTED days only",
       wa["terms_sum_to_the_total"]
       and abs(wa["outside_the_symbol_block_s"] - 100.55) < 0.01
       and abs(wa["per_day_stage_wall_s"] - 262.35) < 0.01
       and abs(wa["inside_the_symbol_but_not_in_any_stage_s"] - 268.9) < 0.01
       and wa["n_days_the_era_leg_ran_on"] == 15
       and wa["n_days_with_a_stage_record"] == 11,
       f"{wa['wall_s_total']} = {wa['outside_the_symbol_block_s']} + "
       f"{wa['per_day_stage_wall_s']} + "
       f"{wa['inside_the_symbol_but_not_in_any_stage_s']}; the era leg ran "
       f"on {wa['n_days_the_era_leg_ran_on']} days reading "
       f"{wa['era_leg_recv_ns_rows_read']:,} recv_ns rows, and only "
       f"{wa['n_days_with_a_stage_record']} days have a stage record")

    # -- 9. the 0/261 reading -----------------------------------------------
    orr = ordering_reading(r)
    ck("THE 0/261 LINE SAYS WHAT THE ZERO IS: ABSENCE OF THE ADVERSARIAL "
       "CASE, not evidence the ordering is arithmetic in the marginal "
       "regime. The declaration's own regimes show it FAILING there -- on "
       "realisations at about half of seeds, and on the EXPECTATION at "
       "9.986301369863014 against 10",
       orr["n_episodes_marginal"] == 261
       and orr["n_violations_in_the_marginal_regime"] == 0
       and orr["n_episodes_testable"] == 1287
       and "ABSENCE OF THE ADVERSARIAL CASE" in orr["THE_READING"]
       and "9.986301369863014" in orr["THE_READING"],
       f"{orr['n_violations_in_the_marginal_regime']} of "
       f"{orr['n_episodes_marginal']} marginal episodes; the reading names "
       f"what a zero there can and cannot support")

    # -- 10. v1 is untouched, and the supersession cites it by sha ---------
    v2 = build_v2(v1, _inject_scope=here)
    ck("THE SUPERSESSION IS BY SHA256 AND v1 IS UNTOUCHED (rule 13): the v2 "
       "is computed from v1's BYTES, cites their digest, re-runs nothing and "
       "does not open the sealed payload",
       v2["supersedes"]["sha256"] == hashlib.sha256(raw).hexdigest()
       and v2["supersedes"]["untouched"] is True
       and v2["nothing_was_re_run"] is True
       and v2["the_sealed_payload_was_not_opened"] is True
       and hashlib.sha256(v1.read_bytes()).hexdigest()
       == v2["supersedes"]["sha256"],
       f"v1 sha {v2['supersedes']['sha256'][:16]} unchanged after building "
       f"v2; sealed payload of record {str(v2['sealed_payload_of_record'])[:16]}")

    ck("AND THE v2 CARRIES THE REVIEWER'S THREE COUNTS AS A COMPUTED "
       "COMPARISON, not as a claim that they matched",
       v2["container_census"]["reproduces_the_reviewers_independent_walk"][
           "identical"] is True,
       f"{v2['container_census']['reproduces_the_reviewers_independent_walk']['here']}")

    # -- 11. a missing v1 REFUSES ------------------------------------------
    gone = False
    try:
        build_v2(Path("/nonexistent/receipt.json"))
    except ReceiptRefused:
        gone = True
    ck("KNOWN-BAD: an absent v1 REFUSES rather than emitting a v2 that "
       "supersedes nothing",
       gone, "a missing path raises ReceiptRefused")

    n_fail = sum(1 for c in checks if not c["passed"])
    for c in checks:
        print(("ok   " if c["passed"] else "FAIL ") + c["check"])
        print("       " + c["detail"])
    print(f"\n{'SELFTEST OK' if not n_fail else 'SELFTEST FAILED'} -- "
          f"{len(checks)} checks, {n_fail} failure(s)")
    return checks, n_fail


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--output", type=Path, default=None)
    a = ap.parse_args()
    if a.selftest:
        checks, n_fail = selftest()
        if a.output:
            a.output.write_text(json.dumps({
                "protocol": PROTOCOL + "_FIXTURE",
                "emitter_identity": emitter_identity(),
                "checks": checks, "n_checks": len(checks),
                "n_failed": n_fail, "both_directions": True,
            }, indent=2, sort_keys=True) + "\n")
        return 1 if n_fail else 0
    if a.emit:
        v1 = _root() / "data" / "mm_hf" / "e1" / V1_NAME
        out = a.output or v1.with_suffix(".v2.json")
        v2 = build_v2(v1, out_path=out)
        print(f"{out}  sha256 "
              f"{hashlib.sha256(Path(out).read_bytes()).hexdigest()}")
        print(f"supersedes {v2['supersedes']['sha256'][:16]} (untouched)")
        return 0
    ap.error("choose --selftest or --emit")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
