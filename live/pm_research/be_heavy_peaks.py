"""THE ONE WRITER of `live/pm_research/be_heavy_peaks.jsonl` (REV 89 §3.2).

WHY THIS MODULE EXISTS. The roll-up was HAND-MAINTAINED, and REV 89 measured
the drift at `ad131f9`: 9 units named in the file against 45 with a launcher
record; `be72frag` and `be87fwd06` had records and no row; `be64book` had a
row and no record; `grep -rn "be_heavy_peaks" live/ scripts/` returned zero
hits, so nothing produced it and nothing read it. Q-BE-332 promised "tomorrow's
close reads the trend from the file rather than from a report someone has to
find" -- a promise a hand-kept file cannot keep unattended (R-605: a practice
that depends on noticing is not a control).

THE SOURCES, AND WHICH FACT COMES FROM WHICH:

  the LEAF peak (the peak of record)  <- the launcher record,
      `data/pm_5min/derived/be_heavy_run_record_<unit>.jsonl`, which is the
      only place a cgroup leaf is ever read WHILE THE RUN IS ALIVE;
  the IN-PROCESS peak                 <- the producer's own receipt, one
      named field per stage (below), because it is the builder's own RSS and
      the builder is the only thing that can measure it.

They are two quantities and are never conflated (BE 74): the leaf charges the
cgroup, page cache included; the in-process number is the process's RSS.
systemd's `MemoryPeak` property is a THIRD number and never stands in for the
leaf -- it is carried in the record and not copied here.

WHAT IS EXCLUDED, AND BY WHAT. Not by a list of names. A record joins the
per-day roll-up only if what IT SAYS puts it in the day chain: its launch
names a payload under `live/pm_research/` AND its arguments name a day. That
is what separates the eleven real runs from the probes (`--selftest`), the
race read (`--open`), and the thirty scratch and `be_hr_falsify_*` units whose
payload is a file in /tmp or a scratchpad. Every record lands in one of the
named classes below and the counts are published in the header, so an
exclusion is a status with a reason, never a silent absence (rule 4).
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import be_data_root as _BDR                                    # noqa: E402

OUT = HERE / "be_heavy_peaks.jsonl"
PROTOCOL = "BE_HEAVY_PEAKS_V2"
CAP_BYTES = 8589934592                    # MemoryMax=8G, rule 20; never raised
GIB = 1024 ** 3
RECORD_STEM = "be_heavy_run_record_"
#: `\b` does NOT work here: in `be_daybook_20260903_btc.pkl` the day is
#: flanked by underscores, which are word characters, so there is no
#: boundary and every `--verify-structure <path>` run parsed as naming no
#: day at all. Measured: five of the eleven day-chain runs were excluded.
DAY_RE = re.compile(r"(?<!\d)(20\d{6})(?!\d)")

#: payload module -> (stage, the receipt that carries the IN-PROCESS peak,
#: the dotted paths to try IN ORDER, the dotted paths that may name the unit).
#: `be_daybook_build.py` is two stages and the ARGUMENTS separate them.
STAGES = {
    "be_gate1_fragment.py": ("fragment", "be_gate1_fragment_receipt_{day}_btc",
                             ("resources.peak_rss_gb",), ("scope.unit",)),
    "be_gate1_state_tape.py": ("tape", "be_gate1_state_tape_receipt_{day}_btc",
                               ("resources.peak_rss_gb",), ("scope.unit",)),
    "be_daybook_build.py": ("book", "be_daybook_receipt_{day}_btc",
                            ("resources.asm_peak_gb_PUBLISHED",),
                            ("journal_copy.unit", "scope.unit")),
    "be_forward_day.py": ("forward_day", "be_forward_day_receipt_{day}",
                          (), ()),
}
STRUCTURE = ("structure_verification",
             "be_daybook_structure_verification_{day}_btc",
             ("peak_rss_gb", "peak_of_record.builder_own_measurement_peak_rss_gb"),
             ("unit",))

#: THE STAGES A DAY'S CHAIN HAS. Rows are emitted for every one of these on
#: every day any of them is seen, so a MISSING stage is a visible row with a
#: named status rather than a gap a reader has to notice.
CHAIN_STAGES = ("fragment", "tape", "book", "structure_verification",
                "forward_day")


class PeaksRefused(RuntimeError):
    """A named refusal. Every message begins with its own reason code."""


def _derived() -> Path:
    return Path(_BDR.data_root()) / "pm_5min" / "derived"


def _dotted(doc, path):
    cur = doc
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _events(record: Path) -> list:
    out = []
    for line in record.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def classify_record(unit: str, events: list) -> dict:
    """Does this launcher record belong in the PER-DAY roll-up?

    DECIDED FROM WHAT THE RECORD SAYS, NEVER FROM THE UNIT'S NAME. A name is
    a literal standing in for a fact, and this seat's recurring defect; the
    launch event already carries the two facts that matter.

    Four named classes, and every record gets exactly one:

      DAY_CHAIN_RUN              a payload under `live/pm_research/` AND a day
                                 in the arguments -- the run belongs;
      NOT_A_TRACKED_PRODUCER     the payload is a file in /tmp or a scratchpad
                                 (the `be_hr_falsify_*` units and the hold /
                                 sleeper / allocator probes);
      NAMES_NO_DAY               a tracked producer run for something other
                                 than a day -- `--selftest`, `--open`;
      NO_LAUNCH_EVENT            a record with nothing to classify.

    A record whose launches name MORE THAN ONE day is `DAY_AMBIGUOUS`: it is
    excluded and SAID, never silently resolved to the first or the last.
    """
    launches = [e for e in events if e.get("event") == "launch"]
    if not launches:
        return {"unit": unit, "in_rollup": False, "why": "NO_LAUNCH_EVENT",
                "stage": None, "day": None}
    payloads = {str(e.get("payload") or "") for e in launches}
    tracked = {p for p in payloads if "live/pm_research/" in p}
    if not tracked:
        return {"unit": unit, "in_rollup": False,
                "why": "NOT_A_TRACKED_PRODUCER", "stage": None, "day": None,
                "payloads": sorted(payloads)}
    days = set()
    for e in launches:
        days |= set(DAY_RE.findall(str(e.get("args") or "")))
    if not days:
        return {"unit": unit, "in_rollup": False, "why": "NAMES_NO_DAY",
                "stage": None, "day": None, "payloads": sorted(payloads)}
    if len(days) > 1:
        return {"unit": unit, "in_rollup": False, "why": "DAY_AMBIGUOUS",
                "stage": None, "day": None, "days_named": sorted(days)}
    base = sorted(Path(p).name for p in tracked)[0]
    stage = STAGES.get(base, (None,))[0]
    if base == "be_daybook_build.py" and any(
            "--verify-structure" in str(e.get("args") or "") for e in launches):
        stage = STRUCTURE[0]
    if stage is None:
        return {"unit": unit, "in_rollup": False, "why": "PAYLOAD_NOT_A_STAGE",
                "stage": None, "day": None, "payloads": sorted(payloads)}
    return {"unit": unit, "in_rollup": True, "why": "DAY_CHAIN_RUN",
            "stage": stage, "day": sorted(days)[0]}


def leaf_peak_from_record(events: list) -> dict:
    """The leaf peak OF RECORD, or a named absence -- never a zero.

    Only a reading taken WHILE THE RUN WAS ALIVE counts: the leaf is released
    at exit and `--capture` on a finished unit reports `LEAF_RELEASED`. A
    `leaf_peak` event is the launcher's own sampler (BE 87); a
    `leaf_peak_recorded_by_hand` event is a scratch watcher from before the
    sampler existed, carried in the record with its provenance rather than in
    a roll-up nobody produces.
    """
    for kind in ("leaf_peak", "leaf_peak_recorded_by_hand"):
        rows = [e for e in events if e.get("event") == kind]
        if not rows:
            continue
        row = rows[-1]
        peak = row.get("peak_of_record_bytes")
        if not isinstance(peak, int) or peak <= 0:
            continue
        return {"leaf_peak_status": "MEASURED", "leaf_peak_bytes": peak,
                "leaf_peak_is_a_lower_bound": bool(row.get("is_a_lower_bound")),
                "leaf_peak_note": row.get("note") or (
                    f"the LAUNCHER's own --sample, {row.get('n_samples')} "
                    f"sample(s), last increase {row.get('last_increase_utc')}"),
                "leaf_peak_source": f"{kind} event in the launcher record"}
    return {"leaf_peak_status": "NOT_RECORDED",
            "leaf_peak_bytes": None,
            # NEVER `false` about a number that does not exist (REV 89 §3.2).
            "leaf_peak_is_a_lower_bound": None,
            "leaf_peak_note": "the leaf was never read while the run was "
                              "alive; it is released at exit and cannot be "
                              "read afterwards",
            "leaf_peak_source": None}


def _receipt_head(derived: Path, stem: str):
    """The receipt for a stage/day: the highest superseding version present."""
    cands = sorted(glob.glob(str(derived / (stem + ".json")))
                   + sorted(glob.glob(str(derived / (stem + ".v*.json")))))
    if not cands:
        return None, None

    def ver(p):
        m = re.search(r"\.v(\d+)\.json$", p)
        return int(m.group(1)) if m else 1
    best = sorted(cands, key=ver)[-1]
    try:
        return Path(best), json.load(open(best))
    except (OSError, json.JSONDecodeError):
        return Path(best), None


def _in_process(derived: Path, stage: str, day: str) -> dict:
    spec = STRUCTURE if stage == STRUCTURE[0] else next(
        (v for v in STAGES.values() if v[0] == stage), None)
    if spec is None:
        return {"in_process_peak_bytes": None, "in_process_peak_gb": None,
                "in_process_source": f"NOT_RECORDED -- no receipt family is "
                                     f"declared for stage {stage!r}"}
    _, stem, fields, unit_fields = spec
    path, doc = _receipt_head(derived, stem.format(day=day))
    if path is None:
        return {"in_process_peak_bytes": None, "in_process_peak_gb": None,
                "in_process_source": f"NOT_RECORDED -- no receipt matching "
                                     f"{stem.format(day=day)}(.vN).json"}
    if doc is None:
        return {"in_process_peak_bytes": None, "in_process_peak_gb": None,
                "in_process_source": f"NOT_RECORDED -- {path.name} is present "
                                     f"and unreadable, which is not absent"}
    if not fields:
        return {"in_process_peak_bytes": None, "in_process_peak_gb": None,
                "in_process_source": f"NOT_RECORDED -- {path.name} carries no "
                                     f"memory field at all, so for this stage "
                                     f"the launcher's leaf sample is the ONLY "
                                     f"measurement that exists"}
    for f in fields:
        gb = _dotted(doc, f)
        if isinstance(gb, (int, float)):
            return {"in_process_peak_bytes": round(float(gb) * GIB),
                    "in_process_peak_gb": gb,
                    "in_process_source": f"{path.name} :: {f} -- the "
                                         f"builder's own peak RSS"}
    return {"in_process_peak_bytes": None, "in_process_peak_gb": None,
            "in_process_source": f"NOT_RECORDED -- {path.name} carries none "
                                 f"of {list(fields)}"}


def _unit_from_receipt(derived: Path, stage: str, day: str):
    spec = STRUCTURE if stage == STRUCTURE[0] else next(
        (v for v in STAGES.values() if v[0] == stage), None)
    if spec is None:
        return None, None
    _, stem, _, unit_fields = spec
    path, doc = _receipt_head(derived, stem.format(day=day))
    if doc is None:
        return None, None
    for f in unit_fields:
        u = _dotted(doc, f)
        if isinstance(u, str) and u:
            return u, f"{path.name} :: {f}"
    return None, None


def _ratio(n, d):
    return None if not isinstance(n, int) else round(n / d, 4)


def build(records_dir=None, derived=None) -> tuple:
    """(header, rows), DERIVED. Nothing in the output is typed by hand."""
    derived = Path(derived) if derived else _derived()
    records_dir = Path(records_dir) if records_dir else derived
    classes, by_unit = {}, {}
    for f in sorted(records_dir.glob(f"{RECORD_STEM}*.jsonl")):
        unit = f.name[len(RECORD_STEM):-len(".jsonl")]
        ev = _events(f)
        c = classify_record(unit, ev)
        classes[unit] = c
        if c["in_rollup"]:
            by_unit[unit] = (c, ev)
    days = {c["day"] for (c, _) in by_unit.values()}
    for stage_spec in list(STAGES.values()) + [STRUCTURE]:
        stem = stage_spec[1]
        pre = stem.split("{day}")[0]
        for p in derived.glob(pre + "*.json"):
            m = DAY_RE.search(p.name)
            if m:
                days.add(m.group(1))
    rows = []
    for day in sorted(d for d in days if d):
        for stage in CHAIN_STAGES:
            runs = sorted(u for u, (c, _) in by_unit.items()
                          if c["day"] == day and c["stage"] == stage)
            ip = _in_process(derived, stage, day)
            if runs:
                ru, _ = _unit_from_receipt(derived, stage, day)
                for unit in runs:
                    c, ev = by_unit[unit]
                    row = {"day": day, "stage": stage,
                           "unit": f"{unit}.service",
                           "unit_source": f"{RECORD_STEM}{unit}.jsonl",
                           "cap": "MemoryMax=8G", "cap_bytes": CAP_BYTES}
                    row.update(leaf_peak_from_record(ev))
                    # THE IN-PROCESS PEAK BELONGS TO THE RUN THAT WROTE THE
                    # RECEIPT, AND TO NO OTHER. Two runs can share a (day,
                    # stage) -- `be72frag` REFUSED and wrote nothing while
                    # `be72frag2` produced the fragment; `be74struct04b` re-ran
                    # `be74struct04`'s day. Handing the survivor's number to
                    # the refused run would publish a measurement of a run
                    # that never happened, which is the shape of defect this
                    # file exists to stop.
                    if ru and ru != f"{unit}.service":
                        row.update({
                            "in_process_peak_bytes": None,
                            "in_process_peak_gb": None,
                            "in_process_source":
                                f"NOT_RECORDED -- the {stage} receipt for "
                                f"{day} names {ru}, not this run. A receipt's "
                                f"peak is that run's and is not shared across "
                                f"runs of the same stage"})
                    else:
                        row.update(ip)
                    row["ratio_leaf_to_cap"] = _ratio(row["leaf_peak_bytes"],
                                                      CAP_BYTES)
                    row["ratio_in_process_to_cap"] = _ratio(
                        row["in_process_peak_bytes"], CAP_BYTES)
                    rows.append(row)
                continue
            u, usrc = _unit_from_receipt(derived, stage, day)
            has_receipt = ip["in_process_source"].startswith(
                "NOT_RECORDED -- no receipt matching") is False
            if u is None and not has_receipt:
                continue
            row = {"day": day, "stage": stage, "unit": u,
                   "unit_source": usrc,
                   "no_launcher_record": (
                       "UNIT_NAMED_BY_RECEIPT_ONLY -- this run left no "
                       f"{RECORD_STEM}<unit>.jsonl. Runs before the service "
                       "form were transient `--scope`s and the launcher that "
                       "writes records did not exist" if u else
                       "NO_UNIT_NAMED -- neither a launcher record nor the "
                       "receipt names the unit this stage ran under"),
                   "cap": "MemoryMax=8G", "cap_bytes": CAP_BYTES}
            row.update({"leaf_peak_status": "NOT_RECORDED",
                        "leaf_peak_bytes": None,
                        "leaf_peak_is_a_lower_bound": None,
                        "leaf_peak_note": "no launcher record exists for this "
                                          "run, so the leaf was never read "
                                          "while it was alive",
                        "leaf_peak_source": None})
            row.update(ip)
            row["ratio_leaf_to_cap"] = None
            row["ratio_in_process_to_cap"] = _ratio(
                row["in_process_peak_bytes"], CAP_BYTES)
            rows.append(row)
    tally = {}
    for c in classes.values():
        tally[c["why"]] = tally.get(c["why"], 0) + 1
    header = {
        "protocol": PROTOCOL,
        "as_of_utc": dt.datetime.now(dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "n_rows": len(rows),
        "cap_bytes": CAP_BYTES,
        "cap_is_never_raised": "rule 20. A run that approaches it is a fact "
                               "about the run, not a case for a bigger cap.",
        "derived_by": "live/pm_research/be_heavy_peaks.py -- ONE writer, from "
                      "the launcher records plus the receipts. Hand-editing "
                      "this file is what REV 89 §3.2 found drifted.",
        "what_this_is": "one row per heavy run: the cgroup LEAF peak (the "
                        "peak of record) from the launcher record, the "
                        "builder's own in-process peak from its receipt, the "
                        "cap and the ratios. Two different quantities, never "
                        "conflated.",
        "absences_are_statuses": "NOT_RECORDED with its reason, never 0 and "
                                 "never the other measurement standing in "
                                 "(rule 4; BE 74 on systemd's property). "
                                 "`leaf_peak_is_a_lower_bound` is null, not "
                                 "false, where there is no number to bound.",
        "records_seen": len(classes),
        "records_by_class": dict(sorted(tally.items())),
        "excluded_by": "classify_record -- what the launch event SAYS (a "
                       "payload under live/pm_research/ AND a day in the "
                       "args), never the unit's name.",
    }
    return header, rows


#: The two correspondence failures, each with its own name.
RECORD_WITHOUT_ROW = "RECORD_WITHOUT_ROW"
ROW_WITHOUT_RECORD = "ROW_WITHOUT_RECORD"


def check(header, rows, records_dir=None, derived=None) -> list:
    """Every day-chain record has a row, and every row is supported.

    THE FIRST is REV 89 §3.2's finding: `be72frag` and `be87fwd06` ran,
    refused and left records, and the hand-kept file knew about neither.

    THE SECOND is its mirror: a row naming a unit that no record and no
    receipt supports is a hand insertion. A row whose unit comes from the
    RECEIPT alone is NOT a violation -- it is the pre-service-form era, and
    the row says so in `no_launcher_record`.
    """
    derived = Path(derived) if derived else _derived()
    records_dir = Path(records_dir) if records_dir else derived
    out = []
    named = {r.get("unit") for r in rows if r.get("unit")}
    for f in sorted(records_dir.glob(f"{RECORD_STEM}*.jsonl")):
        unit = f.name[len(RECORD_STEM):-len(".jsonl")]
        c = classify_record(unit, _events(f))
        if c["in_rollup"] and f"{unit}.service" not in named:
            out.append({"kind": RECORD_WITHOUT_ROW, "unit": unit,
                        "detail": f"{f.name} classifies as DAY_CHAIN_RUN "
                                  f"({c['day']} {c['stage']}) and no row "
                                  f"names it"})
    for r in rows:
        u = r.get("unit")
        if not u:
            continue
        stem = u[:-len(".service")] if u.endswith(".service") else u
        if (records_dir / f"{RECORD_STEM}{stem}.jsonl").exists():
            continue
        if r.get("unit_source") and "receipt" not in str(r["unit_source"]):
            out.append({"kind": ROW_WITHOUT_RECORD, "unit": u,
                        "detail": f"the row claims {r['unit_source']} and no "
                                  f"such record exists"})
        elif not r.get("unit_source"):
            out.append({"kind": ROW_WITHOUT_RECORD, "unit": u,
                        "detail": "a row naming a unit with no launcher "
                                  "record and no receipt field behind it"})
    return out


def write(path=None, records_dir=None, derived=None) -> dict:
    path = Path(path) if path else OUT
    header, rows = build(records_dir, derived)
    v = check(header, rows, records_dir, derived)
    if v:
        raise PeaksRefused(
            f"CORRESPONDENCE_FAILED: {len(v)} violation(s) -- "
            f"{[x['kind'] for x in v]}. The roll-up is not written when it "
            f"disagrees with the records it is derived from. {v[0]['detail']}")
    before = (hashlib.sha256(path.read_bytes()).hexdigest()
              if path.exists() else None)
    body = "\n".join(json.dumps(r, sort_keys=True)
                     for r in [header] + rows) + "\n"
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(body)
        from declaration_chain import plain_create_mode
        os.chmod(tmp, plain_create_mode())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    after = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path), "sha256_before": before, "sha256_after": after,
            "n_rows": len(rows), "records_seen": header["records_seen"],
            "records_by_class": header["records_by_class"]}


def selftest() -> int:
    checks, fails = 0, []

    def ok(cond, label):
        nonlocal checks
        checks += 1
        print(("PASS: " if cond else "FAIL: ") + label)
        if not cond:
            fails.append(label)

    def _rec(d, unit, payload, args, extra=()):
        rows = [{"event": "launch", "unit": unit, "payload": payload,
                 "args": args}] + list(extra)
        (Path(d) / f"{RECORD_STEM}{unit}.jsonl").write_text(
            "\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n")

    # ---- THE EXCLUSION PREDICATE, ON ITS FOUR NAMED CLASSES --------------
    d = Path(tempfile.mkdtemp(prefix="peaks_cls_"))
    _rec(d, "u_real", "live/pm_research/be_gate1_fragment.py", "--day 20990101")
    _rec(d, "u_struct", "live/pm_research/be_daybook_build.py",
         "--verify-structure /x/be_daybook_20990101_btc.pkl")
    _rec(d, "u_probe", "live/pm_research/be_daybook_build.py", "--selftest")
    _rec(d, "u_scratch", "/tmp/tmp.abc/sleeper.py", "")
    (d / f"{RECORD_STEM}u_empty.jsonl").write_text("")
    _rec(d, "u_two", "live/pm_research/be_gate1_fragment.py",
         "--day 20990101 --also 20990102")
    got = {u: classify_record(u, _events(d / f"{RECORD_STEM}{u}.jsonl"))["why"]
           for u in ("u_real", "u_struct", "u_probe", "u_scratch", "u_empty",
                     "u_two")}
    ok(got == {"u_real": "DAY_CHAIN_RUN", "u_struct": "DAY_CHAIN_RUN",
               "u_probe": "NAMES_NO_DAY", "u_scratch": "NOT_A_TRACKED_PRODUCER",
               "u_empty": "NO_LAUNCH_EVENT", "u_two": "DAY_AMBIGUOUS"},
       f"THE EXCLUSION IS A NAMED PREDICATE OVER WHAT THE RECORD SAYS, never "
       f"the unit's name: {got}. Both directions in one fixture -- two runs "
       f"ADMITTED (a `--day` and a `--verify-structure <path>`) and four "
       f"excluded, each under its own name. A name-based rule would have to "
       f"be told about `u_scratch`; this one reads its payload")
    ok(classify_record("x", _events(
        d / f"{RECORD_STEM}u_struct.jsonl"))["day"] == "20990101",
       "AND THE DAY IS FOUND INSIDE A PATH: `be_daybook_20990101_btc.pkl` "
       "flanks the day with underscores, which are WORD characters, so `\\b` "
       "matched nothing -- measured, that dropped five of the eleven "
       "day-chain runs. The pattern is a digit-boundary, not a word one")

    # ---- CELL: A RECORD WITH NO ROW IS REFUSED BY NAME -------------------
    d2 = Path(tempfile.mkdtemp(prefix="peaks_rec_"))
    _rec(d2, "u_real", "live/pm_research/be_gate1_fragment.py", "--day 20990101")
    v = check({}, [], d2, d2)
    ok([x["kind"] for x in v] == [RECORD_WITHOUT_ROW] and v[0]["unit"] == "u_real",
       f"KNOWN-BAD A -- {RECORD_WITHOUT_ROW}: a day-chain record with no row "
       f"is REFUSED BY NAME ({[x['kind'] for x in v]}). This is REV 89 §3.2's "
       f"finding driven as a control: `be72frag` and `be87fwd06` ran, refused "
       f"and left records, and the hand-kept file named neither")
    v_ok = check({}, [{"unit": "u_real.service",
                       "unit_source": f"{RECORD_STEM}u_real.jsonl"}], d2, d2)
    ok(v_ok == [],
       "POSITIVE CONTROL ON THE SAME FIXTURE: with the row present the same "
       "check passes -- a control shown only to refuse has not been shown to "
       "admit (rule 16)")

    # ---- CELL: A ROW WITH NO RECORD IS REFUSED BY NAME -------------------
    d3 = Path(tempfile.mkdtemp(prefix="peaks_row_"))
    v2 = check({}, [{"unit": "ghost.service",
                     "unit_source": f"{RECORD_STEM}ghost.jsonl"}], d3, d3)
    ok([x["kind"] for x in v2] == [ROW_WITHOUT_RECORD],
       f"KNOWN-BAD B -- {ROW_WITHOUT_RECORD}: a row claiming a launcher "
       f"record that does not exist is REFUSED BY NAME "
       f"({[x['kind'] for x in v2]}). A hand insertion cannot survive the "
       f"build")
    v3 = check({}, [{"unit": "be64book.service",
                     "unit_source": "be_daybook_receipt_20260905_btc.json "
                                    ":: scope.unit"}], d3, d3)
    ok(v3 == [],
       "AND ITS BOUNDARY, WHICH IS THE INTERESTING HALF: a row whose unit "
       "comes from the RECEIPT alone is NOT a violation -- that is the "
       "pre-service-form era (`be64book`, REV 89's third name), and the row "
       "carries `UNIT_NAMED_BY_RECEIPT_ONLY` instead of being refused or "
       "dropped. Refusing it would delete four days of history; dropping it "
       "silently is what drifted")

    # ---- CELL: NO `false` ABOUT A NUMBER THAT DOES NOT EXIST -------------
    absent = leaf_peak_from_record([{"event": "exit"}])
    present = leaf_peak_from_record([{"event": "leaf_peak",
                                      "peak_of_record_bytes": 123,
                                      "is_a_lower_bound": True,
                                      "n_samples": 4}])
    ok(absent["leaf_peak_bytes"] is None
       and absent["leaf_peak_is_a_lower_bound"] is None
       and present["leaf_peak_bytes"] == 123
       and present["leaf_peak_is_a_lower_bound"] is True,
       f"NOT_RECORDED CARRIES `leaf_peak_is_a_lower_bound: null`, NOT `false` "
       f"(REV 89 §3.2): {absent['leaf_peak_is_a_lower_bound']!r} where there "
       f"is no number, {present['leaf_peak_is_a_lower_bound']!r} where there "
       f"is one and it is bounded. `false` asserts a property of a number "
       f"that does not exist")
    ok(leaf_peak_from_record([{"event": "outcome",
                               "peak_of_record_bytes": ""}]
                             )["leaf_peak_status"] == "NOT_RECORDED",
       "and an EMPTY peak string from a post-exit capture is NOT_RECORDED, "
       "never 0 -- `--capture` on a finished unit reports LEAF_RELEASED and "
       "the leaf is gone (BE 87)")

    # ---- CELL: THE IN-PROCESS PEAK IS NOT SHARED ACROSS RUNS ------------
    h, rows = build()
    frag06 = {r["unit"]: r for r in rows
              if r["day"] == "20260906" and r["stage"] == "fragment"}
    ok(set(frag06) == {"be72frag.service", "be72frag2.service"}
       and frag06["be72frag.service"]["in_process_peak_bytes"] is None
       and isinstance(frag06["be72frag2.service"]["in_process_peak_bytes"], int),
       f"A REFUSED RUN DOES NOT INHERIT THE SURVIVOR'S MEASUREMENT: "
       f"`be72frag` REFUSED on the missing mask and wrote nothing, so its "
       f"in-process peak is None while `be72frag2`'s is "
       f"{frag06['be72frag2.service']['in_process_peak_bytes']} -- the "
       f"receipt names be72frag2 and a receipt's peak is that run's alone")

    # ---- CELL: THE REAL LEDGER DERIVES CLEAN ----------------------------
    v4 = check(h, rows)
    named = {r["unit"] for r in rows if r.get("unit")}
    ok(v4 == [] and "be72frag.service" in named
       and "be87fwd06.service" in named and "be64book.service" in named,
       f"THE REAL LEDGER: {len(rows)} rows from {h['records_seen']} records "
       f"({h['records_by_class']}), zero violations, and all THREE names REV "
       f"89 §3.2 measured as drifted are present -- be72frag and be87fwd06 "
       f"(record, no row) and be64book (row, no record, now carried with its "
       f"reason)")
    # REV 84 §3.2 -- ONE IMPLEMENTATION, N DETECTORS. This module imports
    # `declaration_chain`, so it RUNS that module's own falsifier as a
    # subprocess cell: a regression there fails every importer at once, and
    # no importer re-implements the logic. (REV 89 §8 row 8 routed four
    # importers that ship no such cell; a new one is not a fifth.)
    import be_rule22 as _R22
    _sf = _R22.shared_falsifier()
    ok(_sf["ok"],
       f"REV 84 §3.2: this battery RUNS `declaration_chain.py --falsify` as "
       f"a subprocess -> rc {_sf['rc']}, {_sf['summary']!r}")

    print()
    if fails:
        print(f"{len(fails)} FAILURES of {checks} checks")
        return 1
    print(f"{checks} checks passed")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--print", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.print:
        h, rows = build()
        print(json.dumps(h, indent=1, sort_keys=True))
        for r in rows:
            print(json.dumps(r, sort_keys=True))
        return 0
    if a.build:
        print(json.dumps(write(), indent=1, sort_keys=True))
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
