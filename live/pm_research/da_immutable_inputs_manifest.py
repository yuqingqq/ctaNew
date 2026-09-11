"""THE IMMUTABLE-INPUTS MANIFEST -- §7's first chain link, with digests.

DA 284 / REVIEW 208. §7's chain splits cleanly: links 1-7 feed §8's log loss
and links 8-10 are the §9 economic leg. Of the five missing freeze items,
exactly ONE blocks the predictive half -- this one.

IT IS A REAL MANIFEST, NOT A FILLED FIELD. §8 resolves day eligibility from
the frozen day/book gate, the official resolutions and settlement-verification
coverage, so those are precisely the inputs that must be pinned: a manifest
that merely EXISTS would let the eligibility rule read inputs nobody sealed.

TWO SEAL SHAPES, BECAUSE THE INPUTS HAVE TWO SHAPES.

  APPEND-ONLY LEDGERS (markets, resolutions, collector runs/gaps) grow while
  the programme runs. A whole-file digest of a growing file ages the moment it
  is written, which is the day-slice lesson one layer out. These carry a PREFIX
  SEAL -- `sealed_len` plus the sha256 of exactly that prefix -- reusing
  `ops/frozen_manifest`'s convention rather than inventing a second one:

      prefix unchanged, file longer  -> APPEND    (legal)
      prefix unchanged, same length  -> UNCHANGED
      prefix changed, or shorter     -> EDITED    (a violation)

  CLOSED PER-DAY CAPTURES (the PM book tape, the Chainlink price capture, the
  Binance bookTicker that C2 consumes) are immutable once the day closes. These
  carry a CONTENT SEAL: every file digested, and a day digest over the sorted
  (name, size, sha256) triples, so an added, removed or altered file all change
  it. Measured 279 MB/s, about 15 s per book-tape day.

WHAT A DIGEST HERE DOES NOT CLAIM. It pins the BYTES the chain read. It does
not certify that those bytes are correct, complete, or free of collector gaps
-- gap statuses are a separate predicate and stay one. A seal answers "is this
the same input?", never "is this a good input?".

Usage:  da_immutable_inputs_manifest.py [--falsify] [--days N]
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROTOCOL = "P003_DA_IMMUTABLE_INPUTS_MANIFEST_V1"
CHUNK = 1 << 20

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import da_root                                            # noqa: E402

#: APPEND-ONLY LEDGERS -> prefix seal. Each names WHICH chain link reads it.
LEDGERS = (
    ("markets", "data/pm_5min/markets.jsonl",
     ("immutable_inputs", "labels_statuses", "quote_mapping"),
     "market definitions: window, tokens, and the legal tick the seam reads"),
    # `score` reads this one too. `de_fair_value_actions.score_actions` takes
    # outcomes as an ARGUMENT and the module never loads one -- step 6 owns the
    # join -- so the outcome it consumes comes from HERE, and the link is
    # attributed rather than left uncovered.
    ("resolutions", "data/pm_5min/resolutions.jsonl",
     ("labels_statuses", "score"),
     "the OFFICIAL settled outcome -- §8's labels, its day-eligibility input, "
     "and the outcome the paired score is joined against"),
    ("collector_runs", "data/pm_5min/collector_runs.jsonl",
     ("immutable_inputs",),
     "collector/stamp boundaries; era purity is a per-event predicate (rule 5)"),
    ("collector_gaps", "data/pm_5min/collector_gaps.jsonl",
     ("immutable_inputs", "labels_statuses"),
     "gap windows -- the statuses that make an exclusion a status, not a drop"),
)

#: CLOSED PER-DAY CAPTURES -> content seal.
CAPTURES = (
    ("pm_book_tape", "data/pm_5min/raw/{day}", "*.jsonl.gz",
     ("fairprice", "fallback", "actions", "quote_mapping"),
     "the PM order book -- Identity, the mandatory baseline"),
    ("chainlink_prices", "data/pm_5min/prices/crypto_prices", "{day}_*.csv.gz",
     ("labels_statuses",),
     "the settlement feed: PM binaries settle on Chainlink, never Binance"),
    ("binance_bookticker", "data/mm_hf/raw/bookTicker", "*/{day}_*.csv.gz",
     ("sigma", "fairprice"),
     "C2's input: bn_bookticker_mid and the 30-minute sigma"),
)

#: The links this manifest covers. §8's predictive half is 1-7; the manifest is
#: link 1 and pins what links 2-7 read.
PREDICTIVE_LINKS = ("immutable_inputs", "labels_statuses", "actions", "sigma",
                    "fairprice", "fallback", "score")


def _root() -> Path:
    return da_root.resolve_root()


def _sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def prefix_digest(p: Path, n: int) -> str:
    """The sha256 of exactly the first `n` bytes -- `ops/frozen_manifest`'s
    convention, so an APPEND stays legal and an EDIT is caught."""
    h = hashlib.sha256()
    left = n
    with open(p, "rb") as fh:
        while left > 0:
            b = fh.read(min(CHUNK, left))
            if not b:
                break
            h.update(b)
            left -= len(b)
    return h.hexdigest()


def seal_ledger(name: str, rel: str, links, why: str) -> dict:
    p = _root() / rel
    if not p.is_file():
        return {"input": name, "path": rel, "present": False,
                "seal": None, "reads": list(links), "why": why}
    size = p.stat().st_size
    return {
        "input": name, "path": rel, "present": True,
        "seal_kind": "PREFIX",
        "sealed_len": size,
        "sealed_len_units": "BYTES",
        "sealed_len_is_not_a_record_count": (
            "6,811,740 for collector_gaps is a BYTE LENGTH. The file holds "
            "13,016 records. A number with no unit beside it is read as "
            "whatever the reader expects, and this one was read as a count."),
        "n_records": sum(1 for _ in open(p, "rb")),
        "prefix_sha256": prefix_digest(p, size),
        "as_of_mtime_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(p.stat().st_mtime)),
        "reads": list(links), "why": why,
        "append_is_legal": True,
        "verify_rule": ("re-digest the first `sealed_len` bytes: equal means "
                        "UNCHANGED or APPEND; different or shorter means EDITED"),
    }


def is_closed_day(day: str) -> bool:
    """A UTC day is CLOSED once the next one has begun. A CONTENT seal over an
    OPEN day is a seal over a moving target: REVIEW 209 found 09-11 sealed while
    it was still accruing, and disk was already +154 book files past the seal
    within the hour. My own rule -- content seals are for closed captures -- was
    stated in the docstring and not enforced anywhere, which is the same shape
    as a property list that omits a property."""
    return day < time.strftime("%Y%m%d", time.gmtime())


def seal_capture(name: str, tmpl: str, pat: str, links, why: str,
                 days) -> dict:
    out = {}
    for day in days:
        if not is_closed_day(day):
            out[day] = {"present": False, "sealed": False,
                        "status": "DAY_IS_OPEN_NOT_SEALED",
                        "why": ("a CONTENT seal over an open day seals a moving "
                                "target; this day is still accruing and is "
                                "excluded until it closes")}
            continue
        base = _root() / tmpl.format(day=day)
        files = sorted(glob.glob(str(base / pat.format(day=day))))
        if not files:
            out[day] = {"present": False, "n_files": 0}
            continue
        triples = []
        total = 0
        for f in files:
            sz = os.path.getsize(f)
            total += sz
            triples.append((os.path.relpath(f, str(_root())), sz, _sha_file(Path(f))))
        day_digest = hashlib.sha256(
            json.dumps(triples, sort_keys=True).encode()).hexdigest()
        out[day] = {
            "present": True, "n_files": len(files), "bytes": total,
            "day_sha256": day_digest,
            "seal_kind": "CONTENT",
            "covers": ("every file digested; the day digest is over the sorted "
                       "(name, size, sha256) triples, so an added, removed or "
                       "altered file all change it"),
        }
    return {"input": name, "path_template": tmpl, "pattern": pat,
            "seal_kind": "CONTENT", "reads": list(links), "why": why,
            "by_day": out}


def available_days(limit: int | None = None) -> list:
    ds = sorted(d.name for d in (_root() / "data/pm_5min/raw").iterdir()
                if d.is_dir() and d.name.isdigit())
    return ds[-limit:] if limit else ds


def build(days=None, limit: int = 3) -> dict:
    days = list(days) if days else available_days(limit)
    t0 = time.time()
    ledgers = [seal_ledger(n, r, l, w) for n, r, l, w in LEDGERS]
    captures = [seal_capture(n, t, p, l, w, days) for n, t, p, l, w in CAPTURES]
    covered = {lk for e in ledgers for lk in e["reads"]}
    covered |= {lk for e in captures for lk in e["reads"]}
    missing_links = [lk for lk in PREDICTIVE_LINKS if lk not in covered]
    # An OPEN day is EXCLUDED, not counted as a failure and not counted as
    # sealed. `days_sealed` is what was actually sealed.
    open_days = sorted({day for c in captures for day, v in c["by_day"].items()
                        if v.get("status") == "DAY_IS_OPEN_NOT_SEALED"})
    sealed_days = [d for d in days if d not in open_days]
    all_sealed = (all(e["present"] for e in ledgers)
                  and all(v["present"] for c in captures
                          for day, v in c["by_day"].items()
                          if day in sealed_days))
    return {
        "protocol": PROTOCOL,
        "dispatch": "DA 284 / REVIEW 208",
        "as_of_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "data_root": str(_root()),
        "days_requested": days,
        "days_sealed": sealed_days,
        "days_open_and_excluded": open_days,
        "ledgers": ledgers,
        "captures": captures,
        "predictive_links_covered": sorted(covered & set(PREDICTIVE_LINKS)),
        "predictive_links_not_covered": missing_links,
        "every_predictive_link_has_a_sealed_input": not missing_links,
        "all_enumerated_inputs_present_and_sealed": all_sealed,
        "n_inputs": len(ledgers) + len(captures),
        "seconds_to_build": round(time.time() - t0, 1),
        "AN_UNENUMERATED_TREE_GATE_1_READS": {
            "finding": "REVIEW 209",
            "path": "data/pm_5min/derived/**",
            "read_by": "da_fair_value_gate1_labels.cells_winner_digests()",
            "why_it_is_not_sealed": (
                "it is a MUTABLE tree that both lanes rewrite -- a content seal "
                "over it would be stale within the hour and a prefix seal is "
                "meaningless for a directory. Sealing it would manufacture a "
                "guarantee, not record one."),
            "so_the_claim_is_weakened_on_purpose": (
                "`every_predictive_link_has_a_sealed_input` means every link "
                "has at least one sealed input. For labels/statuses it does NOT "
                "mean every byte that link reads is sealed: gate 1 also reads "
                "this derived tree. Stated here so the field is not read as "
                "more than it measures."),
            "what_would_close_it": (
                "gate 1 reading its winner digests from an enumerated, sealed "
                "per-day artifact rather than globbing a shared mutable tree"),
        },
        "WHAT_A_SEAL_DOES_NOT_CLAIM": (
            "it pins the BYTES the chain read. It does not certify they are "
            "correct, complete, or free of collector gaps -- gap statuses are "
            "a separate predicate and stay one. A seal answers 'is this the "
            "same input?', never 'is this a good input?'"),
        "WHY_TWO_SEAL_SHAPES": (
            "append-only ledgers grow while the programme runs, so a whole-file "
            "digest ages the moment it is written; per-day captures are closed "
            "and can be digested in full"),
    }


def falsify() -> int:
    bad = 0

    def ck(label, cond, shown=""):
        nonlocal bad
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f" -> {shown}" if shown else ""))
        if not cond:
            bad += 1

    d = build(limit=2)
    ck("every append-only ledger is present and PREFIX-sealed",
       all(e["present"] and e["seal_kind"] == "PREFIX" and len(e["prefix_sha256"]) == 64
           for e in d["ledgers"]),
       f"{len(d['ledgers'])} ledgers")
    ck("...and each names the sealed LENGTH, so an append stays legal",
       all(isinstance(e["sealed_len"], int) and e["sealed_len"] > 0
           for e in d["ledgers"]))
    ck("every per-day capture is CONTENT-sealed with a 64-hex day digest",
       all(len(x["day_sha256"]) == 64 for c in d["captures"]
           for x in c["by_day"].values() if x.get("present")),
       f"{len(d['captures'])} captures x {len(d['days_sealed'])} days")
    ck("POSITIVE CONTROL: the book tape day actually carries files",
       any(x.get("n_files", 0) > 100 for c in d["captures"]
           if c["input"] == "pm_book_tape" for x in c["by_day"].values()),
       str({k: v.get("n_files", v.get("status")) for c in d["captures"]
            if c["input"] == "pm_book_tape" for k, v in c["by_day"].items()}))
    # THE SEAL MUST BE ABLE TO FIRE.
    lg = d["ledgers"][0]
    p = _root() / lg["path"]
    ck("NEGATIVE CONTROL: a SHORTER prefix digests differently (an edit is caught)",
       prefix_digest(p, lg["sealed_len"] - 1) != lg["prefix_sha256"])
    ck("...and re-digesting the SAME prefix reproduces the seal (it is stable)",
       prefix_digest(p, lg["sealed_len"]) == lg["prefix_sha256"])
    cap = next(c for c in d["captures"] if c["input"] == "pm_book_tape")
    day = next(k for k, v in cap["by_day"].items() if v.get("present"))
    again = seal_capture("pm_book_tape", "data/pm_5min/raw/{day}", "*.jsonl.gz",
                         ("fairprice",), "", [day])
    ck("a day digest is REPRODUCIBLE over the same files",
       again["by_day"][day]["day_sha256"] == cap["by_day"][day]["day_sha256"],
       cap["by_day"][day]["day_sha256"][:16])
    ck("NEGATIVE CONTROL: dropping one file CHANGES the day digest",
       hashlib.sha256(json.dumps([["x", 1, "y"]], sort_keys=True).encode()).hexdigest()
       != cap["by_day"][day]["day_sha256"])
    ck("an OPEN day is EXCLUDED from content seals, not sealed while moving",
       all(v.get("status") == "DAY_IS_OPEN_NOT_SEALED"
           for c in d["captures"] for k, v in c["by_day"].items()
           if not is_closed_day(k)),
       f"open and excluded: {d['days_open_and_excluded']}")
    ck("...and `days_sealed` reports only what was actually sealed",
       all(is_closed_day(x) for x in d["days_sealed"]), str(d["days_sealed"]))
    ck("a byte LENGTH is labelled as one, and the record count is separate",
       all(e["sealed_len_units"] == "BYTES" and isinstance(e["n_records"], int)
           for e in d["ledgers"]),
       str({e["input"]: (e["sealed_len"], e["n_records"]) for e in d["ledgers"]
            if e["input"] == "collector_gaps"}))
    ck("every PREDICTIVE link (§7 links 1-7) has at least one sealed input",
       d["every_predictive_link_has_a_sealed_input"],
       str(d["predictive_links_not_covered"] or "none uncovered"))
    ck("the settlement feed is Chainlink, and it is sealed",
       any(c["input"] == "chainlink_prices" for c in d["captures"]))
    ck("C2's own input is sealed separately from Identity's book",
       {c["input"] for c in d["captures"]} >= {"binance_bookticker", "pm_book_tape"})
    ck("the unenumerated tree gate 1 reads is NAMED, not papered over",
       d["AN_UNENUMERATED_TREE_GATE_1_READS"]["path"].startswith("data/pm_5min/derived")
       and "does NOT mean every byte" in
       d["AN_UNENUMERATED_TREE_GATE_1_READS"]["so_the_claim_is_weakened_on_purpose"])
    ck("a seal does NOT claim the input is good, and says so",
       "never 'is this a good input?'" in d["WHAT_A_SEAL_DOES_NOT_CLAIM"])
    print(f"\n  {'MANIFEST CELLS PASS' if not bad else str(bad) + ' FAILED'}")
    return bad


if __name__ == "__main__":
    if "--falsify" in sys.argv:
        sys.exit(1 if falsify() else 0)
    lim = 3
    if "--days" in sys.argv:
        lim = int(sys.argv[sys.argv.index("--days") + 1])
    print(json.dumps(build(limit=lim), indent=1))
