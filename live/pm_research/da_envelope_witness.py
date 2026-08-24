"""R-42 applied to R-12: make the envelope-field list REVEAL that it is envelope.

R-12 stopped 518 duplicate-identity crashes by declaring two fields to be
"envelope" -- not identity-bearing -- for `book` snapshots:

    BOOK_ENVELOPE_FIELDS = ("last_trade_price", "tick_size")

and I verified the fix by re-scanning the corpus and finding 518 -> 0 conflicts.

THAT VERIFICATION FAILS OPEN.  Stripping a field can only ever REDUCE the
conflict count, so "0 conflicts" is exactly what an over-broad strip list also
produces.  Success and over-merging are indistinguishable in that metric; a
strip list that also removed `bids` would have scored perfectly.  The conflict
count is the mirror of a gate that cannot fire.

And the consequence is not cosmetic.  `normalize_clob` does not merely decline
to raise on a collapsed duplicate -- it `continue`s, so **the record is
DROPPED** (`tier1_pipeline.py:1096-1108`).  Anything carried only by the dropped
copy is gone.

R-42's rule: the check does not ASK the rule what it is; it MAKES the rule
reveal it.  Here that means asking, of each stripped field, a question whose
answer can come out wrong:

    DOES ANY CONSUMER READ THIS FIELD OFF A `book` MESSAGE?

Read off the code rather than off the declaration:

  * `last_trade_price` -- BENIGN.  Every consumer in the tree filters
    `event_type == "last_trade_price"`, i.e. the dedicated event stream
    (`flow_intensity.py:730`, `edge_layer1.py:279`, `warning_window.py:308`,
    `queue_and_type.py:878`, and eight more).  Nothing reads the field off a
    `book` snapshot.  Dropping a re-delivery that differs only here loses
    nothing.

  * `tick_size` -- **LOAD-BEARING**.  `tier1_pipeline.py:768-769` reads it off
    the book message into `_BookState.tick_size`, which is emitted as a Tier-1
    column at `:1222` and typed at `:120`.  A dropped re-delivery carrying a
    CHANGED `tick_size` therefore loses a tick update that reaches the output.

So the two fields are not alike, and R-12's justification -- "all 19 conflicting
book keys differed only in `last_trade_price`/`tick_size`" -- never separated
them.  `ParseStats.book_envelope_collapsed` counts both together, so the number
that would show the damage is aggregated with the number that shows the fix
working.

This module splits that counter by field and reports any collapse that dropped a
DIFFERING load-bearing field.  It adopts nothing and changes no CHOSEN value:
`BOOK_ENVELOPE_FIELDS` is untouched, and if the witness fires the remedy is a
ruling, not an edit.  DA-owned verification of a DA-shipped fix.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[1]))  # repo root: tier1_pipeline imports live.*
import tier1_pipeline as tp  # noqa: E402

# Which stripped fields actually reach the Tier-1 output.  Derived by reading
# consumers, not by declaration -- see the module docstring.  A field moving
# into this tuple is a finding, not a config change.
LOAD_BEARING_ENVELOPE: tuple[str, ...] = ("tick_size",)
BENIGN_ENVELOPE: tuple[str, ...] = ("last_trade_price",)

WITNESS_VERSION = "envelope_witness_v1_r42"


def differing_envelope_fields(
    base: Mapping[str, Any],
    duplicate: Mapping[str, Any],
    envelope: Sequence[str] = tp.BOOK_ENVELOPE_FIELDS,
) -> tuple[str, ...]:
    """Which envelope fields actually differ between a record and its collapsed twin.

    Absent-vs-present counts as differing: a re-delivery that ADDS a tick_size
    the first copy lacked is exactly the case that loses an update.
    """
    sentinel = object()
    return tuple(
        field
        for field in envelope
        if base.get(field, sentinel) != duplicate.get(field, sentinel)
    )


# `_BookState.tick_size` is initialised to this (tier1_pipeline.py:748), so a
# discarded update that merely restates it changes nothing observable.
BOOK_STATE_TICK_DEFAULT = "0.01"


def impact_of_drop(
    kept: Mapping[str, Any], dropped: Mapping[str, Any], field: str = "tick_size"
) -> str:
    """What the DROP actually costs, which is not the same as what DIFFERS.

    Direction matters and the first version of this witness ignored it, which
    over-reported by 4 of 8 on the first day measured.  Dropping a copy that
    carries NOTHING loses nothing; dropping one that merely restates the state
    default loses nothing observable; dropping one that carries a value the
    surviving copy CONTRADICTS is a silent selection between two authorities.
    """
    k, d = kept.get(field), dropped.get(field)
    if d is None:
        return "NO_LOSS_DROPPED_CARRIED_NOTHING"
    if d == k:
        return "NO_LOSS_AGREES"
    if k is None:
        return ("NO_EFFECT_RESTATES_DEFAULT" if str(d) == BOOK_STATE_TICK_DEFAULT
                else "SILENT_SELECTION_UNCONTRADICTED")
    return "SILENT_SELECTION_CONTRADICTED"


def classify_collapse(
    base: Mapping[str, Any],
    duplicate: Mapping[str, Any],
    envelope: Sequence[str] = tp.BOOK_ENVELOPE_FIELDS,
    load_bearing: Sequence[str] = LOAD_BEARING_ENVELOPE,
) -> tuple[str, tuple[str, ...]]:
    """(verdict, differing fields).  Pure, so the rule can be fed its own mirror.

    IDENTICAL      -- a true re-delivery; dropping it loses nothing.
    BENIGN         -- differs only where nothing downstream reads.
    LOAD_BEARING   -- differs where a consumer reads: the drop LOSES DATA.
    """
    fields = differing_envelope_fields(base, duplicate, envelope)
    if not fields:
        return "IDENTICAL", ()
    if any(field in load_bearing for field in fields):
        return "LOAD_BEARING", fields
    return "BENIGN", fields


def scan_slug(day: str, slug: str, shards: Sequence[Path]) -> dict[str, Any]:
    """Replay `normalize_clob`'s dedup decision and witness what it discards."""
    stats = tp.ParseStats()
    seen: dict[tuple[Any, ...], dict[str, Any]] = {}
    verdicts: Counter[str] = Counter()
    field_counts: Counter[str] = Counter()
    losses: list[dict[str, Any]] = []
    impacts: Counter[str] = Counter()
    n_book = 0

    for path in shards:
        for recv_ns, seq, _sid, message in tp._iter_wire_file(
            path, stats, source_file_id="witness"
        ):
            if str(message.get("event_type", "")) != "book":
                continue
            n_book += 1
            key = tp._raw_message_key(message)
            prior = seen.get(key)
            if prior is None:
                seen[key] = message
                continue
            verdict, fields = classify_collapse(prior, message)
            verdicts[verdict] += 1
            for field in fields:
                field_counts[field] += 1
            if verdict == "LOAD_BEARING":
                impact = impact_of_drop(prior, message)
                impacts[impact] += 1
                if impact.startswith("SILENT_SELECTION") and len(losses) < 50:
                    losses.append({
                        "slug": slug, "shard": path.name, "recv_ns": recv_ns,
                        "seq": seq, "impact": impact, "fields": list(fields),
                        "kept": {f: prior.get(f) for f in fields},
                        "dropped": {f: message.get(f) for f in fields},
                    })
    return {
        "day": day, "slug": slug, "book_messages": n_book,
        "collapses": dict(verdicts), "differing_fields": dict(field_counts),
        "impacts": dict(impacts), "load_bearing_losses": losses,
    }


def scan_day(day: str) -> dict[str, Any]:
    import da_duplicate_identity_scan as census
    slugs = census.discover_slugs(day)
    verdicts: Counter[str] = Counter()
    fields: Counter[str] = Counter()
    impacts: Counter[str] = Counter()
    losses: list[dict[str, Any]] = []
    n_book = 0
    for slug, shards in sorted(slugs.items()):
        r = scan_slug(day, slug, shards)
        n_book += r["book_messages"]
        verdicts.update(r["collapses"])
        fields.update(r["differing_fields"])
        impacts.update(r["impacts"])
        losses.extend(r["load_bearing_losses"])
    return {
        "witness_version": WITNESS_VERSION, "day": day, "slugs": len(slugs),
        "book_messages": n_book, "collapses": dict(verdicts),
        "differing_fields": dict(fields), "impacts": dict(impacts),
        "silent_selections": losses[:50],
        "n_silent_selections": len(losses),
    }


# ---------------------------------------------------------------------------
# selftests -- including the mirror, which is the point of the exercise
# ---------------------------------------------------------------------------

def _selftests() -> int:
    checks = 0

    def ok(cond, label):
        nonlocal checks
        checks += 1
        if not cond:
            raise AssertionError(f"selftest failed: {label}")

    base = {"event_type": "book", "asset_id": "A", "hash": "h1",
            "timestamp": "1000", "bids": [["0.4", "10"]], "asks": [["0.6", "10"]],
            "tick_size": "0.01", "last_trade_price": "0.5"}

    # 1. a true re-delivery
    ok(classify_collapse(base, dict(base))[0] == "IDENTICAL", "identical re-delivery")

    # 2. BENIGN: differs only where nothing downstream reads
    benign = dict(base, last_trade_price="0.55")
    v, f = classify_collapse(base, benign)
    ok((v, f) == ("BENIGN", ("last_trade_price",)), "last_trade_price alone is benign")

    # 3. THE MIRROR.  A pair differing only in the LOAD-BEARING field must be
    #    reported differently from the benign pair.  If the rule answered the
    #    same for both it would be blind in exactly the way R-12's conflict
    #    count is blind, and this module would be decorative.
    loaded = dict(base, tick_size="0.001")
    v2, f2 = classify_collapse(base, loaded)
    ok((v2, f2) == ("LOAD_BEARING", ("tick_size",)), "tick_size alone is load-bearing")
    ok(v2 != v, "MIRROR: the rule must ANSWER DIFFERENTLY on the two fields")

    # 4. both differ -> load-bearing dominates, and both fields are reported
    both = dict(base, tick_size="0.001", last_trade_price="0.55")
    v3, f3 = classify_collapse(base, both)
    ok(v3 == "LOAD_BEARING" and set(f3) == {"tick_size", "last_trade_price"},
       "load-bearing dominates and reporting is complete")

    # 5. absent-vs-present counts as differing: a re-delivery that ADDS the
    #    field is the case that silently loses an update
    no_tick = {k: v for k, v in base.items() if k != "tick_size"}
    ok(classify_collapse(no_tick, base)[0] == "LOAD_BEARING", "absent -> present differs")
    ok(classify_collapse(base, no_tick)[0] == "LOAD_BEARING", "present -> absent differs")

    # 6. the digest really does collapse these -- i.e. the witness is watching a
    #    live path, not a hypothetical one
    ok(tp._identity_digest(base) == tp._identity_digest(loaded),
       "the shipped digest DOES collapse a differing tick_size")
    ok(tp._raw_message_key(base) == tp._raw_message_key(loaded),
       "and they share a raw key, so the second is the one dropped")

    # 7. NEGATIVE CONTROL on the witness itself: with an over-broad strip list
    #    that swallows a real identity field, the witness must FIRE.  A checker
    #    that stays quiet here could not tell a justified strip from a reckless
    #    one, which is the failure this module exists to rule out.
    over_broad = ("last_trade_price", "tick_size", "bids")
    moved_book = dict(base, bids=[["0.41", "10"]])
    v4, f4 = classify_collapse(base, moved_book, envelope=over_broad,
                               load_bearing=("tick_size", "bids"))
    ok(v4 == "LOAD_BEARING" and "bids" in f4,
       "NEGATIVE CONTROL: an over-broad strip list must be detected, not tolerated")
    # and under the SHIPPED list the same pair is not a collapse at all
    ok(classify_collapse(base, moved_book)[0] == "IDENTICAL",
       "under the shipped list a moved book is not an envelope collapse")
    ok(tp._identity_digest(base) != tp._identity_digest(moved_book),
       "and the shipped digest correctly SEPARATES a moved book -- it still raises")

    # 7b. DIRECTION: what DIFFERS is not what is LOST.  The first version of
    #     this witness reported 8 load-bearing collapses on 2026-08-20; 2 of
    #     them dropped a copy carrying nothing and 4 restated the state default,
    #     so only 2 were real.  A witness that over-reports gets discounted, so
    #     it has to be as sharp against itself as against what it watches.
    ok(impact_of_drop({"tick_size": "0.01"}, {}) == "NO_LOSS_DROPPED_CARRIED_NOTHING",
       "dropping a copy carrying nothing loses nothing")
    ok(impact_of_drop({"tick_size": "0.01"}, {"tick_size": "0.01"}) == "NO_LOSS_AGREES",
       "agreeing copies lose nothing")
    ok(impact_of_drop({}, {"tick_size": "0.01"}) == "NO_EFFECT_RESTATES_DEFAULT",
       "a discarded 0.01 restates _BookState's default and is unobservable")
    ok(impact_of_drop({}, {"tick_size": "0.001"}) == "SILENT_SELECTION_UNCONTRADICTED",
       "a discarded 0.001 IS observable -- the state keeps 0.01")
    ok(impact_of_drop({"tick_size": "0.01"}, {"tick_size": "0.001"})
       == "SILENT_SELECTION_CONTRADICTED",
       "two copies disagreeing on the tick is a selection, not an update")
    ok(BOOK_STATE_TICK_DEFAULT == "0.01", "default matches tier1_pipeline.py:748")

    # 8. the two field lists must stay disjoint and cover the shipped tuple
    ok(not set(LOAD_BEARING_ENVELOPE) & set(BENIGN_ENVELOPE), "lists disjoint")
    ok(set(LOAD_BEARING_ENVELOPE) | set(BENIGN_ENVELOPE) == set(tp.BOOK_ENVELOPE_FIELDS),
       "every shipped envelope field is classified -- a new one fails this test")

    print(f"da_envelope_witness selftests: {checks} checks passed")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--day", action="append", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.selftest or not args.day:
        raise SystemExit(_selftests())

    reports = [scan_day(day) for day in args.day]
    total_loss = sum(r["n_silent_selections"] for r in reports)
    out = {"witness_version": WITNESS_VERSION, "days": reports,
           "total_silent_selections": total_loss}
    text = json.dumps(out, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    print(f"\nSILENT SELECTIONS: {total_loss}", file=sys.stderr)


if __name__ == "__main__":
    main()
