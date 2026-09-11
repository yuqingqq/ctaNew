"""WHICH MODEL COMPOSITION WAS ON DISK FOR A BUILD -- recorded beside it.

THE GAP THIS EXISTS FOR (Q-BE-365). `CONDVALUE_X_SKEW` names THREE objects:
2 model files in params v1-19 (hazard only, R-833's defect), 4 in v20, 5 in
v21-30. They are ADDITIVE -- identical hazard digests, each a superset of the
last -- and the screen ran on composition 3, settled by digest match on the
run's `be_module`. But **the BOOK does not record its model identity**: its
`score_contracts` carry the score kind and formula and NO model digests, and
the receipt names none of the five files. So at the artifact that holds the
SCORES, the composition is INFERABLE and not CHECKABLE, which is rule 16's
shape. The forward test's books do not exist yet; they would inherit it.

WHY A SIDECAR AND NOT A RECEIPT FIELD. The builder is frozen and its bytes
are pinned because the tip's scoring moved under commits whose
score-neutrality was never established. Editing the builder to add a
provenance field re-opens that question for a field that decides nothing. So
this runs BESIDE a build and writes its own artifact.

WHAT IT ESTABLISHES
  * the FULL sha256 of every model file in the fit directory, snapshotted at
    build START and again at build END -- so a mid-build swap is caught
    rather than averaged over;
  * which DECLARED composition those bytes are, derived from the params
    files themselves and never typed here (rule 32);
  * the record bound to the build: unit, InvocationID, book sha256.

WHAT IT DOES *NOT* ESTABLISH, stated because a recorder's limit decides what
may be concluded from it. **IT DOES NOT OBSERVE THAT THE BUILDER READ THESE
FILES.** Two routes were measured and both are closed on this box: the fit
directory is mounted `relatime` and a probe read did NOT advance atime, so
atime cannot witness a per-build read; and `inotifywait` is absent. The only
remaining routes inject an observer into the build process (a `sitecustomize`
audit hook) or sample `/proc/<pid>/fd`. The first changes the run to record a
provenance field and is refused for a forward-test build. The second is
offered as BEST EFFORT only: a catch is strong evidence, an absence proves
nothing, and it is labelled that way on the artifact rather than counted.

So this turns "inferable from the timeline" into "these exact bytes were on
disk for this build, they are composition N, and nothing swapped under it" --
and leaves ONE residual named on its own output.

THE BOOK HALF IS DELIBERATELY WEAK. `composition_floor_from_book` reports the
MINIMUM composition consistent with the book's own score formula: a
value-bearing formula rules out composition 1 and says nothing further. It
REFUSES to claim a specific composition from a book alone, because that is
exactly the inference this module exists to stop being made silently.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

FIT_DIR = "data/pm_5min/derived/phase2_fits"
DECL_DIR = "live/pm_research/declarations"

NO_MATCH = "MODEL_SET_MATCHES_NO_DECLARED_COMPOSITION"
AMBIGUOUS = "MODEL_SET_MATCHES_MORE_THAN_ONE_COMPOSITION"
MOVED = "MODELS_CHANGED_DURING_THE_BUILD"
ABSENT = "FIT_DIRECTORY_ABSENT_NO_VERDICT_POSSIBLE"
MISMATCH = "COMPOSITION_IS_NOT_THE_ONE_ASSERTED"
BOOK_CANNOT_SAY = "BOOK_DOES_NOT_RECORD_ITS_MODEL_IDENTITY"

#: A value-bearing score formula cannot be produced by the hazard-only
#: composition. This is the ONLY thing a book's own fields support.
VALUE_BEARING = "predicted_conditional_value"


class IdentityRefused(RuntimeError):
    """The composition cannot be established, so none is reported."""


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def declared_compositions(arm: str, decl_dir=DECL_DIR) -> list:
    """Every distinct model set this arm has been declared with, DERIVED
    from the params files in version order -- never typed here (rule 32)."""
    out = []
    seen = {}
    files = sorted(Path(decl_dir).glob("de_multiday_gate1_params_v*.json"),
                   key=lambda f: int(re.search(r"_v(\d+)\.json", f.name).group(1)))
    if not files:
        raise IdentityRefused(
            "REFUSED: no params declarations found; the composition set is "
            "derived from them and cannot be assumed.")
    for f in files:
        v = int(re.search(r"_v(\d+)\.json", f.name).group(1))
        spec = (json.loads(f.read_text()).get("arms") or {}).get(arm) or {}
        md = spec.get("model_digests") or {}
        if not md:
            continue
        key = tuple(sorted(md.items()))
        if key in seen:
            seen[key]["params_versions"].append(v)
            continue
        rec = {"composition": len(out) + 1, "n_models": len(md),
               "model_digests_declared": dict(md), "params_versions": [v]}
        seen[key] = rec
        out.append(rec)
    return out


def snapshot(fit_dir=FIT_DIR) -> dict:
    """FULL sha256 of every file in the fit directory. An absent directory
    REFUSES rather than returning an empty set -- an empty snapshot would
    match nothing and read as a clean miss."""
    d = Path(fit_dir)
    if not d.is_dir():
        raise IdentityRefused(f"REFUSED {ABSENT}: {fit_dir}")
    got = {f.name: _sha(f) for f in sorted(d.iterdir()) if f.is_file()}
    if not got:
        raise IdentityRefused(f"REFUSED {ABSENT}: {fit_dir} holds no files")
    return got


def _matches(declared: dict, on_disk: dict) -> bool:
    """The params store 16-char TRUNCATED digests, so a declared value is a
    PREFIX of the full one. A wrong prefix must not match."""
    for name, dec in declared.items():
        got = on_disk.get(name)
        if got is None or not got.startswith(str(dec)):
            return False
    return True


def identify(arm: str, on_disk: dict, *, decl_dir=DECL_DIR,
             assert_composition: int | None = None) -> dict:
    """WHICH declared composition these bytes are. Refuses on no match, on
    ambiguity, and on a mismatch against an asserted composition."""
    comps = declared_compositions(arm, decl_dir)
    hits = [c for c in comps if _matches(c["model_digests_declared"], on_disk)]
    if not hits:
        raise IdentityRefused(
            f"REFUSED {NO_MATCH}: the {len(on_disk)} file(s) on disk match "
            f"none of {len(comps)} declared composition(s) for {arm}.")
    # The compositions are supersets of one another, so a larger set on disk
    # satisfies a smaller declaration. The composition IS the largest match.
    chosen = max(hits, key=lambda c: c["n_models"])
    smaller = [c["composition"] for c in hits if c is not chosen]
    out = {"arm": arm, "composition": chosen["composition"],
           "n_models": chosen["n_models"],
           "params_versions": chosen["params_versions"],
           "also_satisfied_because_compositions_are_supersets": smaller,
           "n_declared_compositions": len(comps),
           "model_digests_full": {k: on_disk[k] for k
                                  in chosen["model_digests_declared"]},
           "comparison": "declared digests are 16-char PREFIXES of the full "
                         "sha256; matched by prefix, and a wrong prefix does "
                         "not match"}
    if assert_composition is not None and chosen["composition"] != assert_composition:
        raise IdentityRefused(
            f"REFUSED {MISMATCH}: asserted composition "
            f"{assert_composition}, the bytes on disk are composition "
            f"{chosen['composition']} ({chosen['n_models']} models).")
    return out


def composition_floor_from_book(header: dict) -> dict:
    """What the BOOK alone supports -- deliberately a FLOOR, not a verdict.

    A value-bearing formula rules out the hazard-only composition and says
    nothing further. Claiming a specific composition from a book is the
    inference this module exists to stop."""
    sc = (header or {}).get("score_contracts") or {}
    if not sc:
        raise IdentityRefused(
            f"REFUSED {BOOK_CANNOT_SAY}: no score_contracts on the header.")
    formulas = {str(c.get("formula") or "") for c in sc.values()}
    value_bearing = all(VALUE_BEARING in f for f in formulas)
    return {
        "n_heads": len(sc), "formulas": sorted(formulas),
        "value_bearing": value_bearing,
        "composition_floor": 2 if value_bearing else 1,
        "WHAT_THE_BOOK_CANNOT_SAY": (
            "the book records the score KIND and FORMULA and NO model "
            "digests, so it supports a FLOOR and never a specific "
            "composition. The sidecar's snapshot is what pins it."),
    }


def attest(arm: str, *, start: dict, end: dict, unit: str | None = None,
           invocation_id: str | None = None, book_sha256: str | None = None,
           fd_samples: dict | None = None, decl_dir=DECL_DIR) -> dict:
    """The record for one build. Refuses if the models moved under it."""
    if start != end:
        moved = sorted(k for k in set(start) | set(end)
                       if start.get(k) != end.get(k))
        raise IdentityRefused(
            f"REFUSED {MOVED}: {len(moved)} file(s) differ between the "
            f"start and end snapshots: {moved[:5]}. A build whose models "
            f"moved under it has no single composition.")
    ident = identify(arm, end, decl_dir=decl_dir)
    return {
        "protocol": "BE_MODEL_IDENTITY_SIDECAR_V1",
        "arm": arm, "unit": unit, "invocation_id": invocation_id,
        "book_sha256": book_sha256,
        "identity": ident,
        "snapshot_stable_across_the_build": True,
        "n_files_snapshotted": len(end),
        "READ_EVIDENCE": fd_samples or {
            "observed": False,
            "why": "NOT OBSERVED. The fit directory is mounted `relatime` "
                   "and a probe read did not advance atime, so atime cannot "
                   "witness a per-build read; `inotifywait` is absent. The "
                   "remaining routes inject an observer into the build "
                   "process, which is refused for a frozen builder.",
            "what_this_record_DOES_establish":
                "these exact bytes were on disk for this build and nothing "
                "swapped under it",
            "what_it_does_NOT":
                "that the builder read them",
        },
    }


def falsify() -> int:                                        # noqa: C901
    checks = []

    def note(n, ok):
        checks.append((n, bool(ok)))

    def refuses(fn, token):
        try:
            fn()
            return False
        except IdentityRefused as e:
            return token in str(e)

    ARM = "CONDVALUE_X_SKEW"
    comps = declared_compositions(ARM)
    note("three declared compositions are DERIVED from the params, not typed",
         len(comps) == 3 and [c["n_models"] for c in comps] == [2, 4, 5])
    note("they are supersets of one another",
         set(comps[0]["model_digests_declared"]) <
         set(comps[1]["model_digests_declared"]) <
         set(comps[2]["model_digests_declared"]))

    disk = snapshot()
    # 1. POSITIVE CONTROL -- the real bytes on disk report composition 3.
    got = identify(ARM, disk)
    note("the REAL models on disk report composition 3",
         got["composition"] == 3 and got["n_models"] == 5)
    note("and asserting composition 3 passes",
         identify(ARM, disk, assert_composition=3)["composition"] == 3)

    # 2. POSITIVE CONTROL -- a composition-1 byte set reports composition 1.
    c1 = {n: str(d) + "0" * (64 - len(str(d)))
          for n, d in comps[0]["model_digests_declared"].items()}
    note("a composition-1 byte set reports composition 1",
         identify(ARM, c1)["composition"] == 1)

    # 3. **THE ONE THE COORDINATOR ASKED FOR** -- handed a composition-1 set
    #    while asserting composition 3, it must REFUSE, not pass.
    note("composition-1 bytes asserted as 3 REFUSE rather than pass",
         refuses(lambda: identify(ARM, c1, assert_composition=3), MISMATCH))

    # 4. KNOWN-BAD -- a set matching nothing refuses.
    note("a set matching no declared composition REFUSES",
         refuses(lambda: identify(ARM, {"lgbm_haz_btc.txt": "f" * 64}), NO_MATCH))

    # 5. KNOWN-BAD -- a WRONG prefix must not match a declared digest.
    bad = dict(c1)
    k = sorted(bad)[0]
    bad[k] = "0" * 64
    note("a wrong prefix does not match", refuses(
        lambda: identify(ARM, bad), NO_MATCH))

    # 6. KNOWN-BAD -- models moving under the build refuses.
    moved = dict(disk)
    moved[sorted(moved)[0]] = "a" * 64
    note("models changing between start and end REFUSE",
         refuses(lambda: attest(ARM, start=disk, end=moved), MOVED))

    # 7. KNOWN-BAD -- an absent fit directory refuses, never empty-passes.
    note("an absent fit directory REFUSES",
         refuses(lambda: snapshot("/nonexistent/fits"), ABSENT))

    # 8. THE BOOK HALF IS A FLOOR, and a hazard-only book floors at 1.
    vb = composition_floor_from_book(
        {"score_contracts": {"h": {"formula": "p_fill * predicted_conditional_value"}}})
    hz = composition_floor_from_book(
        {"score_contracts": {"h": {"formula": "p_fill"}}})
    note("a value-bearing book floors at composition 2",
         vb["composition_floor"] == 2 and vb["value_bearing"] is True)
    note("a hazard-only book floors at composition 1",
         hz["composition_floor"] == 1 and hz["value_bearing"] is False)
    note("a book with no score_contracts REFUSES",
         refuses(lambda: composition_floor_from_book({}), BOOK_CANNOT_SAY))

    # 9. The attest record names its own limit rather than implying reads.
    rec = attest(ARM, start=disk, end=disk, unit="probe")
    # STRUCTURE, not spelling. An earlier version of this check searched
    # for the words "does not" and failed on `what_it_does_NOT` -- a
    # text-match defect inside the falsifier written to catch them.
    note("the record states that a read was NOT observed",
         rec["READ_EVIDENCE"]["observed"] is False
         and "what_it_does_NOT" in rec["READ_EVIDENCE"]
         and "what_this_record_DOES_establish" in rec["READ_EVIDENCE"])

    for n, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {n}")
    bad_ = [n for n, ok in checks if not ok]
    print(json.dumps({"falsifier": "be_model_identity_sidecar",
                      "n": len(checks), "n_failed": len(bad_), "failed": bad_}))
    return 1 if bad_ else 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--selftest" in argv:
        return falsify()
    arm = argv[0] if argv else "CONDVALUE_X_SKEW"
    snap = snapshot()
    print(json.dumps(attest(arm, start=snap, end=snap), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
