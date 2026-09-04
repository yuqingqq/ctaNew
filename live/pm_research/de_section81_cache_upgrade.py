"""Upgrade a §8.1 arms cache to the v2 schema WITHOUT re-running assembly.

WHY THIS EXISTS. The v2 schema adds `terminal_marks` to `fr` and changes
NOTHING ELSE. A full cold re-run rebuilds the tape index and the
assembly too, and on this machine that OOMs inside the standing 8 G cap
(`run-u56625.scope: systemd-oomd killed 1 process(es)`, 2026-09-04
13:22:55Z) AFTER the 248.8 s feed has already succeeded -- so the
expensive half is thrown away by the half that does not need redoing.

THE REUSE IS NOT ASSUMED, IT IS CHECKED. `asm` is keyed by
`(slug, side, t0)`, so it remains valid only if the newly fed reference
is structurally IDENTICAL to the one the assembly was built against.
This module recomputes the reference, compares it FIELD BY FIELD against
the old one, and REFUSES if anything but the new key differs. A cache
upgrade that assumed the reuse would be a stale-cache bug that looks
like a measurement -- which is the same failure the v2 filename bump
exists to prevent, one level up.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

#: The one key v2 adds. Anything else differing is a refusal.
V2_ADDED_KEYS = ("terminal_marks",)

#: Statuses `build_reference` gained with it; their presence is not a
#: structural change to the reference itself.
V2_ADDED_STATUSES = ("TERMINAL_MARK_OK", "TERMINAL_MARK_MISSING",
                     "TERMINAL_MARK_ENDED_IN_GAP")


class CacheUpgradeRefused(RuntimeError):
    """Refused rather than reusing an assembly against a changed reference."""


def reference_signature(ref: dict) -> dict:
    """Everything `asm` is keyed on, plus the payload a consumer reads --
    so a difference anywhere the assembly or the replay could see is
    visible to the comparison."""
    sig = {}
    for slug, sides in sorted(ref.items()):
        for side, gens in sorted(sides.items()):
            for g in gens:
                # KEYED THE WAY `asm` IS KEYED -- (slug, side, t0) --
                # and NOT by `gen`. Keyed by gen, a moved t0 read as a
                # CHANGED generation; keyed by t0 it reads as one key
                # gone and another arrived, which is what it is to the
                # assembly. Found by this module's own falsifier, which
                # is why the falsifier asserts the key and not the diff.
                sig[f"{slug}|{side}|{g['t0']!r}"] = {
                    "gen": g["gen"], "t1": g["t1"], "level": g["level"],
                    "displayed": g.get("displayed"),
                    "status": g.get("status"),
                    "tranches": [
                        (t["t"], t["shares"],
                         t["markout_cents_per_share"],
                         t.get("mid_at_fill"), t.get("level"))
                        for t in g.get("tranches", ())],
                }
    return sig


def compare_references(old: dict, new: dict) -> dict:
    """A field-by-field verdict, with the differing keys NAMED."""
    so, sn = reference_signature(old), reference_signature(new)
    only_old = sorted(set(so) - set(sn))
    only_new = sorted(set(sn) - set(so))
    changed = sorted(k for k in set(so) & set(sn) if so[k] != sn[k])
    return {
        "n_generations_old": len(so), "n_generations_new": len(sn),
        "only_in_old": only_old[:20], "n_only_in_old": len(only_old),
        "only_in_new": only_new[:20], "n_only_in_new": len(only_new),
        "changed": changed[:20], "n_changed": len(changed),
        "identical": not (only_old or only_new or changed),
    }


def upgrade(src: Path, dst: Path, *, coin: str = "btc",
            limit: int | None = 12) -> dict:
    if not src.exists():
        raise CacheUpgradeRefused(f"REFUSED: no source cache at {src}")
    if dst.exists():
        raise CacheUpgradeRefused(
            f"REFUSED: {dst} already exists. An artifact under review is "
            f"moved aside with a timestamp, never overwritten")
    old = pickle.loads(src.read_bytes())
    if "fr" not in old or "asm" not in old:
        raise CacheUpgradeRefused(
            f"REFUSED: {src} is not an arms cache (keys "
            f"{sorted(old) if isinstance(old, dict) else type(old)})")
    if any(k in old["fr"] for k in V2_ADDED_KEYS):
        raise CacheUpgradeRefused(
            f"REFUSED: {src} already carries {V2_ADDED_KEYS}; it is not a "
            f"v1 cache and upgrading it would hide what it already is")
    import de_phase4_diag_runner as R
    t0 = time.time()
    new_fr = R.build_reference(coin, limit=limit)
    feed_s = round(time.time() - t0, 1)
    cmp = compare_references(old["fr"]["reference"], new_fr["reference"])
    if not cmp["identical"]:
        raise CacheUpgradeRefused(
            f"REFUSED: the re-fed reference DIFFERS from the one the "
            f"assembly was built against -- {cmp['n_changed']} changed, "
            f"{cmp['n_only_in_new']} new, {cmp['n_only_in_old']} gone. "
            f"Reusing `asm` here would score generations against a "
            f"reference they were not assembled from. First differing "
            f"keys: {(cmp['changed'] or cmp['only_in_new'] or cmp['only_in_old'])[:3]}")
    for k in V2_ADDED_KEYS:
        if k not in new_fr:
            raise CacheUpgradeRefused(
                f"REFUSED: the re-fed reference has no {k!r}, so this is "
                f"not a v2 build and the filename would lie about it")
    prov = {
        "upgraded_from": str(src), "upgraded_to": str(dst),
        "as_of": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "feed_wall_s": feed_s,
        "asm_reused": True,
        "asm_reuse_justification":
            "the re-fed reference is IDENTICAL to the one the assembly "
            "was built against, compared field by field over every "
            "generation's t0/t1/level/displayed/status and every "
            "tranche's (t, shares, markout, mid_at_fill, level). `asm` "
            "is keyed by (slug, side, t0), so identity on those keys is "
            "what makes the reuse legitimate -- CHECKED, never assumed",
        "reference_comparison": cmp,
        "new_statuses": {k: new_fr["statuses"].get(k)
                         for k in V2_ADDED_STATUSES},
        "why_not_a_cold_rerun":
            "a cold run rebuilds the tape index and assembly too and was "
            "OOM-killed by systemd-oomd inside the standing 8G cap AFTER "
            "the feed had succeeded (run-u56625.scope, 2026-09-04 "
            "13:22:55Z). The feed is the half that changed",
    }
    dst.write_bytes(pickle.dumps({"fr": new_fr, "asm": old["asm"],
                                  "upgrade_provenance": prov}))
    return prov


def selftest() -> int:
    """FALSIFIERS IN BOTH DIRECTIONS (rule 15)."""
    import tempfile
    checks, fails = 0, []

    def ok(c, m):
        nonlocal checks
        checks += 1
        if not c:
            fails.append(m)

    g = lambda t0=1.0, lvl=0.5, tr=(): {
        "gen": 1, "t0": t0, "t1": t0 + 1, "level": lvl, "displayed": 5.0,
        "status": "OK", "tranches": list(tr)}
    A = {"w1": {"BUY_UP": [g()], "SELL_UP": []}}

    ok(compare_references(A, {"w1": {"BUY_UP": [g()], "SELL_UP": []}})
       ["identical"] is True,
       "NEGATIVE CONTROL: two identical references must compare identical")
    # POSITIVE CONTROLS -- each difference the reuse would hide.
    _t0 = compare_references(A, {"w1": {"BUY_UP": [g(t0=2.0)],
                                        "SELL_UP": []}})
    ok(_t0["n_only_in_new"] == 1 and _t0["n_only_in_old"] == 1
       and _t0["n_changed"] == 0,
       f"FALSIFIER: a changed t0 is a DIFFERENT ASSEMBLY KEY -- one key "
       f"gone, one arrived, none 'changed' -- because `asm` is keyed by "
       f"(slug, side, t0). Got {_t0['n_only_in_new']}/"
       f"{_t0['n_only_in_old']}/{_t0['n_changed']}")
    ok(compare_references(A, {"w1": {"BUY_UP": [g(lvl=0.6)],
                                     "SELL_UP": []}})["n_changed"] == 1,
       "FALSIFIER: a changed level must show as CHANGED, not as identical")
    ok(compare_references(
        A, {"w1": {"BUY_UP": [g(tr=({"t": 1.0, "shares": 5.0,
                                     "markout_cents_per_share": 1.0,
                                     "mid_at_fill": 0.5,
                                     "level": 0.5},))],
                   "SELL_UP": []}})["n_changed"] == 1,
       "FALSIFIER: a tranche appearing must show -- the payload the "
       "replay reads is part of the identity, not just the key")
    ok(compare_references(A, {})["n_only_in_old"] == 1,
       "FALSIFIER: a vanished generation must show")

    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        bad = p / "bad.pkl"
        bad.write_bytes(pickle.dumps({"nope": 1}))
        for src, why in ((p / "missing.pkl", "a missing cache"),
                         (bad, "a non-cache pickle")):
            try:
                upgrade(src, p / "out.pkl")
                ok(False, f"REFUSAL: must refuse {why}")
            except CacheUpgradeRefused:
                ok(True, "")
        already = p / "v2.pkl"
        already.write_bytes(pickle.dumps(
            {"fr": {"reference": {}, "terminal_marks": {}}, "asm": {}}))
        try:
            upgrade(already, p / "out2.pkl")
            ok(False, "REFUSAL: must refuse a cache that is ALREADY v2")
        except CacheUpgradeRefused:
            ok(True, "")
        exists = p / "there.pkl"
        exists.write_bytes(b"x")
        src2 = p / "v1.pkl"
        src2.write_bytes(pickle.dumps({"fr": {"reference": {}}, "asm": {}}))
        try:
            upgrade(src2, exists)
            ok(False, "REFUSAL: must refuse to overwrite an existing dst")
        except CacheUpgradeRefused:
            ok(True, "")

    print(json.dumps({"selftest": "PASS" if not fails else "FAIL",
                      "checks": checks, "failures": fails}, indent=1))
    return 0 if not fails else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--src"); ap.add_argument("--dst")
    ap.add_argument("--coin", default="btc")
    ap.add_argument("--limit", type=int, default=12)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not (a.src and a.dst):
        raise SystemExit("REFUSED: --src and --dst are required")
    try:
        print(json.dumps(upgrade(Path(a.src), Path(a.dst), coin=a.coin,
                                 limit=a.limit), indent=1, sort_keys=True))
    except CacheUpgradeRefused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
