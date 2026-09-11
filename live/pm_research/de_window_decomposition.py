"""PER-WINDOW DECOMPOSITION OF A BOOK'S SETTLED D. Reads a BOOK, not a result.

A slug IS a five-minute window, so `settlement_legs_by_slug` already
decomposes the ruled quantity exactly; this groups those per-slug totals
onto the 300 s grid and differences the arm against the zero-cancel
baseline. The sum over ALL windows equals the book's D BY CONSTRUCTION --
which is why that identity is ASSERTED here, not trusted.

IT DECOMPOSES EVERY WINDOW, NOT ONLY THE DECLARED 27. The emit differences
two books over the declared spine; the windows OUTSIDE it are the control:
a gap repair must not move a window that carried no gap, so a non-zero
difference there is a finding, and it can only be seen if those windows
were decomposed too.
"""
from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

PROTOCOL = "P003_DE_WINDOW_DECOMPOSITION_V1"
GRID_S = 300
SUM_MISMATCH = "WINDOW_DECOMPOSITION_DOES_NOT_SUM_TO_THE_BOOKS_D"
BAD_SLUG = "WINDOW_DECOMPOSITION_CANNOT_PLACE_A_SLUG_ON_THE_GRID"


class DecompositionRefused(RuntimeError):
    """A named refusal."""


def window_of(slug: str) -> int:
    """A slug carries its window start as its final field."""
    try:
        start = int(str(slug).rsplit("-", 1)[1])
    except (IndexError, ValueError) as exc:
        raise DecompositionRefused(
            f"REFUSED {BAD_SLUG}: {slug!r} -- {exc}. A slug that cannot be "
            f"placed on the grid cannot be attributed, and attributing it "
            f"to a default window would put real cents in a wrong row.")
    if start % GRID_S:
        raise DecompositionRefused(
            f"REFUSED {BAD_SLUG}: {slug!r} starts at {start}, not on the "
            f"{GRID_S}s grid.")
    return start


def decompose(book_path, params=None) -> dict:
    """Per-window arm/baseline/D for every arm in the params."""
    import de_multiday_gate1_runner as R
    params = params or R.load_params()
    mod, _ = R.import_be_cascade(params)
    t0 = time.time()
    bk = mod.load(Path(book_path))
    slugs = sorted({r["slug"] for r in bk["rows"]})
    winners = R.winner_source(required_slugs=slugs)["winners"]
    base_legs = R.settlement_legs_by_slug(
        mod.replay(bk, mod.flagged_stream(bk["rows"], []), 0.5)["fills"],
        winners)
    out = {"protocol": PROTOCOL, "book": str(book_path),
           "book_sha256": bk["source_sha256"], "grid_s": GRID_S, "arms": {}}
    for arm, spec in params["arms"].items():
        arm_legs = R.settlement_legs_by_slug(
            mod.replay(bk, mod.arm_stream(bk, spec["head"]),
                       spec["theta"])["fills"], winners)
        per: dict = {}
        for legs, key in ((base_legs, "baseline_settled_cents"),
                          (arm_legs, "arm_settled_cents")):
            for slug, leg in legs["per_slug"].items():
                w = per.setdefault(window_of(slug), {
                    "window_start": window_of(slug),
                    "baseline_settled_cents": 0.0,
                    "arm_settled_cents": 0.0})
                w[key] += float(leg["total_cents"])
        for w in per.values():
            w["D_contribution_cents"] = (w["arm_settled_cents"]
                                         - w["baseline_settled_cents"])
        total_base = float(base_legs["total_cents"])
        total_arm = float(arm_legs["total_cents"])
        D = total_arm - total_base
        summed = sum(w["D_contribution_cents"] for w in per.values())
        if abs(summed - D) >= 1e-6:
            raise DecompositionRefused(
                f"REFUSED {SUM_MISMATCH}: {arm} rows sum to {summed!r} and "
                f"the book's D is {D!r}. A decomposition that does not sum "
                f"is not a view of the quantity, it is a different one.")
        out["arms"][arm] = {
            "theta": spec["theta"], "D_cents": D,
            "baseline_total_cents": total_base,
            "arm_total_cents": total_arm,
            "n_windows": len(per),
            "rows_sum_to_D": True,
            "per_window": {str(k): v for k, v in sorted(per.items())}}
    out["elapsed_s"] = round(time.time() - t0, 1)
    out["peak_rss_gib"] = round(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 20), 3)
    return out


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--falsify" in argv:
        ok = cells = 0
        def ck(n, c):
            nonlocal ok, cells
            cells += 1; ok += bool(c); print(f"  [{'PASS' if c else 'FAIL'}] {n}")
        ck("a slug's final field is its window start",
           window_of("btc-updown-5m-1788813900") == 1788813900)
        for bad, why in (("btc-updown-5m-notanint", "non-numeric"),
                         ("btc-updown-5m-1788813901", "off-grid")):
            try:
                window_of(bad); ck(f"REFUSES a {why} slug", False)
            except DecompositionRefused as e:
                ck(f"REFUSES a {why} slug", BAD_SLUG in str(e))
        print(f"\n{ok}/{cells} cells pass")
        return 0 if ok == cells else 1
    book = argv[0]
    out = decompose(book)
    dst = Path("/home/yuqing/ctaNew/data/pm_5min/derived/fwd_v2") / (
        "p003_de_window_decomposition_" + Path(book).stem + ".json")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1, default=str))
    for arm, a in out["arms"].items():
        print(f"  {arm}: D={a['D_cents']:+.6f}c windows={a['n_windows']} "
              f"rows_sum_to_D={a['rows_sum_to_D']}")
    print(f"  peak_rss_gib={out['peak_rss_gib']} elapsed_s={out['elapsed_s']}")
    print(f"  written: {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
