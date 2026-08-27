"""Rebuild the PRED_STATE tape WITH its required inputs. R-184(4)(ii)/R-185.

AUTHORISATION (R-126, in-file): R-184 user audit; R-185(2) bound rules.

WHAT WAS WRONG WITH THE FIRST TAPE. It was built without `gaps` and without
`bn_recv_ns`. Neither omission errors -- that is the whole problem:
  * no bn_recv_ns  -> bn_feed_age_s is None and bn_feed_missing 1.0 for EVERY
                      row, so the freshness family is CONSTANT and carries no
                      information at all while looking like three features
  * no gaps        -> GAP_AT_CUTOFF can never fire, so the population reports
                      zero gap-affected rows and looks cleaner than it is
Both are supplied here, and `assert_required_inputs` REFUSES without them.

BUILT TO THE SCHEMA, NOT TO THE CHECKER. DA's verifier is armed and BE has not
read it. BE builds against `da_pred_state_v1_schema.json`; DA checks against
the same schema; neither reads the other's intent. A tape tuned to a checker
proves only that it was tuned.

CARRIED PER ROW: feature_asof and decision_time (knowledge-time evidence),
state_status (counted, never a silent drop), and the SPLIT LABEL, so the
embargo can be certified on times rather than assumed from identity.
"""
from __future__ import annotations

import gzip, glob, json, os, sys, tempfile
from pathlib import Path

sys.path.insert(0, "/home/yuqing/ctaNew/live/pm_research")
import phase2_state_schema_freeze as PIN
import phase2_embargo as EMB

DERIVED = Path("/home/yuqing/ctaNew/data/pm_5min/derived")
OUT = DERIVED / "phase2_state_tape_v2.json"
FRAGMENT = DERIVED / "harmful_exposure_rows_v3_eraB.json"
TOPUP = DERIVED / "harmful_exposure_rows_v3_topup.json"
ERA_NS = 1787579334881534478
BN = {"btc": "BTCUSDT", "eth": "ETHUSDT"}


def bn_recv_for_window(coin: str, t0: float, span: float = 320.0) -> list:
    """Binance bookTicker receipt times covering one window. Era-pure."""
    import datetime as dt
    sym = BN[coin]
    out = []
    lo_t, hi_t = t0 - 20.0, t0 + span
    h = dt.datetime.fromtimestamp(lo_t, dt.timezone.utc)
    for k in range(3):
        hh = h + dt.timedelta(hours=k)
        for ext in (".csv.gz", ".csv"):
            for f in glob.glob(f"/home/yuqing/ctaNew/data/mm_hf/raw/bookTicker/"
                               f"{sym}/{hh:%Y%m%d_%H}{ext}"):
                op = gzip.open if f.endswith(".gz") else open
                with op(f, "rb") as fh:
                    for line in fh:
                        i = line.find(b",")
                        if i < 1:
                            continue
                        try:
                            r = int(line[:i])
                        except ValueError:
                            continue
                        if r < ERA_NS:
                            continue
                        t = r / 1e9
                        if lo_t <= t <= hi_t:
                            out.append(r)
                break
    out.sort()
    return out


def main() -> int:
    import harmful_hazard_model as hm
    import harmful_state_features as sf

    pin = PIN.build_pin()
    gaps_by_slug = hm.fi.gaps_by_slug(hm.fi.ERA)
    paths = hm.fi._archive_paths(); tokens = hm.fi.token_map()

    rows_out = []
    status_counts: dict = {}
    per_split: dict = {}
    for split, src in (("train", FRAGMENT), ("score", TOPUP)):
        data = json.loads(src.read_text())
        rows = [r for r in data["rows"] if r["status"] == "OK"]
        bywin: dict = {}
        for r in rows:
            bywin.setdefault(r["slug"], []).append(r)
        n_slug = 0
        for slug, wrows in bywin.items():
            coin = slug.split("-")[0]
            t0 = float(wrows[0]["t0"])
            g = gaps_by_slug.get(slug, [])
            bn = bn_recv_for_window(coin, t0)
            # REFUSES rather than degrading -- the R-184 finding, in code
            PIN.assert_required_inputs(g, bn)
            up, dn = tokens[slug]
            tape = sf.build_tape(paths[slug], up, dn, gaps=g, bn_recv_ns=bn)
            feats = sf.features_for_window(tape, wrows)
            for r, fe in zip(wrows, feats):
                st = str(fe.get("state_status", "OK"))
                status_counts[st] = status_counts.get(st, 0) + 1
                rows_out.append({
                    "slug": slug, "coin": coin, "day": r["day"],
                    "t0": r["t0"], "t_start": r["t_start"],
                    "side": r["side"], "gen": r["gen"],
                    "split": split,                       # embargo certifiable
                    "state_status": st,
                    "feature_asof": fe.get("feature_asof"),
                    "decision_time": float(r["t0"]) + float(r["t_start"]),
                    "label_exit_time": EMB.label_exit_time(r),
                    "state": {k: fe.get(k) for k in pin["features_in_order"]},
                })
            n_slug += 1
            if n_slug % 100 == 0:
                print(f"  [{split}] {n_slug}/{len(bywin)} slugs", flush=True)
        per_split[split] = {"slugs": len(bywin), "rows": sum(
            1 for x in rows_out if x["split"] == split)}
        print(f"  [{split}] DONE {per_split[split]}", flush=True)

    tr = [r for r in rows_out if r["split"] == "train"]
    sc = [r for r in rows_out if r["split"] == "score"]
    emb = None
    try:
        emb = EMB.assert_embargo(tr, sc)
        emb_state = "CERTIFIED"
    except EMB.EmbargoViolation as e:
        emb_state = f"VIOLATED (unpurged): {e}"

    out = {
        "protocol": "PHASE2_STATE_TAPE_V2",
        "built_from_schema": pin["derived_from"],
        "n_features": pin["n_features"],
        "features_in_order": pin["features_in_order"],
        "nullable_guard_pairs": pin["nullable_guard_pairs"],
        "required_inputs_supplied": {"gaps": True, "bn_recv_ns": True},
        "status_field": "state_status",
        "state_status_counts": status_counts,
        "per_split": per_split,
        "embargo": {"state": emb_state, "detail": emb,
                    "rule": "label_exit_time + 60s < first score feature time"},
        "n_rows": len(rows_out),
        "rows": rows_out,
    }
    fd, tmp = tempfile.mkstemp(dir=str(OUT.parent), suffix=".tmp")
    with os.fdopen(fd, "w") as fh:
        for chunk in json.JSONEncoder().iterencode(out):
            fh.write(chunk)
        fh.flush(); os.fsync(fh.fileno())
    os.replace(tmp, OUT)
    print(f"\nWROTE {OUT.name}: {len(rows_out):,} rows, "
          f"statuses {status_counts}, embargo {emb_state[:40]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
