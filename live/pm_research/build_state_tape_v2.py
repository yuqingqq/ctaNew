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
OUT = DERIVED / "phase2_state_tape_v5.json"   # DISTINCT: the v2 path holds
# the tape4 diagnostic bytes (GAP_AT_CUTOFF=286, an unknown mid-fix
# intermediate) and stays quarantined. Overwriting it would destroy the one
# artifact that records what a moving-tree build produced.
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
    # R-191: gaps are COIN-LEVEL and ABSOLUTE, read from the collector-gaps
    # LEDGER -- not per-slug. Per-slug assignment made a feed gap invisible to
    # every window except the one it happened to be logged against, which is
    # why BE counted 0 where the ledger counts 289.
    import gap_at_cutoff_count as GC
    coin_gaps_abs = GC.load_coin_gaps()
    print(f"  coin-level absolute gaps from the ledger: "
          f"{ {c: len(v) for c, v in coin_gaps_abs.items() if c in ('btc','eth')} }",
          flush=True)

    def gaps_for(slug: str, coin: str, t0: float):
        """Project the COIN's absolute gaps into THIS window's basis.

        `_in_gap` compares a window-relative cutoff, so the intervals are
        shifted by t0 rather than the comparison being changed. Intervals
        landing outside [0, WINDOW_S] are KEPT deliberately: a gap logged
        against the preceding window overlaps this window's warm-up rows at
        NEGATIVE t_start, and those are precisely the 289."""
        lo, hi = t0 - 400.0, t0 + 700.0
        return [(a - t0, b - t0) for a, b in coin_gaps_abs.get(coin, ())
                if b >= lo and a <= hi]
    paths = hm.fi._archive_paths(); tokens = hm.fi.token_map()

    # STREAMING (R-174). The stopped build held every row in memory and hit
    # 7.4G at 200/471 slugs. Rows are appended to a JSONL spool per slug and
    # the final artifact is streamed FROM the spool, so peak memory is one
    # slug's rows, not the population's.
    spool_fd, spool_path = tempfile.mkstemp(dir=str(OUT.parent), suffix=".jsonl")
    spool = os.fdopen(spool_fd, "w")
    n_rows = 0
    tr_last_exit = float("-inf"); sc_first_feat = float("inf")
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
            g = gaps_for(slug, coin, t0)
            bn = bn_recv_for_window(coin, t0)
            # REFUSES rather than degrading -- the R-184 finding, in code
            PIN.assert_required_inputs(g, bn)
            up, dn = tokens[slug]
            tape = sf.build_tape(paths[slug], up, dn, gaps=g, bn_recv_ns=bn)
            feats = sf.features_for_window(tape, wrows)
            for r, fe in zip(wrows, feats):
                st = str(fe.get("state_status", "OK"))
                status_counts[st] = status_counts.get(st, 0) + 1
                _row = {
                    "slug": slug, "coin": coin, "day": r["day"],
                    "t0": r["t0"], "t_start": r["t_start"],
                    "side": r["side"], "gen": r["gen"],
                    "split": split,                       # embargo certifiable
                    "state_status": st,
                    "feature_asof": fe.get("feature_asof"),
                    # CLOCK BASIS, per the schema: decision_time is
                    # WINDOW-RELATIVE seconds (== t_start) and is legitimately
                    # NEGATIVE for pre-window warm-up rows. BE's first builder
                    # wrote t0 + t_start here -- an absolute epoch under a name
                    # the schema declares window-relative -- so any reader
                    # adding t0 would have double-counted the window start.
                    "decision_time": float(r["t_start"]),
                    "decision_time_epoch": float(r["t0"]) + float(r["t_start"]),
                    "label_exit_time": EMB.label_exit_time(r),
                    "state": {k: fe.get(k) for k in pin["features_in_order"]},
                }
                spool.write(json.dumps(_row) + "\n")
                n_rows += 1
                if split == "train":
                    tr_last_exit = max(tr_last_exit, _row["label_exit_time"])
                else:
                    sc_first_feat = min(sc_first_feat, _row["decision_time_epoch"])
            n_slug += 1
            if n_slug % 100 == 0:
                print(f"  [{split}] {n_slug}/{len(bywin)} slugs", flush=True)
        per_split[split] = {"slugs": len(bywin)}
        print(f"  [{split}] DONE {per_split[split]}", flush=True)

    spool.flush(); os.fsync(spool.fileno()); spool.close()
    # embargo from the running extremes -- no second pass over the rows
    gap = sc_first_feat - tr_last_exit
    emb = {"gap_s": gap, "embargo_s": EMB.EMBARGO_S,
           "last_train_label_exit": tr_last_exit,
           "first_score_feature": sc_first_feat}
    emb_state = "CERTIFIED" if gap >= EMB.EMBARGO_S else (
        f"VIOLATED (unpurged): gap {gap:.3f}s < {EMB.EMBARGO_S}s")

    import subprocess as _sp
    _head = _sp.run(["git", "-C", "/home/yuqing/ctaNew", "rev-parse", "HEAD"],
                    capture_output=True, text=True).stdout.strip()
    _dirty = bool(_sp.run(["git", "-C", "/home/yuqing/ctaNew", "status",
                           "--porcelain"], capture_output=True,
                          text=True).stdout.strip())
    _built_at = _sp.run(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"],
                        capture_output=True, text=True).stdout.strip()
    out = {
        "protocol": "PHASE2_STATE_TAPE_V5",
        # R-196(2): a heavy build launches only from a COMMITTED ref, and the
        # ref travels ON the artifact. tape4 was launched 06:08:02Z from a
        # working tree mid-fix and produced GAP_AT_CUTOFF=286 -- an unknown
        # intermediate of six in-flight fixes, reconcilable to nothing.
        "builder_commit": _head,
        "builder_tree_dirty_at_build": _dirty,
        "built_at_utc": _built_at,
        # LAYOUT: the schema's native form is FLAT; this tape WRAPS the
        # features, so it declares the wrapping key rather than leaving a
        # reader to guess it (schema LAYOUT.note).
        "features_under": "state",
        "clock_basis": {"decision_time": "window_relative_seconds",
                        "decision_time_epoch": "absolute_epoch",
                        "label_exit_time": "absolute_epoch"},
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
        "n_rows": n_rows,
    }
    fd, tmp = tempfile.mkstemp(dir=str(OUT.parent), suffix=".tmp")
    with os.fdopen(fd, "w") as fh:
        head = json.dumps(out)
        fh.write(head[:-1] + ', "rows": [')      # splice the array in
        with open(spool_path) as sp:
            first = True
            for line in sp:
                if not first:
                    fh.write(",")
                fh.write(line.rstrip("\n")); first = False
        fh.write("]}")
        fh.flush(); os.fsync(fh.fileno())
    os.replace(tmp, OUT)
    Path(spool_path).unlink(missing_ok=True)
    print(f"\nWROTE {OUT.name}: {n_rows:,} rows, "
          f"statuses {status_counts}, embargo {emb_state[:44]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
