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

# R-199 seam 20c: the import root is THIS FILE'S OWN DIRECTORY. It was
# hardcoded to the main tree, so a snapshot build imported live-tree modules
# another seat was editing -- the snapshot isolated nothing. Deriving it from
# __file__ means a snapshot copy imports its own siblings by construction.
_ROOT = str(Path(__file__).resolve().parent)
sys.path.insert(0, _ROOT)
import phase2_state_schema_freeze as PIN
import phase2_embargo as EMB


def assert_modules_under_root() -> None:
    """Every pinned module must have loaded from THIS tree, not another.

    Without this, a wrong-tree import is completely silent: the build runs,
    produces a plausible artifact, and the bytes belong to whatever the live
    tree happened to contain. That is exactly how tape5b became
    unattributable."""
    import gap_at_cutoff_count as _GC
    for m in (PIN, EMB, _GC):
        f = str(Path(m.__file__).resolve())
        if not f.startswith(_ROOT):
            raise SystemExit(
                f"REFUSED: {m.__name__} loaded from {f}, which is OUTSIDE this "
                f"build root {_ROOT}. A snapshot that imports another tree "
                f"isolates nothing.")

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


def assert_build_ref() -> str:
    """BUILD_REF must be present and well-formed BEFORE any work begins.

    An earlier version checked it where the header is written -- after the
    whole population had been rebuilt. A refusal that fires at the end is not
    a refusal, it is a wasted build."""
    r = os.environ.get("BUILD_REF", "").strip()
    if len(r) != 40 or any(c not in "0123456789abcdef" for c in r.lower()):
        raise SystemExit(
            "REFUSED at startup: BUILD_REF must be a 40-hex commit ref on the "
            "launch line. A build that determines its own provenance at "
            "runtime cannot be attributed to the code that produced it.")
    return r


PM_DATA_ROOT = Path("/home/yuqing/ctaNew")


def pin_data_root() -> None:
    """Point every data-deriving module at the REAL tree.

    `flow_intensity` computes `REPO = Path(__file__).resolve().parents[2]`
    (:44-45), so a CODE snapshot silently relocates the DATA root with it --
    inside a worktree `fi.PM` became <snapshot>/data/pm_5min, `token_map()`
    returned 0 entries, and every slug raised KeyError. The snapshot rule
    assumed data paths were absolute; they are, in THIS builder, and are not
    in a module it calls. Code isolation and data location are separate
    concerns and must be stated separately."""
    import harmful_hazard_model as _hm
    fi = _hm.fi
    pm = PM_DATA_ROOT / "data/pm_5min"
    # EVERY derived constant, not just PM. RAW/GAPS/MARKETS are computed FROM
    # PM at IMPORT time, so rebinding PM alone leaves them pointing into the
    # snapshot -- which is why the first version of this fix still returned an
    # empty token_map. Fixing a constant does not fix what was derived from it.
    fi.REPO = PM_DATA_ROOT
    fi.PM = pm
    fi.RAW = pm / "raw"
    fi.GAPS = pm / "collector_gaps.jsonl"
    fi.MARKETS = pm / "markets.jsonl"
    for name in ("PM", "RAW", "GAPS", "MARKETS"):
        q = getattr(fi, name)
        if not q.exists():
            raise SystemExit(f"REFUSED: pinned {name} = {q} does not exist.")
    # Values COMPUTED AT IMPORT from a path are stale after rebinding it.
    # DAYS = _discover_days() ran at import and returned 0 in the snapshot, so
    # _archive_paths() -- which iterates DAYS -- stayed empty even though RAW
    # was correct. Third round of this class: PM, then RAW/GAPS/MARKETS, now
    # DAYS. Enumerating constants does not converge; verifying CONSUMERS does.
    if hasattr(fi, "_discover_days"):
        fi.DAYS = fi._discover_days()

    # BEHAVIOURAL, over EVERY input the builder actually consumes. The prior
    # version checked token_map() alone and passed while _archive_paths() was
    # empty -- which produced a 0-row tape that reported "embargo CERTIFIED".
    consumers = {"token_map": len(fi.token_map()),
                 "archive_paths": len(fi._archive_paths()),
                 "gaps_by_slug": len(fi.gaps_by_slug(fi.ERA)),
                 "DAYS": len(fi.DAYS)}
    # R-202(1) SAME-INSTANCE PREFLIGHT. Loading a map is not the lookup the row
    # path performs. This probes REAL slugs from the actual population through
    # the SAME dicts main() will use, and asserts real token pairs come back --
    # the previous check verified the load and the row path still matched zero.
    import json as _j
    _tok = fi.token_map(); _pth = fi._archive_paths()
    _probe = []
    try:
        _d = _j.loads(FRAGMENT.read_text())
        _probe = sorted({r["slug"] for r in _d["rows"][:20000]})[:25]
    except Exception:
        pass
    if _probe:
        _hit = [x for x in _probe if x in _tok and x in _pth]
        _pairs_ok = all(isinstance(_tok[x], tuple) and len(_tok[x]) == 2
                        for x in _hit)
        if len(_hit) < len(_probe) or not _pairs_ok:
            raise SystemExit(
                f"REFUSED: row-path probe matched {len(_hit)}/{len(_probe)} "
                f"real population slugs (token pairs well-formed: {_pairs_ok}). "
                f"The maps LOAD but the lookup the build performs does not "
                f"resolve them -- exactly the state that emitted a 0-row tape.")
        consumers["row_path_probe"] = f"{len(_hit)}/{len(_probe)}"
    empty = [k for k, v in consumers.items() if v == 0]
    if empty:
        raise SystemExit(
            f"REFUSED: these builder inputs load EMPTY after pinning: {empty}. "
            f"Counts: {consumers}. Paths that look right while the data does "
            f"not load is the exact state that produced a zero-row tape.")
    print(f"  data root pinned; inputs {consumers}", flush=True)


def main() -> int:
    _ref0 = assert_build_ref()
    assert_modules_under_root()
    pin_data_root()
    print(f"  modules under snapshot root {_ROOT}", flush=True)
    print(f"  BUILD_REF {_ref0[:12] or '<ABSENT>'}", flush=True)
    import harmful_hazard_model as hm
    import harmful_state_features as sf

    pin = PIN.build_pin()
    # R-191: gaps are COIN-LEVEL and ABSOLUTE, read from the collector-gaps
    # LEDGER -- not per-slug. Per-slug assignment made a feed gap invisible to
    # every window except the one it happened to be logged against, which is
    # why BE counted 0 where the ledger counts 289.
    import gap_at_cutoff_count as GC
    # A pinned LEDGER SNAPSHOT makes the gap population reproducible: the live
    # ledger grows with every collector restart, so two builds minutes apart
    # can legitimately disagree on the count.
    _ledger = os.environ.get("LEDGER_PATH", "").strip()
    if _ledger:
        _lp = Path(_ledger)
        if not _lp.exists():
            raise SystemExit(f"REFUSED: LEDGER_PATH={_lp} does not exist.")
        coin_gaps_abs = GC.load_coin_gaps(_lp)
        import hashlib as _hl
        _ledger_sha = _hl.sha256(_lp.read_bytes()).hexdigest()
        print(f"  pinned ledger {_lp.name} sha {_ledger_sha[:16]}", flush=True)
    else:
        coin_gaps_abs = GC.load_coin_gaps()
        _ledger_sha = None
    print(f"  coin-level absolute gaps from the ledger: "
          f"{ {c: len(v) for c, v in coin_gaps_abs.items() if c in ('btc','eth')} }",
          flush=True)

    def gap_contains(T_abs: float, coin: str):
        """THE ruled predicate, and the ONLY gap comparison in this builder.

        [g_start, g_end) -- lower-INCLUSIVE, upper-EXCLUSIVE -- evaluated on
        the ABSOLUTE instant at FULL PRECISION. No projection.

        R-213: there were TWO comparisons (main path and warm-up/shifted path)
        and they disagreed at both edges: all 4 rows at exactly T==g_start were
        unflagged (effectively strictly-exclusive lower bound), while the one
        negative-t_start row at T==g_end WAS flagged (effectively inclusive
        upper bound). Projecting gaps into window-relative form by subtracting
        t0 from values near 1.79e9 is lossy -- ULP there is 2.4e-7 s -- so
        exact equality survives for some values and not others. Comparing in
        the absolute basis removes the projection, and routing BOTH paths
        through this one function removes the divergence categorically rather
        than by diagnosing either edge."""
        for a, b in coin_gaps_abs.get(coin, ()):
            if a <= T_abs < b:
                return (a, b)
        return None

    def gaps_for(slug: str, coin: str, t0: float):
        """Project the COIN's absolute gaps into THIS window's basis.

        RETAINED FOR THE TAPE HEADER ONLY. The gap DECISION is made by
        gap_contains() on the absolute instant; this projection no longer
        feeds any comparison (R-213). Intervals
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
    status_counts: dict = {}      # EMITTED rows: population statements
    skip_counts: dict = {}        # PRE-EMISSION: the row never entered at all
    no_token_by_coin_day: dict = {}
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
            if slug not in tokens or slug not in paths:
                _why = ("NO_TOKEN_MAP" if slug not in tokens
                        else "NO_ARCHIVE_PATH")
                # rule 4: an exclusion is a COUNTED STATUS. Not a KeyError
                # (which killed tape6 outright), and not a silent .get() skip
                # (which would shrink the population invisibly).
                n = len(wrows)
                skip_counts[_why] = skip_counts.get(_why, 0) + n
                no_token_by_coin_day[(coin, wrows[0]["day"])] = \
                    no_token_by_coin_day.get((coin, wrows[0]["day"]), 0) + n
                continue
            g = gaps_for(slug, coin, t0)   # retained for the tape header only
            bn = bn_recv_for_window(coin, t0)
            # REFUSES rather than degrading -- the R-184 finding, in code
            PIN.assert_required_inputs(g, bn)
            up, dn = tokens[slug]
            # gaps are NOT handed to features_at: its window-relative
            # comparison is the second path R-213 eliminates. The builder owns
            # the single absolute comparison and applies the status below.
            tape = sf.build_tape(paths[slug], up, dn, gaps=(), bn_recv_ns=bn)
            feats = sf.features_for_window(tape, wrows)
            for r, fe in zip(wrows, feats):
                st = str(fe.get("state_status", "OK"))
                # ONE comparison, both paths. A gap is a FEED fact and outranks
                # where the row sits in its window, so it overrides whatever
                # the status chain produced -- including PRE_WINDOW, which is
                # exactly the warm-up population that carries the gaps.
                _T = float(r["t0"]) + float(r["t_start"])
                _hit = gap_contains(_T, coin)
                if _hit is not None:
                    st = "GAP_AT_CUTOFF"
                elif st == "GAP_AT_CUTOFF":
                    st = "OK"      # the old path flagged it; the ruled one does not
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
    # A zero-row build is NOT a result. The status path turned a loud KeyError
    # into a SILENT TOTAL EXCLUSION: 1,764,206 rows statused out, artifact
    # written, exit 0, header claiming "embargo CERTIFIED" over no data. An
    # exclusion status is for a MINORITY of rows; when it swallows the whole
    # population it is a broken input, not a population statement.
    if n_rows == 0:
        raise SystemExit(
            f"REFUSED: the build produced ZERO rows. Statuses: {status_counts}. "
            f"Every row was excluded, which is a broken input path, not a "
            f"result -- writing it would have produced an empty tape claiming "
            f"a certified embargo.")
    # R-202(2) ABSORPTION BOUND. A status absorbs ROW-LEVEL anomalies; it must
    # never absorb a total failure. Any single exclusion status above 1% of
    # input rows refuses the build, with its count named.
    # R-203(1): the bound applies ONLY to PRE-EMISSION SKIPS. BE's version
    # iterated every status, so PRE_WINDOW at 3.85% -- a legitimate population
    # statement about warm-up rows -- would have REFUSED the VALID population
    # at completion, after ~75 minutes of work. A row that is IN the tape and
    # labelled is not an absorbed failure; a row that never entered is.
    _total_in = n_rows + sum(skip_counts.values())
    for _st, _n in sorted(skip_counts.items()):
        if _total_in == 0:
            continue
        _frac = _n / _total_in
        if _frac > 0.01:
            raise SystemExit(
                f"REFUSED: PRE-EMISSION SKIP {_st} covers {_n:,} of "
                f"{_total_in:,} input rows ({_frac:.1%}), above the 1% "
                f"absorption bound. Skips absorb row-level anomalies, never "
                f"total failures. Skips: {skip_counts}; emitted statuses "
                f"(NOT bounded): {status_counts}")
    # embargo from the running extremes -- no second pass over the rows
    gap = sc_first_feat - tr_last_exit
    emb = {"gap_s": gap, "embargo_s": EMB.EMBARGO_S,
           "last_train_label_exit": tr_last_exit,
           "first_score_feature": sc_first_feat}
    emb_state = "CERTIFIED" if gap >= EMB.EMBARGO_S else (
        f"VIOLATED (unpurged): gap {gap:.3f}s < {EMB.EMBARGO_S}s")

    # RULED PROVENANCE INTERFACE (R-199 seam 20): the LAUNCHER passes
    # BUILD_REF; the builder READS it and writes it VERBATIM. No git at
    # runtime -- `git rev-parse HEAD` reported the MAIN tree's head AT
    # COMPLETION, which is not the ref the snapshot was cut from and is not
    # attributable to anything.
    import subprocess as _sp
    _ref = os.environ.get("BUILD_REF", "").strip()
    if len(_ref) != 40 or any(c not in "0123456789abcdef" for c in _ref.lower()):
        raise SystemExit(
            "REFUSED: BUILD_REF must be set to a 40-hex commit ref on the "
            "launch line. A build that determines its own provenance at "
            "runtime cannot be attributed to the code that produced it.")
    _built_at = _sp.run(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"],
                        capture_output=True, text=True).stdout.strip()
    out = {
        "protocol": "PHASE2_STATE_TAPE_V5",
        # R-196(2): a heavy build launches only from a COMMITTED ref, and the
        # ref travels ON the artifact. tape4 was launched 06:08:02Z from a
        # working tree mid-fix and produced GAP_AT_CUTOFF=286 -- an unknown
        # intermediate of six in-flight fixes, reconcilable to nothing.
        "builder_ref": _ref,               # verbatim from the launcher
        "ledger_path": _ledger or str(GC.LEDGER),
        "ledger_sha256": _ledger_sha,
        "ledger_pinned": bool(_ledger),
        "gap_predicate": "[g_start, g_end) absolute basis, full precision, "
                         "single comparison for main and warm-up paths (R-213)",
        "snapshot_path": _ROOT,
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
        "pre_emission_skip_counts": skip_counts,
        "skip_vs_status_note": ("skips are rows that never entered the tape "
                                "(input failures, bounded at 1%); statuses are "
                                "rows that ARE in the tape and labelled "
                                "(population statements, unbounded)"),
        "no_token_map_by_coin_day": {f"{c}|{d}": n
                                     for (c, d), n in sorted(
                                         no_token_by_coin_day.items())},
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
