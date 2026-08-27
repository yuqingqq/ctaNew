"""PHASE 2 — three heads on the top-up, per the committed declaration.

AUTHORISATION (R-126, in-file): R-170/R-173. Governed by `phase2_declaration`,
which was committed BEFORE any Phase-2 number existed (rule 11). Nothing here
chooses a threshold, window, feature set or hyperparameter -- if this file and
the declaration ever disagree, the declaration wins and this file is wrong.

THE ARCHITECTURE, and the reason it is not the obvious one:
  FIT on the CONSUMED FRAGMENT   (already spent; fitting costs nothing new)
  SCORE on the TOP-UP            (held out; R-166(3) reserved it)
Fitting on the top-up would consume the only held-out tape Phase 2 has and
make all three heads in-sample at once.

  A  PM_PLUS_FINE       the FROZEN artifact APPLIED UNCHANGED. No refit, no
                        weighting. R-157(2): the incumbent is not rewritten
                        mid-comparison, so arm A does not even load a fitter.
  B  PLUS_PRED_STATE_V1 A's features + DA's 21 state features, fitted on the
                        fragment WITH w = 1/n_rows(generation).
  C  LGBM_PINNED        same features and weighting, capacity pinned in the
                        declaration, seed pinned, no grid, no early stopping.

Every arm scores the SAME top-up rows. A row any arm cannot feature is dropped
from ALL arms, so the comparison stays paired -- an arm scoring a different
population is not a comparison at all.
"""
from __future__ import annotations

import json, sys
from pathlib import Path

# R-212(1): THE TAPE5 CLASS, IN BE'S OWN FIT STAGE. This was an absolute
# main-tree insert, so a fit run from a snapshot imported its pinned modules
# from the LIVE tree -- the identical defect BE fixed in the builder and never
# looked for here. The import root is THIS FILE'S directory, so a snapshot copy
# imports its own siblings by construction.
_ROOT = str(Path(__file__).resolve().parent)
sys.path.insert(0, _ROOT)
import phase2_declaration as D


PM_DATA_ROOT = Path("/home/yuqing/ctaNew")


def pin_data_root() -> None:
    """Point every data-deriving module at the REAL tree, then PROVE it loads.

    R-214 / fourth appearance of this class. `flow_intensity` computes
    `REPO = Path(__file__).resolve().parents[2]`, so running the FIT from a
    snapshot silently relocated its DATA root: fi.PM pointed inside the
    snapshot, DAYS/_archive_paths/token_map all returned 0, and every fragment
    row dropped as `no_archive` -- 1,125,289 of them, quietly.

    R-212 pinned this stage's IMPORTS and not its DATA. BE had written that
    code isolation and data location are separate concerns, then carried only
    the first across. Values COMPUTED AT IMPORT from a path (DAYS) are stale
    after rebinding it, so they are recomputed here."""
    import harmful_hazard_model as _hm
    fi = _hm.fi
    pm = PM_DATA_ROOT / "data/pm_5min"
    fi.REPO = PM_DATA_ROOT
    fi.PM = pm
    fi.RAW = pm / "raw"
    fi.GAPS = pm / "collector_gaps.jsonl"
    fi.MARKETS = pm / "markets.jsonl"
    for name in ("PM", "RAW", "GAPS", "MARKETS"):
        q = getattr(fi, name)
        if not q.exists():
            raise RuntimeError(f"REFUSED: pinned {name} = {q} does not exist.")
    if hasattr(fi, "_discover_days"):
        fi.DAYS = fi._discover_days()

    # SAME-INSTANCE PREFLIGHT (R-202): probe REAL slugs through the lookups
    # THE FIT PERFORMS, not merely check that maps are non-empty. Loading a
    # map is not the lookup; the builder learned that and this stage did not.
    tok, pth = fi.token_map(), fi._archive_paths()
    counts = {"token_map": len(tok), "archive_paths": len(pth),
              "DAYS": len(fi.DAYS)}
    empty = [k for k, v in counts.items() if v == 0]
    if empty:
        raise RuntimeError(
            f"REFUSED: fit inputs load EMPTY after pinning: {empty}. "
            f"Counts: {counts}. Paths that look right while the data does not "
            f"load is the state that dropped every row as no_archive.")
    probe = []
    for src in (FRAGMENT, TOPUP):
        if src.exists():
            d = json.loads(src.read_text())
            probe += sorted({r["slug"] for r in d["rows"][:20000]})[:12]
    probe = sorted(set(probe))
    if probe:
        hit = [x for x in probe if x in tok and x in pth]
        if len(hit) < len(probe):
            raise RuntimeError(
                f"REFUSED: fit row-path probe matched {len(hit)}/{len(probe)} "
                f"REAL population slugs. The maps load but the lookup the fit "
                f"performs does not resolve them.")
        counts["row_path_probe"] = f"{len(hit)}/{len(probe)}"
    print(f"  fit data root pinned; inputs {counts}", flush=True)


def assert_modules_under_root() -> None:
    """Every pinned module must have loaded from THIS tree.

    A wrong-tree import is silent: the fit runs, the numbers look normal, and
    the code that produced them is whatever the live tree happened to contain.
    Called at BOTH entry points, because either can be run from a snapshot."""
    import phase2_state_schema_freeze as _PIN
    import phase2_embargo as _EMB
    import harmful_action_eval as _AE
    for m in (D, _PIN, _EMB, _AE):
        f = str(Path(m.__file__).resolve())
        if not f.startswith(_ROOT):
            raise RuntimeError(
                f"REFUSED: {m.__name__} loaded from {f}, OUTSIDE this run root "
                f"{_ROOT}. A snapshot that imports another tree isolates "
                f"nothing -- the fit would run bytes nobody pinned.")

DERIVED = Path("/home/yuqing/ctaNew/data/pm_5min/derived")
TOPUP = DERIVED / "harmful_exposure_rows_v3_topup.json"
FRAGMENT = DERIVED / "harmful_exposure_rows_v3_eraB.json"
FROZEN = DERIVED / "harmful_reduced_fine_candidate_v1.json"
OUT = DERIVED / "phase2_three_arm_v1.json"


class PopulationLeak(RuntimeError):
    """A scoring row came from the fitting population."""


def assert_disjoint(fit_slugs: set, score_slugs: set) -> None:
    """The two populations must not intersect. AT ALL.

    This is the check that would have caught the 808-window build before any
    arm ran: the test set contained every fitting slug, and nothing errored."""
    overlap = fit_slugs & score_slugs
    if overlap:
        raise PopulationLeak(
            f"{len(overlap)} slug(s) appear in BOTH the fitting and scoring "
            f"populations (e.g. {sorted(overlap)[:3]}). Arms B and C are fitted "
            f"on the fragment; scoring them on rows they were fitted on is "
            f"in-sample and FLATTERS them. Refusing.")




# R-196(4): PINNED to v5 explicitly. No default-path fallthrough to v2 --
# those bytes are the quarantined tape4 diagnostic (GAP_AT_CUTOFF=286 from a
# moving tree), and a silent fallback would fit the rerun on them.
TAPE_PATH = DERIVED / "phase2_state_tape_v5.json"


def assert_tape_is_v5(path: Path = None) -> dict:
    """REFUSE any tape that is not the committed-ref v5 build."""
    p = path or TAPE_PATH
    if not p.exists():
        raise RuntimeError(f"{p.name} absent: the rerun may not fall back to "
                           f"the quarantined v2/tape4 bytes.")
    with p.open() as fh:
        head = fh.read(4000)
    i = head.index('"rows"')
    meta = json.loads(head[:i].rstrip().rstrip(",") + "}")
    if meta.get("protocol") != "PHASE2_STATE_TAPE_V5":
        raise RuntimeError(f"refusing tape with protocol "
                           f"{meta.get('protocol')!r}; expected V5")
    if meta.get("builder_tree_dirty_at_build"):
        raise RuntimeError("refusing a tape built from a DIRTY tree: tape4's "
                           "286 came from exactly that.")
    return meta


def load_tape_index(split: str) -> dict:
    """Stream the rebuilt tape and index ONE split's rows by identity.

    Streamed, not json.loads'd: the tape is 3.17 GB and R-174 forbids
    materializing it. Keyed by (slug, side, gen, t_start) so the scorer
    CONSUMES this tape by identity rather than re-deriving state without its
    required inputs."""
    import ijson  # noqa: F401  (optional fast path)
    raise RuntimeError("unused")


def _stream_tape_rows(path: Path):
    """Yield row dicts from the tape without holding the file in memory."""
    dec = json.JSONDecoder()
    buf = ""
    with path.open("r") as fh:
        head = fh.read(1 << 16)
        i = head.index('"rows"')
        i = head.index("[", i) + 1
        buf = head[i:]
        while True:
            buf = buf.lstrip()
            while buf.startswith(","):
                buf = buf[1:].lstrip()
            if buf.startswith("]"):
                return
            try:
                obj, end = dec.raw_decode(buf)
            except ValueError:
                chunk = fh.read(1 << 22)
                if not chunk:
                    return
                buf += chunk
                continue
            yield obj
            buf = buf[end:]


def tape_index(split: str, features_in_order=None) -> dict:
    """Index ONE split by identity, storing a COMPACT FLOAT TUPLE per row.

    R-194 seam 15: storing whole row dicts was ~12 GB for 1.7M rows -- the
    R-174 violation a third time. Only the 45 feature values are needed
    downstream, so the index holds a tuple of floats, not the row.
    Encoding happens HERE, from the NESTED state block (seam 11), so the
    caller cannot re-make the outer-row mistake."""
    import phase2_state_schema_freeze as _PIN
    feats = features_in_order or _PIN.build_pin()["features_in_order"]
    idx = {}
    for r in _stream_tape_rows(TAPE_PATH):
        if r.get("split") != split:
            continue
        if str(r.get("state_status", "OK")) != "OK":
            continue
        state = r.get("state") or {}          # NESTED, per features_under
        # The index value carries the ENCODED FEATURES plus the two clock
        # fields the embargo probe needs. Storing only the tuple made
        # stage_fit read r["t0"] on a tuple (TypeError at :333); reading them
        # from the KEY works but re-parses a slug, so they travel explicitly.
        idx[(r["slug"], r["side"], r["gen"], r["t_start"])] = {
            "vec": tuple(_PIN.encode_row(state, feats)),
            "t0": float(r["t0"]), "t_start": float(r["t_start"])}
    return idx

def freeze_thresholds(train_scores, budgets, gen_keys=None):
    """Returns a cutoff for EVERY declared budget -- the shape R-209(2)
    requires of its callers. A partial map is refused downstream, so it must
    never be produced here."""
    """CAUSAL thresholds: per-budget cutoffs resolved from TRAINING scores ONLY.

    R-184(4)(vi). The evaluator's top-k cutoff is knowable only after seeing the
    whole scoring population -- a valid offline RANKING curve, but not a policy
    anyone could have run on the day. These are frozen before scoring, so the
    cancel/keep decision at any scored row depends on nothing after it."""
    if not train_scores:
        raise ValueError("no training scores: a causal threshold cannot be "
                         "resolved from an empty training side")
    # ACTION UNIT. The evaluator ranks GENERATIONS by their max row score, so a
    # cutoff taken from the ROW distribution is not comparable to it and
    # selects the wrong count. When generation keys are supplied the quantile
    # is taken over per-generation MAXIMA, matching what theta is compared to.
    if gen_keys is not None:
        gmax: dict = {}
        for k, v in zip(gen_keys, train_scores):
            if k not in gmax or v > gmax[k]:
                gmax[k] = v
        xs = sorted(gmax.values(), reverse=True)
    else:
        xs = sorted(train_scores, reverse=True)
    out = {}
    for b in budgets:
        k = max(1, int(len(xs) * b))
        out[f"{int(b*100)}%"] = xs[k - 1]
    return out


def head_diagnostics(p_haz, y_haz, v_true, v_pred):
    """The five declared head diagnostics, per arm, reported SEPARATELY.

    R-184(4)(iv): a product ranking cannot show WHICH head improved. Hazard
    quality and conditional-value quality are different claims and a gain in
    one can mask a loss in the other."""
    import math
    n = len(y_haz)
    pos = [p for p, y in zip(p_haz, y_haz) if y]
    neg = [p for p, y in zip(p_haz, y_haz) if not y]
    if pos and neg:
        # RANK-BASED AUC, O(n log n). The pairwise form was O(n^2): at 639k
        # scoring rows that is ~4e11 comparisons and does not finish.
        order = sorted(range(n), key=lambda i: p_haz[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and p_haz[order[j + 1]] == p_haz[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0            # average rank for ties
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        rsum = sum(r for r, y in zip(ranks, y_haz) if y)
        npos, nneg = len(pos), len(neg)
        auc = (rsum - npos * (npos + 1) / 2.0) / (npos * nneg)
    else:
        auc = None
    # length equality asserted: zip() silently truncates, so a short p_haz
    # produced a confident Brier over a prefix rather than an error
    brier = (sum((p - y) ** 2 for p, y in zip(p_haz, y_haz)) / n
             if n and len(p_haz) == n else None)
    # harmful-sign discrimination: does the value head separate harmful from
    # favourable outcomes, independent of magnitude?
    # R-203(6): the conditional-value head is conditional ON A FILL. Scoring
    # it over rows with no fill measures it on a population it never claims to
    # describe. Condition on the hazard-positive rows.
    hs = [(vp, vt) for vp, vt, yy in zip(v_pred, v_true, y_haz)
          if yy and vt != 0.0]
    sign_acc = (sum(1 for vp, vt in hs if (vp > 0) == (vt > 0)) / len(hs)) if hs else None
    _cv = [(vp, vt) for vp, vt, yy in zip(v_pred, v_true, y_haz) if yy]
    mae = (sum(abs(a - b) for a, b in _cv) / len(_cv)) if _cv else None
    # calibration slope of predicted vs realized value
    if len(_cv) > 1:
        _vp = [a for a, _ in _cv]; _vt = [b for _, b in _cv]
        mx = sum(_vp) / len(_vp); my = sum(_vt) / len(_vt)
        num = sum((a - mx) * (b - my) for a, b in zip(_vp, _vt))
        den = sum((a - mx) ** 2 for a in _vp)
        slope = (num / den) if den > 0 else None
    else:
        slope = None
    return {"hazard_auc": auc, "hazard_brier": brier,
            "harmful_sign_discrimination": sign_acc,
            "conditional_value_mae": mae,
            "conditional_value_calibration_slope": slope}


PA_KIND = lambda a: arm_model_kind(a)


def arm_model_kind(arm: str) -> str:
    """The model class each arm MUST use. Asserted by identity in the fixture.

    Arm D exists to isolate the reweighting; if it silently falls through to
    the LGBM branch it duplicates arm C and both D-A and B-D become
    meaningless. Naming the mapping makes that assertable."""
    return {"PM_PLUS_FINE": "frozen_linear_applied",
            "PLUS_PRED_STATE_V1": "weighted_linear",
            "INCUMBENT_REWEIGHTED_ONLY": "weighted_linear",
            "LGBM_PINNED": "lgbm"}[arm]


def _feature_pass(src: Path, population: str, TAPE=None) -> dict:
    """Build PM+fine+state features for every OK row of one population.

    Returns per-coin parallel lists. A row missing ANY family is dropped from
    ALL arms (paired comparison); drops are counted, never silent."""
    import harmful_hazard_model as hm
    import harmful_state_features as sf
    import phase2_state_schema_freeze as PIN
    PIN_FEATURES = PIN.build_pin()["features_in_order"]

    data = json.loads(src.read_text())
    rows = [r for r in data["rows"] if r["status"] == "OK"]
    paths = hm.fi._archive_paths(); tokens = hm.fi.token_map()
    out: dict = {}
    for coin in ("btc", "eth"):
        crows = [r for r in rows if r["coin"] == coin]
        streams: dict = {}; tapes: dict = {}
        PM = []; FN = []; ST = []; kept = []
        drops = {"pm": 0, "fine": 0, "state": 0, "state_status": 0,
                 "no_archive": 0}
        bywin: dict = {}
        for r in crows:
            bywin.setdefault(r["slug"], []).append(r)
        for slug, wrows in bywin.items():
            if slug not in paths or slug not in tokens:
                drops["no_archive"] = drops.get("no_archive", 0) + len(wrows)
                continue          # a COUNTED status, never a KeyError
            up, dn = tokens[slug]
            streams[slug] = hm.window_streams(paths[slug], up, dn)
            # R-187 seam 2: CONSUME the rebuilt tape. The previous call was
            # `sf.build_tape(paths[slug], up, dn)` with NO gaps and NO
            # bn_recv_ns, which recreated inside the scorer the exact
            # missing-input defect the tape rebuild exists to remove: freshness
            # constant for every row, GAP_AT_CUTOFF unreachable. The tape is
            # built once, with its required inputs asserted, and read here.
            if TAPE is None:
                raise RuntimeError(
                    "no rebuilt state tape supplied. Re-deriving state here "
                    "would rebuild it WITHOUT gaps/bn_recv_ns and silently "
                    "restore the defect the rebuild removed.")
            sfeats = [TAPE[(r["slug"], r["side"], r["gen"], r["t_start"])]
                      for r in wrows if (r["slug"], r["side"], r["gen"],
                                         r["t_start"]) in TAPE]
            if len(sfeats) != len(wrows):
                drops["state"] += len(wrows) - len(sfeats)
                wrows = [r for r in wrows if (r["slug"], r["side"], r["gen"],
                                              r["t_start"]) in TAPE]
            for r, sfe in zip(wrows, sfeats):
                fp = hm.features(streams[slug], r["t_start"], r["side"],
                                 r["level"], r["resting"], r["qahead"])
                if fp is None:
                    drops["pm"] += 1; continue
                ff = hm.fine_feats(r["t0"] + r["t_start"], r["side"], coin)
                if ff is None:
                    drops["fine"] += 1; continue
                # R-185(1): NEVER `or 0.0`. That coerced None to 0.0 with no
                # guard, so a missing velocity was indistinguishable from a
                # level that did not move -- and `or` also swallows a
                # legitimate 0.0 and False. encode_row maps None->0.0 only
                # because the PAIRED GUARD FLAG carries the information, and
                # the pin refuses any nullable whose guard is absent.
                # non-OK statuses were already excluded when the index was
                # built; they are counted there, never silently dropped here
                # seam 11: the tape index already encoded from the NESTED
                # state block. Passing the outer row here scored every state
                # feature as 0.0, silently.
                sv = list(sfe["vec"])
                PM.append(fp); FN.append(ff); ST.append(sv)
                kept.append({k: r.get(k) for k in
                             ("slug", "day", "t0", "t_start", "side", "gen",
                              "latency", "coin")})
            streams.pop(slug, None)
        # R-214: an ABSORPTION BOUND on the FIT's drops. The builder's bound
        # covers its own skip_counts and never reached here, so a 100% input
        # failure read as a quiet all-drop instead of a refusal -- the exact
        # defect the bound exists to prevent, in the stage it did not cover.
        _in = len(kept) + sum(drops.values())
        for _k, _n in sorted(drops.items()):
            if _in and _n / _in > 0.01:
                raise RuntimeError(
                    f"REFUSED: fit drop `{_k}` covers {_n:,} of {_in:,} rows "
                    f"({_n/_in:.1%}) for {coin}, above the 1% absorption "
                    f"bound. Drops absorb row-level anomalies, never total "
                    f"input failures. All drops: {drops}")
        out[coin] = {"PM": PM, "FN": FN, "ST": ST, "kept": kept, "drops": drops}
        print(f"  [{population}/{coin}] kept {len(kept)} rows, drops {drops}",
              flush=True)
    return out


def _labels(kept: list):
    Lh = str(D.TARGET_LATENCY_MS)
    y = [1 if (r.get("latency") or {}).get(Lh, {}).get(
             "preventable_shares", 0.0) > 0 else 0 for r in kept]
    tgt = [(r.get("latency") or {}).get(Lh, {}).get(
               "preventable_value_cents", 0.0) for r in kept]
    return y, tgt


FITDIR = DERIVED / "phase2_fits"



DA_VERDICT = DERIVED / "da_tape_gate_verdict_v5.json"


def assert_gate_passed() -> dict:
    """Fitting REQUIRES DA's ALL-PASS gate verdict on the v5 tape.

    R-199 seam 21: BE built `assert_tape_is_v5` against exactly this class and
    never called it. A refusal with no call site is documentation. Both entry
    paths now call this, and an ABSENT verdict is a REFUSAL, not a pass --
    fitting before the gate has spoken is how a contaminated tape becomes a
    result."""
    import hashlib
    if not DA_VERDICT.exists():
        raise RuntimeError(
            f"REFUSED: {DA_VERDICT.name} absent. Fitting requires DA's gate "
            f"verdict on the v5 tape; absence is not permission.")
    v = json.loads(DA_VERDICT.read_text())
    # R-203(3) VERDICT CONTRACT. A bare {"verdict": "PASS"} is a string anyone
    # can write; it is not evidence that a gate ran on THIS tape. The consumer
    # identifies the artifact, RECOMPUTES the verdict from the predicate table
    # rather than trusting a summary field, and binds it to the tape by hash,
    # size and builder ref.
    if v.get("verdict") != "da_tape_gate_verdict_v1":
        raise RuntimeError(
            f"REFUSED: verdict artifact identifies as {v.get('verdict')!r}, "
            f"not the ruled contract 'da_tape_gate_verdict_v1'.")
    table = v.get("predicates") or v.get("predicate_table")
    if not table:
        raise RuntimeError("REFUSED: verdict carries no predicate table; a "
                           "summary field is not a result.")
    # THREE STATES, not two. A predicate marked `applicable: False` is
    # NOT-APPLICABLE -- neither pass nor fail -- and must be EXCLUDED from
    # all_pass. BE's first version flattened it to a boolean, so DA's real
    # emission (which carries embargo_respected as pass:False/applicable:False,
    # "ENFORCED-DOWNSTREAM") was REFUSED as failing. A consumer that cannot
    # represent not-applicable will reject correct verdicts forever.
    if isinstance(table, dict):
        applicable = [(k, bool(v)) for k, v in table.items()]
        skipped = []
    else:
        applicable = [(x.get("predicate"), bool(x.get("pass"))) for x in table
                      if x.get("applicable", True)]
        skipped = [x.get("predicate") for x in table
                   if not x.get("applicable", True)]
    if not applicable:
        raise RuntimeError(
            f"REFUSED: every predicate is marked not-applicable "
            f"({skipped}); a table with nothing evaluated is not a result.")

    # R-207: N/A-VACUITY. Excluding not-applicable predicates from all_pass was
    # correct; ALLOWING A LOAD-BEARING PREDICATE TO BE not-applicable was not.
    # DA demonstrated the bypass: a gate run with NO expectations marks
    # everything N/A, writes all_pass:true, and a consumer that only excludes
    # N/A accepts it. The absence of a check is not the passing of a check.
    # So each load-bearing predicate must be ASSERTED (present, not N/A) AND
    # passing. Only `embargo_respected` may legitimately be N/A -- it is
    # ENFORCED-DOWNSTREAM by the purge, which the receipt evidences separately.
    LOAD_BEARING = ("gap_count_matches_expected", "provenance_matches_expected",
                    "dataset_non_empty", "no_rows_skipped_by_builder",
                    "absorption_within_bound")
    NA_WHITELIST = ("embargo_respected",)
    _by_name = {}
    if isinstance(table, dict):
        _by_name = {k: {"pass": bool(v), "applicable": True}
                    for k, v in table.items()}
    else:
        for _x in table:
            _by_name[_x.get("predicate")] = {
                "pass": bool(_x.get("pass")),
                "applicable": bool(_x.get("applicable", True))}
    _missing = [k for k in LOAD_BEARING if k not in _by_name]
    _na = [k for k in LOAD_BEARING if _by_name.get(k, {}).get("applicable") is False]
    _failed_lb = [k for k in LOAD_BEARING
                  if k in _by_name and _by_name[k]["applicable"]
                  and not _by_name[k]["pass"]]
    if _missing or _na or _failed_lb:
        raise RuntimeError(
            f"REFUSED: load-bearing predicates must be ASSERTED and PASSING. "
            f"missing={_missing} not_applicable={_na} failing={_failed_lb}. "
            f"A gate that checked nothing is not a gate that passed: the "
            f"absence of a check is not the passing of a check.")
    _bad_na = [k for k in skipped if k not in NA_WHITELIST]
    if _bad_na:
        raise RuntimeError(
            f"REFUSED: predicates marked not-applicable outside the whitelist "
            f"{list(NA_WHITELIST)}: {_bad_na}. N/A is a claim about a "
            f"predicate that CANNOT apply here, not a way to skip one.")
    failed = [k for k, ok in applicable if not ok]
    if failed:
        raise RuntimeError(
            f"REFUSED: recomputed all_pass is FALSE. Failing predicates: "
            f"{failed} (of {len(applicable)} applicable; not-applicable and "
            f"excluded: {skipped}). The summary field is not consulted.")
    # TWO SEPARATE QUESTIONS, and conflating them was a defect:
    #   (i)  is this verdict internally valid and BOUND TO ITS OWN SUBJECT?
    #   (ii) is that subject the tape THIS FIT is about to consume?
    # (i) is the contract; (ii) is a caller concern. Checking (ii) inside the
    # contract made the consumer refuse a perfectly valid verdict whenever the
    # module constant pointed elsewhere.
    tp = Path(v.get("tape_path", ""))
    if not tp.exists():
        raise RuntimeError(
            f"REFUSED: the verdict's subject {tp} does not exist; a verdict "
            f"about a missing artifact certifies nothing.")
    subject = tp
    nbytes = subject.stat().st_size
    if v.get("tape_bytes") != nbytes:
        raise RuntimeError(
            f"REFUSED: verdict says {v.get('tape_bytes')} bytes, tape is "
            f"{nbytes} -- the verdict is about a different artifact.")
    h = hashlib.sha256()
    with subject.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    digest = h.hexdigest()
    pref = str(v.get("tape_sha256_prefix", ""))
    if not pref or not digest.startswith(pref):
        raise RuntimeError(
            f"REFUSED: tape sha256 {digest[:16]} does not match the verdict's "
            f"prefix {pref!r}.")
    with subject.open() as fh:
        head = fh.read(4000)
    try:
        meta = json.loads(head[:head.index('"rows"')].rstrip().rstrip(",") + "}")
    except (ValueError, json.JSONDecodeError):
        meta = {}
    # DA's writer carries the tape's header pins under `tape_header_pins`,
    # not at top level. Reading only the top level made the consumer refuse
    # DA's REAL emission over a field that was present all along, one key
    # deeper -- a contract disagreement about LOCATION, not about content,
    # which is the same shape as the features_under/nesting seam.
    pins = v.get("tape_header_pins") or {}
    vref = v.get("builder_ref") or pins.get("builder_ref")
    tref = meta.get("builder_ref")
    if vref is not None and tref is not None and vref != tref:
        raise RuntimeError(
            f"REFUSED: verdict builder_ref {vref!r} != tape builder_ref "
            f"{tref!r} -- the verdict is about a different build.")
    # ABSENCE of builder_ref in the verdict is ACCEPTED, and the reason is
    # that the binding is already complete without it: tape_bytes and
    # tape_sha256_prefix were checked against THIS file, so the tape is
    # byte-identical to the one the gate ran on -- and those bytes CONTAIN the
    # builder_ref. Hash-binding subsumes ref-binding. Refusing on absence
    # rejected DA's own real emission (whose `tape_header_pins` is empty in the
    # contract test) over a field the hash had already pinned.
    # NOTE FOR THE REGISTER: R-203(3) lists builder_ref among the fields to
    # validate, while DA's acceptance test emits a verdict without one and
    # requires acceptance. BE resolves toward the test -- a MISMATCH still
    # refuses -- and flags the discrepancy rather than choosing silently.
    return v

def assert_verdict_subject_is(tape: Path, v: dict) -> None:
    """(ii): the verdict's subject must be the tape THIS fit will consume.

    Separate from the contract check by design -- a verdict can be perfectly
    valid about another artifact, and consuming it here would still be wrong."""
    sub = Path(v.get("tape_path", "")).resolve()
    if sub != tape.resolve():
        raise RuntimeError(
            f"REFUSED: the verdict certifies {sub}, but this fit consumes "
            f"{tape.resolve()}. A valid verdict about a different tape is "
            f"still the wrong verdict.")


FIT_MANIFEST = "fit_manifest.json"


def _tape_identity() -> dict:
    """Everything this fit is bound to: tape, verdict, gate code, fit code."""
    import hashlib, os
    h = hashlib.sha256()
    with TAPE_PATH.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    v = json.loads(DA_VERDICT.read_text()) if DA_VERDICT.exists() else {}
    # R-212(3): bind to the VERDICT ITSELF (path + content hash) and to the
    # CODE on both sides. A manifest that pins only the tape cannot tell that
    # the verdict was swapped, or that a different gate or a different fit
    # produced it -- and each of those changes what the numbers mean.
    vh = (hashlib.sha256(DA_VERDICT.read_bytes()).hexdigest()[:16]
          if DA_VERDICT.exists() else None)
    gate_src = Path(_ROOT) / "da_state_tape_verify.py"
    gate_id = (hashlib.sha256(gate_src.read_bytes()).hexdigest()[:16]
               if gate_src.exists() else None)
    fit_ref = os.environ.get("FIT_CODE_REF", "").strip() or None
    return {"tape_path": str(TAPE_PATH), "tape_sha256_prefix": h.hexdigest()[:16],
            "tape_bytes": TAPE_PATH.stat().st_size,
            "verdict_kind": v.get("verdict"),
            "verdict_tape_sha256_prefix": v.get("tape_sha256_prefix"),
            "verdict_path": str(DA_VERDICT),
            "verdict_sha256_prefix": vh,
            "gate_code_sha256_prefix": gate_id,
            "fit_code_ref": fit_ref}


def assert_fit_complete_and_matching() -> dict:
    """stage_score REQUIRES a completed fit bound to THE TAPE IT SCORES.

    R-209(3): fits wrote in place into a SHARED directory with no completion
    marker, so a killed partial fit left stale per-arm artifacts that
    stage_score consumed as if whole -- the in-place-overwrite class, fifth
    appearance. A run directory plus an atomic promote makes 'partially
    written' unrepresentable; the manifest makes 'written by a different run,
    against a different tape' detectable."""
    mf = FITDIR / FIT_MANIFEST
    if not mf.exists():
        raise RuntimeError(
            f"REFUSED: no {FIT_MANIFEST} in {FITDIR}. A fit without a "
            f"completion marker may be a killed partial; its artifacts look "
            f"identical to a finished run's.")
    m = json.loads(mf.read_text())
    if not m.get("complete"):
        raise RuntimeError(f"REFUSED: {FIT_MANIFEST} is not marked complete.")
    now = _tape_identity()
    # RECHECK EVERY BINDING, not just the tape. Each of these changes what the
    # scored numbers mean, and each is invisible in the numbers themselves.
    for k, why in (
            ("tape_sha256_prefix", "the fit was produced against a different tape"),
            ("tape_bytes", "the tape changed size since the fit"),
            ("verdict_path", "a different verdict artifact is in place"),
            ("verdict_sha256_prefix", "the verdict CONTENT changed since the fit"),
            ("gate_code_sha256_prefix", "a different GATE produced the verdict"),
            ("fit_code_ref", "a different FIT CODE REF produced these artifacts")):
        if m.get(k) != now.get(k):
            raise RuntimeError(
                f"REFUSED: {why} ({k}: fit={m.get(k)!r} now={now.get(k)!r}). "
                f"Scoring under bindings that differ from the fit's is not a "
                f"comparison.")
    import hashlib
    for name, want in (m.get("file_hashes") or {}).items():
        f = FITDIR / name
        if not f.exists():
            raise RuntimeError(f"REFUSED: manifest lists {name}, which is absent.")
        got = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
        if got != want:
            raise RuntimeError(
                f"REFUSED: {name} hash {got} != manifest {want}; the fit "
                f"directory changed after the run completed.")
    return m


def stage_fit() -> None:
    """STAGE 1: fit B and C on the fragment, persist, exit.

    The single-process version held fragment features + top-up features + LGBM
    training matrices simultaneously and was oom-killed at 14G AFTER all four
    feature passes had succeeded. Splitting is not only a memory fix: it is the
    daily-refit shape -- fit once, persist, apply -- so the scoring stage never
    needs the fitting population in memory at all."""
    import harmful_fast_compute as fc
    import lightgbm as lgb
    import numpy as np
    import phase2_embargo as EMB
    assert_modules_under_root()
    pin_data_root()
    assert_tape_is_v5()                              # committed-ref v5 build
    _v = assert_gate_passed()                        # (i) contract valid
    assert_verdict_subject_is(TAPE_PATH, _v)         # (ii) about THIS tape
    # write into a per-run directory; promote only on completion
    import shutil as _sh, os as _os0, time as _tm0
    # R-212(4): a UNIQUE run dir. `rmtree` on a fixed `.run` path would delete
    # a CONCURRENT run's working directory -- destroying another process's
    # in-flight state to make room for this one.
    _run = FITDIR.parent / f"{FITDIR.name}.run-{int(_tm0.time())}-{_os0.getpid()}"
    _run.mkdir(parents=True, exist_ok=False)
    _lock = FITDIR.parent / f"{FITDIR.name}.lock"
    if _lock.exists():
        try:
            _owner = int(_lock.read_text().strip())
            _alive = Path(f"/proc/{_owner}").exists()
        except (ValueError, OSError):
            _owner, _alive = None, False
        if _alive:
            raise RuntimeError(
                f"REFUSED: fit lock held by LIVE pid {_owner}. Two fits "
                f"writing one directory is how a partial run becomes another "
                f"run's input.")
        print(f"  stale fit lock from dead pid {_owner}; reclaiming", flush=True)
    _lock.write_text(str(_os0.getpid()))
    _final = FITDIR
    globals()["FITDIR"] = _run
    print("  indexing rebuilt tape (train split)...", flush=True)
    TP = tape_index("train")
    print(f"  tape rows indexed: {len(TP):,}", flush=True)
    FIT = _feature_pass(FRAGMENT, "fragment", TAPE=TP)
    # APPLY the purge -- recording a violation is not applying it (R-187 seam 3)
    print("  indexing score split for the embargo boundary...", flush=True)
    SP = tape_index("score")
    score_probe = [{"t0": v["t0"], "t_start": v["t_start"]} for v in SP.values()]
    empty_coins = []
    for coin in list(FIT):
        # A coin with ZERO kept rows is a NAMED outcome, not a crash. It
        # previously fell through to assert_embargo and raised
        # "empty side: embargo is undefined" -- a message about the embargo for
        # a condition that has nothing to do with it. Surfaced by the
        # production seam, which ran a population where one coin was absent.
        if not FIT[coin]["kept"]:
            # VACUOUS, not an error (rules 4 + 8). An empty side is a
            # POPULATION STATEMENT: there is nothing to purge, and that fact is
            # carried as an explicit per-side status rather than deleted. BE's
            # first fix dropped the coin entirely, which loses the statement;
            # enforcement of non-emptiness belongs to the GATE
            # (dataset_non_empty), not to the purge.
            empty_coins.append(coin)
            FIT[coin]["embargo_evidence"] = {
                "train_rows_before_purge": 0, "train_rows_after_purge": 0,
                "train_rows_dropped": 0, "purge_status": "VACUOUS_N_0",
                "purge_applicable": False,
                "note": "n=0 on the training side: nothing to purge. This is a "
                        "population statement, not a satisfied embargo and not "
                        "an error. Non-emptiness is the gate's predicate.",
                "EMBARGO_ENFORCED": None}
            print(f"  [purge/{coin}] n=0, purge N/A (vacuous) -- carried as a "
                  f"status, not an error", flush=True)
            continue
        before = len(FIT[coin]["kept"])
        keep_idx = set()
        kept, dropped = EMB.purge_training(FIT[coin]["kept"], score_probe)
        keys = {(r["slug"], r["side"], r["gen"], r["t_start"]) for r in kept}
        for n, r in enumerate(FIT[coin]["kept"]):
            if (r["slug"], r["side"], r["gen"], r["t_start"]) in keys:
                keep_idx.add(n)
        for fam in ("PM", "FN", "ST"):
            FIT[coin][fam] = [v for n, v in enumerate(FIT[coin][fam]) if n in keep_idx]
        FIT[coin]["kept"] = [v for n, v in enumerate(FIT[coin]["kept"]) if n in keep_idx]
        FIT[coin]["purged_rows"] = before - len(FIT[coin]["kept"])
        print(f"  [purge/{coin}] {before:,} -> {len(FIT[coin]['kept']):,} "
              f"({FIT[coin]['purged_rows']:,} rows dropped by the 60s embargo)",
              flush=True)
        # R-189: the enforcement must be VISIBLE AS NUMBERS, not as the
        # fixture's word. Both sides of the seam and the realized gap are
        # recorded, and `gap >= 60` is evaluated rather than asserted in prose.
        _gap = EMB.assert_embargo(FIT[coin]["kept"], score_probe)
        FIT[coin]["embargo_evidence"] = {
            "train_rows_before_purge": before,
            "train_rows_after_purge": len(FIT[coin]["kept"]),
            "train_rows_dropped": FIT[coin]["purged_rows"],
            "score_rows_untouched": len(score_probe),
            "score_side_trimmed": False,
            "realized_gap_s": _gap["gap_s"],
            "required_embargo_s": _gap["embargo_s"],
            "last_train_label_exit": _gap["last_train_label_exit"],
            "first_score_feature": _gap["first_score_feature"],
            "EMBARGO_ENFORCED": _gap["gap_s"] >= _gap["embargo_s"],
            "pre_purge_gap_s": -8.134101152420044,
            "note": "the tape header records VIOLATED-unpurged by design; "
                    "enforcement belongs to the run path and is shown here as "
                    "numbers on both sides of the seam (R-189).",
        }
    del SP, score_probe
    for coin in list(FIT):
        f = FIT[coin]
        if not f["kept"]:
            (FITDIR / f"linear_{coin}.json").write_text(json.dumps(
                {"n_rows": 0, "purge_status": "VACUOUS_N_0",
                 "embargo_evidence": f.get("embargo_evidence"),
                 "fitted": False}))
            continue
        yF, tF = _labels(f["kept"])
        XF = [f["PM"][i] + f["FN"][i] + f["ST"][i] for i in range(len(f["kept"]))]
        Xf, mu, sd = fc.fast_zscale(XF, XF)
        sw = fc.fast_generation_weights(f["kept"])
        _gk = [(r["slug"], r["side"], r["gen"]) for r in f["kept"]]
        # arm A applied UNWEIGHTED on PM+fine, scored on the training side to
        # obtain ITS OWN cutoff (the frozen model is not refitted)
        _fz = json.loads(FROZEN.read_text())["fits"][coin]
        _mA, _sA = _fz["norm_mu"], _fz["norm_sd"]
        WA, WMA = _fz["hazard_weights"], _fz["value_weights"]
        _w = len(f["PM"][0]) + len(f["FN"][0])
        if _w != len(_mA):
            raise RuntimeError(
                f"REFUSED: the frozen candidate expects {len(_mA)} PM+fine "
                f"features but this pipeline produces {_w}. A width mismatch "
                f"between the frozen artifact and the live feature builder "
                f"would otherwise surface as an IndexError deep in the fit, or "
                f"silently truncate if the frozen side were the shorter one.")
        XfA = [[1.0] + [((f["PM"][i] + f["FN"][i])[k] - _mA[k]) / _sA[k]
                        for k in range(len(_mA))]
               for i in range(len(f["kept"]))]
        W = fc.fast_fit_logistic_w(Xf, yF, sw)
        ft = [i for i in range(len(yF)) if yF[i]]
        WM = (fc.fast_fit_ridge_w([Xf[i] for i in ft], [tF[i] for i in ft],
                                  [sw[i] for i in ft], lam=10.0)
              if len(ft) >= 100 else None)
        # ARM D: incumbent features ONLY (PM+fine, no state), WITH the R-157
        # weighting. This is a SEPARATE FIT persisted to a SEPARATE ARTIFACT.
        # Sharing B's branch made D load B's weights and predict identically,
        # so D-A and B-D were both meaningless (R-194 seam 12).
        XD = [f["PM"][i] + f["FN"][i] for i in range(len(f["kept"]))]
        Xd, mud, sdd = fc.fast_zscale(XD, XD)
        Wd = fc.fast_fit_logistic_w(Xd, yF, sw)
        WMd = (fc.fast_fit_ridge_w([Xd[i] for i in ft], [tF[i] for i in ft],
                                   [sw[i] for i in ft], lam=10.0)
               if len(ft) >= 100 else None)
        (FITDIR / f"linear_d_{coin}.json").write_text(json.dumps(
            {"hazard_weights": list(Wd),
             "value_weights": list(WMd) if WMd else None,
             "norm_mu": list(mud), "norm_sd": list(sdd),
             "arm": "INCUMBENT_REWEIGHTED_ONLY",
             "features": "PM+fine only, NO state features",
             "causal_thresholds": freeze_thresholds(
                 [fc.fast_predict_p(Wd, x) *
                  (float(sum(a * b for a, b in zip(WMd, x))) if WMd else 0.0)
                  for x in Xd], D.BUDGETS, gen_keys=_gk)}))
        del XD, Xd
        (FITDIR / f"linear_{coin}.json").write_text(json.dumps(
            {"hazard_weights": list(W), "value_weights": list(WM) if WM else None,
             "norm_mu": list(mu), "norm_sd": list(sd),
             "n_rows": len(f["kept"]), "n_positive": sum(yF),
             "n_actions": len({(r["slug"], r["side"], r["gen"]) for r in f["kept"]}),
             "drops": f["drops"],
             "purged_rows_embargo": f.get("purged_rows", 0),
             "embargo_evidence": f.get("embargo_evidence"),
             "causal_thresholds": freeze_thresholds(
                 [fc.fast_predict_p(W, x) *
                  (float(sum(a * b for a, b in zip(WM, x))) if WM else 0.0)
                  for x in Xf], D.BUDGETS, gen_keys=_gk),
             # ARM A and ARM C need their own training-side cutoffs too: a
             # frozen threshold is per-MODEL, and reusing B's would compare
             # each arm against another arm's score distribution.
             "causal_thresholds_armA": freeze_thresholds(
                 [fc.fast_predict_p(WA, x) *
                  (float(sum(a * b for a, b in zip(WMA, x))) if WMA else 0.0)
                  for x in XfA], D.BUDGETS, gen_keys=_gk)}))
        A = np.asarray(Xf, dtype=np.float64); swa = np.asarray(sw)
        clf = lgb.LGBMClassifier(**D.LGBM_PARAMS)
        clf.fit(A, np.asarray(yF), sample_weight=swa)
        clf.booster_.save_model(str(FITDIR / f"lgbm_haz_{coin}.txt"))
        # arm C's OWN training thresholds, resolved AFTER its model exists
        _pc = clf.predict_proba(A)[:, 1]
        ftm = np.asarray(yF) == 1
        if ftm.sum() >= 100:
            reg = lgb.LGBMRegressor(**D.LGBM_VALUE_PARAMS)
            reg.fit(A[ftm], np.asarray(tF)[ftm], sample_weight=swa[ftm])
            reg.booster_.save_model(str(FITDIR / f"lgbm_val_{coin}.txt"))
            _vc = reg.predict(A)
        else:
            _vc = np.zeros(len(A))
        (FITDIR / f"lgbm_thresholds_{coin}.json").write_text(json.dumps(
            freeze_thresholds((_pc * _vc).tolist(), D.BUDGETS, gen_keys=_gk)))
        print(f"  [fit/{coin}] persisted linear + lgbm; rows {len(f['kept'])}, "
              f"positive {sum(yF)}", flush=True)
        del XF, Xf, A, FIT[coin]["PM"], FIT[coin]["FN"], FIT[coin]["ST"]
    (FITDIR / "empty_coins.json").write_text(json.dumps(empty_coins))
    _ident = _tape_identity()
    if not FIT:
        raise RuntimeError(
            "REFUSED: every coin came back empty. A fit over no population is "
            "not a null result, it is a broken input path.")
    (FITDIR / "fit_slugs.json").write_text(json.dumps(sorted(
        {r["slug"] for c in FIT.values() for r in c["kept"]})))
    # COMPLETION MANIFEST then ATOMIC PROMOTE. Written last, so its presence
    # is the completion signal; promoted by rename, so a consumer never sees a
    # half-populated directory.
    import hashlib as _hh, shutil as _sh2, os as _os2
    _hashes = {f.name: _hh.sha256(f.read_bytes()).hexdigest()[:16]
               for f in sorted(_run.iterdir()) if f.is_file()}
    _mani = dict(_ident)
    _mani.update({"complete": True, "file_hashes": _hashes,
                  "run_finished_utc": __import__("subprocess").run(
                      ["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"],
                      capture_output=True, text=True).stdout.strip(),
                  "arms": list(D.ARMS), "budgets": list(D.BUDGETS)})
    (_run / FIT_MANIFEST).write_text(json.dumps(_mani, indent=1, sort_keys=True))
    if _final.exists():
        _sh2.rmtree(_final.with_suffix(".prev"), ignore_errors=True)
        _final.rename(_final.with_suffix(".prev"))
    _os2.replace(str(_run), str(_final))
    globals()["FITDIR"] = _final
    print(f"STAGE FIT COMPLETE -- promoted {len(_hashes)} artifacts", flush=True)


def stage_score() -> dict:
    """STAGE 2: score all three arms on the top-up. Never loads the fragment."""
    import harmful_hazard_model as hm
    import harmful_action_eval as ae
    import harmful_fast_compute as fc
    import lightgbm as lgb
    import numpy as np

    assert_modules_under_root()
    pin_data_root()
    assert_tape_is_v5()
    _v = assert_gate_passed()
    assert_verdict_subject_is(TAPE_PATH, _v)
    # DA relay: a verdict issued from a DIRTY checker tree is reproducible from
    # no ref at all -- binding to `head` would bind to a ref that never ran.
    # The manifest already binds the checker by FILE SHA256; this additionally
    # refuses a verdict whose own gate_code declares itself dirty. If we ever
    # legitimately need one, that is a ruling, not a default.
    _gc = (_v.get("gate_code") or {}) if isinstance(_v, dict) else {}
    if _gc.get("dirty") is True:
        raise RuntimeError(
            f"REFUSED: the verdict was produced by a DIRTY checker tree "
            f"(gate_code.dirty=true, sha256={_gc.get('sha256', '?')!r:.18}). "
            f"A verdict from uncommitted checker bytes is attributable to no "
            f"ref; accepting it would bind this fit to code that never existed "
            f"as a commit.")
    _fm = assert_fit_complete_and_matching()   # a partial fit cannot be scored
    print(f"  fit manifest OK: {len(_fm.get('file_hashes') or {})} artifacts, "
          f"tape {_fm.get('tape_sha256_prefix')}", flush=True)
    frozen = json.loads(FROZEN.read_text())
    fit_slugs = set(json.loads((FITDIR / "fit_slugs.json").read_text()))
    print("  indexing rebuilt tape (score split)...", flush=True)
    TP = tape_index("score")
    print(f"  tape rows indexed: {len(TP):,}", flush=True)
    SC = _feature_pass(TOPUP, "topup", TAPE=TP)
    assert_disjoint(fit_slugs, {r["slug"] for c in SC.values() for r in c["kept"]})
    print("  populations asserted DISJOINT (fitted slugs read from stage 1)",
          flush=True)

    out = {"protocol": "PHASE2_FOUR_ARM_V2",
           "supersedes_label": "PHASE2_THREE_ARM_V1 (stale: four arms since arm D)", "arms": {}, "population": {},
           "declaration_commit": "d7082b6",
           # rule 10: COMPUTED from the declaration, never written as a
           # literal beside it. A hardcoded count has contradicted its own
           # declaration before.
           "multiplicity_before": D.MULTIPLICITY_BEFORE,
           "multiplicity_after": D.MULTIPLICITY_BEFORE + len(D.WEIGHTED_ARMS_V2),
           "multiplicity_scored_arms": list(D.WEIGHTED_ARMS_V2),
           "n_random": D.N_RANDOM, "decision_metric": D.DECISION_METRIC,
           "lgbm_params": D.LGBM_PARAMS,
           "staged_because": "the single-process run was oom-killed at 14G after "
                             "all four feature passes succeeded; fit and score "
                             "are now separate processes (the daily-refit shape)",
           "da_caveat_field": "RESERVED for Q-DA-79 post-gap queue-validity finding"}

    _empty = json.loads((FITDIR / "empty_coins.json").read_text()) \
        if (FITDIR / "empty_coins.json").exists() else []
    for coin in ("btc", "eth"):
        if coin in _empty or not (FITDIR / f"linear_{coin}.json").exists():
            print(f"  {coin}: not fitted (absent from the population); skipped",
                  flush=True)
            continue
        sc = SC[coin]
        lin = json.loads((FITDIR / f"linear_{coin}.json").read_text())
        srows = [hm.keptrow(r) for r in sc["kept"]]
        nA = len({(r["slug"], r["side"], r["gen"]) for r in sc["kept"]})
        out["population"][coin] = {
            "score_rows": len(sc["kept"]), "score_actions": nA,
            "score_windows": len({r["slug"] for r in sc["kept"]}),
            "score_drops": sc["drops"], "fit_rows": lin["n_rows"],
            "embargo_evidence": lin.get("embargo_evidence"),
            "fit_actions": lin["n_actions"], "fit_positive": lin["n_positive"],
            "fit_drops": lin["drops"]}
        # R-174: NO duplicate materialization. The previous version built
        # XS_lin (a full concatenated copy of 638k x 81) AND a float64 matrix
        # for LGBM, on top of the feature-pass lists already resident -- three
        # copies of the same data, which is the shape that oom-killed the
        # single-process run. Rows are now concatenated ON DEMAND inside each
        # arm, and LGBM is fed in CHUNKS so no full matrix ever exists.
        n_sc = len(sc["kept"])

        def _raw(i):
            return sc["PM"][i] + sc["FN"][i] + sc["ST"][i]

        for arm in D.ARMS:
            # BOUND PER ARM, before any branch. `p_head if p_head else ...`
            # guarded TRUTHINESS, not BINDING: on arm A the names were never
            # assigned and the guard itself raised NameError (:484/:550).
            p_head: list = []; v_head: list = []; thr = None
            if arm == "PM_PLUS_FINE":
                fz = frozen["fits"][coin]
                mu, sd = fz["norm_mu"], fz["norm_sd"]
                W, WM = fz["hazard_weights"], fz["value_weights"]
                ecv = []
                for j in range(len(sc["kept"])):
                    raw = sc["PM"][j] + sc["FN"][j]
                    x = [1.0] + [(raw[i] - mu[i]) / sd[i] for i in range(len(mu))]
                    ph = fc.fast_predict_p(W, x)
                    vh = float(sum(a * b for a, b in zip(WM, x)))
                    p_head.append(ph); v_head.append(vh); ecv.append(ph * vh)
                thr = (fz.get("causal_thresholds")
                       or lin.get("causal_thresholds_armA"))
            elif arm == "INCUMBENT_REWEIGHTED_ONLY":
                lind = json.loads((FITDIR / f"linear_d_{coin}.json").read_text())
                mu, sd = lind["norm_mu"], lind["norm_sd"]
                W, WM = lind["hazard_weights"], lind["value_weights"]
                ecv = []; p_head = []; v_head = []
                for j in range(n_sc):
                    raw = sc["PM"][j] + sc["FN"][j]      # NO state features
                    x = [1.0] + [(raw[i] - mu[i]) / sd[i] for i in range(len(mu))]
                    ph = fc.fast_predict_p(W, x)
                    vh = float(sum(a * b for a, b in zip(WM, x))) if WM else 0.0
                    p_head.append(ph); v_head.append(vh); ecv.append(ph * vh)
                thr = lind.get("causal_thresholds")
            elif arm == "PLUS_PRED_STATE_V1":
                # R-205: NAMES ONLY -- this comment previously labelled the
                # PLUS_PRED_STATE_V1 branch with the wrong letter and taught a
                # swapped convention. PLUS_PRED_STATE_V1 is identity-dispatched
                # to the weighted linear; it shares that model class with
                # INCUMBENT_REWEIGHTED_ONLY, which is what makes
                # (PLUS_PRED_STATE_V1 - INCUMBENT_REWEIGHTED_ONLY) isolate the
                # STATE FEATURES. Falling through to LGBM would duplicate
                # LGBM_PINNED.
                assert PA_KIND(arm) == "weighted_linear", arm
                mu, sd = lin["norm_mu"], lin["norm_sd"]
                W, WM = lin["hazard_weights"], lin["value_weights"]
                ecv = []
                for j in range(n_sc):
                    raw = _raw(j)                       # transient, freed each loop
                    x = [1.0] + [(raw[i] - mu[i]) / sd[i] for i in range(len(mu))]
                    ph = fc.fast_predict_p(W, x)
                    vh = float(sum(a * b for a, b in zip(WM, x))) if WM else 0.0
                    p_head.append(ph); v_head.append(vh); ecv.append(ph * vh)
                thr = lin.get("causal_thresholds")
            else:
                mu, sd = lin["norm_mu"], lin["norm_sd"]
                hb = lgb.Booster(model_file=str(FITDIR / f"lgbm_haz_{coin}.txt"))
                vf = FITDIR / f"lgbm_val_{coin}.txt"
                vb = lgb.Booster(model_file=str(vf)) if vf.exists() else None
                CH = 50_000                             # ~32MB per chunk, not ~2GB
                ecv = []
                for lo in range(0, n_sc, CH):
                    hi = min(lo + CH, n_sc)
                    S = np.empty((hi - lo, len(mu) + 1), dtype=np.float64)
                    S[:, 0] = 1.0
                    for j in range(lo, hi):
                        raw = _raw(j)
                        S[j - lo, 1:] = [(raw[i] - mu[i]) / sd[i]
                                         for i in range(len(mu))]
                    p = hb.predict(S)
                    v = vb.predict(S) if vb is not None else np.zeros(hi - lo)
                    ecv.extend((p * v).tolist())
                    # arm C's heads: the LGBM hazard probability and value
                    # prediction. Leaving them empty made head_diagnostics
                    # divide by len(v_pred)==0 -- the empty-heads class again,
                    # this time on the one arm whose branch does not build them
                    # row by row.
                    p_head.extend(p.tolist()); v_head.extend(v.tolist())
                    del S
                # LGBM_PINNED loads ITS OWN frozen thresholds, persisted by
                # stage_fit after its model existed. Without this the arm ran
                # RETROSPECTIVE_TOPK while the other three ran causal -- and
                # every per-arm number still looked entirely normal.
                _tf = FITDIR / f"lgbm_thresholds_{coin}.json"
                thr = json.loads(_tf.read_text()) if _tf.exists() else None
                if thr is None:
                    raise RuntimeError(
                        f"REFUSED: LGBM_PINNED has no frozen thresholds for "
                        f"{coin}; running it retrospectively while the other "
                        f"arms are causal is not a paired comparison.")
            gate = ae.evaluate_policy(srows, ecv, latency_ms=D.TARGET_LATENCY_MS,
                                      budgets=D.BUDGETS, n_random=D.N_RANDOM,
                                      theta_frozen=thr)
            Lh = str(D.TARGET_LATENCY_MS)
            y_sc = [1 if (r.get("latency") or {}).get(Lh, {}).get(
                        "preventable_shares", 0.0) > 0 else 0 for r in sc["kept"]]
            v_sc = [(r.get("latency") or {}).get(Lh, {}).get(
                        "preventable_value_cents", 0.0) for r in sc["kept"]]
            # REAL head outputs, retained separately (R-194). Passing
            # min(1,|ecv|) as a probability and ecv as the value head measured
            # neither head: |product| is not a hazard probability and the
            # product is not a conditional value.
            heads = head_diagnostics(p_head, y_sc, v_sc, v_head)
            out["arms"].setdefault(coin, {})[arm] = {
                "gate": gate, "head_diagnostics": heads,
                "model_kind": PA_KIND(arm),
                # R-203(2): EACH ARM's OWN thresholds. The receipt previously
                # carried B's for every arm, so three of four rows described a
                # cutoff their model never used.
                "causal_thresholds": thr,
                "threshold_source": {
                    "PM_PLUS_FINE": "frozen candidate artifact",
                    "PLUS_PRED_STATE_V1": "linear_{coin}.json",
                    "INCUMBENT_REWEIGHTED_ONLY": "linear_d_{coin}.json",
                    "LGBM_PINNED": "lgbm_thresholds_{coin}.json"}.get(arm)}
            print(f"  {coin} {arm:<20} n_actions={gate.get('n_actions')}", flush=True)
            for b, g in gate["budgets"].items():
                print(f"      @{b}: net {g['net_cents']:+9.1f}c  "
                      f"rand_max {g['random_net_max']:+8.1f}  "
                      f"beats_NET={g['beats_random_max_on_NET']}", flush=True)
        # free this coin's features before the next coin is built
        SC[coin]["PM"] = []; SC[coin]["FN"] = []; SC[coin]["ST"] = []
        del sc, srows
    # R-204(2): every arm must have been evaluated in the SAME mode. If a
    # rewiring ever leaves one arm on RETROSPECTIVE_TOPK while the others are
    # causal, the arms are not comparable -- and the per-arm numbers would
    # still look entirely normal. Asserted at the ARTIFACT, not in prose.
    _modes = {}
    for _c, _arms in out["arms"].items():
        for _a, _v in _arms.items():
            _modes[f"{_c}/{_a}"] = _v["gate"].get("threshold_mode")
    _distinct = sorted(set(_modes.values()))
    out["evaluation_mode_check"] = {
        "per_arm": _modes,
        "distinct_modes": _distinct,
        "ALL_ARMS_SAME_MODE": len(_distinct) == 1,
        "mode": _distinct[0] if len(_distinct) == 1 else None,
        "why": "arms evaluated under different threshold modes are not "
               "comparable, and each arm's numbers still look normal alone",
    }
    if len(_distinct) != 1:
        raise RuntimeError(
            f"REFUSED: arms were evaluated under DIFFERENT threshold modes: "
            f"{_modes}. A paired comparison across modes is not a comparison.")
    import os, tempfile
    fd, tmp = tempfile.mkstemp(dir=str(OUT.parent), suffix=".tmp")
    with os.fdopen(fd, "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True); fh.flush(); os.fsync(fh.fileno())
    os.replace(tmp, OUT)
    print(f"\nWROTE {OUT.name}", flush=True)
    return out


def run() -> dict:
    """SUPERSEDED. The single-process path was oom-killed at 14G and R-174
    mandates the staged pattern. Kept as a refusal rather than deleted so a
    caller reaching for it gets told why, instead of finding it missing and
    reinventing it."""
    raise SystemExit(
        "REFUSED: the single-process path is superseded. It held the fragment "
        "features, the top-up features and the LGBM matrices at once and was "
        "oom-killed at 14G after all four feature passes had succeeded. "
        "Use --stage-fit then --stage-score (R-174: restructure, never raise "
        "the cap).")


def selftest() -> int:
    checks = 0

    def ok(c, label):
        nonlocal checks
        if not c:
            raise AssertionError(label)
        checks += 1

    assert_disjoint({"a", "b"}, {"c", "d"})
    ok(True, "KNOWN-GOOD: disjoint populations pass")
    try:
        assert_disjoint({"a", "b"}, {"b", "c"})
        ok(False, "an overlapping slug must be refused")
    except PopulationLeak as e:
        ok("FLATTERS" in str(e),
           "POSITIVE CONTROL: a single shared slug is REFUSED, naming the "
           "direction of the harm -- this is the 808-window failure, caught "
           "before any arm runs rather than after")
    ok(D.ARMS[0] == "PM_PLUS_FINE" and "PM_PLUS_FINE" not in D.WEIGHTED_ARMS,
       "arm A is the unweighted frozen incumbent, per the declaration")
    ok(D.N_RANDOM >= 200 and D.DECISION_METRIC == "net_cents",
       "declared null and decision metric come from the declaration, not "
       "from this file")
    ok(D.MULTIPLICITY_AFTER == 4, "multiplicity 4 after scoring two candidates")
    ok(len(D.PRED_STATE_V1) == 21, "PRED_STATE_V1 is the declared 21 features")
    ok("early_stopping" not in D.LGBM_PARAMS,
       "arm C has no early stopping -- nothing about it is chosen after "
       "seeing the scoring side")
    print(f"phase2_arms selftest: {checks} checks OK")
    return 0


def main() -> int:
    if "--selftest" in sys.argv:
        return selftest()
    selftest()
    if not TOPUP.exists():
        raise SystemExit(f"REFUSED: {TOPUP.name} does not exist. Phase 2 has "
                         f"no test surface; build it before scoring.")
    if "--stage-fit" in sys.argv:
        stage_fit(); return 0
    if "--stage-score" in sys.argv:
        stage_score(); return 0
    raise SystemExit("specify --stage-fit or --stage-score (staged after the "
                     "single-process run was oom-killed at 14G)")


if __name__ == "__main__":
    raise SystemExit(main())
