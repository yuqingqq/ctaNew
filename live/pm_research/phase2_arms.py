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


def assert_modules_under_root() -> dict:
    """Every LATTICE module loaded from THIS tree, and its RUNTIME BYTES equal
    the bytes being attested. R-230(3).

    This checked FOUR modules while CODE_IDENTITY_FILES hashes TWELVE, so a
    wrong-tree harmful_hazard_model was ACCEPTED while the lattice hashed the
    repo copy -- the identity attested to bytes that never ran. Every lattice
    module is now imported at entry (an unimportable one is a refusal, not a
    skip) and checked twice:

      (a) __file__ resolves under _ROOT          -- a PROXY for provenance
      (b) sha256(the file actually loaded) == sha256(the file being recorded)

    (b) is the real invariant: RUNTIME BYTES == ATTESTED BYTES. Path prefix
    only says where a module claims to live."""
    import importlib
    checked = {}
    for fname in CODE_IDENTITY_FILES:
        mod_name = fname[:-3]
        try:
            m = importlib.import_module(mod_name)
        except Exception as e:
            raise RuntimeError(
                f"REFUSED: lattice module {mod_name} could not be imported "
                f"({type(e).__name__}: {e}). A module whose bytes are attested "
                f"but which never loads means the identity describes code that "
                f"did not run; an unimportable dependency is a refusal, not a "
                f"skip.")
        f = getattr(m, "__file__", None)
        if not f:
            raise RuntimeError(
                f"REFUSED: lattice module {mod_name} has no __file__, so the "
                f"bytes that ran cannot be identified.")
        fp = Path(f).resolve()
        if not str(fp).startswith(_ROOT):
            raise RuntimeError(
                f"REFUSED: {mod_name} loaded from {fp}, OUTSIDE this run root "
                f"{_ROOT}. A snapshot that imports another tree isolates "
                f"nothing -- the run would execute bytes nobody pinned.")
        # The sha is taken from the RESOLVED __file__ -- the bytes the module
        # actually loaded from -- and that value is what the identity records.
        # Hashing _ROOT/fname here instead would compare a file to ITSELF and
        # prove nothing (the trap DA hit in its own sweep: one source supplying
        # both the input and the expected answer).
        checked[fname] = _file_sha16(fp)
    return checked

DERIVED = Path("/home/yuqing/ctaNew/data/pm_5min/derived")
TOPUP = DERIVED / "harmful_exposure_rows_v3_topup.json"
FRAGMENT = DERIVED / "harmful_exposure_rows_v3_eraB.json"
FROZEN = DERIVED / "harmful_reduced_fine_candidate_v1.json"
# R-216: the receipt path follows the protocol rename. The label was renamed to
# PHASE2_FOUR_ARM_V2 and this was not, so the four-arm score OVERWROTE the
# committed three-arm receipt in place -- destroying DA's Q-DA-79 caveat block
# and 11 provenance fields on a SUPERSEDED artifact that exists to be provenance
# (rule 13: never edit a frozen artifact; the old receipt stays). Recovered from
# git. The seam sandbox redirected PA.OUT for tests, which is exactly why the
# production default went unexercised.
OUT = DERIVED / "phase2_four_arm_v2.json"

# R-228(5): the receipt's OWN version and supersession, emitted by the
# generator. These were added POST-GENERATION at v2.1 because receipt schema is
# a fit-time decision under R-225 -- editing phase2_arms.py between fit and
# score changes fit_code_sha256_prefix and the manifest correctly refuses its
# own score. Declaring them here means the closing cycle emits them, and a
# reader never has to trust a hand-added field.
PROTOCOL_VERSION = "v2.2"
SUPERSEDED_RECEIPT = DERIVED / "phase2_four_arm_v2.SUPERSEDED_BY_v2_2.json"
SUPERSEDED_REASON = (
    "R-228 enforcement rerun (audit #9). The v2.1 chain still failed OPEN in "
    "four places: an EMPTY file_hashes map passed the completeness loop "
    "vacuously, a manifest listing one artifact of fourteen passed while "
    "thirteen went unverified, an all-zeros fit_code_ref matched an all-zeros "
    "env value, and result-bearing code and data artifacts sat outside the "
    "identity lattice. This receipt is produced under the closed chain. "
    "Numbers expected identical; divergence would itself be a finding."
)


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

# ---------------------------------------------------------------- R-215 -----
# The fit population for ALL FOUR ARMS is: fragment rows whose TAPE row carries
# state_status == OK. Parity is the estimand -- same rows, different features --
# so this exclusion is applied ONCE, in tape_index(), and every arm inherits it.
#
# Rows the tape marked unusable are DESIGN exclusions: they are counted under
# their own status name and exempt from the fit's absorption bound. A key that
# is ABSENT from the tape is a FAILURE (`state_join_failed`), expected 0 and
# bounded at 1%. The two were previously the same counter, named for the
# failure -- so 26,339 warm-up rows (99.8% negative t_start) refused a fit as
# though the join had broken.
#
# Any status NOT listed here maps to `<status>_excluded` and is therefore NOT
# in DESIGN_EXCLUSIONS -- so a new or unexpected status trips the bound instead
# of inheriting an exemption it was never ruled into.
STATUS_DROP_NAME = {
    "PRE_WINDOW":       "pre_window_excluded",
    "GAP_AT_CUTOFF":    "gap_at_cutoff_excluded",   # own line, never folded
    "NO_LEVEL_HISTORY": "no_level_history_excluded",
}
DESIGN_EXCLUSIONS = frozenset(STATUS_DROP_NAME.values())
BOUNDED_DROPS     = ("state_join_failed",)          # expected 0, bound 1%

# ---- TWO-STAGE PARITY REGISTRATION (R-216), keyed by TAPE IDENTITY ---------
# THE STAGE IS PART OF THE DECLARATION. be-fit3 refused because a number
# declared for the post-join population was checked against the post-PURGE one:
# 578,917 and 577,598 are both correct counts of DIFFERENT stages, and the
# embargo purge (R-189) sits deliberately between them. A population count
# without its pipeline stage is not a declaration, it is an ambiguity.
#
# So BOTH stages are registered, and the purge is an accounted step:
#     ok_n            post-join, pre-purge: fragment rows whose tape row is OK
#     embargo_purged  training rows removed by the 60s embargo
#     fitted_n        the matrix every arm is fitted on
# and the predicate asserts fitted_n + embargo_purged == ok_n == registered.
#
# EVIDENTIAL STATUS IS NOT UNIFORM ACROSS THESE THREE, and the receipt must say
# so rather than let a reader assume all were declared blind:
#   ok_n            PRE-REGISTERED before any score existed (R-215(1)).
#   embargo_purged  RE-DECLARED after observing be-fit3's purge line, with the
#   fitted_n        cause named (registration under-specification, not a
#                   population movement -- every drop cell was identical to the
#                   diagnostic and state_join_failed stayed 0). Deliberate
#                   re-declaration under rule 11, NOT "adopting the new number".
# A future tape re-registers all three blind; only THIS tape carries the split.
PREREGISTERED_FIT_N = {
    "c7ab02ebcf27d2fc": {                      # tape6e, builder_ref ed9d572
        "btc": {"ok_n": 578917, "embargo_purged": 1319, "fitted_n": 577598},
        "eth": {"ok_n": 505904, "embargo_purged":  406, "fitted_n": 505498},
    },
}
REGISTRATION_PROVENANCE = {
    "ok_n": "PRE-REGISTERED before any score existed (R-215(1))",
    "embargo_purged": "RE-DECLARED post-observation, cause named (R-216)",
    "fitted_n": "RE-DECLARED post-observation, cause named (R-216)",
}


def registration(tape_sha16: str, coin: str, table: dict = None):
    """The two-stage registration for this coin ON THIS TAPE, or None."""
    tbl = PREREGISTERED_FIT_N if table is None else table
    return tbl.get(tape_sha16, {}).get(coin)


def preregistered_n(tape_sha16: str, coin: str, table: dict = None):
    """The registered PRE-PURGE (ok_n) population, or None."""
    r = registration(tape_sha16, coin, table)
    return None if r is None else r["ok_n"]


def assert_registration_arithmetic(table: dict = None) -> int:
    """fitted_n + embargo_purged == ok_n, for every registered entry.

    A registration whose own three numbers disagree cannot adjudicate a fit.
    Checked at import-time cost of nothing, and callable so a seam can feed it
    a deliberately inconsistent table and require a refusal."""
    tbl = PREREGISTERED_FIT_N if table is None else table
    n = 0
    for sha, coins in tbl.items():
        for coin, r in coins.items():
            if r["fitted_n"] + r["embargo_purged"] != r["ok_n"]:
                raise RuntimeError(
                    f"REFUSED: registration for {coin} on tape {sha} is "
                    f"internally inconsistent: fitted_n {r['fitted_n']:,} + "
                    f"embargo_purged {r['embargo_purged']:,} != ok_n "
                    f"{r['ok_n']:,}. The registration must reconcile before it "
                    f"can adjudicate anything.")
            n += 1
    return n


assert_registration_arithmetic()      # the table adjudicates nothing until it reconciles


def assert_preregistered_population(FIT: dict, tape_sha16: str,
                                    table: dict = None) -> dict:
    """STAGE 1: the PRE-PURGE population, checked where the number is true.

    Callable, and called immediately after the feature pass -- the stage the
    declared ok_n actually describes."""
    ev = {}
    for coin, f in FIT.items():
        n = len(f["kept"])
        r = registration(tape_sha16, coin, table)
        ev[coin] = {"population_pre_purge": n,
                    "registered_ok_n": None if r is None else r["ok_n"],
                    "registered_embargo_purged": (None if r is None
                                                  else r["embargo_purged"]),
                    "registered_fitted_n": None if r is None else r["fitted_n"],
                    "preregistration_key": tape_sha16,
                    "registration_provenance": dict(REGISTRATION_PROVENANCE),
                    "matches_ok_n": None if r is None else n == r["ok_n"]}
        if r is not None and n != r["ok_n"]:
            raise RuntimeError(
                f"REFUSED: {coin} PRE-PURGE population is {n:,} rows but ok_n "
                f"{r['ok_n']:,} is REGISTERED for tape {tape_sha16}. Same stage, "
                f"different counts: this is a real population move. Name the "
                f"cause and re-declare deliberately, never adopt the new "
                f"number. drops={f.get('drops')}")
    return ev


def assert_fitted_population(coin: str, fitted_n: int, purged: int,
                             ev: dict) -> dict:
    """STAGE 2: the fitted matrix, AND the purge reconciliation.

    Asserts fitted_n == registered fitted_n AND that the observed purge closes
    the arithmetic back to ok_n -- so the gap between the declared and fitted
    counts is attributed to a named step, never absorbed (rule 4)."""
    pre = ev.get("population_pre_purge")
    reconciles = (pre is not None and pre - purged == fitted_n)
    out = {"fitted_population": fitted_n, "purged_rows_embargo": purged,
           "population_pre_purge": pre, "purge_reconciles": reconciles,
           "registered_fitted_n": ev.get("registered_fitted_n"),
           "registered_embargo_purged": ev.get("registered_embargo_purged"),
           "matches_registered_fitted_n": (
               None if ev.get("registered_fitted_n") is None
               else fitted_n == ev["registered_fitted_n"]),
           "matches_registered_purge": (
               None if ev.get("registered_embargo_purged") is None
               else purged == ev["registered_embargo_purged"])}
    if not reconciles:
        raise RuntimeError(
            f"REFUSED: {coin} purge does not reconcile: pre-purge {pre!r} - "
            f"purged {purged:,} != fitted {fitted_n:,}. Rows left the "
            f"population through an unaccounted path.")
    for _k, _lbl in (("matches_registered_fitted_n", "fitted_n"),
                     ("matches_registered_purge", "embargo_purged")):
        if out[_k] is False:
            raise RuntimeError(
                f"REFUSED: {coin} {_lbl} disagrees with its registration "
                f"(observed vs registered: {out}). Re-declare deliberately "
                f"with the cause named; never adopt the new number.")
    return out


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
        # R-215: index EVERY row and CARRY ITS STATUS. Skipping non-OK rows
        # here made "excluded by design" indistinguishable from "join failed":
        # the same exclusion was applied at index time and then counted AGAIN
        # at join time under a name meaning the join broke -- while the counter
        # built to observe it (drops['state_status']) read zero, because the
        # filter upstream meant _feature_pass never saw a status at all.
        _st = str(r.get("state_status", "OK"))
        state = r.get("state") or {}          # NESTED, per features_under
        # The index value carries the ENCODED FEATURES plus the two clock
        # fields the embargo probe needs. Storing only the tuple made
        # stage_fit read r["t0"] on a tuple (TypeError at :333); reading them
        # from the KEY works but re-parses a slug, so they travel explicitly.
        idx[(r["slug"], r["side"], r["gen"], r["t_start"])] = {
            "vec": tuple(_PIN.encode_row(state, feats)) if _st == "OK" else None,
            "status": _st,
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


def acquire_fit_lock(lock: Path, pid: int = None) -> int:
    """ATOMIC exclusive acquisition. R-225(3).

    The previous form was check-then-write: `if lock.exists(): ...` followed by
    `lock.write_text(pid)`. Two processes can both pass the check and both
    write, so the lock did not exclude -- exactly the condition it existed to
    prevent, and it would have failed silently with both runs writing.

    O_CREAT|O_EXCL makes creation atomic: the kernel decides the winner. A lock
    whose holder is DEAD is reclaimed, but reclamation is itself a race, so the
    reclaimer re-attempts the atomic create rather than assuming it won."""
    import os as _o, errno as _e
    pid = _o.getpid() if pid is None else pid
    for _attempt in range(3):
        try:
            fd = _o.open(str(lock), _o.O_CREAT | _o.O_EXCL | _o.O_WRONLY, 0o644)
            with _o.fdopen(fd, "w") as fh:
                fh.write(str(pid))
            return pid
        except FileExistsError:
            try:
                owner = int(lock.read_text().strip())
                alive = Path(f"/proc/{owner}").exists()
            except (ValueError, OSError):
                owner, alive = None, False
            if alive:
                raise RuntimeError(
                    f"REFUSED: fit lock {lock.name} held by LIVE pid {owner}. "
                    f"Two fits writing one directory is how a partial run "
                    f"becomes another run's input.")
            print(f"  stale fit lock from dead pid {owner}; reclaiming",
                  flush=True)
            try:
                lock.unlink()
            except FileNotFoundError:
                pass          # another reclaimer won; retry the atomic create
    raise RuntimeError(
        f"REFUSED: could not acquire {lock.name} after 3 attempts; it is being "
        f"contended by other processes. Refusing rather than racing.")


def release_fit_lock(lock: Path, pid: int = None) -> bool:
    """Release ONLY if we still hold it. R-225(3).

    The lock was never released at all, so every run left one behind for the
    next to reclaim as stale -- which made 'a lock exists' carry no information
    and trained the reclaim path to always fire. Ownership is checked so a slow
    process cannot delete a lock that has since been reclaimed by someone
    else."""
    import os as _o
    pid = _o.getpid() if pid is None else pid
    try:
        if int(lock.read_text().strip()) != pid:
            return False
        lock.unlink()
        return True
    except (FileNotFoundError, ValueError, OSError):
        return False


def assert_fit_absorption_within_bound(drops: dict, n_kept: int,
                                       coin: str = "?",
                                       bound: float = 0.01) -> dict:
    """Fit-path absorption: per-status AND TOTAL. CALLABLE. R-225(2).

    Mirrors the builder's guard, for the same reason: a per-status bound cannot
    see a failure that arrives spread across names. Ten non-design categories at
    0.9% each pass every per-status check and still absorb 9% of the
    population. A total failure is under no obligation to use one name.

    DESIGN exclusions (rows the tape itself marked unusable) are exempt BY NAME
    from BOTH forms -- they are population statements, not absorbed failures.
    Anything not on that list is bounded, so a new or unexpected drop can never
    inherit an exemption it was never ruled into.

    CALLABLE because the inline form could only be tested by seam 31e SEARCHING
    ITS SOURCE TEXT for the word 'absorption'. A guard whose test greps for a
    word has not been shown to fire."""
    drops = dict(drops or {})
    missing = [b for b in BOUNDED_DROPS if b not in drops]
    if missing:
        raise RuntimeError(
            f"REFUSED: bounded drop counter(s) {missing} absent from the drops "
            f"table. A renamed or deleted counter removes the only thing "
            f"standing between a real join failure and a silent all-drop. "
            f"Present: {sorted(drops)}")
    total_in = n_kept + sum(drops.values())
    bounded = {k: v for k, v in drops.items() if k not in DESIGN_EXCLUSIONS}
    ev = {"coin": coin, "n_kept": n_kept, "total_input": total_in,
          "bound": bound, "drops": drops,
          "design_exempt": {k: v for k, v in drops.items()
                            if k in DESIGN_EXCLUSIONS},
          "bounded_drops": bounded, "bounded_total": sum(bounded.values()),
          "per_status_fractions": {}, "bounded_total_fraction": 0.0}
    if not total_in:
        return ev
    ev["bounded_total_fraction"] = sum(bounded.values()) / total_in
    for k, n in sorted(bounded.items()):
        frac = n / total_in
        ev["per_status_fractions"][k] = frac
        if frac > bound:
            raise RuntimeError(
                f"REFUSED: fit drop `{k}` covers {n:,} of {total_in:,} rows "
                f"({frac:.1%}) for {coin}, above the {bound:.0%} absorption "
                f"bound. Drops absorb row-level anomalies, never total input "
                f"failures. All drops: {drops}")
    if ev["bounded_total_fraction"] > bound:
        raise RuntimeError(
            f"REFUSED: fit drops TOTAL {sum(bounded.values()):,} of "
            f"{total_in:,} rows ({ev['bounded_total_fraction']:.1%}) for "
            f"{coin}, above the {bound:.0%} absorption bound. No single drop "
            f"exceeded it; the failure arrived spread across {len(bounded)} "
            f"names, which a per-status bound cannot see. Bounded: {bounded}")
    return ev


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
        # R-215: every exclusion is counted under ITS OWN NAME. `state` and
        # `state_status` are gone: the first lumped design exclusions under a
        # name meaning the join broke, the second could never fire.
        drops = {"pm": 0, "fine": 0, "no_archive": 0,
                 "pre_window_excluded": 0, "gap_at_cutoff_excluded": 0,
                 "no_level_history_excluded": 0,
                 "state_join_failed": 0}
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
            # R-215: `state_join_failed` means the key is ABSENT from the tape
            # -- a real join failure, expected 0, and it keeps the 1% bound.
            # Rows PRESENT under a non-OK status are counted by that status, so
            # a genuine join failure can never hide behind a design exclusion.
            _keep, sfeats = [], []
            for r in wrows:
                _e = TAPE.get((r["slug"], r["side"], r["gen"], r["t_start"]))
                if _e is None:
                    drops["state_join_failed"] += 1
                    continue
                _st = _e.get("status", "OK")
                if _st != "OK":
                    _nm = STATUS_DROP_NAME.get(_st, f"{_st.lower()}_excluded")
                    drops[_nm] = drops.get(_nm, 0) + 1
                    continue
                _keep.append(r); sfeats.append(_e)
            wrows = _keep
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
        # R-215: DESIGN exclusions (rows the tape itself marked unusable) are
        # exempt BY NAME. A blanket carve-out would let a real join failure
        # hide behind one; each name is listed, so anything unlisted is bounded.
        assert_fit_absorption_within_bound(drops, len(kept), coin)
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
    # BE F3: this set was hardcoded while the VERDICT declares its own
    # load_bearing_asserted -- six entries, adding half_open_containment_landed.
    # A consumer that ignores the producer's declaration can accept a verdict
    # where a predicate the producer called load-bearing is N/A or absent. The
    # required set is now the UNION: this file's floor, plus whatever the
    # verdict itself says is load-bearing.
    LOAD_BEARING = ("gap_count_matches_expected", "provenance_matches_expected",
                    "dataset_non_empty", "no_rows_skipped_by_builder",
                    "absorption_within_bound")
    LOAD_BEARING = tuple(sorted(set(LOAD_BEARING)
                                | set(v.get("load_bearing_asserted") or ())))
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


# R-230(2): capture and comparison must consume ONE declared set. The fit's
# at-write drift tuple listed SEVEN keys while the identity captured more, so
# the R-228 bindings (topup, frozen incumbent, topup build receipt) were
# captured and never compared -- an incumbent swap DURING fitting passed the
# recheck. Two hand-maintained lists diverged once and would again.
#
# The rule is now: compare EVERYTHING captured, with explicit named exceptions.
# The exception list is empty by design and each future entry must carry its
# reason; a key that cannot be compared should not be in the identity.
IDENTITY_DRIFT_EXEMPT: tuple = ()


def identity_drift(pre: dict, post: dict) -> dict:
    """Every captured key compared, minus declared exemptions. R-230(2)."""
    keys = (set(pre) | set(post)) - set(IDENTITY_DRIFT_EXEMPT)
    return {k: (pre.get(k), post.get(k)) for k in sorted(keys)
            if pre.get(k) != post.get(k)}


def _file_sha16(path) -> str:
    """sha256 prefix of a file, or None if it is absent."""
    import hashlib
    from pathlib import Path as _P
    path = _P(path)
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# The modules whose CONTENT defines a fit/score run. Declared explicitly so the
# identity is deterministic across entry points and a new dependency is an
# explicit decision rather than a silent change of meaning.
CODE_IDENTITY_FILES = (
    "phase2_arms.py", "phase2_declaration.py", "phase2_embargo.py",
    "phase2_state_schema_freeze.py", "harmful_action_eval.py",
    "harmful_hazard_model.py", "harmful_fast_compute.py",
    "harmful_state_features.py",
    # R-228(2): result-bearing dependencies that were OUTSIDE the lattice. Code
    # that shapes a number but is not hashed means the identity attests to a
    # subset of what produced the result -- a partial identity reads as a whole
    # one, which is the fail-open shape one level down from the manifest.
    "harmful_exposure_rows.py",        # owns any_fill_ahead + the latency cut
    "flow_intensity.py",
    "flow_fill_development.py",
    "harmful_candidate_manifest.py",
)

# Result-bearing DATA artifacts, bound by content. R-228(2). The code lattice
# says which program ran; these say what it ran ON. A frozen incumbent or a
# scoring population that can be swapped without detection makes every
# between-arm comparison unattributable.
DATA_IDENTITY_ARTIFACTS = {
    "fragment": lambda: FRAGMENT,
    "topup": lambda: TOPUP,
    "frozen_incumbent": lambda: FROZEN,
    "topup_build_receipt": lambda: DERIVED / "da_development_topup_v3.json",
    "verdict": lambda: DA_VERDICT,
}


def measured_code_identity() -> dict:
    """The code that IS RUNNING, by content -- never by a passed label.

    R-216 debt 3. FIT_CODE_REF is supplied by the launcher, so a manifest that
    records only it verifies that someone passed a string. This hashes the
    modules actually imported from _ROOT, so the manifest states a measured
    fact. Recorded BESIDE the env label, not instead of it: the label stays
    the human-readable ref, and a disagreement between them is detectable."""
    import hashlib
    from pathlib import Path as _P
    root = _P(_ROOT).resolve()
    # A DECLARED file list, not "whatever happens to be imported". Hashing the
    # live sys.modules made the identity depend on entry point -- stage_fit and
    # stage_score would report different shas for the SAME tree, which is the
    # opposite of an identity. Absent files hash to None and are visible.
    #
    # R-230(3): the sha comes from the module's RESOLVED __file__ when it is
    # loaded, so the identity attests to the BYTES THAT RAN rather than to the
    # bytes at a path we assume it used. A sys.modules-injected wrong-tree copy
    # is returned by import without touching sys.path, so a path-derived hash
    # would record the repo copy while other bytes executed.
    import sys as _sys
    files = {}
    for n in CODE_IDENTITY_FILES:
        m = _sys.modules.get(n[:-3])
        f = getattr(m, "__file__", None) if m is not None else None
        files[n] = _file_sha16(_P(f).resolve()) if f else _file_sha16(root / n)
    combined = hashlib.sha256(
        "".join(f"{k}:{v}" for k, v in sorted(files.items())).encode()
    ).hexdigest()[:16]
    decl = root / "phase2_declaration.py"
    return {"combined": combined, "files": files,
            "declaration": _file_sha16(decl), "n_files": len(files)}


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
            # R-216 debt 3: fit_ref is DECLARED (env). The manifest therefore
            # verified that a LABEL was passed, not that the code ran -- one
            # could score from a different tree, pass the right label, and the
            # check would pass clean. gate_code was already bound by CONTENT;
            # the fit now is too, and both travel so a reader can compare them.
            "fit_code_sha256_prefix": measured_code_identity()["combined"],
            "fit_code_files": measured_code_identity()["files"],
            "declaration_sha256_prefix": measured_code_identity()["declaration"],
            # R-216 debt 4: the FRAGMENT defines the population ok_n registers,
            # and nothing bound it. A regenerated fragment would move ok_n (and
            # correctly refuse) but the receipt could not say WHICH fragment
            # produced the number.
            "fragment_path": str(FRAGMENT),
            "fragment_sha256_prefix": _file_sha16(FRAGMENT),
            "fragment_bytes": FRAGMENT.stat().st_size if FRAGMENT.exists() else None,
            # R-228(2): the SCORED population, the FROZEN incumbent, and the
            # top-up's own build receipt. Each of these can change what the
            # numbers mean while every previously-bound field stays identical.
            "topup_path": str(TOPUP),
            "topup_sha256_prefix": _file_sha16(TOPUP),
            "topup_bytes": TOPUP.stat().st_size if TOPUP.exists() else None,
            "topup_build_receipt_sha256_prefix": _file_sha16(
                DERIVED / "da_development_topup_v3.json"),
            "frozen_incumbent_path": str(FROZEN),
            "frozen_incumbent_sha256_prefix": _file_sha16(FROZEN),
            "tape_bytes": TAPE_PATH.stat().st_size,
            "verdict_kind": v.get("verdict"),
            "verdict_tape_sha256_prefix": v.get("tape_sha256_prefix"),
            "verdict_path": str(DA_VERDICT),
            "verdict_sha256_prefix": vh,
            "gate_code_sha256_prefix": gate_id,
            "fit_code_ref": fit_ref}


# The artifacts a COMPLETE fit produces. Declared, so completeness is a
# question the checker can ask -- not inferred from what a manifest happens to
# list. lgbm_val_{coin} is CONDITIONAL (written only when a coin has >=100
# positives), so it is verified when present and never required.
FIT_BASE_ARTIFACTS = ("empty_coins.json", "fit_slugs.json",
                      "fit_population_parity.json",
                      "val_models.json")          # R-230(1)
FIT_PER_COIN_ARTIFACTS = ("linear_{c}.json", "linear_d_{c}.json",
                          "lgbm_haz_{c}.txt", "lgbm_thresholds_{c}.json")


def expected_fit_artifacts(m: dict) -> set:
    """The exact set a complete fit must have hashed. R-228(1).

    Coins come from empty_coins.json, which is ITSELF in the required set and
    hash-verified -- so shrinking the expected set by forging that file breaks
    its own hash. At least one coin must be fitted; a manifest claiming every
    coin empty describes no fit at all."""
    empty = []
    ec = FITDIR / "empty_coins.json"
    if ec.exists():
        try:
            empty = list(json.loads(ec.read_text()))
        except (ValueError, OSError):
            empty = []
    coins = [c for c in ("btc", "eth") if c not in empty]
    if not coins:
        raise RuntimeError(
            "REFUSED: empty_coins.json marks every coin empty. A fit over no "
            "population is not a null result, it is a broken input path.")
    out = set(FIT_BASE_ARTIFACTS)
    for c in coins:
        out |= {t.format(c=c) for t in FIT_PER_COIN_ARTIFACTS}
    # R-230(1): lgbm_val_{coin} is REQUIRED exactly when the fit recorded that
    # it wrote one. It used to be "verified when present, never required", so
    # deleting it between fit and score passed completeness untouched.
    for c, wrote in (val_models_record() or {}).items():
        if wrote and c in coins:
            out.add(f"lgbm_val_{c}.txt")
    return out


def val_models_record() -> dict:
    """Which coins the FIT recorded writing a val model for. R-230(1)."""
    f = FITDIR / "val_models.json"
    if not f.exists():
        return {}
    try:
        return dict(json.loads(f.read_text()))
    except (ValueError, OSError):
        return {}


def assert_ref_resolves_to_recorded_code(m: dict) -> dict:
    """The declared ref must be a REAL commit carrying the RECORDED bytes.

    R-228(1). fit_code_ref was only ever compared to the env value the scorer
    was launched with, so an all-zeros ref matched an all-zeros ref and the
    manifest attested to a commit that does not exist. The label is now checked
    against the content it claims: every file in the recorded fit_code_files
    must hash, AT THAT COMMIT, to the value the manifest recorded.

    FAIL-CLOSED. A git failure is a REFUSAL, never a skip -- an unverifiable
    binding that passes is exactly the shape this audit is about."""
    import subprocess, hashlib
    ref = m.get("fit_code_ref")
    files = m.get("fit_code_files") or {}
    if not ref or not isinstance(ref, str) or len(ref) != 40 \
            or any(c not in "0123456789abcdef" for c in ref.lower()):
        raise RuntimeError(
            f"REFUSED: fit_code_ref {ref!r} is not a 40-hex commit ref.")
    if not files:
        raise RuntimeError(
            "REFUSED: the manifest records no fit_code_files, so its ref "
            "cannot be checked against the code it names.")
    r = subprocess.run(["git", "-C", _ROOT, "cat-file", "-t", ref],
                       capture_output=True, text=True)
    if r.returncode != 0 or r.stdout.strip() != "commit":
        raise RuntimeError(
            f"REFUSED: fit_code_ref {ref} does not resolve to a commit "
            f"(git said {r.stdout.strip()!r} / {r.stderr.strip()[:80]!r}). A "
            f"ref naming no commit attests to nothing.")
    rel = subprocess.run(["git", "-C", _ROOT, "rev-parse", "--show-prefix"],
                         capture_output=True, text=True)
    if rel.returncode != 0:
        raise RuntimeError("REFUSED: cannot locate the code tree in git; the "
                           "recorded ref cannot be verified and an "
                           "unverifiable binding must not pass.")
    prefix = rel.stdout.strip()
    bad = {}
    for name, want in sorted(files.items()):
        blob = subprocess.run(
            ["git", "-C", _ROOT, "show", f"{ref}:{prefix}{name}"],
            capture_output=True)
        if blob.returncode != 0:
            bad[name] = f"absent at {ref[:7]}"
            continue
        got = hashlib.sha256(blob.stdout).hexdigest()[:16]
        if got != want:
            bad[name] = f"{got} != recorded {want}"
    if bad:
        raise RuntimeError(
            f"REFUSED: fit_code_ref {ref[:7]} does not carry the recorded "
            f"code: {bad}. The ref is a LABEL; it must name the commit whose "
            f"content was actually measured, or it is decoration.")
    return {"ref": ref, "files_verified": len(files)}


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
    #
    # R-225(1): the previous form was `m.get(k) != now.get(k)`, which has TWO
    # holes the user's audit found. (a) A key absent from the list was never
    # compared at all -- fit_code_sha256_prefix and fragment_sha256_prefix were
    # WRITTEN into the manifest and never ENFORCED, so a manifest carrying no
    # measured identity was ACCEPTED. Proving a hash changes is not the same as
    # proving scoring rejects a wrong one. (b) `.get() != .get()` passes
    # VACUOUSLY when BOTH sides are missing: two Nones compare equal, so an
    # absent binding read as agreement. Presence is now required explicitly,
    # and "missing" is a different refusal from "mismatched".
    REQUIRED = (
        ("tape_sha256_prefix", "the fit was produced against a different tape"),
        ("tape_bytes", "the tape changed size since the fit"),
        ("verdict_path", "a different verdict artifact is in place"),
        ("verdict_sha256_prefix", "the verdict CONTENT changed since the fit"),
        ("gate_code_sha256_prefix", "a different GATE produced the verdict"),
        ("fit_code_ref", "a different FIT CODE REF produced these artifacts"),
        ("fit_code_sha256_prefix", "DIFFERENT FIT CODE produced these artifacts "
                                   "(measured, not the env label)"),
        ("fragment_sha256_prefix", "a different FRAGMENT defines the population"),
        ("topup_sha256_prefix", "a different TOP-UP is the scored population"),
        ("frozen_incumbent_sha256_prefix",
         "a different FROZEN INCUMBENT is arm A's model"),
        ("topup_build_receipt_sha256_prefix",
         "the top-up's own build receipt changed"),
    )
    for k, why in REQUIRED:
        if k not in m or m.get(k) is None:
            raise RuntimeError(
                f"REFUSED: {FIT_MANIFEST} carries no {k}. A manifest that "
                f"cannot state this binding cannot authorise scoring against "
                f"it -- and an absent hash must never read as a matching one. "
                f"Re-run the fit with code that records it.")
        if now.get(k) is None:
            raise RuntimeError(
                f"REFUSED: cannot MEASURE {k} at score time, so the manifest's "
                f"value cannot be checked. An uncheckable binding is not a "
                f"binding.")
        if m[k] != now[k]:
            raise RuntimeError(
                f"REFUSED: {why} ({k}: fit={m[k]!r} now={now[k]!r}). "
                f"Scoring under bindings that differ from the fit's is not a "
                f"comparison.")
    import hashlib
    # R-228(1): the loop below verifies whatever file_hashes HOLDS. An EMPTY map
    # iterates zero times and passes -- a manifest asserting nothing was read as
    # a manifest asserting everything. It also never asked what SHOULD be there,
    # so a manifest listing one artifact of fourteen passed while thirteen went
    # unverified. Completeness is now required against an EXPECTED set.
    fh = m.get("file_hashes") or {}
    if not fh:
        raise RuntimeError(
            "REFUSED: the fit manifest's file_hashes is EMPTY. Zero artifacts "
            "verified is not zero artifacts changed; an empty map must never "
            "read as a satisfied check.")
    expected = expected_fit_artifacts(m)
    missing = sorted(expected - set(fh))
    if missing:
        raise RuntimeError(
            f"REFUSED: the fit manifest does not cover {len(missing)} required "
            f"artifact(s): {missing}. Verifying only what a manifest chooses to "
            f"list lets anything it omits change unobserved. Listed: "
            f"{sorted(fh)}")
    for name, want in fh.items():
        f = FITDIR / name
        if not f.exists():
            raise RuntimeError(f"REFUSED: manifest lists {name}, which is absent.")
        got = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
        if got != want:
            raise RuntimeError(
                f"REFUSED: {name} hash {got} != manifest {want}; the fit "
                f"directory changed after the run completed.")
    assert_ref_resolves_to_recorded_code(m)
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
    acquire_fit_lock(_lock)
    # R-225(3): the lock is RELEASED in a finally. It was never released at
    # all, so every run left one behind for the next to reclaim as stale --
    # which made 'a lock exists' carry no information and trained the
    # reclaim path to fire on every run.
    try:
        # R-228(4): the identity capture moved INSIDE the try. It was between
        # acquisition and the try block, so an identity failure -- an
        # unreadable tape, a vanished fragment -- raised with the lock HELD and
        # never released, leaving a live lock behind on exactly the paths that
        # are hardest to reproduce. The finally now covers everything after
        # acquisition.
        # R-225(4): captured BEFORE anything is loaded. Captured after, it
        # describes whatever the inputs happened to be once reading finished,
        # so an input perturbed DURING the run is recorded as though it had
        # always been that way.
        _ident_pre = _tape_identity()
        print(f"  input identity captured BEFORE load: tape "
              f"{_ident_pre['tape_sha256_prefix']} fragment "
              f"{_ident_pre['fragment_sha256_prefix']} topup "
              f"{_ident_pre['topup_sha256_prefix']}", flush=True)
        _final = FITDIR
        globals()["FITDIR"] = _run
        print("  indexing rebuilt tape (train split)...", flush=True)
        TP = tape_index("train")
        print(f"  tape rows indexed: {len(TP):,}", flush=True)
        FIT = _feature_pass(FRAGMENT, "fragment", TAPE=TP)
        # The pre-registration is checked HERE, on the pre-purge population, because
        # that is the stage it declares. Checked after the purge it compares two
        # different quantities and refuses on their difference.
        _ident = _ident_pre            # the pre-load capture, not a re-read
        _tape_sha16 = _ident["tape_sha256_prefix"]
        _prereg = assert_preregistered_population(FIT, _tape_sha16)
        for _c, _e in sorted(_prereg.items()):
            if _e["registered_ok_n"] is not None:
                print(f"  [prereg/{_c}] pre-purge {_e['population_pre_purge']:,} == "
                      f"registered ok_n {_e['registered_ok_n']:,} OK", flush=True)
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
        _parity: dict = {}
        _val_written: dict = {}
        for coin in list(FIT):
            f = FIT[coin]
            if not f["kept"]:
                (FITDIR / f"linear_{coin}.json").write_text(json.dumps(
                    {"n_rows": 0, "purge_status": "VACUOUS_N_0",
                     "embargo_evidence": f.get("embargo_evidence"),
                     "fitted": False}))
                continue
            yF, tF = _labels(f["kept"])
            # R-215(1): PARITY IS THE ESTIMAND -- same rows, different features. It
            # is COMPUTED from each arm's actual design matrix, never assumed from
            # the fact that they share a loop: measuring the thing that matters is
            # the point, and a future edit that filters one arm's matrix must fail
            # here rather than produce a quietly unpaired comparison.
            _arm_n: dict = {}
            XF = [f["PM"][i] + f["FN"][i] + f["ST"][i] for i in range(len(f["kept"]))]
            Xf, mu, sd = fc.fast_zscale(XF, XF)
            _arm_n["PLUS_PRED_STATE_V1"] = len(Xf)
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
            _arm_n["PM_PLUS_FINE"] = len(XfA)
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
            _arm_n["INCUMBENT_REWEIGHTED_ONLY"] = len(XD)
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
            _arm_n["LGBM_PINNED"] = int(A.shape[0])
            clf = lgb.LGBMClassifier(**D.LGBM_PARAMS)
            clf.fit(A, np.asarray(yF), sample_weight=swa)
            clf.booster_.save_model(str(FITDIR / f"lgbm_haz_{coin}.txt"))
            # arm C's OWN training thresholds, resolved AFTER its model exists
            _pc = clf.predict_proba(A)[:, 1]
            ftm = np.asarray(yF) == 1
            # R-230(1): whether a val model exists is now a RECORDED FACT, not
            # something inferred from a file being present. "Absent" and "never
            # written" were indistinguishable, so a val model deleted between
            # fit and score silently degraded arm C to hazard-only ranking.
            if ftm.sum() >= 100:
                reg = lgb.LGBMRegressor(**D.LGBM_VALUE_PARAMS)
                reg.fit(A[ftm], np.asarray(tF)[ftm], sample_weight=swa[ftm])
                reg.booster_.save_model(str(FITDIR / f"lgbm_val_{coin}.txt"))
                _vc = reg.predict(A)
                _val_written[coin] = True
            else:
                # the LEGITIMATE no-val path, now an EXPLICIT recorded state
                # rather than an absence anyone must interpret
                _vc = np.zeros(len(A))
                _val_written[coin] = False
                print(f"  [fit/{coin}] NO val model: {int(ftm.sum())} positives "
                      f"< 100. Recorded explicitly; arm C ranks on hazard alone "
                      f"for this coin.", flush=True)
            (FITDIR / f"lgbm_thresholds_{coin}.json").write_text(json.dumps(
                freeze_thresholds((_pc * _vc).tolist(), D.BUDGETS, gen_keys=_gk)))
            # ---- R-215(1): four-arms-one-n parity, COMPUTED ----
            _expect = len(f["kept"])
            _bad = {a: n for a, n in _arm_n.items() if n != _expect}
            if set(_arm_n) != set(D.ARMS) or _bad:
                raise RuntimeError(
                    f"REFUSED: fit population parity broken for {coin}. Declared "
                    f"population {_expect:,} rows; per-arm design matrices "
                    f"{_arm_n}. Arms must be fitted on the SAME rows and differ "
                    f"only in features -- an unpaired arm makes every between-arm "
                    f"delta uninterpretable. Missing arms: "
                    f"{sorted(set(D.ARMS) - set(_arm_n))}; mismatched: {_bad}.")
            # STAGE 2: the fitted matrix and the purge reconciliation. Both the
            # count and the arithmetic back to ok_n are asserted, so the gap
            # between declared and fitted is attributed to a named step (rule 4).
            _pp = _prereg.get(coin, {})
            _fit_ev = assert_fitted_population(coin, _expect,
                                               f.get("purged_rows", 0), _pp)
            _parity[coin] = dict(_fit_ev)
            _parity[coin].update({
                "per_arm_n": dict(_arm_n),
                "all_arms_same_n": (len(set(_arm_n.values())) == 1
                                    and set(_arm_n) == set(D.ARMS)),
                "n_arms": len(_arm_n), "n_arms_declared": len(D.ARMS),
                "registered_ok_n": _pp.get("registered_ok_n"),
                "matches_ok_n": _pp.get("matches_ok_n"),
                "preregistration_key": _pp.get("preregistration_key"),
                "registration_provenance": _pp.get("registration_provenance"),
                "drops": dict(f["drops"])})
            print(f"  [fit/{coin}] persisted linear + lgbm; rows {len(f['kept'])}, "
                  f"positive {sum(yF)}; parity {len(_arm_n)}/{len(D.ARMS)} arms "
                  f"@ fitted_n={_expect:,} (ok_n {_pp.get('population_pre_purge'):,} "
                  f"- {f.get('purged_rows', 0):,} purged, reconciles)", flush=True)
            del XF, Xf, A, FIT[coin]["PM"], FIT[coin]["FN"], FIT[coin]["ST"]
        (FITDIR / "empty_coins.json").write_text(json.dumps(empty_coins))
        if not FIT:
            raise RuntimeError(
                "REFUSED: every coin came back empty. A fit over no population is "
                "not a null result, it is a broken input path.")
        # R-230(1): inside the hash lattice, so forging it breaks its own hash
        (FITDIR / "val_models.json").write_text(
            json.dumps(_val_written, indent=1, sort_keys=True))
        (FITDIR / "fit_population_parity.json").write_text(
            json.dumps(_parity, indent=1, sort_keys=True))
        (FITDIR / "fit_slugs.json").write_text(json.dumps(sorted(
            {r["slug"] for c in FIT.values() for r in c["kept"]})))
        # COMPLETION MANIFEST then ATOMIC PROMOTE. Written last, so its presence
        # is the completion signal; promoted by rename, so a consumer never sees a
        # half-populated directory.
        import hashlib as _hh, shutil as _sh2, os as _os2
        # R-225(4): RECHECK at write. The pre-load capture is only a claim
        # until something verifies the inputs did not move while they were
        # being read. A divergence here means the run's numbers describe a
        # population that no longer exists, which is a refusal, not a warning.
        _ident_post = _tape_identity()
        _drift = identity_drift(_ident_pre, _ident_post)
        if _drift:
            raise RuntimeError(
                f"REFUSED: inputs CHANGED DURING the run: {_drift}. The "
                f"artifacts just produced describe a population that no longer "
                f"exists, and a manifest written now would attest to a state "
                f"that never produced them. Re-run against a quiet tree.")
        _hashes = {f.name: _hh.sha256(f.read_bytes()).hexdigest()[:16]
                   for f in sorted(_run.iterdir()) if f.is_file()}
        _mani = dict(_ident_post)
        _mani["identity_captured_before_load"] = True
        _mani["identity_rechecked_at_write"] = True
        _mani.update({"complete": True, "file_hashes": _hashes,
                      "run_finished_utc": __import__("subprocess").run(
                          ["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"],
                          capture_output=True, text=True).stdout.strip(),
                      "arms": list(D.ARMS), "budgets": list(D.BUDGETS),
                      "fit_population_parity": _parity})
        (_run / FIT_MANIFEST).write_text(json.dumps(_mani, indent=1, sort_keys=True))
        if _final.exists():
            _sh2.rmtree(_final.with_suffix(".prev"), ignore_errors=True)
            _final.rename(_final.with_suffix(".prev"))
        _os2.replace(str(_run), str(_final))
        globals()["FITDIR"] = _final
        print(f"STAGE FIT COMPLETE -- promoted {len(_hashes)} artifacts", flush=True)
    finally:
        if release_fit_lock(_lock):
            print(f"  released {_lock.name}", flush=True)

# R-230(4) / R-229 top debt: the population and reach disclosure, EMITTED BY
# THE GENERATOR. It was re-attached by hand for three consecutive cycles, and
# these are exactly the fields a fresh generation drops -- the ones whose silent
# absence turns a development result into an apparent validation.
#
# COMPUTED, not asserted. Every field below is derived from the declaration or
# from the scored rows; none is a prose conclusion sitting beside a table
# (rule 10). A reader can check each one.
VALIDATION_MIN_COMPLETE_DAYS = 5          # CLAUDE.md rule 11


def population_reach_disclosure(rows) -> dict:
    """What this receipt's numbers can and cannot support. R-230(4)."""
    import datetime as _dt
    ts = [r["t0"] + r["t_start"] for r in rows] if rows else []
    label = str(getattr(D, "POPULATION", "?"))
    out = {
        "population_label": label,
        "is_development_population": "development" in label.lower(),
        "declared_in": "phase2_declaration.POPULATION",
    }
    if not ts:
        out.update({"G_complete_utc_days": 0, "dates_present": [],
                    "span_hours": 0.0})
    else:
        lo, hi = min(ts), max(ts)
        by_date = {}
        for t in ts:
            d = _dt.datetime.fromtimestamp(t, _dt.timezone.utc).date().isoformat()
            by_date[d] = by_date.get(d, 0) + 1
        complete = []
        for d in sorted(by_date):
            d0 = _dt.datetime.fromisoformat(d).replace(
                tzinfo=_dt.timezone.utc).timestamp()
            if lo <= d0 and hi >= d0 + 86400:
                complete.append(d)
        out.update({
            "G_complete_utc_days": len(complete),
            "complete_days": complete,
            "dates_present": sorted(by_date),
            "rows_by_date": {k: by_date[k] for k in sorted(by_date)},
            "span_hours": (hi - lo) / 3600.0,
            "day_completeness_definition":
                "a UTC date is COMPLETE iff the population span covers all of "
                "it: min(t) <= 00:00:00 and max(t) >= the next 00:00:00",
        })
    g = out["G_complete_utc_days"]
    out["intervals_claimable"] = g >= VALIDATION_MIN_COMPLETE_DAYS
    out["meets_validation_day_bar"] = g >= VALIDATION_MIN_COMPLETE_DAYS
    out["is_a_validation"] = bool(
        out["meets_validation_day_bar"] and not out["is_development_population"])
    out["why"] = (
        f"G={g} complete UTC day(s) against a bar of "
        f"{VALIDATION_MIN_COMPLETE_DAYS} (rule 11), on a population labelled "
        f"{label!r}. is_a_validation is COMPUTED from those two facts, not "
        f"asserted: a development population or too few complete days makes "
        f"these numbers a development result, whatever their size.")
    return out


def _supersedes_block() -> dict:
    """What this receipt replaces, emitted BY THE GENERATOR. R-228(5).

    The superseded artifact is hashed at write time if it is present. Its
    ABSENCE is reported as an explicit status rather than an empty field: a
    receipt that silently claims to supersede nothing is indistinguishable from
    one whose predecessor was deleted."""
    present = SUPERSEDED_RECEIPT.exists()
    return {
        "path": str(SUPERSEDED_RECEIPT.name),
        "present_at_write": present,
        "sha256_prefix": _file_sha16(SUPERSEDED_RECEIPT) if present else None,
        "bytes": SUPERSEDED_RECEIPT.stat().st_size if present else None,
        "reason": SUPERSEDED_REASON,
        "note": ("preserve-then-write: the superseded receipt is renamed to "
                 "this path and committed BEFORE this run, so it survives "
                 "unedited at a resolvable path (rule 13). present_at_write "
                 "false means that preservation step did not happen."),
    }


def _fit_identity_from_manifest() -> dict:
    """The FIT's identity, READ from its manifest. R-225(1).

    Never measured here: at score time `measured_code_identity()` measures the
    SCORER. Reporting that under the fit's name is a category error that makes
    a mismatched pair look matched."""
    mf = FITDIR / FIT_MANIFEST
    if not mf.exists():
        return {"present": False, "why": f"{FIT_MANIFEST} absent"}
    m = json.loads(mf.read_text())
    return {"present": True, "source": FIT_MANIFEST,
            "declared_env_ref": m.get("fit_code_ref"),
            "measured_sha256_prefix": m.get("fit_code_sha256_prefix"),
            "fragment_sha256_prefix": m.get("fragment_sha256_prefix"),
            "fragment_bytes": m.get("fragment_bytes"),
            "declaration_sha256_prefix": m.get("declaration_sha256_prefix"),
            "note": "read from the fit's manifest, not measured at score time"}


def _read_fit_parity() -> dict:
    """The parity block, READ from the fit artifact, never recomputed.

    R-216 debt 5. Recomputing it here could produce a receipt that disagrees
    with the artifact it describes; reading it means the receipt reports what
    the fit actually recorded, and says so when the file is absent."""
    fp = FITDIR / "fit_population_parity.json"
    if not fp.exists():
        return {"present": False,
                "why": f"{fp.name} absent from the fit directory; parity is "
                       f"resolvable only via the completion manifest"}
    d = json.loads(fp.read_text())
    return {"present": True, "source": fp.name,
            "all_coins_parity_holds": all(
                bool(c.get("all_arms_same_n")) and bool(c.get("purge_reconciles"))
                for c in d.values()),
            "per_coin": d}


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
    # R-228(2): the SCORE side captures identity BEFORE load and RECHECKS at
    # write, mirroring the fit. Only the fit did this, so an input perturbed
    # during SCORING -- the stage that produces the published numbers -- was
    # invisible. The asymmetry meant the more consequential half was the
    # unguarded one.
    _ident_pre = _tape_identity()
    print(f"  score-side identity captured BEFORE load: tape "
          f"{_ident_pre['tape_sha256_prefix']} topup "
          f"{_ident_pre['topup_sha256_prefix']} frozen "
          f"{_ident_pre['frozen_incumbent_sha256_prefix']}", flush=True)
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
           # R-228(5): generator-owned, not hand-added after the fact.
           "protocol_version": PROTOCOL_VERSION,
           "supersedes": _supersedes_block(),
           # R-230(4): generator-owned, computed from the scored rows below
           "population_and_reach": None,
           "supersedes_label": "PHASE2_THREE_ARM_V1 (stale: four arms since arm D)", "arms": {}, "population": {},
           # BE F2: this was the literal "d7082b6" -- a ref whose ARMS has
           # THREE entries, beside a four-arm receipt. The governing
           # declaration is the v2 amendment (arm D, causal thresholds, five
           # head diagnostics, EMBARGO_S) introduced at 1cc2163. The MEASURED
           # sha is authoritative; the label is human-readable and must be
           # verifiable against it, which is why both travel.
           "declaration_commit_declared": "1cc2163",
           "declaration_commit_superseded_label": "d7082b6 (v1, THREE arms — "
                                                  "wrong for this receipt)",
           "declaration_sha256_prefix": measured_code_identity()["declaration"],
           "declaration_arms": list(D.ARMS),
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
           "da_caveat_field": "RESERVED for Q-DA-79 post-gap queue-validity finding",
           # R-216 debt 5: parity was resolvable only by following the manifest
           # chain. Automated readers resolve receipt FIELDS, so it is a field.
           # Read from the fit's own artifact, never recomputed here -- a
           # receipt that recomputes a predicate can disagree with the artifact
           # it claims to describe.
           "fit_population_parity": _read_fit_parity(),
           # R-225(1): these came from measured_code_identity() AT SCORE TIME,
           # i.e. the SCORER's identity printed under the FIT's name. The fit's
           # identity is a property of the fit and can only be read from its
           # manifest; the scorer's is recorded separately, under its own name.
           "fit_code_identity": _fit_identity_from_manifest(),
           "score_code_identity": {
               "declared_env_ref": _tape_identity().get("fit_code_ref"),
               "measured_sha256_prefix": measured_code_identity()["combined"],
               "note": "the code that produced the SCORES. Recorded beside the "
                       "fit's own identity, never in place of it."}}

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
                # R-230(1): the val model is loaded against the fit's RECORD,
                # not against whether a file happens to be there. This was
                # `Booster(vf) if vf.exists() else None`, and the None path fed
                # np.zeros -- so a val model deleted between fit and score
                # degraded arm C to hazard-only ranking, produced a full set of
                # plausible numbers, and refused nothing.
                vf = FITDIR / f"lgbm_val_{coin}.txt"
                _rec = val_models_record()
                if coin not in _rec:
                    raise RuntimeError(
                        f"REFUSED: the fit recorded no val-model state for "
                        f"{coin}. Absent and never-written are different facts "
                        f"and must not be inferred from a file listing.")
                if _rec[coin]:
                    if not vf.exists():
                        raise RuntimeError(
                            f"REFUSED: the fit RECORDED writing a val model for "
                            f"{coin} and {vf.name} is absent. Substituting zeros "
                            f"would silently rank arm C on hazard alone while "
                            f"reporting it as the value-weighted arm.")
                    vb = lgb.Booster(model_file=str(vf))
                else:
                    if vf.exists():
                        raise RuntimeError(
                            f"REFUSED: the fit recorded NO val model for {coin} "
                            f"yet {vf.name} exists. An artifact the fit did not "
                            f"write must not be scored.")
                    vb = None
                    print(f"  [{coin}/LGBM_PINNED] no val model (recorded): "
                          f"ranking on hazard alone, by declaration",
                          flush=True)
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
    # R-230(4): the disclosure is computed from the ROWS ACTUALLY SCORED, so it
    # cannot describe a population other than the one that produced these
    # numbers. Emitted by the generator; not re-attached by hand.
    _all_rows = [r for c in SC.values() for r in c["kept"]]
    out["population_and_reach"] = population_reach_disclosure(_all_rows)
    _pr = out["population_and_reach"]
    print(f"  population/reach: {_pr['population_label']} | G="
          f"{_pr['G_complete_utc_days']} complete UTC days | "
          f"is_a_validation={_pr['is_a_validation']}", flush=True)

    # R-228(2): RECHECK at write. A capture nobody re-verifies is a claim.
    _ident_post = _tape_identity()
    _drift = identity_drift(_ident_pre, _ident_post)
    if _drift:
        raise RuntimeError(
            f"REFUSED: inputs CHANGED DURING scoring: {_drift}. The numbers "
            f"just produced describe inputs that no longer exist, and writing "
            f"a receipt now would publish them as though they did.")
    out["score_input_identity"] = {
        "captured_before_load": True, "rechecked_at_write": True,
        "tape_sha256_prefix": _ident_post.get("tape_sha256_prefix"),
        "topup_sha256_prefix": _ident_post.get("topup_sha256_prefix"),
        "frozen_incumbent_sha256_prefix":
            _ident_post.get("frozen_incumbent_sha256_prefix"),
        "topup_build_receipt_sha256_prefix":
            _ident_post.get("topup_build_receipt_sha256_prefix"),
    }
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
