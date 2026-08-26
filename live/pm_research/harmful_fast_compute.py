"""OPT-IN accelerated twins of harmful_hazard_model's stdlib fits, plus the
columnar row cache and the equivalence harness that will make the adoption
gate runnable.

SURFACE AUTHORISATION (R-126, in-file): coordinator dispatch, R-155(2)
pre-work; ADOPTION GATED on full-scale cent-exact equivalence — this module
is not wired into any pipeline. Nothing imports it today; the incumbent
`harmful_hazard_model.py` stays bit-for-bit untouched (R-157(2): an incumbent
rewritten mid-comparison is no longer the thing that was confirmed).

WHAT THIS IS. The builder's fits are deliberately pure-stdlib and single-
threaded: hand-rolled Newton/IRLS over ~1.1M rows, ~80 min per full run. This
file provides numpy twins that mirror the EXACT algorithms — same iteration
scheme, same z-clamp (±30), same convergence criterion (max|step| < 1e-9),
same singularity bar (|pivot| < 1e-12), same regularisation semantics
(logistic: per-sample penalty (lam*w[a])/len(X) with the INTERCEPT UNPENALISED
in the gradient but lam added to the FULL Hessian diagonal, exactly as the
stdlib does it; ridge: lam once on the full diagonal). The weighted logistic
keeps the R-159 fix EXACTLY: the penalty divisor is len(X), NEVER sum(sw) —
the sum(sw) variant lives on below only as a named falsifier the harness must
flag.

WHERE EXACT MIRRORING IS IMPOSSIBLE, and only there, the twins differ:
  1. REDUCTION ORDER. The stdlib accumulates g/H/mu/sd/z sequentially row by
     row; numpy reduces pairwise/blocked (and matmul delegates to BLAS). Same
     summands, different association -> ~1e-16-relative drift per reduction.
  2. np.exp vs math.exp may differ by <=1 ulp on some inputs (SIMD dispatch).
  3. The stdlib subtracts the logistic penalty term interleaved per sample;
     the twin subtracts the algebraically identical total n*((lam*w)/n),
     keeping the stdlib's own intermediate rounding of (lam*w)/n.
Every per-element step that is NOT a reduction (the Gauss-Jordan elimination,
the clamp, the sigmoid arithmetic, the penalty intermediates, the weighted
products' association) is arranged to be operation-for-operation identical.
The CONSEQUENCE of 1-3 is never asserted away: `equivalence_report` measures
it per fixture, and the full-scale R-155(2) gate is the only adopter.

DETERMINISM. float64 throughout; accumulation walks fixed-size chunks in a
fixed order; no threading knobs are set anywhere. Within a chunk the matmul
is BLAS-deterministic for a fixed build; run-to-run stability at full scale
is re-checked by the gate, not assumed here.

DECLARED PASS BAR (declared before any result, R-109/rule 6 discipline):
  per-row |Δscore| < PASS_MAX_ABS_SCORE_CENTS (1e-7 cents) at realistic
  magnitudes, AND every score rounds to the same cent (np.rint equality).
  1e-7 is chosen so that even a fully one-signed accumulation across 2e6 rows
  moves a receipt aggregate by < 0.2 cents, below the 0.5-cent resolution at
  which receipts round. Replication-invariance (the R-159 lesson) must hold
  to INV_BAR (1e-12, the model selftest's own tolerance); error GROWING with
  the replication factor is a bug signature, not float noise, and is computed
  as a predicate, never printed as prose (rule 10).

    python3 live/pm_research/harmful_fast_compute.py --selftest --tmp DIR
    python3 live/pm_research/harmful_fast_compute.py --equivalence
    python3 live/pm_research/harmful_fast_compute.py --bench --tmp DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

try:
    import numpy as np
except ImportError as _e:                                     # pragma: no cover
    raise ImportError(
        "harmful_fast_compute REQUIRES numpy (present in "
        "/home/yuqing/pricer-sol/venv). This opt-in module never falls back "
        "silently to the stdlib path: a caller who imported it wants the fast "
        "path or a loud refusal.") from _e

# Fixed-size accumulation chunk. Chunks are combined SEQUENTIALLY in index
# order, so the only reduction whose order is build-dependent is the matmul
# inside one chunk. 1<<17 rows x ~60 features x 8 B keeps the per-chunk temp
# ~63 MB — the era-scale OOM in zscale's history is why this is bounded.
CHUNK = 1 << 17

PASS_MAX_ABS_SCORE_CENTS = 1e-7   # per-row; rationale in the module docstring
INV_BAR = 1e-12                   # replication invariance, model selftest's bar
INV_MIN_SIGNAL = 1e-10            # growth predicate floor: below this, noise

CACHE_SCHEMA_VERSION = "harmful_rows_columnar_v1"
DEFAULT_CACHE_DIR = Path("/home/yuqing/ctaNew/data/ml/cache/pm_harmful_columnar")


# ------------------------------------------------------------------ fast fits
def _fast_solve(H: np.ndarray, g: np.ndarray, w: np.ndarray):
    """Vectorised twin of harmful_hazard_model._solve — Gauss-Jordan with
    partial pivoting. Elimination here is ELEMENTWISE-IDENTICAL to the stdlib:
    each row's multiplier f = M[r][c]/M[c][c] (one rounding), then
    M[r][cc] -= f*M[c][cc] (multiply, subtract — two roundings), no fused ops,
    no reductions, first-max pivot ties in both. The stdlib eliminates EVERY
    row r != c each pass (rows above c included, with their ~1-ulp residuals),
    so the twin does too — f is computed for all rows and only f[c] zeroed."""
    k = g.shape[0]
    M = np.empty((k, k + 1), dtype=np.float64)
    M[:, :k] = H
    M[:, k] = g
    for c in range(k):
        piv = c + int(np.argmax(np.abs(M[c:, c])))
        if piv != c:
            M[[c, piv]] = M[[piv, c]]
        if abs(M[c, c]) < 1e-12:
            return None
        f = M[:, c] / M[c, c]
        f[c] = 0.0
        M[:, c:] -= f[:, None] * M[c, c:][None, :]
    d = M[:, k] / M.diagonal()
    w2 = w + d
    return w2, bool(np.max(np.abs(d)) < 1e-9)


def _as_matrix(X) -> np.ndarray:
    Xa = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    if Xa.ndim != 2:
        raise ValueError(f"X must be rows x features; got ndim={Xa.ndim}")
    return Xa


def _logit_accum(Xa: np.ndarray, ya: np.ndarray, sw, w: np.ndarray):
    """One IRLS pass: data gradient and Hessian, chunked in fixed order.
    Association mirrors the stdlib per term: weighted e is (si*e)*x, weighted
    curvature is ((si*ww)*x_a)*x_b — only the reduction order differs."""
    n, k = Xa.shape
    g = np.zeros(k, dtype=np.float64)
    H = np.zeros((k, k), dtype=np.float64)
    for s in range(0, n, CHUNK):
        xb = Xa[s:s + CHUNK]
        z = np.clip(xb @ w, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-z))
        e = ya[s:s + CHUNK] - p
        ww = p * (1.0 - p)
        if sw is not None:
            sb = sw[s:s + CHUNK]
            e = sb * e
            ww = sb * ww
        g += xb.T @ e
        H += (xb * ww[:, None]).T @ xb
    return g, H


def fast_fit_logistic(X, y, lam=1e-3, it=120):
    """Twin of harmful_hazard_model.fit_logistic. Same signature, list out."""
    Xa = _as_matrix(X)
    ya = np.asarray(y, dtype=np.float64)
    n, k = Xa.shape
    w = np.zeros(k, dtype=np.float64)
    diag = np.arange(k)
    for _ in range(it):
        g, H = _logit_accum(Xa, ya, None, w)
        # stdlib subtracts (lam*w[a])/len(X) once per SAMPLE, intercept
        # exempt; the total is n*((lam*w)/n), keeping the same intermediate.
        pen = (lam * w) / n
        pen[0] = 0.0
        g -= pen * n
        H[diag, diag] += lam
        got = _fast_solve(H, g, w)
        if got is None:
            break
        w, done = got
        if done:
            break
    return w.tolist()


def fast_fit_logistic_w(X, y, sw, lam=1e-3, it=120):
    """Twin of fit_logistic_w. DIVISOR IS len(X), NOT sum(sw) — R-159. The
    sum(sw) variant over-shrinks by len(X)/sum(sw), scaling with rows-per-
    generation, which differs by coin; it survives below ONLY as the named
    falsifier `_broken_fit_logistic_w_sumsw` that the harness must flag."""
    Xa = _as_matrix(X)
    ya = np.asarray(y, dtype=np.float64)
    swa = np.asarray(sw, dtype=np.float64)
    n, k = Xa.shape
    n_samp = n or 1
    w = np.zeros(k, dtype=np.float64)
    diag = np.arange(k)
    for _ in range(it):
        g, H = _logit_accum(Xa, ya, swa, w)
        pen = (lam * w) / n_samp
        pen[0] = 0.0
        g -= pen * n_samp
        H[diag, diag] += lam
        got = _fast_solve(H, g, w)
        if got is None:
            break
        w, done = got
        if done:
            break
    return w.tolist()


def _ridge_accum(Xa: np.ndarray, ya: np.ndarray, sw):
    n, k = Xa.shape
    H = np.zeros((k, k), dtype=np.float64)
    g = np.zeros(k, dtype=np.float64)
    for s in range(0, n, CHUNK):
        xb = Xa[s:s + CHUNK]
        yb = ya[s:s + CHUNK]
        if sw is None:
            H += xb.T @ xb
            g += xb.T @ yb
        else:
            xw = xb * sw[s:s + CHUNK][:, None]     # (si*x_a), stdlib's order
            H += xw.T @ xb
            g += xw.T @ yb
    return H, g


def fast_fit_ridge(X, y, lam=1.0):
    """Twin of fit_ridge: XtX + lam on the FULL diagonal (index 0 included,
    exactly as the stdlib), solved from w=0; [0.0]*k on a singular system."""
    Xa = _as_matrix(X)
    ya = np.asarray(y, dtype=np.float64)
    k = Xa.shape[1]
    H, g = _ridge_accum(Xa, ya, None)
    diag = np.arange(k)
    H[diag, diag] += lam
    r = _fast_solve(H, g, np.zeros(k, dtype=np.float64))
    return r[0].tolist() if r else [0.0] * k


def fast_fit_ridge_w(X, y, sw, lam=1.0):
    """Twin of fit_ridge_w."""
    Xa = _as_matrix(X)
    ya = np.asarray(y, dtype=np.float64)
    swa = np.asarray(sw, dtype=np.float64)
    k = Xa.shape[1]
    H, g = _ridge_accum(Xa, ya, swa)
    diag = np.arange(k)
    H[diag, diag] += lam
    r = _fast_solve(H, g, np.zeros(k, dtype=np.float64))
    return r[0].tolist() if r else [0.0] * k


def fast_generation_weights(kept: list,
                            key=lambda r: (r.get("slug"), r.get("side"),
                                           r.get("gen"))) -> list:
    """Verbatim mirror of generation_weights — dict counting is O(n) and not
    a bottleneck, so the twin changes NOTHING (identical output by
    construction; still verified against the real one in the selftest rather
    than assumed, rule 16)."""
    counts: dict = {}
    for r in kept:
        counts[key(r)] = counts.get(key(r), 0) + 1
    return [1.0 / counts[key(r)] for r in kept]


def fast_predict_p(w, x):
    """Character-identical to predict_p (scalar; nothing to vectorise)."""
    return 1 / (1 + math.exp(-max(-30, min(30, sum(a * b for a, b in zip(w, x))))))


def fast_predict_p_batch(w, X) -> np.ndarray:
    """Vectorised scoring for whole populations; same clamp and sigmoid."""
    z = np.clip(np.asarray(X, dtype=np.float64) @
                np.asarray(w, dtype=np.float64), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def fast_zscale(train_X, all_X):
    """Twin of zscale: population sd (divisor n), sd==0.0 -> 1.0, prepends the
    1.0 intercept column. DECLARED DIVERGENCE: the stdlib scales its list-of-
    lists IN PLACE (its memory fix); the twin returns a NEW (n, k+1) float64
    ndarray — with the columnar cache feeding ndarrays there is no python-list
    blowup to avoid, and callers converting from lists should drop the source.
    Same refusal, same message, on an empty training set."""
    if len(train_X) == 0:
        raise ValueError(
            "zscale received an EMPTY training set. This is a symptom, not a "
            "cause: something upstream dropped every training row. Callers "
            "should refuse with a named error before reaching here.")
    T = _as_matrix(train_X)
    n, k = T.shape
    mu = T.sum(axis=0) / n
    sd = np.sqrt(((T - mu) ** 2).sum(axis=0) / n)
    sd = np.where(sd == 0.0, 1.0, sd)
    A = _as_matrix(all_X)
    out = np.empty((A.shape[0], k + 1), dtype=np.float64)
    out[:, 0] = 1.0
    out[:, 1:] = (A - mu) / sd
    return out, mu.tolist(), sd.tolist()


# One-import-switch parity: `from harmful_fast_compute import fit_logistic`
# is the drop-in for `from harmful_hazard_model import fit_logistic` — after,
# and only after, the R-155(2) gate passes at full scale.
fit_logistic = fast_fit_logistic
fit_logistic_w = fast_fit_logistic_w
fit_ridge = fast_fit_ridge
fit_ridge_w = fast_fit_ridge_w
generation_weights = fast_generation_weights
predict_p = fast_predict_p
zscale = fast_zscale


# ------------------------------------------------------------- columnar cache
# The parsed-row structures the builder consumes, saved once as compact
# columnar .npz keyed by (dataset sha256, cache schema version) so REPEAT runs
# skip the 1.24 GB JSON parse. The loader VERIFIES the stored key against the
# requested one and REFUSES any mismatch — a cache that can silently serve
# stale features is how 163/176 symbol histories once died (CLAUDE.md
# pitfall 3): mismatch is a refusal, absence is a miss, nothing is coerced.
F64_FIELDS = ("t0", "t_start", "level", "resting", "qahead", "v_cancel_cents")
I64_FIELDS = ("gen",)
STR_FIELDS = ("slug", "day", "coin", "side", "status")
LATENCY_FIELDS = ("preventable_shares", "preventable_value_cents",
                  "stale_shares")


class CacheRefused(RuntimeError):
    """Named refusal: stale key, wrong schema, corrupt file, or a row the
    columnar layout cannot represent losslessly. Never a warning."""


def dataset_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _check_sha(dataset_sha: str) -> str:
    if (not isinstance(dataset_sha, str) or len(dataset_sha) != 64
            or any(c not in "0123456789abcdef" for c in dataset_sha)):
        raise CacheRefused(
            f"dataset_sha must be 64 lowercase hex chars (sha256 of the "
            f"dataset artifact bytes); got {dataset_sha!r}")
    return dataset_sha


def cache_path(dataset_sha: str, cache_dir=None,
               schema: str = CACHE_SCHEMA_VERSION) -> Path:
    d = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    return d / f"{schema}__{_check_sha(dataset_sha)[:16]}.npz"


def _num(v, where: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise CacheRefused(
            f"non-numeric value {v!r} at {where}: the columnar cache refuses "
            f"to coerce rather than store a lossy representation")
    return float(v)


def save_columnar(rows: list, dataset_sha: str, cache_dir=None,
                  schema: str = CACHE_SCHEMA_VERSION,
                  extra_meta: dict | None = None) -> Path:
    """Write the consumed row fields as columns. Presence masks distinguish
    an absent field from a stored default, so reconstruction is exact for the
    fields the builder reads. Atomic write (tmp + fsync + os.replace), same
    discipline as the candidate manifest: a half-written cache that still
    parses is worse than none."""
    _check_sha(dataset_sha)
    if not isinstance(rows, list) or not rows:
        raise CacheRefused("save_columnar needs a non-empty list of row dicts")
    n = len(rows)
    cols: dict[str, np.ndarray] = {
        "meta_dataset_sha": np.array(dataset_sha),
        "meta_cache_schema": np.array(schema),
        "meta_n": np.array(n, dtype=np.int64),
        "meta_extra_json": np.array(json.dumps(extra_meta or {},
                                               sort_keys=True)),
    }
    for f in F64_FIELDS:
        vals = np.zeros(n, dtype=np.float64)
        mask = np.zeros(n, dtype=np.uint8)
        for i, r in enumerate(rows):
            v = r.get(f)
            if v is None:
                continue
            vals[i] = _num(v, f"rows[{i}].{f}")
            mask[i] = 1
        cols[f"f64__{f}"] = vals
        cols[f"f64m__{f}"] = mask
    for f in I64_FIELDS:
        vals = np.zeros(n, dtype=np.int64)
        mask = np.zeros(n, dtype=np.uint8)
        for i, r in enumerate(rows):
            v = r.get(f)
            if v is None:
                continue
            if isinstance(v, bool) or not isinstance(v, int):
                raise CacheRefused(
                    f"non-int value {v!r} at rows[{i}].{f}: refused")
            vals[i] = v
            mask[i] = 1
        cols[f"i64__{f}"] = vals
        cols[f"i64m__{f}"] = mask
    for f in STR_FIELDS:
        raw = []
        mask = np.zeros(n, dtype=np.uint8)
        for i, r in enumerate(rows):
            v = r.get(f)
            if v is None:
                raw.append("")
                continue
            if not isinstance(v, str):
                raise CacheRefused(
                    f"non-str value {v!r} at rows[{i}].{f}: refused")
            raw.append(v)
            mask[i] = 1
        cols[f"str__{f}"] = np.array(raw, dtype=np.str_)
        cols[f"strm__{f}"] = mask
    lat_keys = sorted({L for r in rows for L in (r.get("latency") or {})})
    cols["meta_lat_keys"] = np.array(lat_keys, dtype=np.str_)
    for L in lat_keys:
        if not isinstance(L, str):
            raise CacheRefused(f"latency key {L!r} is not a str: refused")
        for f in LATENCY_FIELDS:
            vals = np.zeros(n, dtype=np.float64)
            mask = np.zeros(n, dtype=np.uint8)
            for i, r in enumerate(rows):
                sub = (r.get("latency") or {}).get(L)
                if sub is None or f not in sub:
                    continue
                vals[i] = _num(sub[f], f"rows[{i}].latency[{L!r}].{f}")
                mask[i] = 1
            cols[f"lat__{L}__{f}"] = vals
            cols[f"latm__{L}__{f}"] = mask
    p = cache_path(dataset_sha, cache_dir, schema)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.parent / f".{p.name}.tmp-{os.getpid()}.npz"
    try:
        np.savez_compressed(tmp, **cols)
        fd = os.open(tmp, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, p)
    finally:
        if tmp.exists():
            tmp.unlink()
    return p


def load_columnar(dataset_sha: str, cache_dir=None,
                  schema: str = CACHE_SCHEMA_VERSION) -> dict | None:
    """None on a MISS (no file for this key — caller parses the JSON and may
    save_columnar). CacheRefused on a file that EXISTS but fails any check:
    stored sha != requested sha, stored schema != requested schema, corrupt
    or truncated members, wrong lengths. A present-but-wrong cache is never
    served and never silently ignored."""
    _check_sha(dataset_sha)
    p = cache_path(dataset_sha, cache_dir, schema)
    if not p.exists():
        return None
    try:
        with np.load(p, allow_pickle=False) as z:
            cols = {k: np.array(z[k]) for k in z.files}   # materialise: CRC
    except Exception as e:
        raise CacheRefused(f"cache file {p} is unreadable/corrupt "
                           f"({type(e).__name__}: {e}); refusing — "
                           f"delete it and rebuild from the dataset") from e
    for m in ("meta_dataset_sha", "meta_cache_schema", "meta_n",
              "meta_lat_keys", "meta_extra_json"):
        if m not in cols:
            raise CacheRefused(f"cache file {p} lacks {m}; refusing")
    got_sha = str(cols["meta_dataset_sha"][()])
    got_schema = str(cols["meta_cache_schema"][()])
    if got_sha != dataset_sha:
        raise CacheRefused(
            f"cache file {p} is keyed to dataset sha {got_sha[:16]}.. but "
            f"{dataset_sha[:16]}.. was requested — STALE cache, refusing to "
            f"serve it")
    if got_schema != schema:
        raise CacheRefused(
            f"cache file {p} holds schema {got_schema!r}, caller expects "
            f"{schema!r}; refusing")
    n = int(cols["meta_n"][()])
    for name, arr in cols.items():
        if not name.startswith("meta_") and arr.shape != (n,):
            raise CacheRefused(
                f"cache column {name} has shape {arr.shape}, expected ({n},); "
                f"refusing a truncated/inconsistent cache")
    return {"columns": cols, "n": n, "dataset_sha": got_sha,
            "schema": got_schema,
            "extra": json.loads(str(cols["meta_extra_json"][()]))}


def rows_from_columns(cols: dict) -> list[dict]:
    """Reconstruct the row dicts (consumed fields only) from the columns."""
    n = int(cols["meta_n"][()])
    rows: list[dict] = [{} for _ in range(n)]
    for f in F64_FIELDS:
        vals = cols[f"f64__{f}"]; mask = cols[f"f64m__{f}"]
        for i in np.nonzero(mask)[0]:
            rows[i][f] = float(vals[i])
    for f in I64_FIELDS:
        vals = cols[f"i64__{f}"]; mask = cols[f"i64m__{f}"]
        for i in np.nonzero(mask)[0]:
            rows[i][f] = int(vals[i])
    for f in STR_FIELDS:
        vals = cols[f"str__{f}"]; mask = cols[f"strm__{f}"]
        for i in np.nonzero(mask)[0]:
            rows[i][f] = str(vals[i])
    for L in (str(x) for x in cols["meta_lat_keys"]):
        for f in LATENCY_FIELDS:
            vals = cols[f"lat__{L}__{f}"]; mask = cols[f"latm__{L}__{f}"]
            for i in np.nonzero(mask)[0]:
                rows[i].setdefault("latency", {}).setdefault(L, {})[f] = \
                    float(vals[i])
    return rows


# ------------------------------------------------------- equivalence harness
def _scores_logistic(w, X) -> np.ndarray:
    return fast_predict_p_batch(w, X)


def _scores_linear(w, X) -> np.ndarray:
    return np.asarray(X, dtype=np.float64) @ np.asarray(w, dtype=np.float64)


def _replicate_fixture(fx: dict, r: int):
    """Row 0 duplicated r-fold with its weight mass split 1/r per copy — the
    R-159 invariance construction. Requires sw: duplicating rows of an
    UNWEIGHTED fit legitimately changes it, so replication without weights is
    a harness-misuse refusal, not a soft skip."""
    if "sw" not in fx or fx["sw"] is None:
        raise ValueError(
            f"fixture {fx.get('name')!r} declares replicate= but has no sw; "
            f"replication invariance is a WEIGHTED-fit property")
    X = list(fx["X"]) + [fx["X"][0]] * (r - 1)
    y = list(fx["y"]) + [fx["y"][0]] * (r - 1)
    m = fx["sw"][0] / r
    sw = [m] + list(fx["sw"][1:]) + [m] * (r - 1)
    return X, y, sw


def equivalence_report(stdlib_fn: Callable, fast_fn: Callable,
                       fixtures: Sequence[dict]) -> dict:
    """Run both implementations over the fixtures; report raw differences and
    COMPUTED predicates (rule 10: no hardcoded verdict strings).

    Fixture dict: name, kind ("logistic"/"logistic_w"/"ridge"/"ridge_w"),
    X (rows), y, optional sw, optional kwargs, score_scale_cents (the factor
    that puts this fixture's scores at realistic receipt magnitudes: 1.0 for
    ridge fixtures already in cents; 100.0 for probabilities, a conservative
    bound on the |tox_mag| cents that multiply p in the decision score),
    optional replicate=(factors...) for the invariance arm.

    Per fixture: max abs/rel coefficient diff, max abs/rel score diff in
    cents, rint-same-cent predicate, the declared bar predicate, and — when
    replicate is present — per-implementation invariance errors at each
    factor plus the error-grows-with-factor bug-signature predicate."""
    out: dict[str, Any] = {
        "declared_pass_bar": {
            "max_abs_score_diff_cents": PASS_MAX_ABS_SCORE_CENTS,
            "rounds_to_same_cent": True,
            "replication_invariance_max_err": INV_BAR,
        },
        "fixtures": [], "all_pass": None,
    }
    for fx in fixtures:
        weighted = fx.get("sw") is not None
        args = [fx["X"], fx["y"]] + ([fx["sw"]] if weighted else [])
        kw = dict(fx.get("kwargs", {}))
        t0 = time.perf_counter(); w_std = stdlib_fn(*args, **kw)
        t_std = time.perf_counter() - t0
        t0 = time.perf_counter(); w_fast = fast_fn(*args, **kw)
        t_fast = time.perf_counter() - t0
        a = np.asarray(w_std, dtype=np.float64)
        b = np.asarray(w_fast, dtype=np.float64)
        coef_abs = float(np.max(np.abs(a - b)))
        coef_rel = float(np.max(np.abs(a - b) /
                                np.maximum(np.abs(a), 1e-12)))
        scorer = (_scores_logistic if fx["kind"].startswith("logistic")
                  else _scores_linear)
        scale = float(fx.get("score_scale_cents", 1.0))
        s_a = scorer(w_std, fx["X"]) * scale
        s_b = scorer(w_fast, fx["X"]) * scale
        d = np.abs(s_a - s_b)
        max_abs_score = float(np.max(d))
        max_rel_score = float(np.max(d / np.maximum(np.abs(s_a), 1e-9)))
        same_cent = bool(np.array_equal(np.rint(s_a), np.rint(s_b)))
        bar_ok = bool(max_abs_score < PASS_MAX_ABS_SCORE_CENTS)
        rec: dict[str, Any] = {
            "name": fx["name"], "kind": fx["kind"],
            "n": len(fx["X"]), "k": len(fx["X"][0]), "weighted": weighted,
            "t_stdlib_s": t_std, "t_fast_s": t_fast,
            "speedup": (t_std / t_fast) if t_fast > 0 else None,
            "coef_stdlib": [float(v) for v in a],
            "coef_fast": [float(v) for v in b],
            "max_abs_coef_diff": coef_abs, "max_rel_coef_diff": coef_rel,
            "score_scale_cents": scale, "n_scores": int(s_a.shape[0]),
            "max_abs_score_diff_cents": max_abs_score,
            "max_rel_score_diff": max_rel_score,
            "rounds_to_same_cent": same_cent, "score_bar_ok": bar_ok,
            "replication": None,
        }
        inv_ok_all = True
        factors = fx.get("replicate")
        if factors:
            factors = sorted(int(r) for r in factors)
            inv: dict[str, Any] = {"factors": factors}
            for label, fn, base in (("stdlib", stdlib_fn, a),
                                    ("fast", fast_fn, b)):
                errs: dict[str, float] = {}
                for r in factors:
                    Xr, yr, swr = _replicate_fixture(fx, r)
                    wr = np.asarray(fn(Xr, yr, swr, **kw), dtype=np.float64)
                    errs[str(r)] = float(np.max(np.abs(wr - base)))
                elist = [errs[str(r)] for r in factors]
                grows = bool(len(elist) >= 2
                             and elist[-1] > 3.0 * max(elist[0], 1e-300)
                             and elist[-1] > INV_MIN_SIGNAL)
                inv[label] = {
                    "err_by_factor": errs,
                    "inv_ok": bool(max(elist) <= INV_BAR),
                    "error_grows_with_factor": grows,
                }
                inv_ok_all = inv_ok_all and inv[label]["inv_ok"]
            rec["replication"] = inv
        rec["pass"] = bool(bar_ok and same_cent and inv_ok_all)
        out["fixtures"].append(rec)
    out["all_pass"] = bool(all(f["pass"] for f in out["fixtures"]))
    return out


def default_fixtures(seed: int = 20260826) -> dict[str, list[dict]]:
    """Synthetic fixtures, sub-second each, spanning the declared conditions:
    well-conditioned, near-separable (non-separable ON PURPOSE — the model
    selftest documents that a cleanly separable fixture has no unique optimum
    and turns summation noise into huge coefficient noise), non-uniform
    generation weights, single-feature, and the 10x/50x replication arm.
    Plain-python floats so both implementations consume identical bits."""
    rng = random.Random(seed)

    def gauss_rows(n, kf):
        return [[1.0] + [rng.gauss(0.0, 1.0) for _ in range(kf)]
                for _ in range(n)]

    Xw = gauss_rows(400, 4)
    yw = [1 if (0.8 * x[1] - 0.5 * x[2] + 0.25 * x[3] - 0.2
                + rng.gauss(0.0, 1.2)) > 0 else 0 for x in Xw]
    Xn = gauss_rows(240, 2)
    yn = [1 if (x[1] + 0.6 * x[2]) > 0 else 0 for x in Xn]
    for i in rng.sample(range(240), 4):
        yn[i] = 1 - yn[i]
    Xs1 = [[1.0, rng.gauss(0.0, 1.0)] for _ in range(300)]
    ys1 = [1 if (1.3 * x[1] + rng.gauss(0.0, 1.0)) > 0 else 0 for x in Xs1]
    Xr = gauss_rows(400, 4)
    yr = [12.0 + 45.0 * x[1] - 30.0 * x[2] + 8.0 * x[3]
          + rng.gauss(0.0, 25.0) for x in Xr]
    Xr1 = [[1.0, rng.gauss(0.0, 1.0)] for _ in range(300)]
    yr1 = [-20.0 + 60.0 * x[1] + rng.gauss(0.0, 20.0) for x in Xr1]

    # generation-realistic non-uniform weights: sizes 1..8, per-gen latent
    Xg: list[list[float]] = []; ygl: list[int] = []
    ygr: list[float] = []; swg: list[float] = []
    sizes = (1, 3, 1, 8, 2, 5, 1, 4, 2, 6)
    gi = 0
    while len(Xg) < 520:
        gs = sizes[gi % len(sizes)]; gi += 1
        latent = rng.gauss(0.0, 0.7)
        for _ in range(gs):
            x = [1.0, rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0)]
            Xg.append(x)
            ygl.append(1 if (0.9 * x[1] - 0.6 * x[2] + latent
                             + rng.gauss(0.0, 1.2)) > 0 else 0)
            ygr.append(30.0 * x[1] - 18.0 * x[2] + 10.0 * latent
                       + rng.gauss(0.0, 25.0))
            swg.append(1.0 / gs)

    return {
        "logistic": [
            {"name": "log_wellcond", "kind": "logistic", "X": Xw, "y": yw,
             "score_scale_cents": 100.0},
            {"name": "log_nearsep", "kind": "logistic", "X": Xn, "y": yn,
             "score_scale_cents": 100.0},
            {"name": "log_single_feature", "kind": "logistic",
             "X": Xs1, "y": ys1, "score_scale_cents": 100.0},
        ],
        "logistic_w": [
            {"name": "logw_generation_weights", "kind": "logistic_w",
             "X": Xg, "y": ygl, "sw": swg, "score_scale_cents": 100.0,
             "replicate": (10, 50)},
        ],
        "ridge": [
            {"name": "ridge_wellcond_cents", "kind": "ridge",
             "X": Xr, "y": yr, "score_scale_cents": 1.0,
             "kwargs": {"lam": 10.0}},   # the builder's tox_mag lam
            {"name": "ridge_single_feature_cents", "kind": "ridge",
             "X": Xr1, "y": yr1, "score_scale_cents": 1.0},
        ],
        "ridge_w": [
            {"name": "ridgew_generation_weights_cents", "kind": "ridge_w",
             "X": Xg, "y": ygr, "sw": swg, "score_scale_cents": 1.0,
             "replicate": (10, 50)},
        ],
    }


def _import_model():
    """The REAL incumbent, not a copy — equivalence vs a transcription would
    gate nothing (rule 16: verify at the artifact the claim names)."""
    d = str(Path(__file__).resolve().parent)
    if d not in sys.path:
        sys.path.insert(0, d)
    import harmful_hazard_model as hm
    return hm


def run_default_equivalence(hm=None, seed: int = 20260826) -> dict:
    """The runnable shape of the future adoption gate: every stdlib fit vs
    its twin over the default fixtures. The full-scale gate feeds this same
    `equivalence_report` real-population fixtures instead."""
    hm = hm or _import_model()
    fams = default_fixtures(seed)
    pairs = {
        "fit_logistic": (hm.fit_logistic, fast_fit_logistic,
                         fams["logistic"]),
        "fit_logistic_w": (hm.fit_logistic_w, fast_fit_logistic_w,
                           fams["logistic_w"]),
        "fit_ridge": (hm.fit_ridge, fast_fit_ridge, fams["ridge"]),
        "fit_ridge_w": (hm.fit_ridge_w, fast_fit_ridge_w, fams["ridge_w"]),
    }
    src = Path(hm.__file__)
    out: dict[str, Any] = {
        "protocol": "HARMFUL_FAST_COMPUTE_EQUIVALENCE_V1",
        "model_file": str(src),
        "model_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
        "numpy_version": np.__version__,
        "fixture_seed": seed,
        "families": {}, "all_pass": None,
    }
    for name, (sf, ff, fxs) in pairs.items():
        out["families"][name] = equivalence_report(sf, ff, fxs)
    out["all_pass"] = bool(all(v["all_pass"]
                               for v in out["families"].values()))
    return out


# --------------------------------------------- deliberate falsifiers (rule 15)
# NEVER exported, never aliased, names say broken. A harness that has not
# proven it can flag a wrong implementation is not evidence of a right one.
def _broken_fit_logistic_w_sumsw(X, y, sw, lam=1e-3, it=120):
    """THE R-159 BUG, resurrected on purpose: penalty divisor sum(sw) instead
    of len(X). Net shrinkage becomes len(X)/sum(sw) — scales with rows-per-
    generation, and its replication-invariance error GROWS with the factor."""
    Xa = _as_matrix(X)
    ya = np.asarray(y, dtype=np.float64)
    swa = np.asarray(sw, dtype=np.float64)
    n, k = Xa.shape
    n_samp = float(np.sum(swa)) or 1.0            # the bug
    w = np.zeros(k, dtype=np.float64)
    diag = np.arange(k)
    for _ in range(it):
        g, H = _logit_accum(Xa, ya, swa, w)
        pen = (lam * w) / n_samp
        pen[0] = 0.0
        g -= pen * n
        H[diag, diag] += lam
        got = _fast_solve(H, g, w)
        if got is None:
            break
        w, done = got
        if done:
            break
    return w.tolist()


def _broken_fit_ridge_intercept_unpenalized(X, y, lam=1.0):
    """Wrong regularisation semantics: skips lam on diagonal index 0. The
    stdlib penalises the FULL diagonal; the harness must catch the drift."""
    Xa = _as_matrix(X)
    ya = np.asarray(y, dtype=np.float64)
    k = Xa.shape[1]
    H, g = _ridge_accum(Xa, ya, None)
    diag = np.arange(1, k)                        # the bug: index 0 exempt
    H[diag, diag] += lam
    r = _fast_solve(H, g, np.zeros(k, dtype=np.float64))
    return r[0].tolist() if r else [0.0] * k


def _broken_zscale_ddof1(train_X, all_X):
    """Sample sd (n-1) where the stdlib uses population sd (n)."""
    if len(train_X) == 0:
        raise ValueError("empty train set")
    T = _as_matrix(train_X)
    n, k = T.shape
    mu = T.sum(axis=0) / n
    sd = np.sqrt(((T - mu) ** 2).sum(axis=0) / (n - 1))   # the bug
    sd = np.where(sd == 0.0, 1.0, sd)
    A = _as_matrix(all_X)
    out = np.empty((A.shape[0], k + 1), dtype=np.float64)
    out[:, 0] = 1.0
    out[:, 1:] = (A - mu) / sd
    return out, mu.tolist(), sd.tolist()


# ------------------------------------------------------------------- selftest
def selftest(tmp_dir: str | None = None) -> int:
    checks = 0

    def ok(c, label):
        nonlocal checks
        if not c:
            raise AssertionError(label)
        checks += 1

    hm = _import_model()
    print(f"numpy {np.__version__}; comparing against {hm.__file__}")

    # ---- scalar predict_p is character-identical -> exact equality
    rngt = random.Random(3)
    for _ in range(5):
        w = [rngt.gauss(0, 2) for _ in range(4)]
        x = [1.0] + [rngt.gauss(0, 3) for _ in range(3)]
        ok(fast_predict_p(w, x) == hm.predict_p(w, x),
           "fast_predict_p is bit-identical to predict_p (same expression)")
    wbig = [40.0, 40.0]
    ok(fast_predict_p(wbig, [1.0, 1.0]) == hm.predict_p(wbig, [1.0, 1.0]),
       "the +/-30 clamp path is identical too")

    # ---- batch scoring vs the stdlib loop
    Xb = [[1.0, rngt.gauss(0, 1), rngt.gauss(0, 1)] for _ in range(200)]
    wb = [0.3, -1.1, 0.7]
    pb = fast_predict_p_batch(wb, Xb)
    ok(float(np.max(np.abs(pb - np.array([hm.predict_p(wb, x)
                                          for x in Xb])))) < 1e-12,
       "batch scoring matches the stdlib loop to <1e-12 (dot order only)")

    # ---- zscale twin: parity, sd==0 branch, refusal wording, falsifier
    Xz = [[rngt.gauss(0, 2), 2.0, rngt.gauss(5, 3)] for _ in range(60)]
    z_std, mu_s, sd_s = hm.zscale([list(r) for r in Xz],
                                  [list(r) for r in Xz])
    z_fast, mu_f, sd_f = fast_zscale(Xz, Xz)
    ok(z_fast.shape == (60, 4)
       and float(np.max(np.abs(z_fast - np.asarray(z_std)))) < 1e-12,
       "fast_zscale matches zscale to <1e-12 incl the intercept column")
    ok(sd_s[1] == 1.0 and sd_f[1] == 1.0,
       "a constant column hits the sd==0 -> 1.0 branch in BOTH")
    for fn in (hm.zscale, fast_zscale):
        try:
            fn([], [[1.0]])
            ok(False, "zscale must refuse an empty training set")
        except ValueError as e:
            ok("symptom, not a cause" in str(e),
               "both refuse an empty train set with the SAME named message")
    zb, _, _ = _broken_zscale_ddof1(Xz, Xz)
    ok(float(np.max(np.abs(zb - np.asarray(z_std)))) > 1e-6,
       "FALSIFIER: the ddof=1 zscale variant exceeds tolerance — the parity "
       "check can actually fire")

    # ---- generation_weights parity (incl missing keys)
    kept = [{"slug": "s", "side": "B", "gen": 1},
            {"slug": "s", "side": "B", "gen": 1},
            {"slug": "s", "side": "B", "gen": 2},
            {"slug": "t", "side": "S"},           # gen absent -> None key
            {"slug": "t", "side": "S"}]
    ok(fast_generation_weights(kept) == hm.generation_weights(kept)
       == [0.5, 0.5, 1.0, 0.5, 0.5],
       "generation_weights twin matches the real one exactly")

    # ---- singular-system refusal parity (the solver's refuse branch fires
    # identically: all-zero column + lam=0 -> exactly zero pivot in both)
    Xsing = [[1.0, 0.0, rngt.gauss(0, 1)] for _ in range(50)]
    ysing = [rngt.gauss(0, 1) for _ in range(50)]
    ok(hm.fit_ridge(Xsing, ysing, lam=0.0) == [0.0] * 3
       and fast_fit_ridge(Xsing, ysing, lam=0.0) == [0.0] * 3,
       "both ridge paths refuse an exactly singular system with [0.0]*k")
    yb01 = [1 if rngt.random() > 0.5 else 0 for _ in range(50)]
    ok(hm.fit_logistic(Xsing, yb01, lam=0.0) == [0.0] * 3
       and fast_fit_logistic(Xsing, yb01, lam=0.0) == [0.0] * 3,
       "both logistic paths break on the singular Hessian and return zeros")

    # ---- THE equivalence run: every stdlib fit vs its twin
    rep = run_default_equivalence(hm)
    nfx = sum(len(v["fixtures"]) for v in rep["families"].values())
    ok(nfx >= 7, "fixture battery covers all declared conditions")
    for fam, v in rep["families"].items():
        for fx in v["fixtures"]:
            ok(fx["pass"] is True,
               f"KNOWN-GOOD: {fam}/{fx['name']} — max|dscore| "
               f"{fx['max_abs_score_diff_cents']:.3e}c under the declared bar")
    ok(rep["all_pass"] is True, "twin implementations pass the declared bar "
                                "on every fixture")
    fams = default_fixtures()
    lw = [f for f in rep["families"]["fit_logistic_w"]["fixtures"]
          if f["replication"]][0]
    ok(lw["replication"]["fast"]["inv_ok"] is True
       and lw["replication"]["fast"]["error_grows_with_factor"] is False,
       "correct twin: replication invariance holds at 10x and 50x")

    # ---- FALSIFIER A: the resurrected R-159 divisor bug must be flagged
    repA = equivalence_report(hm.fit_logistic_w, _broken_fit_logistic_w_sumsw,
                              fams["logistic_w"])
    ok(repA["all_pass"] is False,
       "POSITIVE CONTROL: the harness FLAGS the sum(sw) penalty divisor")
    fxA = repA["fixtures"][0]
    ok(fxA["max_abs_score_diff_cents"] > PASS_MAX_ABS_SCORE_CENTS,
       "the divisor bug exceeds the declared score bar")
    ok(fxA["replication"]["fast"]["error_grows_with_factor"] is True,
       "and its invariance error GROWS with the replication factor — the "
       "bug signature the 10x/50x arm exists to catch")
    ok(fxA["replication"]["stdlib"]["inv_ok"] is True,
       "while the stdlib reference stays invariant on the same fixture")

    # ---- FALSIFIER B: wrong ridge regularisation must be flagged
    repB = equivalence_report(hm.fit_ridge,
                              _broken_fit_ridge_intercept_unpenalized,
                              fams["ridge"])
    ok(repB["all_pass"] is False,
       "POSITIVE CONTROL: the harness FLAGS the unpenalised-intercept ridge")

    # ---- harness misuse: replicate without weights is a refusal
    try:
        equivalence_report(hm.fit_logistic, fast_fit_logistic,
                           [{"name": "bad", "kind": "logistic",
                             "X": [[1.0, 0.5]] * 4, "y": [0, 1, 0, 1],
                             "replicate": (10,)}])
        ok(False, "replicate without sw must refuse")
    except ValueError as e:
        ok("WEIGHTED-fit property" in str(e),
           "replicate on an unweighted fixture refuses with a named error")

    # ---- columnar cache: round-trip, then every refusal arm
    import tempfile
    td = Path(tempfile.mkdtemp(prefix="hfc_selftest_", dir=tmp_dir))
    rows = [
        {"slug": "btc-x-1", "day": "2026-08-24", "coin": "btc",
         "side": "BUY_UP", "status": "OK", "gen": 3, "t0": 1787579400.0,
         "t_start": 12.25, "level": 0.51, "resting": 5.0, "qahead": 120.0,
         "v_cancel_cents": -3.5,
         "latency": {"50": {"preventable_shares": 2.0,
                            "preventable_value_cents": 7.25,
                            "stale_shares": 0.0},
                     "5": {"preventable_shares": 0.0,
                           "preventable_value_cents": 0.0,
                           "stale_shares": 1.0}}},
        {"slug": "eth-y-2", "day": "2026-08-25", "coin": "eth",
         "side": "SELL_UP", "status": "GAP", "t0": 1787579700.5,
         "t_start": 0.0, "level": 0.4875, "resting": 2.0, "qahead": 0.0},
        {"slug": "min-3"},                        # nearly everything absent
    ]
    sha_a = "ab" * 32
    p = save_columnar(rows, sha_a, cache_dir=td,
                      extra_meta={"dataset_schema": "toy_v1"})
    got = load_columnar(sha_a, cache_dir=td)
    ok(got is not None and got["n"] == 3
       and got["extra"] == {"dataset_schema": "toy_v1"},
       "positive control: a well-keyed cache LOADS (the refuser can pass)")
    back = rows_from_columns(got["columns"])
    ok(back == rows,
       "round-trip is EXACT on the consumed fields (masks preserve absence)")
    ok(load_columnar("cd" * 32, cache_dir=td) is None,
       "an unknown key is a MISS (None), not an error")
    # stale key: file renamed to another sha's slot must be refused
    p_stale = cache_path("cd" * 32, cache_dir=td)
    p_stale.write_bytes(p.read_bytes())
    try:
        load_columnar("cd" * 32, cache_dir=td)
        ok(False, "stale cache must refuse")
    except CacheRefused as e:
        ok("STALE" in str(e), "a mis-keyed cache file REFUSES, never serves")
    # schema mismatch: same bytes presented under a different schema name
    p_schema = cache_path(sha_a, cache_dir=td, schema="harmful_rows_vNEXT")
    p_schema.write_bytes(p.read_bytes())
    try:
        load_columnar(sha_a, cache_dir=td, schema="harmful_rows_vNEXT")
        ok(False, "schema mismatch must refuse")
    except CacheRefused as e:
        ok("schema" in str(e), "a schema mismatch REFUSES")
    # corruption: flip one byte mid-file -> refused, not partially served
    blob = bytearray(p.read_bytes())
    blob[len(blob) * 3 // 5] ^= 0xFF
    p.write_bytes(bytes(blob))
    try:
        load_columnar(sha_a, cache_dir=td)
        ok(False, "corrupt cache must refuse")
    except CacheRefused:
        ok(True, "a corrupted cache file REFUSES on load")
    try:
        save_columnar(rows, "not-a-sha", cache_dir=td)
        ok(False, "malformed key must refuse")
    except CacheRefused:
        ok(True, "save refuses a malformed dataset sha")
    try:
        save_columnar([{"slug": "x", "level": "high"}], sha_a, cache_dir=td)
        ok(False, "non-numeric numeric field must refuse")
    except CacheRefused:
        ok(True, "save refuses a row the layout cannot hold losslessly")

    print(f"harmful_fast_compute selftest: {checks} checks OK")
    return 0


# ---------------------------------------------------------------------- bench
def bench_cmd(tmp_dir: str | None = None, n: int = 100_000,
              seed: int = 20260827) -> int:
    """Synthetic-scale timing: stdlib vs twin on a 100k-row fixture, plus the
    JSON-parse vs columnar-cache path. Prints measurements and the same
    equivalence metrics at this scale; no verdicts beyond computed predicates.
    Synthetic data only — the 1.24 GB dataset is NOT read here."""
    hm = _import_model()
    rng = random.Random(seed)
    print(f"bench n={n} numpy={np.__version__}")

    def _one(label, kind, sf, ff, X, y, sw=None, kwargs=None, scale=1.0):
        kwargs = kwargs or {}
        args = [X, y] + ([sw] if sw is not None else [])
        t0 = time.perf_counter(); w_s = sf(*args, **kwargs)
        ts = time.perf_counter() - t0
        tf = math.inf
        for _ in range(3):
            t0 = time.perf_counter(); w_f = ff(*args, **kwargs)
            tf = min(tf, time.perf_counter() - t0)
        a = np.asarray(w_s); b = np.asarray(w_f)
        scorer = _scores_logistic if kind == "logistic" else _scores_linear
        ds = np.abs(scorer(w_s, X) * scale - scorer(w_f, X) * scale)
        mx = float(np.max(ds))
        print(f"  {label:<28} stdlib {ts:8.2f}s  fast {tf:8.4f}s  "
              f"speedup {ts / tf:9.1f}x  max|dcoef| "
              f"{float(np.max(np.abs(a - b))):.3e}  max|dscore| {mx:.3e}c  "
              f"bar_ok={mx < PASS_MAX_ABS_SCORE_CENTS}  "
              f"same_cent={bool(np.array_equal(np.rint(scorer(w_s, X) * scale), np.rint(scorer(w_f, X) * scale)))}")
        return ts, tf

    X5 = [[1.0] + [rng.gauss(0.0, 1.0) for _ in range(4)] for _ in range(n)]
    y5 = [1 if (0.8 * x[1] - 0.5 * x[2] + 0.25 * x[3] - 0.2
                + rng.gauss(0.0, 1.2)) > 0 else 0 for x in X5]
    _one("fit_logistic 100k x 5", "logistic", hm.fit_logistic,
         fast_fit_logistic, X5, y5, scale=100.0)

    sw = []
    sizes = (1, 3, 1, 8, 2, 5, 1, 4, 2, 6)
    gi = 0
    while len(sw) < n:
        gs = sizes[gi % len(sizes)]; gi += 1
        sw.extend([1.0 / gs] * gs)
    sw = sw[:n]
    _one("fit_logistic_w 100k x 5", "logistic", hm.fit_logistic_w,
         fast_fit_logistic_w, X5, y5, sw=sw, scale=100.0)

    X8 = [[1.0] + [rng.gauss(0.0, 1.0) for _ in range(7)] for _ in range(n)]
    y8 = [12.0 + 45.0 * x[1] - 30.0 * x[2] + 8.0 * x[3] - 20.0 * x[5]
          + rng.gauss(0.0, 25.0) for x in X8]
    _one("fit_ridge 100k x 8", "ridge", hm.fit_ridge, fast_fit_ridge,
         X8, y8, kwargs={"lam": 10.0})

    # builder-width point: the real design matrices are ~55 columns (54 PM
    # features + intercept; 61+1 with the fine arm), and the stdlib inner
    # loop is O(n*k^2) INTERPRETED — narrow-k rows understate the gap. 10k
    # rows keeps the stdlib run in seconds at this width.
    kb = 55
    nb = 10_000
    Xb55 = [[1.0] + [rng.gauss(0.0, 1.0) for _ in range(kb - 1)]
            for _ in range(nb)]
    yb55 = [1 if (0.8 * x[1] - 0.5 * x[2] + 0.25 * x[3] - 0.2
                  + rng.gauss(0.0, 1.2)) > 0 else 0 for x in Xb55]
    _one(f"fit_logistic 10k x {kb} (builder-width)", "logistic",
         hm.fit_logistic, fast_fit_logistic, Xb55, yb55, scale=100.0)

    # cache path vs JSON parse, synthetic rows shaped like the dataset's
    import tempfile
    td = Path(tempfile.mkdtemp(prefix="hfc_bench_", dir=tmp_dir))
    rows = []
    for i in range(n):
        rows.append({
            "slug": f"synthetic-window-{i // 200}", "day": "2026-08-24",
            "coin": "btc" if i % 2 else "eth",
            "side": "BUY_UP" if i % 3 else "SELL_UP", "status": "OK",
            "gen": i // 7, "t0": 1787579334.0 + i, "t_start": (i % 300) * 1.0,
            "level": 0.5, "resting": 5.0, "qahead": float(i % 40),
            "v_cancel_cents": rng.gauss(0.0, 4.0),
            "latency": {"50": {"preventable_shares": 1.0,
                               "preventable_value_cents": rng.gauss(0.0, 6.0),
                               "stale_shares": 0.0},
                        "5": {"preventable_shares": 0.0,
                              "preventable_value_cents": 0.0,
                              "stale_shares": 0.0}}})
    blob = json.dumps({"schema": "bench", "rows": rows})
    t0 = time.perf_counter(); json.loads(blob)
    t_json = time.perf_counter() - t0
    sha = hashlib.sha256(blob.encode()).hexdigest()
    t0 = time.perf_counter(); p = save_columnar(rows, sha, cache_dir=td)
    t_save = time.perf_counter() - t0
    t0 = time.perf_counter(); got = load_columnar(sha, cache_dir=td)
    t_load = time.perf_counter() - t0
    t0 = time.perf_counter(); back = rows_from_columns(got["columns"])
    t_rows = time.perf_counter() - t0
    print(f"  cache: json.loads {t_json:6.2f}s ({len(blob) / 1e6:.0f} MB)  "
          f"save {t_save:6.2f}s  load(columns) {t_load:6.2f}s  "
          f"+rows_from_columns {t_rows:6.2f}s  npz "
          f"{p.stat().st_size / 1e6:.0f} MB  roundtrip_exact={back == rows}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--equivalence", action="store_true",
                    help="print the full stdlib-vs-fast equivalence report")
    ap.add_argument("--bench", action="store_true",
                    help="time stdlib vs fast fits on a 100k-row synthetic "
                         "fixture (seconds; the 1.24 GB dataset is not read)")
    ap.add_argument("--tmp", default=None,
                    help="scratch dir for cache selftest/bench artifacts")
    a = ap.parse_args()
    if a.selftest:
        return selftest(a.tmp)
    if a.equivalence:
        print(json.dumps(run_default_equivalence(), indent=2))
        return 0
    if a.bench:
        return bench_cmd(a.tmp)
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
