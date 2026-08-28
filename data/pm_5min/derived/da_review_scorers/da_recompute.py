"""DA INDEPENDENT RECOMPUTATION of Phase-2 net_cents (R-219 charter item 1).

INDEPENDENCE BOUNDARY, stated rather than implied:
  SHARED (data layer, and my own gate certified the tape):
      harmful_hazard_model.window_streams/features/fine_feats  -> PM, FN
      phase2_state_schema_freeze.encode_row                    -> ST
      the committed fit artifacts (weights, boosters, thresholds)
      the certified tape and the topup exposure rows
  MINE (everything that turns those into the NUMBER):
      the tape join and population construction
      normalisation and design-matrix assembly
      model application (logistic / LGBM)
      action de-duplication, first-crossing, threshold application
      harm / sacrifice / net accounting

I do NOT import phase2_arms or harmful_action_eval.
"""
import sys, json, math, collections
sys.path.insert(0, "/home/yuqing/ctaNew/live/pm_research")
from pathlib import Path
import numpy as np
import da_state_tape_verify as G          # my own streaming reader
import harmful_hazard_model as hm
import phase2_state_schema_freeze as PIN

# R-230(3) applied to DA's own stack: a result must attest the BYTES THAT
# ACTUALLY LOADED, not the module names it meant to import. Without this the
# independence claim is "it imports the shared feature builders" with no way to
# say WHICH copy -- a wrong-tree module would have produced a confident 15/15.
_TREE = Path("/home/yuqing/ctaNew/live/pm_research").resolve()
def _runtime_identity():
    import hashlib
    out = {}
    for name, mod in (("da_state_tape_verify", G), ("harmful_hazard_model", hm),
                      ("phase2_state_schema_freeze", PIN)):
        f = getattr(mod, "__file__", None)
        rp = Path(f).resolve() if f else None
        rec = {"loaded_from": str(rp) if rp else None,
               "under_expected_tree": bool(rp and str(rp).startswith(str(_TREE))),
               "sha256": hashlib.sha256(rp.read_bytes()).hexdigest() if rp and rp.exists() else None}
        c = getattr(mod, "__cached__", None)
        if c and Path(c).exists():
            rec["pyc_sha256"] = hashlib.sha256(Path(c).read_bytes()).hexdigest()[:16]
        out[name] = rec
        if not rec["under_expected_tree"]:
            raise SystemExit(
                f"REFUSED: {name} loaded from {rec['loaded_from']}, outside "
                f"{_TREE}. A verification run must not certify numbers produced "
                f"by feature code from a tree it cannot name.")
    return out
RUNTIME_IDENTITY = _runtime_identity()

D = Path("/home/yuqing/ctaNew/data/pm_5min/derived")
TAPE = D / "phase2_state_tape_v5.json"
TOPUP = D / "harmful_exposure_rows_v3_topup.json"
FIT = D / "phase2_fits"
COIN = "btc"; L = "50"; LAT_MS = 50

feats_order = PIN.build_pin()["features_in_order"]

# ---- 1. my own tape index for the SCORE split ---------------------------
print("indexing tape (score split)...", flush=True)
TP = {}
for r in G.iter_tape(TAPE):
    if r.get("split") != "score":
        continue
    st = str(r.get("state_status", "OK"))
    TP[(r["slug"], r["side"], r["gen"], r["t_start"])] = (
        tuple(PIN.encode_row(r.get("state") or {}, feats_order)) if st == "OK" else None,
        st)
print(f"  score rows indexed: {len(TP):,}", flush=True)

# ---- 2. my own population construction ----------------------------------
paths = hm.fi._archive_paths(); tokens = hm.fi.token_map()
bywin = collections.defaultdict(list)
n_topup = 0
for r in G.iter_tape(TOPUP):
    if r.get("coin") != COIN or r.get("status") != "OK":
        continue
    n_topup += 1
    bywin[r["slug"]].append(r)
print(f"  topup {COIN} OK rows: {n_topup:,} in {len(bywin):,} slugs", flush=True)

drops = collections.Counter()
PMl = []; FNl = []; STl = []; kept = []
for slug, wrows in bywin.items():
    if slug not in paths or slug not in tokens:
        drops["no_archive"] += len(wrows); continue
    up, dn = tokens[slug]
    stream = hm.window_streams(paths[slug], up, dn)
    for r in wrows:
        e = TP.get((r["slug"], r["side"], r["gen"], r["t_start"]))
        if e is None:
            drops["state_join_failed"] += 1; continue
        vec, st = e
        if st != "OK":
            drops[f"{st.lower()}_excluded"] += 1; continue
        fp = hm.features(stream, r["t_start"], r["side"], r["level"],
                         r["resting"], r["qahead"])
        if fp is None:
            drops["pm"] += 1; continue
        ff = hm.fine_feats(r["t0"] + r["t_start"], r["side"], COIN)
        if ff is None:
            drops["fine"] += 1; continue
        PMl.append(fp); FNl.append(ff); STl.append(list(vec))
        kept.append({k: r.get(k) for k in
                     ("slug", "t0", "t_start", "side", "gen", "latency")})
print(f"  kept rows: {len(kept):,}  drops: {dict(drops)}", flush=True)
nA = len({(r["slug"], r["side"], r["gen"]) for r in kept})
print(f"  actions: {nA:,}   rows_per_action: {len(kept)/nA:.10f}", flush=True)

# ---- 3. my own design matrix -------------------------------------------
lin = json.loads((FIT / f"linear_{COIN}.json").read_text())
mu = np.asarray(lin["norm_mu"], dtype=np.float64)
sd = np.asarray(lin["norm_sd"], dtype=np.float64)
X = np.empty((len(kept), len(mu) + 1), dtype=np.float64)
X[:, 0] = 1.0
for j in range(len(kept)):
    X[j, 1:] = (np.asarray(PMl[j] + FNl[j] + STl[j], dtype=np.float64) - mu) / sd
del PMl, FNl, STl
print(f"  design matrix: {X.shape}", flush=True)

# ---- 4. my own model application ---------------------------------------
def logistic(z): return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))

scores = {}
W = np.asarray(lin["hazard_weights"], dtype=np.float64)
WM = np.asarray(lin["value_weights"], dtype=np.float64)
scores["PLUS_PRED_STATE_V1"] = logistic(X @ W) * (X @ WM)

import lightgbm as lgb
hb = lgb.Booster(model_file=str(FIT / f"lgbm_haz_{COIN}.txt"))
vb = lgb.Booster(model_file=str(FIT / f"lgbm_val_{COIN}.txt"))
scores["LGBM_PINNED"] = np.asarray(hb.predict(X)) * np.asarray(vb.predict(X))

# ---- 5. my own evaluation ----------------------------------------------
def val(r):
    lat = r.get("latency") or {}
    afa = (lat.get(L, {}).get("preventable_shares", 0.0) > 0
           or any(v.get("preventable_shares", 0.0) > 0
                  or v.get("stale_shares", 0.0) > 0 for v in lat.values()))
    return lat[L]["preventable_value_cents"] if (afa and L in lat) else 0.0

gens = collections.defaultdict(list)
for i, r in enumerate(kept):
    gens[(r["slug"], r["side"], r["gen"])].append(i)
for k in gens:
    gens[k].sort(key=lambda i: kept[i]["t_start"])

THR = {"PLUS_PRED_STATE_V1": json.loads((FIT / f"linear_{COIN}.json").read_text())["causal_thresholds"],
       "LGBM_PINNED": json.loads((FIT / f"lgbm_thresholds_{COIN}.json").read_text())}

out = {}
for arm, sc in scores.items():
    gmax = {k: max(sc[i] for i in gens[k]) for k in gens}
    res = {}
    for bkey, theta in sorted(THR[arm].items()):
        theta = float(theta)
        cancelled = [k for k in gens if gmax[k] >= theta]
        net = harm = sacr = 0.0
        for gk in cancelled:
            cross = next(i for i in gens[gk] if sc[i] >= theta)
            v = val(kept[cross])
            net += v
            if v > 0: harm += v
            else: sacr += -v
        res[bkey] = {"n_cancelled": len(cancelled), "net_cents": net,
                     "harm_avoided_cents": harm, "sacrifice_cents": sacr,
                     "rho": (harm / sacr) if sacr else None}
    out[arm] = res

print("\n=== DA INDEPENDENT RECOMPUTATION ===")
print(json.dumps({"coin": COIN, "runtime_identity": RUNTIME_IDENTITY,
                  "n_rows": len(kept), "n_actions": nA,
                  "drops": dict(drops), "arms": out}, indent=1, sort_keys=True))
