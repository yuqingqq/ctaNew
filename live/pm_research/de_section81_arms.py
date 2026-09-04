"""§8.1 integration ablation -- the runnable arms, on the real
QR_SKEW_ONLY opportunity population, with real predictors.

Landed from the scratch harness that produced Q-DE-70 so the
numbers have a committed producer. The archive-root injection at
the top is the two-root property (round 45): `phase2_arms.DERIVED`
is hardcoded to the fit tree and `flow_intensity.PM` is
`__file__`-relative, so a run from a worktree must point them at
one tree or select zero windows.
"""
"""§8.1 arms on the real population with REAL PREDICTORS and LEGIBLE ARM
IDENTITY.

Round 52's run was refused as arm results: arms 2 and 5 were the same
computation, and every arm used `_pricing_scorer` -- a declared synthetic.
Here each arm records WHICH predictor it loaded, by artifact and sha, so
"is this arm real or a stub" is answered by evidence in the emission
rather than by trusting the name."""
import json, resource, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import flow_intensity as fi
fi.PM = Path("/home/yuqing/ctaNew/data/pm_5min"); fi.RAW = fi.PM / "raw"
fi.MARKETS = fi.PM / "markets.jsonl"; fi.GAPS = fi.PM / "collector_gaps.jsonl"
fi.DAYS = fi._discover_days()
import de_phase4_diag_runner as R
import de_lane4_real_parity as L
import harmful_stateful_policy as HSP
import de_rho_estimator as RHO
import de_matched_random_control as MRC
import de_score_stream as SS
import harmful_hazard_model as hm

def gb(): return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20, 2)
COIN = "btc"; LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 12
BUDGET, LAT, NO_CANCEL_THETA = 0.10, 250, 2.0
T0 = time.time()

import pickle
CACHE = Path(f"{Path(__file__).parent}/arms53_cache_{LIMIT}.pkl")
if CACHE.exists():
    _c = pickle.loads(CACHE.read_bytes())
    fr, asm = _c["fr"], _c["asm"]
    print(json.dumps({"cache": "HIT", "path": str(CACHE)}), flush=True)
else:
    _c = None
fr = fr if _c else R.build_reference(COIN, limit=LIMIT)
ref = fr["reference"]
POP = {"coin": COIN, "windows": len(ref),
       "generations": sum(len(s[x]) for s in ref.values() for x in HSP.SIDES),
       "statuses": fr["statuses"], "feed_wall_s": round(time.time()-T0, 1),
       "as_of": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
print(json.dumps({"population": POP}), flush=True)

# ---- REAL head scores for THIS population -------------------------------
splits = R.DECLARED_SPLIT_SETS[R.RULED_SPLIT_SET]
if _c is None:
  t = time.time(); tape = R.build_tape_index(splits)
  print(json.dumps({"tape_index_s": round(time.time()-t,1),
                    "rows": tape["n_tape_rows"], "peak_gb": gb()}), flush=True)
  frag = Path(f"{Path(__file__).parent}/arms53_frag.json")
  R.fragment_slice(frag, n_windows=LIMIT, only_slugs=list(ref))
  t = time.time()
  asm = R.assemble_streaming({COIN: ref}, splits=splits, coins=(COIN,),
                             chunk_windows=6, source=frag, tape=tape)
  print(json.dumps({"assembly_s": round(time.time()-t,1),
                    "kept": asm["assembly"]["kept_by_coin"],
                    "n_scores": {f"{k[1]}": len(v[0])
                                 for k, v in asm["by_arm"].items()},
                    "peak_gb": gb()}), flush=True)
  CACHE.write_bytes(pickle.dumps({"fr": fr, "asm": asm}))

def stream(head):
    """The event stream for one head, through the manifest-bound adapter.

    THE POPULATION IS FILTERED TO GENERATIONS THE ASSEMBLY SCORED. The
    scorer refuses a generation whose rows the feature pass dropped --
    correctly, since scoring it would be scoring from nothing -- so the
    exclusion happens HERE, before scoring, and is COUNTED (rule 4)."""
    gen_scores = asm["by_arm"][(COIN, head)][0]
    rows, dropped = [], 0
    for s_, sides in sorted(ref.items()):
        for sd in HSP.SIDES:
            for g in sides[sd]:
                if (s_, sd, float(g["t0"])) in gen_scores:
                    rows.append({"t": g["t0"], "slug": s_, "side": sd,
                                 "gen": g["gen"]})
                else:
                    dropped += 1
    EXCL[head] = {"scored_generations": len(rows),
                  "excluded_no_assembled_score": dropped,
                  "reference_generations": len(rows) + dropped,
                  "excluded_fraction": round(dropped/max(len(rows)+dropped,1), 4),
                  "reason": "the feature pass dropped every row of these "
                            "generations; scoring them would be scoring "
                            "from nothing (rule 4: counted, never dropped)"}
    v = SS.verify_head(head, COIN)
    return SS.score_events(rows, head=head, coin=COIN,
                           scorer=R._head_scorer(head, COIN, gen_scores),
                           verified=v), v

EXCL = {}


def ident(head, verified):
    """ARM IDENTITY, as evidence rather than a name: which predictor was
    loaded, by artifact and sha."""
    return {"predictor": head, "artifacts": verified,
            "n_artifacts": len(verified),
            "source": "de_score_stream.verify_head -> the fit manifest"}

def fields(res, rho_out):
    e, c = res["economics"], res["counters"]
    inv = e.get("inventory", {}) or {}
    out = {}
    for f, spec in L.SECTION_8_1_FIELDS.items():
        src = spec.get("source")
        if src is None:
            out[f] = {"status": "NOT_AVAILABLE", "reason": spec["why"]}
        elif src == "de_rho_estimator.rho":
            out[f] = {"status": "OK", "value": rho_out["rho"],
                      "statuses": rho_out["statuses"]}
        elif src.startswith("the caller"):
            out[f] = {"status": "NOT_RUN_THIS_ROUND",
                      "reason": "needs the 9-rung latency axis; one rung run"}
        else:
            leaf = src.split(".")[-1]
            # ROUND 52 LOOKED IN THE WRONG CONTAINER for four fields.
            # Search all three and RECORD WHICH ONE ANSWERED, so a value
            # carries where it came from.
            for cname, blk in (("holds", res.get("holds") or {}),
                               ("inventory", res.get("inventory") or {}),
                               ("economics.inventory", inv),
                               ("counters", c), ("economics", e),
                               ("result", res)):
                if leaf in blk:
                    out[f] = {"status": "OK", "value": blk[leaf],
                              "found_in": cname}
                    break
            else:
                out[f] = {"status": "MISSING_AT_RUNTIME",
                          "reason": f"{leaf!r} absent from economics, "
                                    f"counters, economics.inventory and "
                                    f"the result",
                          "searched": ["economics.inventory", "counters",
                                       "economics", "result"]}
    return out

def replay(scores, *, cancel, theta):
    params = R.cell_params(
        {"coin": COIN, "latency_ms": LAT, "budget": BUDGET,
         "enable_reduce": False,
         "charge_reset_cost_at_generation_start": False},
        theta_cancel=(theta if cancel else NO_CANCEL_THETA),
        protection_mode=HSP.PROTECTION_MODES[0],
        repost_fill_model=HSP.REPOST_FILL_MODELS[0])
    res = HSP.replay_policy(ref, scores, params)
    f = R.received_fills(res, ref, R._decision_times(scores))
    return res, RHO.rho(f, LAT, proxy={"rho_captured_over_sacrificed": None})

OUT = {"population": POP, "arms": {}, "spec": "§8.1", "coin": COIN,
       "latency_ms": LAT, "budget": BUDGET,
       "evidence_class": "DEVELOPMENT_EVIDENCE_NOT_A_VALIDATION"}

CV = "q1_arrival_composed_lgbm"; HZ = "incumbent_linear_d"
cv_ev, cv_v = stream(CV)
hz_ev, hz_v = stream(HZ)
theta_cv = R.theta_for(COIN, CV, BUDGET)
theta_hz = R.theta_for(COIN, HZ, BUDGET)
_mx = max((float(e["score"]) for e in cv_ev), default=0.0)
assert _mx < NO_CANCEL_THETA, f"arm 1 would cancel: max score {_mx}"

def add(name, res, rho_out, identity, **extra):
    OUT["arms"][name] = dict(
        {"arm": name, "n_cancels": res["counters"].get("cancels_issued", 0),
         "identity": identity, "fields": fields(res, rho_out)}, **extra)
    print(json.dumps({"arm": name,
                      "n_cancels": res["counters"].get("cancels_issued", 0),
                      "predictor": identity["predictor"],
                      "rho": rho_out["rho"]}), flush=True)

r1, h1 = replay(cv_ev, cancel=False, theta=theta_cv)
add("QR_SKEW_ONLY", r1, h1,
    {"predictor": "NONE", "artifacts": {},
     "note": "the neutral opportunity population; no cancellation, "
             "asserted n_cancels==0"})
assert OUT["arms"]["QR_SKEW_ONLY"]["n_cancels"] == 0

r5, h5 = replay(cv_ev, cancel=True, theta=theta_cv)
add("CONDVALUE_X_SKEW", r5, h5, ident(CV, cv_v),
    semantics=L.ARM_X_SKEW_SEMANTICS, theta=theta_cv)

rH, hH = replay(hz_ev, cancel=True, theta=theta_hz)
add("HAZARD_OVER_SKEWED_REF", rH, hH, ident(HZ, hz_v),
    note="NOT §8.1 arm 3 -- arm 3 requires NEUTRAL placement, which is "
         "ABSENT. This is the hazard head over the SKEWED reference and "
         "is named so it cannot be read as arm 3", theta=theta_hz)

gidx = R._gen_index(ref)
# THE DEMAND IS OVER ABOVE-THRESHOLD EVENTS, NEVER OVER REALISED ACTIONS.
# I built it from CANCEL_ISSUED trajectory entries -- i.e. actions -- and
# that is EXACTLY round 37's defect, which DE37-R2 closed inside
# `run_cell` and marked as the falsifier's input (`_known_bad_demand`,
# "no run passes it"). The guard protects `run_cell`'s path; this script
# bypasses `run_cell` and so reintroduced the known-bad from outside it.
#
# WHY IT BREAKS P2 AND P3, in the code's own words: "Every action is an
# above event and not conversely -- the policy is stateful, a HELD side
# suppresses later crossings -- so in any stratum with a non-acting above
# event the demand was too small, `permuted_stream` returned ok=False
# with a TRUNCATED-ZIP stream (above values dropped, below values
# duplicated)". Dropped above values are why the stratum score multisets
# differ (P2) and why the drawn set does not carry exactly the above
# values (P3).
treated = [{"slug": f"{e['slug']}|{e['side']}|{e['gen']}"}
           for e in cv_ev if float(e["score"]) >= theta_cv]
_actions = sum(1 for c in r5["trajectory"] if c.get("kind") == "CANCEL_ISSUED")
DEMAND = {"above_threshold_events": len(treated),
          "realised_actions": _actions,
          "non_acting_above_events": len(treated) - _actions,
          "why": "the demand is over ABOVE events; the ACTION count keeps "
                 "its own job at P4, matched AFTER the replay (DE37-R2)"}
print(json.dumps({"demand": DEMAND}), flush=True)
pbk = {}
for e in cv_ev:
    k = f"{e['slug']}|{e['side']}|{e['gen']}"
    pbk[k] = {"slug": k, "side": e["side"], "hour": R._hour_of(e["slug"])}
pool = [pbk[k] for k in sorted(pbk)]
# P4 IS MATCHED AFTER THE REPLAY, AND A DRAW THAT FAILS IT IS REJECTED
# AND REDRAWN -- not reported. §8.1 arm 7 is matched on ACTION COUNT, and
# a stateful policy cannot be made to cancel exactly the drawn set, so
# the realised count is only knowable once the control has been replayed.
# One seed gave 496 realised actions against the treated arm's 333; that
# is a reshuffle, not a matched control. The attempt budget is the frozen
# one and EXHAUSTING IT REFUSES rather than accepting a mismatch.
def realised(res):
    """Per-stratum realised action count, straight off ONE replay.

    `_realised_by_stratum` reads a four-leg arm dict at its reported leg;
    this script replays one leg, so the map is built directly rather than
    by faking an arm shape -- summing across legs of one and reading one
    leg of the other compares two different quantities, which the helper's
    own docstring calls "a 100% rejection rate wearing a matching rule's
    name"."""
    out = {}
    for c in res["trajectory"]:
        if c.get("kind") != "CANCEL_ISSUED":
            continue
        st = (c["side"], R._hour_of(c["slug"]))
        out[st] = out.get(st, 0) + 1
    return out

rc_treated = realised(r5)
attempts = accepted = 0
rej = {"PERM_NOT_OK": 0, "P1": 0, "P2": 0, "P3": 0, "P4": 0}
chosen = None
for seed in range(R.DRAW_ATTEMPT_BUDGET):
    attempts += 1
    d_ = MRC.draw(pool, treated, seed=seed)
    c_, ok_ = R.permuted_stream(cv_ev, d_, theta_cv, gidx)
    if not ok_:
        rej["PERM_NOT_OK"] += 1; continue
    pre = R.stream_predicates(cv_ev, c_, d_, theta_cv, gidx)
    bad = [k for k in ("P1_key_multisets_equal",
                       "P2_stratum_score_multisets_equal",
                       "P3_drawn_carry_above_and_only_drawn")
           if not pre[k]]
    if bad:
        rej[bad[0][:2]] += 1; continue
    r_, h_ = replay(c_, cancel=True, theta=theta_cv)
    rc_ctrl = realised(r_)
    post = R.stream_predicates(cv_ev, c_, d_, theta_cv, gidx,
                               rc_treated=rc_treated, rc_control=rc_ctrl)
    if not post["P4_realised_action_counts_equal"]:
        rej["P4"] += 1; continue
    accepted += 1; chosen = (d_, c_, r_, h_, post, seed); break
if chosen is None:
    print(json.dumps({"RANDOM_MATCHED": "REFUSED",
                      "reason": "no draw satisfied P1-P4 within the "
                                "attempt budget; a control that cannot be "
                                "matched is refused, never reported as a "
                                "weaker control",
                      "attempts": attempts, "rejections": rej}), flush=True)
    OUT["arms"]["RANDOM_MATCHED"] = {
        "arm": "RANDOM_MATCHED", "status": "REFUSED_NO_MATCHED_DRAW",
        "attempts": attempts, "rejections": rej,
        "VALID_AS_A_CONTROL": False}
else:
    drawn, ctrl, r7, h7, post, seed_used = chosen
    add("RANDOM_MATCHED", r7, h7,
        {"predictor": "MATCHED_RANDOM_PERMUTATION", "artifacts": cv_v,
         "note": "the CONDVALUE stream's above-threshold values permuted "
                 "within (side, hour); shares CONDVALUE's artifacts BY "
                 "CONSTRUCTION"},
        matched={"above_threshold_demand": len(treated), "drawn": len(drawn),
                 "seed_accepted": seed_used, "attempts": attempts,
                 "rejections": rej, "predicates": post,
                 "VALID_AS_A_CONTROL": True,
                 "demand": DEMAND})

# DISTINCTNESS, ASSERTED: identical outputs across arms is what round 52
# shipped. The emission carries the check rather than a reader making it.
# A REFUSED ARM HAS NO SIGNATURE AND MUST STILL BE REPRESENTABLE. The
# first version indexed `n_cancels` on every arm and died when the control
# refused -- a distinctness check that cannot describe a refusal is a
# check that only works when nothing went wrong.
sig = {a: ("REFUSED:" + v.get("status", "?") if "fields" not in v
           else (v["n_cancels"], v["fields"]["rho"].get("value"),
                 v["fields"]["post_fill_markout_cents"].get("value")))
       for a, v in OUT["arms"].items()}
OUT["arm_distinctness"] = {
    "signatures": sig,
    "n_arms": len(sig), "n_distinct": len(set(map(str, sig.values()))),
    "all_distinct": len(set(map(str, sig.values()))) == len(sig),
    "predictors": {a: v.get("identity", {}).get("predictor", "NONE_REFUSED")
                   for a, v in OUT["arms"].items()},
    "n_distinct_predictors": len({
        v.get("identity", {}).get("predictor", "NONE_REFUSED")
        for v in OUT["arms"].values()}),
    "arms_with_values": sorted(a for a, v in OUT["arms"].items()
                               if "fields" in v),
    "arms_refused": sorted(a for a, v in OUT["arms"].items()
                           if "fields" not in v),
    "floor_available": any("fields" in v and
                           v.get("matched", {}).get("VALID_AS_A_CONTROL")
                           for v in OUT["arms"].values())}
OUT["population_exclusions"] = EXCL
OUT["peak_rss_gb"] = gb(); OUT["total_wall_s"] = round(time.time()-T0, 1)
json.dump(OUT, open(f"{Path(__file__).parent}/arms53.json","w"), indent=1, default=str)
print(json.dumps(OUT["arm_distinctness"]), flush=True)
print("ARMS53 COMPLETE peak_gb=%s wall=%ss" % (OUT["peak_rss_gb"], OUT["total_wall_s"]), flush=True)
