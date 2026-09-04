"""FINDING (Q-DA-243): no-path-effect is STRUCTURAL -- 8 cells over both heads, 3 budgets and 3 latency rungs, all identical; reported as SIX non-vacuous because at budget 0.05 the arm declines nothing.

DA: is no-path-effect STRUCTURAL, or true only where I measured it?

Round 44 established it at one cell (btc, latency 250, budget 0.10). I stated
the mechanism was parameter-independent and that I had measured it in one
place. This TESTS that claim across the declared budget and latency axes,
inside the existing cache -- no new feed, no sealed day.
"""
import sys, json, pickle, collections
sys.path.insert(0, "live/pm_research")
import harmful_stateful_policy as HSP
import de_phase4_diag_runner as R
import de_score_stream as SS

d = pickle.loads(open("/home/yuqing/ctaNew/data/pm_5min/derived/"
                      "de_section81_cache_12.pkl", "rb").read())
ref, asm = d["fr"]["reference"], d["asm"]
COIN = "btc"
NO_CANCEL_THETA = 2.0
FIELDS = ("px_cents", "size", "mid_cents_at_fill", "mid_cents_at_markout",
          "gen_start_ns")

_streams: dict = {}
def stream(head):
    if head in _streams:
        return _streams[head]
    gs = asm["by_arm"][(COIN, head)][0]
    rows = []
    for s_, sides in sorted(ref.items()):
        for sd in HSP.SIDES:
            for g in sides[sd]:
                if (s_, sd, float(g["t0"])) in gs:
                    rows.append({"t": g["t0"], "slug": s_, "side": sd,
                                 "gen": g["gen"]})
    ev = SS.score_events(rows, head=head, coin=COIN,
                         scorer=R._head_scorer(head, COIN, gs),
                         verified=SS.verify_head(head, COIN))
    _streams[head] = ev
    return ev

def replay(scores, *, cancel, theta, lat):
    params = R.cell_params(
        {"coin": COIN, "latency_ms": lat, "budget": 0.10,
         "enable_reduce": False,
         "charge_reset_cost_at_generation_start": False},
        theta_cancel=(theta if cancel else NO_CANCEL_THETA),
        protection_mode=HSP.PROTECTION_MODES[0],
        repost_fill_model=HSP.REPOST_FILL_MODELS[0])
    res = HSP.replay_policy(ref, scores, params)
    return R.received_fills(res, ref, R._decision_times(scores))

def key(f):
    return (round(f["fill_ns"] / 1e9, 9), f["side"], round(f["px_cents"], 6))

def compare(head, budget, lat):
    ev = stream(head)
    theta = R.theta_for(COIN, head, budget)
    base = replay(ev, cancel=False, theta=theta, lat=lat)
    arm = replay(ev, cancel=True, theta=theta, lat=lat)
    bi, ai = collections.defaultdict(list), collections.defaultdict(list)
    for f in base:
        bi[key(f)].append(f)
    for f in arm:
        ai[key(f)].append(f)
    shared = set(bi) & set(ai)
    diff = collections.Counter()
    multi = 0
    for k in shared:
        if len(bi[k]) != 1 or len(ai[k]) != 1:
            multi += 1
            continue
        b, a = bi[k][0], ai[k][0]
        for fl in FIELDS:
            bv, av = b.get(fl), a.get(fl)
            if bv is None or av is None:
                if bv is not av:
                    diff[fl + ":NONE"] += 1
            elif bv != av:
                diff[fl] += 1
    return {"head": head, "budget": budget, "latency_ms": lat,
            "theta": theta, "n_base": len(base), "n_arm": len(arm),
            "n_declined": len(base) - len(arm), "n_shared": len(shared),
            "n_arm_only": len(set(ai) - set(bi)),
            "n_multiplicity": multi,
            "fields_that_differ": dict(diff),
            "IDENTICAL": (not diff) and (not (set(ai) - set(bi)))}

CV, HZ = "q1_arrival_composed_lgbm", "incumbent_linear_d"
CELLS = [(CV, 0.10, 250),        # round 44's cell, as the control
         (CV, 0.05, 250), (CV, 0.15, 250),
         (CV, 0.10, 5), (CV, 0.10, 50),
         (CV, 0.05, 5), (CV, 0.15, 50),
         (HZ, 0.15, 5)]
out = []
for h, b, l in CELLS:
    try:
        r = compare(h, b, l)
    except Exception as e:                                    # noqa: BLE001
        r = {"head": h, "budget": b, "latency_ms": l,
             "REFUSED": f"{type(e).__name__}: {e}"[:150]}
    out.append(r)
    print(json.dumps(r), flush=True)
print()
ok = [r for r in out if r.get("IDENTICAL") is True]
bad = [r for r in out if r.get("IDENTICAL") is False]
ref_ = [r for r in out if "REFUSED" in r]
print(json.dumps({"n_cells": len(out), "n_identical": len(ok),
                  "n_with_differences": len(bad), "n_refused": len(ref_),
                  "STRUCTURAL_ACROSS_TESTED_CELLS": not bad and not ref_},
                 indent=1))
