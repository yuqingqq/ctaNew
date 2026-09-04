"""FINDING (Q-DA-242, HAZARD half): the same check at the HZ head -- 4,208 shared fills, 0 arm-only, 0 fields differing.

DA: THE ELEMENT-WISE CHECK. Do RETAINED fills carry identical values in
both replays, or does cancelling change the fills it did not decline?

Q-DA-240 established the fill sets are SUBSETS and that both legs accumulate
in one loop. Neither establishes this. A held side suppresses later crossings,
so a retained fill's level, mid, markout, shares or timing could differ between
the baseline and an acting arm -- and if they do, the delta is not "the value
of the declined fills" and every per-fill ratio built on it is contaminated.
"""
import sys, json, pickle, collections
sys.path.insert(0, "live/pm_research")
import harmful_stateful_policy as HSP
import de_phase4_diag_runner as R
import de_score_stream as SS

CACHE = ("/home/yuqing/ctaNew/data/pm_5min/derived/"
         "de_section81_cache_12.pkl")
d = pickle.loads(open(CACHE, "rb").read())
ref, asm = d["fr"]["reference"], d["asm"]
COIN, LAT, BUDGET = "btc", 250, 0.10
CV = "incumbent_linear_d"
NO_CANCEL_THETA = 2.0

def stream(head):
    gs = asm["by_arm"][(COIN, head)][0]
    rows = []
    for s_, sides in sorted(ref.items()):
        for sd in HSP.SIDES:
            for g in sides[sd]:
                if (s_, sd, float(g["t0"])) in gs:
                    rows.append({"t": g["t0"], "slug": s_, "side": sd,
                                 "gen": g["gen"]})
    v = SS.verify_head(head, COIN)
    return SS.score_events(rows, head=head, coin=COIN,
                           scorer=R._head_scorer(head, COIN, gs),
                           verified=v)

def replay(scores, *, cancel, theta):
    params = R.cell_params(
        {"coin": COIN, "latency_ms": LAT, "budget": BUDGET,
         "enable_reduce": False,
         "charge_reset_cost_at_generation_start": False},
        theta_cancel=(theta if cancel else NO_CANCEL_THETA),
        protection_mode=HSP.PROTECTION_MODES[0],
        repost_fill_model=HSP.REPOST_FILL_MODELS[0])
    res = HSP.replay_policy(ref, scores, params)
    return R.received_fills(res, ref, R._decision_times(scores))

ev = stream(CV)
theta = R.theta_for(COIN, CV, BUDGET)
base = replay(ev, cancel=False, theta=theta)
arm = replay(ev, cancel=True, theta=theta)

def key(f):
    return (round(f["fill_ns"] / 1e9, 9), f["side"], round(f["px_cents"], 6))

def index(fills):
    ix = collections.defaultdict(list)
    for f in fills:
        ix[key(f)].append(f)
    return ix

bi, ai = index(base), index(arm)
shared = sorted(set(bi) & set(ai))
FIELDS = ("px_cents", "size", "mid_cents_at_fill", "mid_cents_at_markout",
          "gen_start_ns")
diff = collections.Counter()
worst = {f: 0.0 for f in FIELDS}
n_cmp = n_multi = 0
for k in shared:
    b_list, a_list = bi[k], ai[k]
    if len(b_list) != 1 or len(a_list) != 1:
        n_multi += 1
        continue
    b, a = b_list[0], a_list[0]
    n_cmp += 1
    for f in FIELDS:
        bv, av = b.get(f), a.get(f)
        if bv is None or av is None:
            if bv is not av:
                diff[f + ":NONE_MISMATCH"] += 1
            continue
        if bv != av:
            diff[f] += 1
            worst[f] = max(worst[f], abs(float(bv) - float(av)))
print(json.dumps({
    "n_baseline_fills": len(base), "n_arm_fills": len(arm),
    "n_declined": len(base) - len(arm),
    "n_shared_keys": len(shared), "n_compared_1to1": n_cmp,
    "n_keys_with_multiplicity": n_multi,
    "n_arm_keys_not_in_baseline": len(set(ai) - set(bi)),
    "fields_that_differ": dict(diff),
    "max_abs_difference": {k: v for k, v in worst.items() if v},
    "ELEMENT_WISE_IDENTICAL": not diff,
}, indent=1))
