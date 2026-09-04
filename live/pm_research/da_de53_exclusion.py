"""FINDING (Q-DA-237): DE53's 1,309 excluded generations are NOT distributionally like the 31,122 -- selective on duration and slug at the permutation floor; median duration 0.215s excluded vs 0.052s retained, and all nine generations lasting >=16s are excluded against a 4.21% base rate.

DA: the fourth oracle, run on DE53's REAL exclusion.

The membership test is reimplemented from the reference and the assembly
output rather than read from DE's counter (R-235: separate implementations on
purpose). It must reproduce DE's own 1,309 / 29,813 / 31,122 before anything
it says about the distribution is worth reading.
"""
import sys, json, pickle, datetime as dt
sys.path.insert(0, "live/pm_research")
import da_population_audit as PA

CACHE = "/home/yuqing/ctaNew/data/pm_5min/derived/de_section81_cache_12.pkl"
d = pickle.loads(open(CACHE, "rb").read())
ref, asm = d["fr"]["reference"], d["asm"]

def hour_of(slug):
    return dt.datetime.fromtimestamp(int(slug.rsplit("-", 1)[1]),
                                     dt.timezone.utc).hour

out = {}
for (coin, head), val in sorted(asm["by_arm"].items()):
    gen_scores = val[0]
    excluded, retained = [], []
    for slug, sides in sorted(ref.items()):
        for side, gens in sorted(sides.items()):
            for g in gens:
                rec = {"slug": slug, "side": side, "hour": hour_of(slug),
                       "duration": float(g["t1"]) - float(g["t0"]),
                       "n_tranches": len(g.get("tranches") or []),
                       "level": g.get("level")}
                key = (slug, side, float(g["t0"]))
                (retained if key in gen_scores else excluded).append(rec)
    rep = {"head": head,
           "n_excluded": len(excluded), "n_retained": len(retained),
           "n_reference": len(excluded) + len(retained),
           "excluded_fraction": round(len(excluded) /
                                      max(len(excluded)+len(retained), 1), 4)}
    rep["reproduces_DE_counts"] = (
        rep["n_excluded"] == 1309 and rep["n_retained"] == 29813
        and rep["n_reference"] == 31122)
    if excluded:
        rep["audit"] = PA.compare(
            excluded, retained,
            attrs=("slug", "side", "hour", "duration", "n_tranches"),
            n_permutations=400, seed=11)
    out[head] = rep
print(json.dumps(out, indent=1, default=str))
