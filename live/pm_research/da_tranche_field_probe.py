"""FINDING (Q-DA-234): of the five inputs claimed present on every tranche, `side` is MISSING_KEY on all 4,315 -- it is the containing dict's key, not a field; the other four are present and non-None.

DA independent check: are the five claimed inputs on EVERY tranche?

Read-only. Calls DE's own producer and counts field presence per tranche --
the question is not whether the code names the field but whether a builder
walking the real reference would find it on every row or discover a hole
halfway through.
"""
import sys, json, collections
sys.path.insert(0, "live/pm_research")
import de_phase4_diag_runner as R

COIN = sys.argv[1] if len(sys.argv) > 1 else "btc"
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 40
FIELDS = ("level", "mid_at_fill", "shares", "side", "markout_cents_per_share")

out = R.build_reference(COIN, limit=LIMIT)
ref = out["reference"]
cnt = {f: collections.Counter() for f in FIELDS}
n_tr = n_gen = 0
gen_level_present_tranche_level_none = 0
for slug, sides in ref.items():
    for side, gens in sides.items():
        for g in gens:
            n_gen += 1
            for t in g["tranches"]:
                n_tr += 1
                for f in FIELDS:
                    if f not in t:
                        cnt[f]["MISSING_KEY"] += 1
                    elif t[f] is None:
                        cnt[f]["PRESENT_BUT_NONE"] += 1
                    else:
                        cnt[f]["PRESENT"] += 1
                if t.get("level") is None and g.get("level") is not None:
                    gen_level_present_tranche_level_none += 1
print(json.dumps({
    "coin": COIN, "limit": LIMIT,
    "n_slugs": len(ref), "n_generations": n_gen, "n_tranches": n_tr,
    "per_field": {f: dict(cnt[f]) for f in FIELDS},
    "tranche_level_None_while_generation_level_present":
        gen_level_present_tranche_level_none,
    "statuses": out["statuses"],
}, indent=1))
