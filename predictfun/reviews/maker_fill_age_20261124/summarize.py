"""Present every supported good-fill interval, without selecting a timing rule."""
import collections,csv,gzip,json,statistics,subprocess
from pathlib import Path
H=Path(__file__).resolve().parent
def read(p):
 with (gzip.open if str(p).endswith('.gz') else open)(p,'rt') as f:return list(csv.DictReader(f))
assert json.loads((H/'checks.json').read_text())['pass_all'];fills=read(H/'fills.csv.gz');orders=read(H/'orders.csv.gz');result=json.loads((H/'result.json').read_text());tables=[]
for grade in ['G2_bound_constrained','G4_scenario_only']:
 g=[r for r in orders if r['grade']==grade and r['timing_status']=='ok'];e=[r for r in g if r['economics_status']=='counted_fills_available'];good=[r for r in e if r['good_order']=='True'];ages=[float(r['first_age_max_s']) for r in good]
 tables.append(dict(grade=grade,orders=len(g),valued_orders=len(e),good_aggregate_orders=len(good),first_age_lower_all_zero=all(float(r['first_age_min_s'])==0 for r in good),median_age_upper_s=statistics.median(ages) if ages else None,upper_le1=sum(x<=1 for x in ages),upper_le5=sum(x<=5 for x in ages),upper_le10=sum(x<=10 for x in ages),upper_le30=sum(x<=30 for x in ages),upper_gt30=sum(x>30 for x in ages)))
good=[r for r in fills if r['timing_status']=='ok' and r['good_fill']=='True'];out=[]
for r in good:
 q=float(r['q_valid']);out.append(dict(day=r['day'],mid=r['mid'],rank=r['rank'],side=r['side'],price_c=r['tick'],order_hash=r['order_hash'],sid=r['sid'],mk_idx=r['mk_idx'],grade=r['grade'],record_class=r['record_class'],age_min_s=r['age_min_s'],age_max_s=r['age_max_s'],shares=q,gross10_c_share=100*float(r['gross10_usd'])/q,gross60_c_share=100*float(r['gross60_usd'])/q))
with (H/'good_fill_intervals.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(out[0]),lineterminator='\n');w.writeheader();w.writerows(out)
(H/'readings.json').write_text(json.dumps(dict(order_table=tables,good_fills=len(good),good_orders=len({r['order_hash'] for r in good}),good_days=dict(collections.Counter(r['day'] for r in good)),all_good_min_lt5=all(float(r['age_min_s'])<5 for r in good),good_min_gt0=sum(float(r['age_min_s'])>0 for r in good),upper_le1=sum(float(r['age_max_s'])<=1 for r in good),upper_le5=sum(float(r['age_max_s'])<=5 for r in good),all_comparisons_insufficient=all(r['verdict']=='insufficient' for r in result['matched'])),indent=2)+'\n')
lines=[]
for r in out:lines.append(f"| {r['day']}/{r['mid']} | #{r['rank']} {r['side']} {r['price_c']}¢ | {'bounded' if r['grade'].startswith('G2') else 'scenario'} | {float(r['age_min_s']):.3f}–{float(r['age_max_s']):.3f}s | {r['shares']:.3f} | {r['gross10_c_share']:+.2f} / {r['gross60_c_share']:+.2f} | `{r['order_hash'][:12]}` |")
ot='\n'.join(f"| {'G2 bounded-conditional' if r['grade'].startswith('G2') else 'G4 scenario-only'} | {r['orders']} | {r['valued_orders']} | {r['good_aggregate_orders']} | {r['median_age_upper_s']:.3f}s | {r['upper_le1']} / {r['upper_le5']} / {r['upper_le10']} / {r['upper_le30']} |" for r in tables)
report=f'''# How early were the orders behind good fills posted?

As of {result['asof']}. Freeze **{subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()}**. Consumed Oct24–27 only; no fit, new data, replay, validation or strategy change. USER question measured with explicit timing uncertainty rather than treating a window midpoint as an actual post.

**No required posting lead is established.** The available records permit timing ranges. The first-fill lower bound is zero by construction: the first fill proves the order existed by its execution, but does not independently prove how much earlier. A later execution of the same hash can establish an earlier identity, not uninterrupted queue residence. How early the order *needed* to be posted requires a counterfactual queue/fill comparison; observed age alone cannot supply that.

## Broad order population, not individual good-fill labels

All4,149 top-three fill-linked hashes are retained. Good aggregate order means that the sum of its counted fills' gross10s and gross60s markouts are both positive. These outcomes are **not the first fill's value**. First-age upper endpoints below are conditional maximum leads, not measured rests; their median is not median actual age. G4 assumptions are weaker than G2.

| Grade | Orders with timing | Orders with counted values | Good aggregate orders | Median upper bound among good orders | Upper bound ≤1 / ≤5 / ≤10 / ≤30s |
|---|---:|---:|---:|---:|---:|
{ot}

Every good-order first-age lower endpoint is zero. Refused timing grades and missing counted-value statuses stay in orders.csv.gz. The complete census therefore does not establish that the profitable aggregate orders had to rest for5s,10s or longer.

## Exact fill-level subset

The saved fresh-level/front-candidate receipt contains1,725 distinct fill legs:1,526 outside the top-three posting census;199 matched top-three legs. Of those,153 have enabled G2/G4 timing and46 refused timing. The153 split into48 G2 (29 orders) and105 G4 (83 orders). Only36 G2 and51 G4 fills have the common valid1s/10s/60s value cohort. This is a small, selected subset of top-maker fills, **not all their good fills**, and the front candidates are not certified front-of-queue observations.

Good fill is declared as gross value positive at **both10s and60s**. It is a stricter descriptive outcome than positive1s spread capture; fees and later hedging costs are absent, so it is not net profit. There are6 such G2 fills and13 G4 fills. None has a5s minimum inferred age. Four G4 fills have upper bounds≤1s; this is conditional short-post compatibility, not evidence that fast posting caused their gains. One G4 good fill has a positive minimum exceeding1s; the rest still allow a near-fill post.

Every qualifying fill is printed below (multiple rows under one hash are related executions). Bounds use the original hard match window, not the narrower book-decrease scenario. Markouts are inherited from the accepted settlement-clock valuation, up to2s after the possible match clock. They are not precisely10/60s after actual matching. Continuous rest, actual queue rank and hidden cancel/re-entry remain unidentified.

| Date/market | Maker/quote | Timing grade | Posting-to-fill age range | Shares | Gross10s /60s c/share | Hash prefix |
|---|---|---|---:|---:|---:|---|
{chr(10).join(lines)}

## Matched comparison and interpretation

The frozen comparison was min-age≥5s versus max-age≤1s, separately G2 and G4, matching market/day/maker/side/price,first-versus-later record class and creation context. **Both have zero common strata**. The500 declared allocations per grade are saved as not-evaluated zeros, not a null p-value or evidence of no effect. No thresholds or matching were relaxed after seeing this. We cannot state that earlier orders achieve a higher good-fill rate on these data.

The useful concrete result is the interval table: some good fills are compatible with short posting leads; others admit much longer leads, but their early posting is not identified. We cannot conclude that queue age is irrelevant, nor that all good fills require early queue building. The better next estimand is how much earlier the same-price order must join to capture its **entire subsequent fill sequence**, including early losses and later gains; that requires queue competition and cumulative volume, not first-fill markout alone.

## Checks, limits and next priority

Independent checker reconciles every source join,duplicate count,grade,interval,quantity,value,cell and matched statistic. All six planted corruptions are refused. Synthetic interval controls include positive lower bounds,overlapping windows,unknown execution lower bounds and impossible chronology. No source or strategy file changed. Existing dirty work is preserved by owned-state staging. Result-bearing jobs use PF_SLOT=codex/run_light; resource receipt accounts for all attempts within2.88CPU seconds,task2.00/line5.30 plus.05reserve.

Largest limitation: the cached exact-fill subset covers only199 of the top-three fills; first-post upper bounds are first-fill-derived, and a continuous queue path is latent. ONE next priority: a declared **full top-three same-hash fill-sequence age/value census** from the existing accepted valued-fill artifact (all13,227 top-three BOOK rows), preserving whole-sequence economics and hard match windows; then test whether long-age evidence actually increases before a queue counterfactual. The1GB artifact was deliberately not reread in this small packet. This requires a bounded CPU top-up if it exceeds the remaining allowance; no new days or live activity are needed. Do not infer a required lead time from the upper-bound histogram.
'''
(H/'report.md').write_text(report);print((H/'readings.json').read_text());print('positive minimum examples:',[r for r in out if float(r['age_min_s'])>0])
block='''\n<!-- FILL_AGE_20261124_CODEX -->
## 2026-11-24 — posting lead versus longer-horizon good fills

[Report](reviews/maker_fill_age_20261124/report.md). USER asks how early makers must post for good fills. All4149 order-level timing/aggregate-value rows retained; exact fill subset1725 rows,199 top-three,153 enabled G2/G4 timing,87 common-valued. Good at both gross10s/60s:6 G2 +13 G4 fills. All19 min-age<5s;four G4 max-age<=1s;one G4 positive minimum>1s. Not a required lead or continuous rest/front proof. First-fill lower bounds0 by construction; aggregate good-order economics do not label the first fill.

Frozen min-age>=5s vs max-age<=1s comparison haszero common strata after market/maker/side/price/record-class/context matching. No threshold relaxation. Independent source/interval/value/cell/null checks pass;six corruptions refused. No new model/day/strategy/profitability claim. ONE next priority:full13227 top-three valued-fill sequence census using existing1GB artifact,whole sequences and hard windows,before queue counterfactual;needs bounded CPU top-up if beyond remaining resource allowance. Do not mistake median age upper bound for actual resting time.
<!-- /FILL_AGE_20261124_CODEX -->
\n'''
blocks={k:block for k in ['predictfun/HANDOFF.md','predictfun/FINDINGS.md']};blocks['predictfun/reviews/maker_trace_20261122/PENDING.md']=block.replace('(reviews/','(../');blocks['predictfun/STATUS.yml']='''fill_age_20261124_codex:
  report: predictfun/reviews/maker_fill_age_20261124/report.md
  stage: conditional_posting_lead_and_fill_quality_checked
  full_order_census: 4149
  exact_top3_fill_subset: 199
  timing_and_common_value_fills: 87
  good10_and60_fills: 19
  matched_early_recent_strata: 0
  required_posting_lead_identified: false
  next_priority: full_same_hash_fill_sequence_age_value_census_with_resource_gate
''';(H/'state_blocks.json').write_text(json.dumps(blocks,indent=2)+'\n')
