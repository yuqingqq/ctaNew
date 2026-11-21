"""Review 216ff603's close-out hunks (FINDINGS P100 = the in-band correction of P99, HANDOFF stop section, STATUS stop block), every number read
from the committed receipt (outputs/eval_rows.csv.gz, outputs/actions.csv, context_checks.json, logs/). argv[1] = the time from `date`.
Writes edits/F_stop.md, edits/H_stop.md, edits/S_stop.yml in the scratchpad. Usage: fill_stop.py <time>"""
import glob
import gzip
import io
import json
import re
import sys
from pathlib import Path

import pandas as pd

R = Path('/home/yuqing/ctaNew/predictfun/reviews/movement_shrink_20261121'); S = Path('/tmp/claude-1001/-home-yuqing-ctaNew/aea5bfb9-4ea4-41d0-bf73-bb98d818e3d4/scratchpad')
NOW = sys.argv[1]; REV = 'predictfun/reviews/automatic/20261121_530eac3b'
E = pd.read_csv(io.BytesIO(gzip.decompress((R / 'outputs' / 'eval_rows.csv.gz').read_bytes())), dtype={'side': str, 'head': str})
AC = pd.read_csv(R / 'outputs' / 'actions.csv', dtype={'side': str, 'head': str, 'rule': str}); cx = json.loads((R / 'context_checks.json').read_text())
RL = re.compile(r'^(\S+) tag=(\S+) rc=(\d+) peak_rss_mib=(\d+) wall_s=([\d.]+) cpu_user_s=([\d.]+)')
jobs, cpu, peak = 0, 0., 0
for p in glob.glob(str(R / 'logs' / '**' / '*.log'), recursive=True):
    for ln in Path(p).read_text().splitlines():
        m = RL.match(ln)
        if m:
            jobs += 1; cpu += float(m.group(6)); peak = max(peak, int(m.group(4)))


def moves(h):
    g = E[E['head'] == h]; o = g.mask_original.astype(str).eq('True'); c = g.mask_candidate.astype(str).eq('True')
    return dict(n=len(g), orig=int(o.sum()), cand=int(c.sum()), dropped=int((o & ~c).sum()), added=int((~o & c).sum()), both=int((o & c).sum()))


def J(h, side, rule):
    g = AC[(AC['head'] == h) & (AC.side == side) & (AC.rule == rule)]; assert len(g) == 1; return float(g.J.iloc[0])


en, rv = moves('entry'), moves('first_review')
subset = {h: (m['added'] == 0) for h, m in (('entry', en), ('first_review', rv))}
S_DIR = (f"entry: {en['dropped']} of {en['orig']} original POSTs become WAIT, {en['added']} WAITs become POST, {en['both']} POSTs kept "
         f"({'a strict reduction' if subset['entry'] else 'not a subset'}); first review: {rv['dropped']} of {rv['orig']} original KEEPs become PULL, {rv['added']} PULLs "
         f"become KEEP, {rv['both']} KEEPs kept by both ({'a strict reduction' if subset['first_review'] else 'not a subset: the candidate KEEP loss comes from newly selected exposure'})")
S_SIDE = (f"hold-all local value on Oct 27: entry bid {J('entry', '0', 'all'):+.3f} c, ask {J('entry', '1', 'all'):+.3f} c; first review bid "
          f"{J('first_review', '0', 'all'):+.3f} c, ask {J('first_review', '1', 'all'):+.3f} c per opportunity")
F = (f"| P100 | Correction to P99 (review 216ff603 item 1; as of {NOW}): P99's \"only reduces posting / keeping\" holds at entry only. Computed decision changes "
     f"(candidate = prior vs the original sign rule): {S_DIR}. \"None beats every rule\" is the pooled comparison only: {S_SIDE}. Hold-all is a local "
     f"neutral-hold value, not factual PI0 P&L; entry and review values are never summed. | No reading changes (both heads stay INCONCLUSIVE; the candidate is "
     f"not promoted). The bid-side values are descriptive facts of one consumed day, not a rule to adopt; an all-sides / all-states \"never retain\" "
     f"conclusion is too broad. |\n")
H = (f"## {NOW} (session aea5bfb9) — REVIEW 216ff603: PI0-MOVEMENT-SHRINK-v1 ACCEPTED, NOT PROMOTED; THE NEXT PRIORITY NEEDS THE USER; LOOP STOPPED\n"
     f"- [Review]({REV.replace('predictfun/', '')}/report.md), [proposal]({REV.replace('predictfun/', '')}/next_priority.md). No blocking finding.\n"
     f"- **Item 1 (P2) cleared:** FINDINGS P100 corrects P99 in band — {S_DIR}; {S_SIDE}.\n"
     f"- **Item 2 (P3) cleared for this receipt:** `reviews/movement_shrink_20261121/context_checks.py` compares every decision-context column of the committed "
     f"{cx['rows']} eval rows (key identity, head, weight, mid, decision cell, status, strata) to their bound sources: {cx['n_clean_failures']} failures; falsifier "
     f"{sum(v['flagged'] for v in cx['falsifier'].values())} / {len(cx['falsifier'])} (mid +1 c, hour +1, side flip, horizon bin, doubled weight). For reuse: consume the committed "
     f"bytes or run this comparison. No rebuild; no fix-only review (the review asked for none).\n"
     f"- **Round totals as of {NOW}:** {jobs} run_light jobs, {cpu / 3600.:.4f} core-hours, peak {peak} MiB (the receipt's 0.064 was an earlier as-of).\n"
     f"- **Decision needed (USER):** authorize Codex's proposal — a six-date (Oct 24–25, Oct 31, Nov 1–3; all consumed; Oct 23 stays the teacher day, Oct 28 out) "
     f"fixed-PI0, unchanged-target neutral-exposure label coverage build: metadata preflight of the dates, at most 192 worlds per date (1,152), at most 4,608 local "
     f"action queries (absent POST, absent WAIT, first-review KEEP, first-review PULL) with exact inclusion probabilities, a fixed 24-world pilot capped at 0.5 "
     f"core-hours and a write-once projection, up to 8 core-hours on the four development slots, 1,536 / 2,048 MiB. Without the go: retain the artifacts, launch "
     f"nothing. Separate USER choices: untouched validation (at least five complete days; no candidate is ready), a new target / model search, or stopping this line.\n"
     f"- The loop (USER 2026-11-19) stopped here by its STOP rule; no review request is pending.\n\n")
St = (f"p005_loop_stop_20261121:\n  status: awaiting_user_decision\n  asof: \"{NOW}\"\n  reason: codex_review_216ff603_next_priority_needs_user\n"
      f"  proposal: {REV}/next_priority.md\n  decision_requested: {{six_date_coverage_build: pending, ceiling_core_hours: 8, pilot_core_hours: 0.5, worlds_max: 1152, queries_max: 4608, "
      f"dates: [\"20261024\", \"20261025\", \"20261031\", \"20261101\", \"20261102\", \"20261103\"]}}\n"
      f"  review_216ff603: {{blocking: [], item1_decision_directions: corrected_FINDINGS_P100, item2_context_columns: {'context_checks_all_ok' if cx['all_ok'] else 'context_checks_failed'}}}\n"
      f"  decision_moves: {{entry_post_to_wait: {en['dropped']}, entry_wait_to_post: {en['added']}, review_keep_to_pull: {rv['dropped']}, review_pull_to_keep: {rv['added']}, review_kept_by_both: {rv['both']}}}\n"
      f"  shrink_round_totals: {{jobs: {jobs}, core_hours: {cpu / 3600.:.4f}, peak_mib: {peak}}}\n  candidate_promoted: false\n  loop: stopped\n")
for n, t in (('F_stop.md', F), ('H_stop.md', H), ('S_stop.yml', St)):
    (S / 'edits' / n).write_text(t)
print(F); print(H); print(St)
