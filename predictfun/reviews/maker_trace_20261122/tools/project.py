"""PROJECTION GATE of MAKER-TRACE-RECONSTRUCTION-v1 (session aea5bfb9, 2026-11-22). After the measured prefix (extract 20261024 + the
lifecycle of the prefix market), projects the rest from measurement and writes logs/projection.json ONCE (write-once; a re-run refuses):
  PREFIX the prefix extraction's whole-process CPU from the run_light ledger (per-hour sums run ~10% low), split into REST / WS by the
         per-hour shares recorded in extract_20261024.json (rest.hours / ws.hours);
  REST   per remaining day: bytes of its planned REST hour files x the prefix's REST cpu per byte;
  WS     per remaining WS day: bytes of its planned WS hour files x the prefix's WS cpu per byte (an upper bound on Nov 8, where
         frames of markets without an own fill are skipped before parsing);
  LIFE   (BOOK markets - 1 + GT markets) x the prefix market's lifecycle cpu x LIFE_SAFETY;
  OTHER  declared allowances for econ / truth / report / checks (not yet measurable; stated here, never fitted);
  run_rest = spent + projected + the 0.05 h review reserve <= the USER cap 1.35 h AND line_before + the same <= the 4.85 ceiling
  (first approval 0.90 / 4.40, topped up by the USER on 2026-11-22).
Sizes come from os.stat of allowed hour files only (pf_mt_common.hour_path refuses any other hour before access). Usage: project.py"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, '/home/yuqing/ctaNew/predictfun/research')
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pf_mt_common as C            # noqa: E402
import cpu_ledger as LG             # noqa: E402

PREFIX_DAY, LIFE_SAFETY = '20261024', 1.5
OTHER_H = dict(econ=0.08, truth=0.01, report=0.01, checks=0.04, selftests_after=0.01)
REST_DAYS, WS_DAYS = ('20261023', '20261025', '20261026', '20261027', C.GT_DAY), ('20261025', '20261026', '20261027', C.GT_DAY)
OUTP = C.RECEIPT / 'logs' / 'projection.json'
# AMENDMENT 1 (2026-11-23): the first projection refused (1.4046 h > 1.35); the USER approved cap 1.45 h, line 4.85 unchanged. 'project.py v2'
# writes logs/projection_v2.json (write-once) from the same measurements, recording the superseded projection's sha256 and the approval.
if len(sys.argv) > 1 and sys.argv[1] == 'v2':
    import hashlib
    _v1 = C.RECEIPT / 'logs' / 'projection.json'
    C.need(_v1.exists() and json.loads(_v1.read_text()).get('run_rest') is False, 'v2 supersedes only a refused first projection')
    SUPERSEDES = dict(path='logs/projection.json', sha256=hashlib.sha256(_v1.read_bytes()).hexdigest(),
                      approval='USER AskUserQuestion 2026-11-23: +0.10 h, cap 1.45, line ceiling 4.85 unchanged')
    OUTP = C.RECEIPT / 'logs' / 'projection_v2.json'
else:
    SUPERSEDES = None


def plan_bytes(family, day):
    stems = C.day_hours(day, extra_before=1, extra_after=1)[0] if family == 'rest' else C.day_hours(day)[0]
    tot = 0
    for s in stems:
        p = C.hour_path(family, s)
        tot += os.path.getsize(p) if p.exists() else 0
    return tot


def one_run(rows, prefix):
    """The single successful ledger run whose tag is prefix (optionally with a retry suffix '_r<n>') -> its cpu_s."""
    hit = [r for r in rows if (r['tag'] == prefix or r['tag'].startswith(prefix + '_r')) and r['rc'] == '0']
    C.need(len(hit) == 1, f'{prefix}: not in the ledger exactly once with rc 0 ({len(hit)} found)')
    return hit[0]['cpu_s']


def main():
    C.need(not OUTP.exists(), f'{OUTP} exists (write-once)')
    ex = json.loads((C.OUT / f'extract_{PREFIX_DAY}.json').read_text())
    RH, WH = ex['rest']['hours'], ex['ws']['hours']
    rest_h = sum(h.get('cpu_s', 0.) for h in RH); ws_h = sum(h.get('cpu_s', 0.) for h in WH)
    rest_b = sum(max(h.get('bytes', 0), 0) for h in RH); ws_b = sum(max(h.get('bytes', 0), 0) for h in WH)
    C.need(rest_b > 0 and ws_b > 0 and rest_h > 0 and ws_h > 0, 'the prefix extraction has no measured REST / WS cost')
    book = json.loads(Path(C.REPO / 'predictfun/reviews/ladder_async_fast_20261122/manifest.json').read_text())['complete']
    mid0 = sorted(int(m) for m in book[PREFIX_DAY])[0]
    rows = LG.ledger(); spent = sum(r['cpu_s'] for r in rows) / 3600.
    proc = one_run(rows, f'mt_x_extract_{PREFIX_DAY}')                # whole-process CPU of the prefix extraction
    C.need(proc >= rest_h + ws_h - 1., 'the ledger CPU of the prefix extraction is below its per-hour sum')
    rest_cpu, ws_cpu = proc * rest_h / (rest_h + ws_h), proc * ws_h / (rest_h + ws_h)
    lc = one_run(rows, f'mt_life_{PREFIX_DAY}_{mid0}')
    n_gt = 52; n_book = sum(len(v) for v in book.values())   # GT windows with an own fill (the truth build: 52 of 220); the run's own list is
    #                                                           decided from public legs at extraction, so 52 is a projection input only
    proj = dict(rest_h=sum(plan_bytes('rest', d) for d in REST_DAYS) * rest_cpu / rest_b / 3600.,
                ws_h=sum(plan_bytes('predict_ws', d) for d in WS_DAYS) * ws_cpu / ws_b / 3600.,
                life_h=(n_book - 1 + n_gt) * lc * LIFE_SAFETY / 3600., **{f'{k}_h': v for k, v in OTHER_H.items()})
    more = sum(proj.values()); total = spent + more + LG.RESERVE
    res = dict(spent_h=round(spent, 4), projected_more_h=round(more, 4), components=({k: round(v, 4) for k, v in proj.items()}), reserve_h=LG.RESERVE,
               projected_batch_h=round(total, 4), cap_h=LG.CAP, line_before_max=LG.LINE_BEFORE, projected_line=round(LG.LINE_BEFORE + total, 4), ceiling=LG.CEILING,
               measured=dict(prefix_process_cpu_s=proc, per_hour_sum_cpu_s=rest_h + ws_h, rest_cpu_s=rest_cpu, rest_bytes=rest_b, ws_cpu_s=ws_cpu, ws_bytes=ws_b,
                             life_prefix=dict(day=PREFIX_DAY, mid=mid0, cpu_s=lc)),
               n_book=n_book, n_gt=n_gt, life_safety=LIFE_SAFETY, run_rest=bool(total <= LG.CAP and LG.LINE_BEFORE + total <= LG.CEILING))
    res['supersedes'] = SUPERSEDES
    OUTP.write_text(json.dumps(res, indent=1) + '\n'); print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
