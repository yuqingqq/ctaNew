"""R2-E REAL-ROW CONTROLS on one promoted pilot partition (review 4393ad19's acceptance list: chronological arm binding / reversal / parity;
intact versus corrupt saved records). Declared before any pilot output is read (2026-11-20, session aea5bfb9); consumed Oct 24-25 only.
 (1) reversal: every ok pair of the partition is re-run with its arms in reverse order. pf_r2_labels.pair_bound(reverse=True) gets a fresh
     teacher per arm, on the fork rebuilt from the world's own census row with the stored key. It must equal the stored result: status, Y,
     every arm's value / cash / position / mark / fills hash / decisions hash / reconciliation, and its fill list
     (pf_r2_label_sample.arm_ledgers(reverse=True)). The rebuilt fork's fingerprint must equal the stored one.
 (2) stale binding (the R2-D falsifier, on real rows): on the partition's first ok pair, arm teachers bound to a copy of the fork instead of
     their arm must report binding failures (binding_bad > 0 summed over the arms).
 (3) persisted corruption: on the partition's first ok row, each of the four review corruptions (Y + 1, another fork, reconciled false,
     binding_bad 1) is refused by the shared validator (no record: each check alone). The intact part passes it with its record and bytes.
Under the 2,048 MiB watchdog; the never-reset high-water <= 1,536 MiB is checked after the loading, every world and at the end.
Usage (run_light): r2e_controls.py <out> <part> -> <out>/controls/part_<k>.json"""
import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, '/home/yuqing/ctaNew/predictfun/research')
import pf_r2_label_sample as S                            # noqa: E402

L, RN, C, KP, X, T = S.L, S.RN, S.C, S.KP, S.X, S.T
KEYS = ('value', 'cash', 'position', 'mark', 'fills_sha256', 'decisions_sha256', 'reconciled', 'fills')


def stale_factory(M, sup, world, B0):
    """Arm teachers bound to a copy of the fork (B0), never to the arm they serve (the falsifier)."""
    made = []

    def make(B):
        t_ = L.ArmTeacher(M, sup); t_.bind(copy.deepcopy(B0), world); made.append((t_, None)); return t_
    make.made = made
    return make


def main(out, k):
    out = Path(out); plan, psha = S.load_plan(out); part = plan['partitions'][k]; day = part['day']; cd = out / 'controls'; cd.mkdir(exist_ok=True)
    mon = RN.Sampler(cd / f'part_{k:03d}_status.json', S.GUARD); mon.start(); t0 = time.time(); hw = []
    try:
        kp = Path(plan['kp_dir']); src = KP.Tape()
        M, sup = S.verify_current(plan, [day], kp, src, S.real_files, S.hours_tracked, None, T.REAL)
        rec = S.accepted_part(out, plan, psha, part); C.require(rec is not None, f'part {k} is not promoted')
        rp, fp, pp = S.part_paths(out, k); b, pb = fp.read_bytes(), pp.read_bytes(); rows, pools = S.gz_rows(b), S.gz_rows(pb)
        sel = {S.ckey(S.wkey(w)): w for w in plan['sample'][day]['selected']}; members = [sel[S.ckey(m)] for m in part['members']]
        allw = {S.ckey(S.wkey(w)): w for w in S.worlds_all(day, kp, keys=[(w['m'], w['t_ms']) for w in members])}; mk = sorted({w['m'] for w in members})
        fair, depth, hrs, trk = X.real_inputs(day, mk)
        C.require(S.day_identity(day, kp, src, hrs, trk, S.real_files) == plan['sample'][day]['inputs'], 'changed input')
        data = src.load(day, mk, src.offset(day)); prs = {int(W_[0]['m']): W_ for W_ in src.pairs(data)}; data = None
        hw.append(S.hwm()); rev, stale = [], None
        for w in members:
            ok = [r for r in rows if (r['m'], r['t_ms']) == (w['m'], w['t_ms']) and r['status'] == 'ok']
            if not ok:
                continue
            aw = allw[S.ckey(S.wkey(w))]; world, wst = L.world_bind(day, dict(m=w['m'], t=aw['t'], o=aw['o']), prs, fair, depth, X.real_features)
            C.require(world is not None, f'{w}: an ok world did not bind')
            cand, _ = L.census_world(day, 0, dict(m=w['m'], t=aw['t'], o=world['o']), world, M, sup); by = {S.ckey(S.row_key(c)): c for c in cand}
            for r in ok:
                ctx = L.fork_for(by[S.ckey(r['row_key'])], world, M, sup); fac = L.teacher_factory(M, sup, world)
                res = L.pair_bound(ctx, fac, r['head'], reverse=True); st = r['result']
                eq = res.get('status') == 'ok' and ctx['fingerprint'] == r['fork_fingerprint'] and abs(res['Y'] - st['Y']) <= S.TOL_V
                if eq:
                    led = S.arm_ledgers(fac, res, reverse=True)
                    eq = all(res['arms'][n][x] == st['arms'][n][x] for n in S.NAMES[r['head']] for x in KEYS) and all(
                        led[n]['fill_list'] == st['arms'][n]['fill_list'] and led[n]['sizes'] == st['arms'][n]['sizes'] for n in S.NAMES[r['head']])
                rev.append(dict(m=r['m'], t_ms=r['t_ms'], side=r['side'], head=r['head'], pick=r['pick'], equal=bool(eq), Y=st['Y'], Y_reversed=res.get('Y')))
                if stale is None:
                    sf = stale_factory(M, sup, world, ctx['state']); rs = L.pair_bound(ctx, sf, r['head'])
                    stale = dict(m=r['m'], t_ms=r['t_ms'], side=r['side'], head=r['head'], status=rs.get('status'),
                                 binding_bad=sum(t_.bind_bad for t_, _ in sf.made), binding_checked=sum(t_.bind_checked for t_, _ in sf.made))
            world = None; hw.append(S.hwm()); C.require(hw[-1] <= S.PEAK, f'the controls high-water passed {S.PEAK} MiB')
        i = [n for n, r in enumerate(rows) if r['status'] == 'ok'][0]; arm0 = S.NAMES[rows[i]['head']][0]

        def corrupt(fn):
            rr = json.loads(json.dumps(rows)); fn(rr[i]); return bool(S.validate(plan, psha, part, rr, pools))
        cor = {'Y + 1': corrupt(lambda r: r['result'].__setitem__('Y', r['result']['Y'] + 1.)),
               'another fork': corrupt(lambda r: r['result'].__setitem__('fork_fingerprint', '0' * 64)),
               'reconciled false': corrupt(lambda r: r['result'].__setitem__('reconciled', False)),
               'binding_bad 1': corrupt(lambda r: r['result']['arms'][arm0]['teacher'].__setitem__('binding_bad', 1))}
        intact = not S.validate(plan, psha, part, rows, pools, rec, (b, pb)); hw.append(S.hwm())
        res_ = dict(part=k, day=day, plan_sha256=psha, part_sha256=rec['sha256'], reversal=dict(n=len(rev), equal=sum(x['equal'] for x in rev), rows=rev), stale_binding=stale,
                    corruptions_refused=cor, intact_valid=intact, hwm_mib=max(hw), elapsed_s=time.time() - t0, sources=S.sources_now(), asof=C.stamp())
        RN.atomic_json(cd / f'part_{k:03d}.json', res_); return res_
    finally:
        mon.live = False


if __name__ == '__main__':
    r_ = main(sys.argv[1], int(sys.argv[2]))
    print(json.dumps(dict(part=r_['part'], reversal=f"{r_['reversal']['equal']}/{r_['reversal']['n']}", stale=r_['stale_binding'], corruptions=r_['corruptions_refused'],
                          intact=r_['intact_valid'], hwm_mib=r_['hwm_mib'])))
