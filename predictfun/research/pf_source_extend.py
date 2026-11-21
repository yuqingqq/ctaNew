"""SOURCE-EXTEND-v1 (P-2026-005; 2026-11-21, session aea5bfb9). USER 2026-11-21 (the preflight choice "Extend source, all 6"): the world source of the
six-date coverage build for the approved consumed dates that never had it. The frozen R2 / R3 neutral source chain runs UNCHANGED:
pf_queue_place_test.day_run (v1: the queue-place grid rows) then pf_keep_pull.day_run (orders.parquet: the grid's JOIN states with their
statuses and KEEP branch; fills.parquet). Only the modules' declared day lists (QP.DAYS, QL.DAYS, KP.DAYS) and v1's output root (QP.OUT) are
swapped for the call and restored after it, as the ledger's own _v1_on routes v1; keep / pull takes its v1 and out roots as arguments. No
frozen file is edited. Outputs go to a new root OUT/<builder>/<day>, never into the 11-17 directories.
DAYS: NEW_DAYS = 20261031, 20261101, 20261102, 20261103 (consumed replay days of the standing rules; the USER go); CONTROL_DAY = 20261024 (its
 11-17 outputs exist). allowed() refuses, before anything runs, a day in the loader's sealed block [SEAL_MS), a day ending after its forward
 boundary FWD_MS, and any day not declared here. The chain's padding: every book part through the loader's part_allowed (SKIP_FOREIGN_PARTS),
 the fair price through fair_fix (hours -1..23 through part_allowed), everything else day-named files (registry, imputation offset, step 5).
POSITIVE CONTROL (control()): the wrapper on CONTROL_DAY into OUT/control reproduces the 11-17 outputs: queue-place rows.parquet and keep / pull
 orders.parquet / fills.parquet equal as frames (columns, dtypes, every value; parquet bytes may differ only in metadata); the receipts' stat,
 status, reconciliation, fair_missing (keep / pull) and markets / stat (v1) equal; keep / pull's recorded frozen-source hashes equal. Written
 once to OUT/control.json. run_day refuses a new day unless the control passed and this module is the committed one.
Usage (run_light): pf_source_extend.py selftest | control | day <day>"""
import contextlib
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_queue_place_test as QP                          # noqa: E402  frozen v1 (imported, never edited)
import pf_queue_place_ledger as QL                        # noqa: E402  frozen
import pf_keep_pull as KPm                                # noqa: E402  frozen
import pf_stop_ev_minutes_inputs as IN                    # noqa: E402  the one guarded loader (SEAL_MS, FWD_MS, part_allowed)

C, pd = KPm.C, KPm.pd
VERSION = 'source_extend_v1'
NEW_DAYS = ('20261031', '20261101', '20261102', '20261103')
CONTROL_DAY = '20261024'
OUT = Path(KPm.I.K.IO) / 'source_extend_20261121'
OLD_QP, OLD_KP = Path(QP.OUT), Path(KPm.OUT)
REPO = HERE.parents[1]


def day_ms(day):
    return IN._ms(f'{day[:4]}-{day[4:6]}-{day[6:]}')


def allowed(day):
    """A declared day outside the sealed block that ends no later than the forward boundary."""
    d0 = day_ms(day)
    return bool(day in NEW_DAYS + (CONTROL_DAY,) and not (IN.SEAL_MS[0] <= d0 < IN.SEAL_MS[1]) and d0 + 86400000 <= IN.FWD_MS)


@contextlib.contextmanager
def declared(day, qp_out):
    """The frozen builders see exactly this day declared and v1 writes under qp_out; everything is restored after, also on an error."""
    C.require(allowed(day), f'{day}: not a declared, unsealed, pre-forward day')
    keep = (QP.DAYS, QL.DAYS, KPm.DAYS, QP.OUT)
    QP.DAYS = QL.DAYS = KPm.DAYS = (day,); QP.OUT = Path(qp_out)
    try:
        yield
    finally:
        QP.DAYS, QL.DAYS, KPm.DAYS, QP.OUT = keep


def build(day, root):
    """The frozen chain for one day into root -> (v1 day dir, keep / pull day dir)."""
    root = Path(root)
    with declared(day, root / 'queue_place'):
        QP.day_run(day)
        d = KPm.day_run(day, v1=root / 'queue_place', out=root / 'keep_pull')
    return root / 'queue_place' / day, Path(d)


def frames_equal(a, b):
    """DataFrame.equals: the same shape, column labels in order, dtypes and every value."""
    A, B = pd.read_parquet(a), pd.read_parquet(b)
    return bool(A.equals(B))


def compare(qd, kd, oq, ok_):
    """The control's comparisons of a rebuilt day with the 11-17 one (header)."""
    qn, qo = (json.loads((x / 'receipt.json').read_text()) for x in (qd, oq)); kn, ko = (json.loads((x / 'receipt.json').read_text()) for x in (kd, ok_))
    return dict(qp_rows=frames_equal(qd / 'rows.parquet', oq / 'rows.parquet'), kp_orders=frames_equal(kd / 'orders.parquet', ok_ / 'orders.parquet'),
                kp_fills=frames_equal(kd / 'fills.parquet', ok_ / 'fills.parquet'), qp_receipt=bool(qn['markets'] == qo['markets'] and qn['stat'] == qo['stat']),
                kp_receipt=bool(all(kn[k] == ko[k] for k in ('stat', 'status', 'reconciliation', 'fair_missing'))),
                kp_frozen_sources=bool(kn['provenance']['source'] == ko['provenance']['source']))


def committed():
    r = subprocess.run(['git', '-C', str(REPO), 'show', 'HEAD:predictfun/research/pf_source_extend.py'], capture_output=True)
    return r.returncode == 0 and r.stdout == Path(__file__).read_bytes()


def control(root=OUT):
    root = Path(root); cj = root / 'control.json'; C.require(not cj.exists(), 'the control exists (write-once)'); C.require(committed(), 'the wrapper is not committed')
    t0 = time.time(); qd, kd = build(CONTROL_DAY, root / 'control'); res = compare(qd, kd, OLD_QP / CONTROL_DAY, OLD_KP / CONTROL_DAY)
    js = dict(version=VERSION, day=CONTROL_DAY, checks=res, all_ok=bool(all(res.values())), seconds=time.time() - t0, asof=C.stamp())
    root.mkdir(parents=True, exist_ok=True); cj.write_text(json.dumps(js, indent=1, sort_keys=True) + '\n'); print(json.dumps(js, indent=1))
    return js['all_ok']


def run_day(day, root=OUT):
    root = Path(root); C.require(day in NEW_DAYS, f'{day}: not a new day of this extension'); cj = root / 'control.json'
    C.require(cj.exists() and json.loads(cj.read_text()).get('all_ok') is True, 'the positive control has not passed'); C.require(committed(), 'the wrapper is not committed')
    t0 = time.time(); qd, kd = build(day, root); kr = json.loads((kd / 'receipt.json').read_text()); qr = json.loads((qd / 'receipt.json').read_text())
    js = dict(version=VERSION, day=day, qp=dict(markets=qr['markets'], stat=qr['stat'], outputs=qr['outputs']), kp=dict(stat=kr['stat'], status=kr['status'], outputs=kr['outputs'],
              reconciliation=kr['reconciliation'], provenance=kr['provenance']), seconds=time.time() - t0, asof=C.stamp())
    (root / 'days').mkdir(parents=True, exist_ok=True); (root / 'days' / f'{day}.json').write_text(json.dumps(js, indent=1, sort_keys=True) + '\n')
    print(json.dumps(dict(day=day, orders=kr['stat'].get('orders'), status=kr['status'], seconds=round(js['seconds'])), indent=1))
    return js


def selftest():
    import pf_source_extend_selftest as ST_
    ST_.run(sys.modules[__name__])


if __name__ == '__main__':
    a = sys.argv[1:]
    if a == ['selftest']:
        selftest()
    elif a == ['control']:
        sys.exit(0 if control() else 1)
    elif a[:1] == ['day'] and len(a) == 2:
        run_day(a[1])
    else:
        sys.exit(__doc__)
