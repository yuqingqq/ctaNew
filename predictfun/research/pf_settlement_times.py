"""SETTLEMENT TIMES OF THE TRADE LEGS (2026-11-18, session aea5bfb9; USER: "Can you clear the issues", after the one-day test of the corrected
timing). Every REST match record carries a settlementId, a UUIDv7 whose first 48 bits are the venue's creation time in ms (RFC 9562). On
live's 99 confirmed own legs (the 11-09 fill-clock audit; their kernel trade_id IS the settlementId) that time lies inside the Matched status
second, after the placement and before the first fill notice on 99 / 99 - the sub-second venue timestamp the timing work lacked
(pf_keep_pull_exec_windows K8 checks it). This extracts it for every leg of a consumed day, row-aligned with the day's legs_timed2 (a leg =
transactionHash[2:18] + maker hash[2:18] with its content and first receipt, as pf_step1 / pf_txmap; joined on the content + its rank within
equal content), and checks it before anything is written (declared before the data is read):
 C1 one settlement per taker order: among taker orders (pf_txmap tk) with >= 2 mapped legs, those whose legs carry more than one settlementId
    are <= 0.1%, and the same check with the ids shuffled across legs fires (>= 50% of them)
 C2 UUID version 7: every mapped leg's id (a leg without one, or another version, is status 4)
 C3 the REST cap: t_set <= (executedAt + 1) * 1000 (the Confirmed status follows the settlement) on >= 99.9% of mapped legs - violations are
    counted, not repaired
 C4 the positive control: the hour reader saw at least one match record per hour read and at least one mapped leg
Statuses (row-aligned with legs_timed2; t_set NaN unless 0): 0 mapped; 1 absent from the pages read; 2 equal-content legs counted differently
in the pages and in legs_timed2; 3 equal-content legs with different settlementIds (which row is which is not identifiable); 4 no settlementId
or not UUIDv7.
Writes data/predict_fun/research/settlement_times_v2_20261118/<day>/legs_set.parquet (t_set_ms, set_status) + settlement.json (statistics,
input hashes, the hours read and the hours excluded). Consumed days only (KP.DAYS).
V2 (2026-11-18, the readiness loop): v1 padded each day by 1 h before and 2 h after (pf_io.hours_of) with no day rule, so day 20261027
OPENED the sealed 20261028_00 and _01 REST match hours (670,454 records read, used only to join 10-27's legs; v1's 20261027 output is
VOID, its other days superseded). v2 never opens a sealed (10-28..30, the replay loader's SEAL_MS) or forward (>= FWD_MS) hour: the page
reader refuses any such hour stem before calling its reader, and day_run drops them from the padded list (recorded in the receipt as
hours_excluded). A leg absent from the pages read whose executedAt second lies within the padding (2 h before / 1 h after) of an excluded
hour gets status 5 (absent: boundary hour excluded) instead of 1. Days 10-23..26 never reach an excluded hour: their v2 output must equal v1's.
Usage (run_light, PF_SLOT=claude): pf_settlement_times.py selftest | day <day>
"""
import collections
import json
import re
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1]))
import pf_keep_pull as KP                                 # noqa: E402
from predictfun.research import pf_io as io               # noqa: E402

C, pd, I = KP.C, KP.pd, KP.I
OUT = Path(I.K.IO) / 'settlement_times_v2_20261118'      # v2: sealed / forward hours never opened; v1 (settlement_times_20261118) stays as provenance
EXTRA_BEFORE_H, EXTRA_AFTER_H = 1, 2                      # the padded hours around a day (REST pages arrive after their matches)
CKEY = ['mid', 'exec_s', 'first_recv_ns', 'side', 'ptick', 'shares', 'price_executed', 'taker_index', 'taker_quote']
STATUS = ('mapped', 'absent', 'count_mismatch', 'ambiguous_content', 'no_v7_id', 'absent_boundary_hour_excluded')
MAX_VIOLATION, MIN_SHUFFLED, MIN_CAP = 0.001, 0.5, 0.999


def uuid7_ms(sid):
    """The 48-bit creation time (ms) of a UUIDv7 string, or NaN if it is not one."""
    if not isinstance(sid, str) or len(sid) != 36 or sid[14] != '7':
        return np.nan
    try:
        return float(int(sid[:8] + sid[9:13], 16))
    except ValueError:
        return np.nan


def hour_start_ms(h):
    """The start (ms) of an hour stem YYYYMMDD_HH, refused if it is not one."""
    C.require(isinstance(h, str) and re.fullmatch(r'\d{8}_\d{2}', h) is not None, f'not an hour stem: {h!r}')
    return I._ms(f'{h[:4]}-{h[4:6]}-{h[6:8]}') + int(h[9:11]) * 3600000


def hour_allowed(h):
    """False for an hour in the sealed block (the replay loader's SEAL_MS: 10-28..30) or at / after the forward boundary (FWD_MS)."""
    h0 = hour_start_ms(h)
    return not (I.SEAL_MS[0] <= h0 < I.SEAL_MS[1]) and h0 < I.FWD_MS


def read_pages(hours, rows=None):
    """Every leg of the REST match pages of `hours` (rows: a reader of (recv, params, path, payload), io.rest_rows by default) -> DataFrame
    CKEY + sid, the first receipt kept per leg; per hour the match records seen; legs whose id changed across pages (counted)."""
    rows = rows or io.rest_rows; legs, seen, changed = {}, {}, 0
    for h in hours:
        C.require(hour_allowed(h), f'sealed or forward hour {h}: never opened')                     # v2: refused before the reader is called
        n = 0
        for recv, params, path, pl in rows(h, ('/v1/orders/matches',)):
            for x in pl.get('data') or []:
                try:
                    ts = int(io.iso_s(x['executedAt']))
                except (KeyError, TypeError, ValueError):
                    continue
                n += 1; mid = (x.get('market') or {}).get('id'); t = x['taker']; txf = x['transactionHash']; sid = x.get('settlementId')
                ts_ = uuid7_ms(sid); sh = hash(sid) if isinstance(sid, str) else 0          # the id's time and a 64-bit hash (no strings held: memory)
                for m in x['makers']:
                    k = txf[2:18] + m['hash'][2:18]; old = legs.get(k)
                    if old is not None:
                        changed += int(old[10] != sh)
                        if recv < old[2]:
                            legs[k] = old[:2] + (recv,) + old[3:]
                        continue
                    side, py = io.yes_side(m)
                    legs[k] = (mid, ts, recv, 0 if side == 'bid' else 1, int(round(py * 100)), int(m['amount']) / io.E18, int(x['priceExecuted']) / io.E18,
                               t['outcome']['indexSet'], 0 if t['quoteType'] == 'Bid' else 1, ts_, sh)
        seen[h] = n; C.guard('settlement pages')
    return pd.DataFrame(list(legs.values()), columns=CKEY + ['t_set', 'sid_h']), seen, changed


def join(L, M):
    """Row-aligned settlement ids (hashes) and times for legs L (CKEY) from the page legs M (CKEY + t_set + sid_h): content + rank within equal
    content -> (sid_h, t_set, status)."""
    L = L[CKEY].copy(); L['_row'] = np.arange(len(L)); gl = L.groupby(CKEY, sort=False, dropna=False); L['_n'] = gl.cumcount(); L['_cnt'] = gl[CKEY[0]].transform('size')
    M = M.copy(); gm = M.groupby(CKEY, sort=False, dropna=False); M['_n'] = gm.cumcount(); M['_cnt'] = gm[CKEY[0]].transform('size')
    M['_nid'] = gm.sid_h.transform('nunique')
    J = L.merge(M[CKEY + ['_n', '_cnt', '_nid', 't_set', 'sid_h']], on=CKEY + ['_n'], how='left', suffixes=('', '_m'))
    C.require(len(J) == len(L) and bool((J._row.values == np.arange(len(L))).all()), 'the join changed the rows')
    t = J.t_set.values.astype(float); st = np.zeros(len(J), np.int8)
    st[J._cnt_m.isna().values] = 1
    st[(st == 0) & (J._cnt.values != J._cnt_m.values)] = 2
    st[(st == 0) & (J._nid.values > 1)] = 3
    st[(st == 0) & ~np.isfinite(t)] = 4
    return np.where(st == 0, J.sid_h.fillna(0).values, 0).astype(np.int64), np.where(st == 0, t, np.nan), st


def one_per_order(tk, sid_h, mapped, rng):
    """C1: the share of taker orders with >= 2 mapped legs whose legs carry > 1 settlementId; the same with the ids shuffled across legs."""
    ok = (tk >= 0) & np.asarray(mapped, bool)
    def share(s):
        f = pd.DataFrame(dict(tk=tk[ok], s=s)); f = f[f.tk.map(f.tk.value_counts()).values >= 2]
        return float((f.groupby('tk').s.nunique() > 1).mean()) if len(f) else 0.
    s = np.asarray(sid_h, np.int64)[ok]
    return share(s), share(rng.permutation(s)), int(ok.sum())


def day_run(day, out=OUT, rows=None, legs_of=None, hours=None):
    start = time.time(); C.require(day in KP.DAYS, 'undeclared day (consumed days only)'); d = Path(out) / day; C.require(not d.exists(), 'settlement output exists')
    S1 = Path(f'{I.IO}/step1/{day}'); hours = hours or io.hours_of(day, extra_after=EXTRA_AFTER_H, extra_before=EXTRA_BEFORE_H)
    excluded = [h for h in hours if not hour_allowed(h)]; hours = [h for h in hours if hour_allowed(h)]          # v2: never opened
    L, tk, inputs = legs_of(day) if legs_of else (pd.read_parquet(S1 / 'legs_timed2.parquet', columns=CKEY), pd.read_parquet(S1 / 'legs_tk.parquet', columns=['tk']).tk.values,
                                                 {str(S1 / f): C.sha(S1 / f) for f in ('legs_timed2.parquet', 'legs_tk.parquet')})
    C.require(len(tk) == len(L), 'legs_tk not row-aligned')
    M, seen, changed = read_pages(hours, rows); M = M[M.mid.isin(set(L.mid))].reset_index(drop=True)
    sid, t, st = join(L, M); rng = np.random.default_rng(int(day))
    if excluded:                                                                                   # v2: absent legs whose pages could lie in an excluded hour
        ex = np.asarray(L.exec_s.values, float) * 1000.; near = np.zeros(len(L), bool)
        for h in excluded:
            h0 = hour_start_ms(h); near |= (ex >= h0 - EXTRA_AFTER_H * 3600000.) & (ex < h0 + 3600000. + EXTRA_BEFORE_H * 3600000.)
        st = np.where((st == 1) & near, 5, st)
    v1, v1s, nmap = one_per_order(np.asarray(tk, np.int64), sid, st == 0, rng)
    cap = (L.exec_s.values.astype(float) + 1.) * 1000.; m0 = st == 0; capok = float((t[m0] <= cap[m0]).mean()) if m0.any() else 0.
    lag = (L.exec_s.values.astype(float) * 1000. - t)[m0]
    R = dict(day=day, legs=int(len(L)), page_legs=int(len(M)), hours=hours, hours_excluded=excluded, records_per_hour=seen, ids_changed_across_pages=int(changed),
             status={STATUS[s]: int((st == s).sum()) for s in range(len(STATUS))}, mapped_share=float(m0.mean()),
             C1=dict(violation_share=v1, shuffled_share=v1s, mapped_legs_with_taker_order=nmap), C3=dict(within_rest_cap_share=capok, violations=int((t[m0] > cap[m0]).sum())),
             rest_second_minus_t_set_ms={f'p{q}': float(np.percentile(lag, q)) for q in (0, 1, 5, 50, 95, 99, 100)} if len(lag) else {})
    ok = dict(C1=v1 <= MAX_VIOLATION and v1s >= MIN_SHUFFLED, C2=bool(m0.any()), C3=capok >= MIN_CAP, C4=all(n > 0 for n in seen.values()) and bool(m0.any()))
    R['controls'] = ok
    if not all(ok.values()):
        sys.exit(f'SETTLEMENT TIMES FAILED -> no output: {ok} {json.dumps({k: R[k] for k in ("status", "C1", "C3")})}')
    d.mkdir(parents=True); pd.DataFrame(dict(t_set_ms=t, set_status=st)).to_parquet(d / 'legs_set.parquet', index=False)
    C.dump(d / 'settlement.json', dict(asof=C.stamp(), R=R, inputs=inputs, outputs={'legs_set.parquet': C.sha(d / 'legs_set.parquet')}, provenance=KP.provenance(),
                                       source_sha256=C.sha(Path(__file__).resolve()), seconds=time.time() - start))
    print(json.dumps(R), flush=True)
    return R


def _u7(ms, tail='7fe1-88c0-f6f4c395258e'):
    h = f'{int(ms):012x}'; return f'{h[:8]}-{h[8:12]}-{tail}'


def selftest():
    """Synthetic pages only: the UUIDv7 time; a mapped leg, an absent one, equal-content legs with different ids, a non-v7 id, a repeated page
    with an earlier receipt; C1 fires on shuffled ids and refuses two ids in one taker order; C3 refuses ids later than their REST second."""
    bad = []
    if not (uuid7_ms(_u7(1794122135551)) == 1794122135551. and np.isnan(uuid7_ms('01a1ba03-3879-4fe1-88c0-f6f4c395258e')) and np.isnan(uuid7_ms(None))
            and uuid7_ms('01a1ba03-3879-7fe1-88c0-f6f4c395258e') == float(0x01a1ba033879)):
        bad.append('the UUIDv7 time')
    day = KP.DAYS[0]; s0 = 1792800000; H0 = '20261023_05'                                    # v2: planted pages sit in a real, allowed hour stem
    if not (hour_allowed('20261027_23') and not hour_allowed('20261028_00') and not hour_allowed('20261030_23') and hour_allowed('20261031_00')
            and not hour_allowed('20261104_00') and not hour_allowed('20261109_12')):
        bad.append('the sealed / forward hour rule')

    def mk(tx, taker, makers, ts, sid, amt=5):
        return dict(executedAt=f"{np.datetime_as_string(np.datetime64(ts, 's'))}Z", market=dict(id=7), taker=dict(hash=taker, outcome=dict(indexSet=1), quoteType='Bid'),
                    transactionHash=tx, priceExecuted=str(int(.5 * io.E18)), settlementId=sid,
                    makers=[dict(hash=h, signer='s', amount=str(int(amt * io.E18)), price=str(int(.5 * io.E18)), outcome=dict(indexSet=1), quoteType='Ask') for h in makers])
    recs = [(100, mk('0x' + 'a' * 64, '0xt1', ['0x' + '1' * 64, '0x' + '2' * 64], s0, _u7(s0 * 1000 + 400))),              # one order, two equal legs, one id
            (50, mk('0x' + 'a' * 64, '0xt1', ['0x' + '1' * 64, '0x' + '2' * 64], s0, _u7(s0 * 1000 + 400))),               # the same page, an earlier receipt
            (100, mk('0x' + 'b' * 64, '0xt2', ['0x' + '3' * 64], s0 + 1, _u7((s0 + 1) * 1000 + 10))),
            (100, mk('0x' + 'c' * 64, '0xt3', ['0x' + '4' * 64], s0 + 2, _u7((s0 + 2) * 1000 + 20))),                     # equal content to the next leg, another id
            (100, mk('0x' + 'd' * 64, '0xt4', ['0x' + '5' * 64], s0 + 2, _u7((s0 + 2) * 1000 + 30))),
            (100, mk('0x' + 'e' * 64, '0xt5', ['0x' + '6' * 64], s0 + 3, '01a1ba03-3879-4fe1-88c0-f6f4c395258e'))]        # not v7
    recs += [(100, dict(mk(f'0x{i:02x}' + 'f' * 62, f'0xu{i}', [], s0 + 10 + i, _u7((s0 + 10 + i) * 1000 + 100)),             # (a leg key reads hex 2-18)
                        makers=[dict(hash=f'0x{i:02x}{j}' + 'e' * 61, signer='s', amount=str(int((1 + i + 50 * j) * io.E18)), price=str(int(.5 * io.E18)), outcome=dict(indexSet=1),
                                     quoteType='Ask') for j in (1, 2)])) for i in range(30)]                                  # 30 two-leg orders with their own ids
    pages = lambda recs_: (lambda h, p: iter([(r, None, '/v1/orders/matches', dict(data=[x])) for r, x in recs_] if h == H0 else []))
    M, seen, changed = read_pages([H0], pages(recs))
    if not (len(M) == 66 and seen == {H0: 36} and changed == 0 and int(M.first_recv_ns.min()) == 50):
        bad.append(f'page legs: {len(M)} {seen} {changed}')
    L = M[CKEY].copy(); L.loc[len(L)] = [7, s0 + 99, 1, 0, 50, 5., .5, 1, 0]                                             # a leg absent from the pages
    L.loc[len(L)] = L.loc[2].values                                                                                     # a second leg with leg 2's content: counts differ
    sid, t, st = join(L, M); want = [0, 0, 2, 3, 3, 4] + [0] * 60 + [1, 1]
    if st.tolist() != want:
        bad.append(f'statuses: {st.tolist()} != {want}')
    if not (t[0] == t[1] == s0 * 1000 + 400. and np.isnan(t[3]) and t[6] == (s0 + 10) * 1000 + 100.):
        bad.append(f'times: {t[:8].tolist()}')
    tk = np.r_[0, 0, 1, 2, 3, 4, np.repeat(np.arange(5, 35), 2), -1, -1].astype(np.int64)
    v, vs, n_ = one_per_order(tk, sid, st == 0, np.random.default_rng(1))
    if not (v == 0. and vs >= MIN_SHUFFLED and n_ == 62):
        bad.append(f'one settlement per taker order: {v} / shuffled {vs} / {n_}')
    sid2 = np.array(sid, np.int64); sid2[1] = hash(_u7(1.))
    if one_per_order(tk, sid2, st == 0, np.random.default_rng(1))[0] != 1. / 31:
        bad.append('two settlement ids in one taker order not counted')
    with tempfile.TemporaryDirectory(dir=str(I.K.IO)) as tmp:
        legs_of = lambda d_: (L.copy(), tk, {})
        try:
            R = day_run(day, out=tmp, rows=pages(recs), legs_of=legs_of, hours=[H0])
            P = pd.read_parquet(Path(tmp) / day / 'legs_set.parquet')
            if not (R['controls'] == dict(C1=True, C2=True, C3=True, C4=True) and len(P) == len(L) and P.set_status.tolist() == st.tolist()):
                bad.append(f'rehearsal: {R["controls"]} {P.set_status.tolist()}')
        except (SystemExit, ValueError) as e:
            bad.append(f'rehearsal refused: {e}')
        late = [(r, dict(x, settlementId=_u7((s0 + 200) * 1000 + i))) for i, (r, x) in enumerate(recs)]                # ids later than their REST second (distinct)
        same = [(r, dict(x, settlementId=_u7(s0 * 1000 + 400))) for r, x in recs]                                     # one id on every leg: only the shuffle can tell
        for nm_, kw, msg in (('ids later than their REST second', dict(rows=pages(late)), "'C3': False"),
                             ('one id on every leg', dict(rows=pages(same)), "'C1': False"),
                             ('two settlement ids in one taker order', dict(rows=pages(recs), legs_of=lambda d_: (L.copy(), np.zeros(len(L), np.int64), {})), "'C1': False")):
            o_ = Path(tmp) / ('z' + str(abs(hash(nm_)) % 1000))
            try:
                day_run(day, out=o_, **dict(dict(rows=pages(recs), legs_of=legs_of, hours=[H0]), **kw)); bad.append(f'accepted {nm_}')
            except SystemExit as e:
                if msg not in str(e):
                    bad.append(f'{nm_} refused by another control: {e}')
            if (o_ / day).exists():
                bad.append(f'a refused run wrote output ({nm_})')
        # v2: a sealed hour is never opened - the reader refuses it before calling, day_run drops it and marks the boundary legs
        asked = []
        def rec_pages(recs_):
            def f(h, p):
                asked.append(h); return pages(recs_)(h, p)
            return f
        try:
            read_pages(['20261028_00'], rec_pages(recs)); bad.append('a sealed hour was read')
        except (SystemExit, ValueError):
            pass
        if asked:
            bad.append(f'the reader was called for {asked}')
        sb = I._ms('2026-10-28') / 1000.                                                            # the sealed boundary (s)
        L2 = L.copy(); L2.loc[len(L2)] = [7, sb - 1800, 1, 0, 50, 5., .5, 1, 0]                     # absent, 30 min before the excluded hour: status 5
        L2.loc[len(L2)] = [7, sb - 3 * 3600, 1, 0, 51, 5., .5, 1, 0]                                 # absent, 3 h before it: stays status 1
        tk2 = np.r_[tk, -1, -1].astype(np.int64)
        try:
            R2 = day_run(day, out=Path(tmp) / 'sealed', rows=rec_pages(recs), legs_of=lambda d_: (L2.copy(), tk2, {}), hours=[H0, '20261028_00'])
            P2 = pd.read_parquet(Path(tmp) / 'sealed' / day / 'legs_set.parquet')
            if not (R2['hours'] == [H0] and R2['hours_excluded'] == ['20261028_00'] and '20261028_00' not in asked and P2.set_status.tolist()[-2:] == [5, 1]
                    and P2.set_status.tolist()[:-2] == st.tolist() and R2['status']['absent_boundary_hour_excluded'] == 1):
                bad.append(f"sealed boundary: hours {R2['hours']} / excluded {R2['hours_excluded']} / asked {sorted(set(asked))} / statuses {P2.set_status.tolist()[-3:]}")
        except (SystemExit, ValueError) as e:
            bad.append(f'the sealed-boundary rehearsal refused: {e}')
    if bad:
        sys.exit('SETTLEMENT SELFTEST FAILED: ' + ' | '.join(bad))
    print('settlement selftests passed (synthetic only): the UUIDv7 time; mapped / absent / ambiguous-content / not-v7 legs; the first receipt kept; '
          'one settlement per taker order and its shuffled control; ids later than their REST second and two ids in one taker order refused with no output; '
          'v2: a sealed hour refused before its reader is called, dropped from a padded list, and the legs it could hold marked status 5', flush=True)


if __name__ == '__main__':
    a = sys.argv[1:]
    if a == ['selftest']:
        selftest()
    elif len(a) == 2 and a[0] == 'day':
        selftest(); day_run(a[1])
    else:
        sys.exit('usage: pf_settlement_times.py selftest | day <day>')
