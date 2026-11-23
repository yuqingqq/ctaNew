"""MATCHED-MARKET-TURNOVER-GAP-v1 (P-2026-005 MAKER-TRACE; contract matched_gap_contract.md, frozen in 46cd3cc56; review 8f2a2e753's ONE next
priority; session aea5bfb9 development, 2026-11-23). DESCRIPTIVE ONLY, on consumed days (Oct 24-27: four point estimates, no intervals): where the
observed gap lies between our simulated CURRENT_ASYNC_FAST10 ledger and the ex-ante top makers on the SAME 19 BOOK markets and intervals. Not an
avoided-loss estimate, an imitation claim, a validation or a profitability claim; it measures no model skill. Every rule below is declared before
any data run; synthetic proof (hand-computed decompositions, reading cases, refusals): matched_gap_selftest.py.
POPULATION reviews/ladder_async_fast_20261122/manifest.json (pf_mt_econ.BOOK_MANIFEST) 'complete': its days must be measurement days (refused
       before any other path is built); its pilot / rest parts must partition them; the CLI run requires exactly N_BOOK = 19 markets.
OURS   the accepted records data/predict_fun/research/ladder_async_fast_v1_20261122/{pilot,rest}/<day>.jsonl.gz, key CURRENT_ASYNC_FAST10
       (pf_ladder_async_fast.run_arm -> pf_ladder_rest.ledger_record). Bound: the day record <day>.json (version async_execution_sensitivity_v1,
       DAY_DONE, members = the manifest part, the arm listed, the manifest's sha256, the gz bytes' sha256) and the accepted measurement (the
       sha256 of measure/arms.csv in measure/audit.json; the arm's arms.csv value = the record's value within VALUE_TOL_C cents).
       Record format (pf_branch_engine.Engine.advance / value; pf_ladder_async_fast.run_day): fills = [t, gid, side, q, p, it]: t = ms from the
       episode start (the engine clock; trades at END count, so 0 <= t <= end), gid the generation, side 0 = bid (buys YES, s = +1) / 1 = ask
       (sells YES, s = -1), q shares, p the order's price in whole cents (YES space), it the public trade row. ident.start_abs_ms = the episode
       start (epoch ms: the first 50 ms grid step >= ws whose received book is valid), ident.end_abs_ms = start + end_rel = ws + 298,000 (the
       supported pre-expiry END); record end = settings.end_ms = ident.end_rel; path_info agrees (t_start, end_rel, ws = ident.opening_ms).
       mark = the YES mid at END used for the value (cents; QP.build_path: the last valid mark within 1 s before END). VALUE = cash + position
       x mark with cash = -sum(sg q p), position = sum(sg q): reproduced EXACTLY by the engine's own arithmetic in its order (bit-equal cash,
       position and value; refused otherwise) and by this module's accounting (total = value / 100 within REL_TOL). Leg: t_abs = start_abs_ms +
       t (the engine's simulated fill time: the trade anchor - 50 ms), role maker (our engine only posts), fee 0 (as recorded), record = it.
THEIRS econ/fills_valued.parquet (v2) rows of the ex-ante top 10 (econ/identities.json ex_ante.top10_shares; ranks 1-3 must be the top 3 of
       decision_contract.md, else refuse) in the BOOK markets, read through the row-group statistics of day (measurement days only; a group whose
       day range leaves the consumed days refuses before it is read) and of mid (a group without a BOOK market is never read); every row of a BOOK
       market must carry that market's manifest day. Completeness per address x market against econ/maker_market (summed over days): ok rows
       (n_mk + n_tk), excluded rows (n_excl), net position (sum s q of the ok lot rows vs net_pos) and fee_fill, REL_TOL with a 1-unit floor.
       Lots: status ok rows with self-trades out (their gross and fees stay as the self line); excluded rows are counted by status and place.
       Leg: s, q, p (YES space, USD), t = settle_ms (the UUIDv7 creation time: an upper bound of the match), role, fee = fee_fill (USD at the
       fill price; positive = paid), record = sid.
UNITS  USD (ours: p / 100, mark / 100); per-share columns in cents (_c_sh).
INTERVAL per market: [start, END] = our ident start_abs_ms / end_abs_ms; the END mark = our recorded mark, the SAME for every party. Boundary by
       '<' at both ends: t < start -> before (carried in), END < t -> after (outside), else inside (a fill exactly at start or at END is
       inside; our engine counts trades at END). markets_<day> (v2) must agree: ws x 1000 = ident.opening_ms, start >= ws, END <= we - 2 s
       (else refuse). no_interval (a status): the record's status is not ok or its mark is not a finite price in (0, 100) cents.
OPENING MARK pf_mt_econ.mid_at on the day's tob frames: the last frame with upd_ms <= floor(start) (ordered upd_ms, recv_ns), its mid when the
       frame is usable (no bad flag, two-sided, not crossed) and not stale (> 1,000 ms), else its MARK_ST status. The tob mid at floor(END) is a
       diagnostic beside our END mark (never used).
ORDER  one rule for both parties: chronological on the absolute ms clock (asserted inside the FIFO), ties by each party's recorded sequence
       (theirs: econ's FIFO order settle_ms, sid lexicographic, px_rank, mk_idx, maker before taker; ours: the engine's fill order).
FIFO   fifo(): ONE implementation for every party, view and check. A deque of open lots, all on one side; a leg closes the oldest opposite lots
       first (unit j bought pairs unit j sold), its unmatched rest opens a lot. A completed close = (entry lot, exit leg, shares): P&L = shares x
       (sell - buy price), fees of both legs prorated by shares (fee x shares / leg q), holding time = exit t - entry t, both roles, the entry
       kind (within / carried). A close below DUST_SH shares stays in every sum, flagged dust; never counted, never in a quantile.
       CHECK: the same function over each address x market's whole recorded history (every ok lot row, no interval) reproduces econ's mq / rq
       per row within SH_TOL_ABS + SH_TOL_REL x traded shares (else refuse).
VIEWS  carry (PRIMARY; recorded inventory) and zero (carried-in ignored; beside, never instead). Carry: the FIFO over the address's ok lot legs
       before start leaves its carried lots (one side), kept with their actual acquisition times and roles, re-based to the opening mark with no
       fee (paid before the interval); the interval FIFO starts from them. Carried lots below DUST_SH shares in total are dropped (counted). An
       active address with carried lots and no ok opening mark: carry status no_opening_mark (its zero view stands). Ours starts at 0, as
       simulated: its two views are equal.
DECOMPOSITION per address (or our arm) x market x view: close net (incl. carried lots closed inside, at the opening-mark basis) + carry
       (unclosed carried shares x (END mark - opening mark)) + resid (unclosed within lots: sum s rq (END mark - p)) - fee_unclosed (fee x rq / q)
       + self (sum s q (END mark - p) - fees of the in-interval self-trade rows) = total. IDENTITY: total = the direct recompute sum s q (END mark -
       p) - fees over the in-interval ok rows + [carry view] (sum s q before start) x (END mark - opening mark), within REL_TOL x max(1 USD,
       |direct|), else refuse. exposure = carry + resid. r1 FIX (integrator, after the first run c0f237ffe): the party-level checks below.
       PARTY IDENTITY (per party x market x view): the party total = an INDEPENDENT party-level direct recompute = every active member's
       in-window flow + [carry view] the carried position summed directly over EVERY member x (END mark - opening mark). CROSS-VIEW (per party x
       market with both views ok): carry total - zero total = carried_pos_recorded x (END mark - opening mark) within REL_TOL x max(1, |rhs|):
       refuses identical views whenever carried_pos x (END - open) != 0 (equal views with END = opening mark are correct, not refused).
PATHS  max |position| along the leg path (FIFO order) from the starting position (carry: the carried position; zero: 0). Inventory actions (self-
       trades out; one action = one address's legs of one record and role: theirs a sid, ours a public trade row): position before = after
       every action at a strictly earlier ms (the declared strict-earlier convention); nonflat |before| > POS_EPS; opposing = s x before < 0;
       absolute reduction |after| < |before| - POS_EPS; overshoot = opposing with |after| > |before| + POS_EPS; simultaneous = actions sharing
       their ms with another action of the same address x market.
PARTIES ours; rank1, rank2, rank3 (single addresses), top3 (ranks 1-3 pooled), top10 (ranks 1-10). A pooled party = the sum of its members'
       accountings (the FIFO never pairs two addresses' legs). A party is ACTIVE in a market iff a member has an ok row inside (the contract's
       no_activity rule, at the party level). Zero view: its active members. Carry view (r1 fix): its active members (with their carried lots)
       AND every member with carried lots but no ok row inside (carry_only: its lots unclosed, valued at END mark - opening mark) - the party's
       recorded inventory entering the interval, never only its active members'. Additive fields summed, holding-time quantiles over the
       members' closes, max |position| = the largest member's; row counts and the recorded carried inventory over every member;
       carried_pos_in = the carried position that entered the view (= carried_pos_recorded in the carry view, 0 in the zero view). Member status
       per view: ok, carry_only (carry view), no_opening_mark (carry view, carried lots without an ok opening mark), rows_excluded, no_activity.
       Party status: no_interval (market); not active -> rows_excluded if a member has excluded rows inside, else no_activity; no_opening_mark
       (carry view) if any accounted member's; else ok.
READING (frozen in the contract) per comparison party x market x view where ours and the party are both ok (active): o, t = our / their close
       net per matched share (NaN without a non-dust close); dT = our - their close net, dE = our - their exposure; the first matching row:
       our_turnover_loses: o < 0 and t > 0; exposure_gap: (o >= 0 and t >= 0, or o < 0 and t < 0) and |dE| > |dT|; both_adverse: o < 0 and
       t < 0; mixed_unsupported: otherwise. EVERY reading of a party (and its pooled reading) is mixed_unsupported when fewer than MIN_MARKETS
       (5) markets have both active; reading_raw keeps the market rule alone, beside. Pooled (secondary, with its market mix): the same rules on
       sum close net / sum matched shares and sum dT / dE over the both-active markets. Reported: per party and view the count of each reading
       with its denominators (markets; our / their matched shares), the fee and self parts of the difference beside (dF, dS), per day sums;
       4 days, no intervals.
OUTPUTS <out> = pf_mt_common.RECEIPT / <subdir> (default 'matched_gap'; run --out <subdir> writes a new one, e.g. matched_gap_r1, never touching
       an earlier output; write-once: refused before any input if it exists; built in a partial directory renamed at the end): per_market.csv
       (market x party x view), closes_by_role.csv (party x view x scope x entry kind x role pair), answer.json, receipt.json (inputs' sha256,
       sources, git, counts, statuses, checks, CPU, memory).
GUARDS MT_RUN=v2 (CLI); measurement days only; VmHWM self-guard pf_mt_common.GUARD_MIB (1,450 MiB, never raised); CPU self-guard CPU_CAP_S;
       every input unchanged during the run; run refuses unless this tool, its selftest and the contract are tracked and unmodified (the freeze).
       rehearse <day>: the same build on one day with no write; prints checks, counts, statuses and timings only (no P&L, no reading).
Usage (run_light): MT_RUN=v2 matched_gap.py run [--out <subdir>] | MT_RUN=v2 matched_gap.py rehearse <day>    selftest: matched_gap_selftest.py"""
import collections
import gzip
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parents[2] / 'research'
for _p in (str(RESEARCH), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import pf_mt_common as C  # noqa: E402
import pf_mt_econ as E  # noqa: E402
import closes as X  # noqa: E402

need, OK = C.need, C.STATUS_OK
NAN = float('nan')
VERSION = 'matched_gap_v1'
CONTRACT = 'predictfun/reviews/maker_trace_20261122/matched_gap_contract.md (frozen in 46cd3cc56)'
REVIEW = 'predictfun/reviews/automatic/20261123_maker_trace_v2_review/report.md (8f2a2e753), ONE next priority'
CONTRACT_FILE = C.REPO / 'predictfun/reviews/maker_trace_20261122/matched_gap_contract.md'
TOOL_FILES = (HERE / 'matched_gap.py', HERE / 'matched_gap_selftest.py', CONTRACT_FILE)
ARM = 'CURRENT_ASYNC_FAST10'
DAY_VERSION = 'async_execution_sensitivity_v1'
LADDER = C.REPO / 'data/predict_fun/research/ladder_async_fast_v1_20261122'
N_BOOK = 19
PARTS = ('pilot', 'rest')
NTOP, KEY_MUL = X.NTOP, X.KEY_MUL
PARTIES = ('ours', 'rank1', 'rank2', 'rank3', 'top3', 'top10')
COMPARE = PARTIES[1:]
MEMBERS = dict(rank1=(0,), rank2=(1,), rank3=(2,), top3=(0, 1, 2), top10=tuple(range(NTOP)))
VIEWS = ('carry', 'zero')
PRIMARY = 'carry'
READINGS = ('our_turnover_loses', 'exposure_gap', 'both_adverse', 'mixed_unsupported')
STATUSES = ('ok', 'no_interval', 'no_opening_mark', 'rows_excluded', 'no_activity')
ROLES, PAIRS = X.ROLES, X.PAIRS
KINDS = ('within', 'carried')
WITHIN, CARRIED = 0, 1
MIN_MARKETS = 5
REL_TOL = 1e-6
SH_TOL_ABS, SH_TOL_REL = X.SH_TOL_ABS, X.SH_TOL_REL
DUST_SH = X.DUST_SH
POS_EPS = 1e-9
VALUE_TOL_C = 1e-9
QUANTS = X.QUANTS
CPU_CAP_S = 120.
_MUT_EXCLUDE_CARRY_ONLY = False      # planted-error switch (selftest only): the first run's defect, carried lots of members without a fill inside left out
FV_COLS = ('day', 'mid', 'sid', 'mk_idx', 'px_rank', 'role', 'addr', 's', 'q', 'p', 'self_trade', 'settle_ms', 'fee_fill', 'mq', 'rq', 'status')
FV_DICT = frozenset(('day', 'role', 'addr', 'status'))
MM_COLS = ('day', 'mid', 'addr', 'n_mk', 'n_tk', 'n_excl', 'net_pos', 'fee_fill')
MM_DICT = frozenset(('day', 'addr'))
TOB_COLS = tuple(E.TOB_COLS) + ('day',)
TOB_DICT = frozenset(('day', 'status'))
MKT_SPEC = dict(mid='int64', ws='num', we='num', status='str')
LT, LS, LQ, LP, LR, LF, LK, LKIND = range(8)          # leg / lot tuple: t (abs ms), s, q, p (USD), role (0 maker / 1 taker), fee (USD), record, kind
ENGINE_LINES = ('sg = lambda s: 1. if s == 0 else -1.',
                'cash = -sum(sg(s) * q * p for _, _, s, q, p, _ in self.fills); pos = sum(sg(s) * q for _, _, s, q, _, _ in self.fills)',
                'return cash, pos, cash + pos * mark')      # pf_branch_engine.Engine.value (the selftest checks both sources)
HOLD_COLS = [f'hold_p{int(round(q * 100))}_s' for q in QUANTS] + ['hold_mean_s', 'hold_med_count_s']
ROW_KEYS = ('n_rows_pre', 'n_rows_in', 'n_rows_post', 'n_self_in', 'n_excl_pre', 'n_excl_in', 'n_excl_post', 'carried_pos_recorded',
            'carried_lots_recorded', 'carried_dust_dropped')
MET_ADD = ('buy_sh', 'sell_sh', 'n_closes', 'n_dust', 'close_sh', 'close_pnl', 'close_fee', 'close_net', 'carried_close_sh', 'carried_close_net',
           'unclosed_sh', 'unclosed_carried_sh', 'carry', 'resid', 'fee_unclosed', 'self_gross', 'self_fee', 'total', 'direct', 'end_pos',
           'n_actions', 'n_nonflat', 'n_opposing', 'n_abs_reduce', 'n_overshoot', 'n_simultaneous', 'carried_pos_in')
MSTAT = ('ok', 'carry_only', 'no_activity', 'rows_excluded', 'no_opening_mark')
PM_COLS = (['day', 'm', 'party', 'view', 'primary', 'status', 'n_members'] + [f'n_members_{s_}' for s_ in MSTAT] +
           ['market_status', 'market_reason', 'start_ms', 'end_ms', 'end_mark', 'open_mark', 'open_status', 'open_age_ms',
            'tob_mid_end', 'tob_end_status', 'end_mark_minus_tob_c', 'value_recorded_c', 'value_exact'] + list(ROW_KEYS) + ['carried_first_ms'] +
           list(MET_ADD) + ['close_net_c_sh', 'exposure', 'identity_dev', 'party_identity_dev', 'max_abs_pos'] + HOLD_COLS +
           ['both_active', 'pair_status', 'reading', 'reading_raw', 'o_c_sh', 't_c_sh', 'd_turnover', 'd_exposure', 'd_fee_unclosed', 'd_self',
            'd_total', 'exposure_larger'])
CR_COLS = ['party', 'view', 'scope', 'entry_kind', 'role_pair', 'n_markets', 'n_closes', 'n_dust', 'sh', 'pnl', 'fee', 'net', 'c_gross_sh',
           'c_net_sh'] + HOLD_COLS
RULES = dict(
    population='the 19 BOOK markets: reviews/ladder_async_fast_20261122/manifest.json complete (measurement days; pilot / rest partition)',
    ours='CURRENT_ASYNC_FAST10 accepted records (day record version / members / arm / manifest sha256 / gz sha256; arms.csv value); fills [t ms '
         'from the episode start, gid, side 0 bid s=+1 / 1 ask s=-1, q, p cents, trade row]; t_abs = ident.start_abs_ms + t; role maker; fee 0; '
         'value reproduced bit-exactly by the engine arithmetic and by the accounting',
    theirs='fills_valued v2, ex-ante top 10 (ranks 1-3 = top 3), BOOK markets; status ok; self-trades out of the lots (self line); t = settle_ms '
           '(upper bound); fee = fee_fill (USD at the fill price); completeness vs maker_market; econ mq / rq reproduced by the same FIFO',
    interval="[start, END] = our ident start_abs_ms / end_abs_ms; '<' at both ends (t < start carried in, END < t outside, start and END inside); "
             'END mark = our recorded mark for every party; markets_<day> ws / we must agree',
    opening_mark='pf_mt_econ.mid_at at floor(start): last tob frame with upd_ms <= start, usable, not stale (> 1,000 ms)',
    order='chronological on absolute ms (asserted); ties: theirs settle_ms, sid, px_rank, mk_idx, maker first; ours the engine fill order',
    fifo='one deque FIFO for every party / view / check; close P&L = shares x (sell - buy); fees prorated by shares; hold = exit t - entry t; '
         f'dust < {DUST_SH:g} sh kept in sums, never counted',
    views='carry (primary): carried lots = the pre-start FIFO remainder at the opening mark, no fee, actual acquisition time; zero: carried '
          'ignored (beside)',
    identity=f'close net + carry + resid - fee_unclosed + self = direct (sum s q (END - p) - fees [+ C0 (END - open)]) within {REL_TOL:g} x '
             'max(1, |direct|); ours also = recorded value / 100',
    paths='max |position| from the starting position along the FIFO leg order; actions = legs of one record and role; strict-earlier position '
          f'before; opposing / absolute reduction / overshoot / simultaneous (POS_EPS {POS_EPS:g})',
    parties='ours; rank1-3; top3 / top10 = sums over members (no cross-address pairing); active iff a member has an ok row inside; zero view: '
            'active members; carry view (r1 fix): active members + every member with carried lots and no ok row inside (carry_only)',
    party_checks='party total = independent party direct (active members in-window flow + [carry] carried position over EVERY member x '
                 '(END - open)); cross-view: carry total - zero total = carried_pos_recorded x (END - open)',
    reading='first match: our_turnover_loses (o < 0, t > 0); exposure_gap ((o >= 0, t >= 0) or (o < 0, t < 0), |dE| > |dT|); both_adverse '
            f'(o < 0, t < 0); mixed_unsupported otherwise; all mixed_unsupported when fewer than {MIN_MARKETS} both-active markets; pooled: '
            'same rules on the sums (secondary)')


# ------------------------------------------------------------------ helpers ------------------------------------------------------------------
def _now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _int(v):
    return type(v) is int


def default_paths():
    """The real inputs and output (MT_RUN switches the maker-trace roots: v2 -> maker_trace_v2_20261123 and RECEIPT / v2)."""
    return dict(manifest=Path(E.BOOK_MANIFEST), ladder=LADDER, econ=Path(C.OUT) / 'econ', root=Path(C.OUT), out=Path(C.RECEIPT) / 'matched_gap')


def engine_value(fills, mark):
    """pf_branch_engine.Engine.value's arithmetic, verbatim and in its order (ENGINE_LINES; the selftest compares both sources): cash,
    position and END value in cents."""
    sg = lambda s: 1. if s == 0 else -1.  # noqa: E731
    cash = -sum(sg(s) * q * p for _, _, s, q, p, _ in fills); pos = sum(sg(s) * q for _, _, s, q, _, _ in fills)  # noqa: E702
    return cash, pos, cash + pos * mark


def side_of(t, start, end):
    """The declared boundary rule, '<' at both ends: -1 before [start, END] (carried in), +1 after it, 0 inside (start and END included)."""
    t = np.asarray(t, dtype=float)
    return np.where(t < start, -1, np.where(end < t, 1, 0)).astype(np.int8)


def check_identity(total, direct, where, devs, key='identity'):
    """The decomposition total against the direct recompute (module rules: IDENTITY) -> |deviation|; refuses outside REL_TOL."""
    dv = abs(total - direct) if (math.isfinite(total) and math.isfinite(direct)) else math.inf
    devs[key] = max(devs.get(key, 0.), dv)
    need(dv <= REL_TOL * max(1., abs(direct) if math.isfinite(direct) else 1.),
         f'identity: {where}: decomposition {total!r} vs direct {direct!r} (|dev| {dv:.3g} > {REL_TOL:g} x max(1, |direct|))')
    return dv


def check_cross_view(t_carry, t_zero, cpos, end_mark, open_mark, where, devs, cnt):
    """CROSS-VIEW (module rules): carry total - zero total = the recorded carried position x (END mark - opening mark). Refuses identical views
    whenever carried_pos x (END - open) != 0 (carried lots that never entered the carry view, the first run's defect) and any other gap."""
    want = cpos * (end_mark - open_mark) if cpos != 0. else 0.
    got = t_carry - t_zero
    dv = abs(got - want) if (math.isfinite(got) and math.isfinite(want)) else math.inf
    devs['cross_view'] = max(devs.get('cross_view', 0.), dv)
    cnt['cross_view_checks'] = cnt.get('cross_view_checks', 0) + 1
    need(dv <= REL_TOL * max(1., abs(want) if math.isfinite(want) else 1.),
         f'cross-view: {where}: carry total - zero total = {got!r} but carried_pos {cpos!r} x (END - open) = {want!r} (carried lots outside the '
         f'carry view?)')
    return dv


def hold_stats(hold_ms, sh, dust):
    """Share-weighted holding-time quantiles of the non-dust closes (the smallest hold whose cumulative share weight reaches q; no
    interpolation), the share-weighted mean and the lower count median, in seconds (closes.py's rule)."""
    m = ~np.asarray(dust, bool)
    hs, w = np.asarray(hold_ms, float)[m], np.asarray(sh, float)[m]
    out = {c: NAN for c in HOLD_COLS}
    if len(hs):
        o = np.argsort(hs, kind='stable')
        hs, w = hs[o], w[o]
        cw = np.cumsum(w)
        for q in QUANTS:
            out[f'hold_p{int(round(q * 100))}_s'] = float(hs[min(int(np.searchsorted(cw, q * cw[-1], 'left')), len(hs) - 1)]) / 1000.
        out['hold_mean_s'] = float((hs * w).sum() / w.sum()) / 1000. if w.sum() > 0 else NAN
        out['hold_med_count_s'] = float(hs[(len(hs) - 1) // 2]) / 1000.
    return out


def _cpc(x, den):
    return 100. * x / den if den > 0 else NAN


def _reg(stat, key, path):
    """Register an input at the moment it is opened: (path, bytes, mtime_ns); the build refuses if it changes before the end."""
    p = Path(path)
    need(p.is_file(), f'input missing: {key}')
    stat[key] = (p, p.stat().st_size, p.stat().st_mtime_ns)
    return p


# ------------------------------------------------------------------ THE FIFO and the accounting ------------------------------------------------------------------
def fifo(legs, init=()):
    """THE FIFO (module rules): init = carried lots (one side, older than every leg), legs chronological (asserted) -> dict(items, n0, closes
    [(entry item, exit item, shares)], rem (unmatched shares per item at the end), matched (per item), book (open items, oldest first))."""
    items = list(init) + list(legs)
    n0 = len(init)
    need(all(it[LKIND] == CARRIED for it in items[:n0]) and all(it[LKIND] == WITHIN for it in items[n0:]), 'fifo: carried lots must precede the legs')
    need(len({it[LS] for it in items[:n0]}) <= 1, 'fifo: carried lots on both sides')
    rem = [float(it[LQ]) for it in items]
    need(all(r > 0. for r in rem), 'fifo: a leg or lot without a positive quantity')
    matched = [0.] * len(items)
    book = collections.deque(range(n0))
    closes = []
    t_last = max((it[LT] for it in items[:n0]), default=-math.inf)
    for j in range(n0, len(items)):
        t, s = items[j][LT], items[j][LS]
        need(t >= t_last, f'fifo: legs are not chronological ({t!r} after {t_last!r})')
        t_last = t
        left = rem[j]
        while left > 0. and book and items[book[0]][LS] == -s:
            i = book[0]
            r = rem[i]
            m = r if r <= left else left
            closes.append((i, j, m))
            rem[i] = r - m
            left -= m
            matched[i] += m
            matched[j] += m
            if rem[i] <= 0.:
                book.popleft()
        rem[j] = left
        if left > 0.:
            book.append(j)
    return dict(items=items, n0=n0, closes=closes, rem=rem, matched=matched, book=list(book))


def paths(legs, pos0):
    """max |position| along the leg path from pos0, the end position and the inventory-action counts (module rules: PATHS)."""
    pos, mx = float(pos0), abs(float(pos0))
    acts = []
    k, n = 0, len(legs)
    while k < n:
        t, rec, role = legs[k][LT], legs[k][LK], legs[k][LR]
        a = 0.
        while k < n and legs[k][LT] == t and legs[k][LK] == rec and legs[k][LR] == role:
            d = legs[k][LS] * legs[k][LQ]
            a += d
            pos += d
            mx = max(mx, abs(pos))
            k += 1
        acts.append((t, a))
    per_t = collections.Counter(t for t, _ in acts)
    run_, before_t, prev = float(pos0), float(pos0), None
    nf = no = nr = nv = 0
    for t, a in acts:
        if t != prev:
            before_t, prev = run_, t
        b, af = before_t, before_t + a
        if abs(b) > POS_EPS:
            nf += 1
            if a * b < 0:
                no += 1
                nv += int(abs(af) > abs(b) + POS_EPS)
            nr += int(abs(af) < abs(b) - POS_EPS)
        run_ += a
    return dict(max_abs_pos=mx, end_pos=pos, n_actions=len(acts), n_nonflat=nf, n_opposing=no, n_abs_reduce=nr, n_overshoot=nv,
                n_simultaneous=sum(1 for t, _ in acts if per_t[t] > 1))


EMPTY_CL = dict(sh=np.zeros(0), pnl=np.zeros(0), fee=np.zeros(0), hold=np.zeros(0), rp=np.zeros(0, np.int64), kind=np.zeros(0, np.int64),
                dust=np.zeros(0, bool))


def account(fr, legs, self_rows, end_mark, open_mark, c0, pos0, where, devs, cnt):
    """Decomposition, identity, paths and actions of one address (or our arm) x market x view from its FIFO (module rules) -> (metrics, closes).
    c0 = the directly summed carried position (carry view with carried lots; None otherwise); pos0 = the paths' starting position."""
    items, rem, cls = fr['items'], fr['rem'], fr['closes']
    n = len(cls)
    cl = dict(sh=np.zeros(n), pnl=np.zeros(n), fee=np.zeros(n), hold=np.zeros(n), rp=np.zeros(n, np.int64), kind=np.zeros(n, np.int64))
    for k, (i, j, m) in enumerate(cls):
        e, x = items[i], items[j]
        need(x[LKIND] == WITHIN, f'{where}: a close exits on a carried lot')
        cl['sh'][k] = m
        cl['pnl'][k] = e[LS] * m * (x[LP] - e[LP])
        cl['fee'][k] = e[LF] * m / e[LQ] + x[LF] * m / x[LQ]
        cl['hold'][k] = x[LT] - e[LT]
        cl['rp'][k] = 2 * e[LR] + x[LR]
        cl['kind'][k] = e[LKIND]
    cl['dust'] = cl['sh'] < DUST_SH
    need(bool((cl['hold'] >= 0).all()), f'{where}: a close before its entry')
    carry = resid = fee_un = un_sh = un_car = 0.
    for i in fr['book']:
        it, r = items[i], rem[i]
        if it[LKIND] == CARRIED:
            carry += it[LS] * r * (end_mark - it[LP])
            un_car += r
        else:
            resid += it[LS] * r * (end_mark - it[LP])
            fee_un += it[LF] * r / it[LQ]
            un_sh += r
    sgr = float(sum(s * q * (end_mark - p) for s, q, p, _ in self_rows))
    sfe = float(sum(f for _, _, _, f in self_rows))
    close_pnl, close_fee = float(cl['pnl'].sum()), float(cl['fee'].sum())
    close_net = close_pnl - close_fee
    total = close_net + carry + resid - fee_un + sgr - sfe
    direct = float(sum(lg[LS] * lg[LQ] * (end_mark - lg[LP]) for lg in legs)) - float(sum(lg[LF] for lg in legs)) + sgr - sfe
    if c0 is not None:
        direct += c0 * (end_mark - open_mark)
    dev = check_identity(total, direct, where, devs)
    cnt['identity_checks'] = cnt.get('identity_checks', 0) + 1
    car = cl['kind'] == CARRIED
    met = dict(buy_sh=float(sum(lg[LQ] for lg in legs if lg[LS] == 1)), sell_sh=float(sum(lg[LQ] for lg in legs if lg[LS] == -1)),
               n_closes=int((~cl['dust']).sum()), n_dust=int(cl['dust'].sum()), close_sh=float(cl['sh'].sum()), close_pnl=close_pnl, close_fee=close_fee,
               close_net=close_net, carried_close_sh=float(cl['sh'][car].sum()), carried_close_net=float((cl['pnl'] - cl['fee'])[car].sum()),
               unclosed_sh=un_sh, unclosed_carried_sh=un_car, carry=carry, resid=resid, fee_unclosed=fee_un, self_gross=sgr, self_fee=sfe, total=total,
               direct=direct, identity_dev=dev, carried_pos_in=float(sum(items[i][LS] * items[i][LQ] for i in range(fr['n0']))))
    met.update(paths(legs, pos0))
    return met, cl


# ------------------------------------------------------------------ ours ------------------------------------------------------------------
def load_ours(P, man, man_sha, days, stat, cnt):
    """Our accepted records of the days (module rules: OURS) -> ({(day, m): (path_info, record)}, accepted arms.csv values, input hashes)."""
    lad = Path(P['ladder'])
    recs, hashes = {}, {}
    for d in days:
        for part in PARTS:
            mem = sorted(int(x) for x in man['parts'][part][d])
            if not mem:
                continue
            jp, gp = lad / part / f'{d}.json', lad / part / f'{d}.jsonl.gz'
            need(jp.is_file() and gp.is_file(), f'ours: the {part} record of {d} is missing')
            _reg(stat, f'ours/{part}/{d}.json', jp)
            _reg(stat, f'ours/{part}/{d}.jsonl.gz', gp)
            hdr = json.loads(jp.read_text())
            need(hdr.get('version') == DAY_VERSION and str(hdr.get('day')) == d and hdr.get('part') == part and hdr.get('status') == 'DAY_DONE',
                 f'ours: {part}/{d}: not a finished {DAY_VERSION} day record')
            need(sorted(int(x) for x in (hdr.get('members') or [])) == mem, f'ours: {part}/{d}: the members differ from the manifest part')
            need(ARM in (hdr.get('arms') or []), f'ours: {part}/{d}: the arm {ARM} is not in the day record')
            need(hdr.get('manifest_sha256') == man_sha, f'ours: {part}/{d}: the day record is bound to another manifest')
            b = gp.read_bytes()
            gsha = hashlib.sha256(b).hexdigest()
            need(gsha == (hdr.get('sha256') or {}).get('gz'), f'ours: {part}/{d}: the gz bytes differ from the day record sha256')
            hashes[f'ours/{part}/{d}.json'] = dict(path=str(jp), sha256=X._sha(jp), bytes=jp.stat().st_size)
            hashes[f'ours/{part}/{d}.jsonl.gz'] = dict(path=str(gp), sha256=gsha, bytes=len(b))
            seen = []
            for line in gzip.decompress(b).decode().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                m = int(row.get('m'))
                need(m in mem and str(row.get('day')) == d and (d, m) not in recs, f'ours: {part}/{d}: a row of market {m} outside the part or twice')
                need(isinstance(row.get(ARM), dict), f'ours: {part}/{d} {m}: no {ARM} record')
                recs[(d, m)] = (row.get('path_info') or {}, row[ARM])
                seen.append(m)
            need(sorted(seen) == mem, f'ours: {part}/{d}: the rows differ from the members')
            cnt['ours_day_records'] = cnt.get('ours_day_records', 0) + 1
    md = lad / 'measure'
    ap, cp = md / 'audit.json', md / 'arms.csv'
    need(ap.is_file() and cp.is_file(), 'ours: the accepted measurement (measure/audit.json, arms.csv) is missing')
    _reg(stat, 'ours/measure/audit.json', ap)
    _reg(stat, 'ours/measure/arms.csv', cp)
    audit = json.loads(ap.read_text())
    csha = X._sha(cp)
    need(csha == (audit.get('outputs') or {}).get('arms.csv'), 'ours: measure/arms.csv differs from the sha256 in measure/audit.json')
    hashes['ours/measure/audit.json'] = dict(path=str(ap), sha256=X._sha(ap), bytes=ap.stat().st_size)
    hashes['ours/measure/arms.csv'] = dict(path=str(cp), sha256=csha, bytes=cp.stat().st_size)
    A = pd.read_csv(cp, usecols=['day', 'm', 'arm', 'value'], dtype={'day': str, 'arm': str}, float_precision='round_trip')
    A = A[(A['arm'] == ARM).to_numpy()]
    acc = {}
    for d_, m_, v_ in A[['day', 'm', 'value']].itertuples(index=False, name=None):
        k = (str(d_), int(m_))
        need(k not in acc, f'ours: arms.csv holds {ARM} of {k} twice')
        acc[k] = float(v_)
    return recs, acc, hashes


def our_market(d, m, pinfo, rec):
    """One market's interval, END mark and legs from our record (module rules: OURS, INTERVAL) -> dict (status ok / no_interval)."""
    out = dict(day=d, m=m, status=OK, reason='', start=NAN, end=NAN, end_mark=NAN, mark_c=NAN, value_c=NAN, value_exact=None, legs=[], n_fills=0)
    ident = rec.get('ident')
    need(isinstance(ident, dict) and _num(ident.get('instrument_id')) and int(ident['instrument_id']) == m and str(ident.get('day')) == d,
         f'ours {d} {m}: the record ident names another market')
    s0, e0, er, op = (ident.get(k) for k in ('start_abs_ms', 'end_abs_ms', 'end_rel', 'opening_ms'))
    if not (_num(s0) and _num(e0) and _num(er) and _num(op) and er > 0):
        out.update(status='no_interval', reason='interval_unreadable')
        return out
    s0, e0, er, op = float(s0), float(e0), float(er), float(op)
    need(e0 == s0 + er, f'ours {d} {m}: ident end_abs_ms != start_abs_ms + end_rel')
    need(_num(rec.get('end')) and float(rec['end']) == er, f'ours {d} {m}: the record end differs from ident end_rel')
    se = (rec.get('settings') or {}).get('end_ms')
    need(se is None or (_num(se) and float(se) == er), f'ours {d} {m}: settings end_ms differs from ident end_rel')
    need(all(_num(pinfo.get(k)) for k in ('t_start', 'end_rel', 'ws')) and float(pinfo['t_start']) == s0 and float(pinfo['end_rel']) == er
         and float(pinfo['ws']) == op, f'ours {d} {m}: path_info (t_start, end_rel, ws) differs from the ident')
    out.update(start=s0, end=e0, opening_ms=op)
    mk, st = rec.get('mark'), rec.get('status')
    if st != 'ok':
        out.update(status='no_interval', reason=f'record_status_{st}')
        return out
    if not (_num(mk) and 0. < float(mk) < 100.):
        out.update(status='no_interval', reason='mark_unreadable')
        return out
    fills = rec.get('fills')
    need(isinstance(fills, list), f'ours {d} {m}: fills is not a list')
    legs = []
    for k, f in enumerate(fills):
        need(isinstance(f, list) and len(f) == 6, f'ours {d} {m}: fill {k} is not [t, gid, side, q, p, it]: {f!r}')
        t, gid, side, q, p, it = f
        need(_num(t) and _int(gid) and _int(side) and side in (0, 1) and _num(q) and q > 0 and _num(p) and 1. <= p <= 99. and _int(it),
             f'ours {d} {m}: malformed fill {k}: {f!r}')
        need(0. <= float(t) <= er, f'ours {d} {m}: fill {k} at {t!r} ms lies outside its own episode [0, {er!r}]')
        legs.append((s0 + float(t), 1 if side == 0 else -1, float(q), float(p) / 100., 0, 0., int(it), WITHIN))
    need(bool((side_of([lg[LT] for lg in legs], s0, e0) == 0).all()) if legs else True, f'ours {d} {m}: a fill outside [start, END]')
    cash, pos, val = engine_value(fills, float(mk))
    exact = bool(cash == rec.get('cash') and pos == rec.get('position') and val == rec.get('value'))
    need(exact, f'ours {d} {m}: the recorded cash / position / value do not reproduce exactly from the fills by the engine arithmetic '
                f'({cash!r} / {pos!r} / {val!r} vs {rec.get("cash")!r} / {rec.get("position")!r} / {rec.get("value")!r})')
    out.update(end_mark=float(mk) / 100., mark_c=float(mk), value_c=float(val), value_exact=exact, legs=legs, n_fills=len(fills))
    return out


def our_member(mk, devs, cnt):
    """Our arm in one market: row counts and both (equal) views; the accounting total must equal the recorded value / 100."""
    L = mk['legs']
    rows = dict(n_rows_pre=0, n_rows_in=len(L), n_rows_post=0, n_self_in=0, n_excl_pre=0, n_excl_in=0, n_excl_post=0, carried_pos_recorded=0.,
                carried_lots_recorded=0, carried_dust_dropped=0, carried_first_ms=NAN)
    if not L:
        return rows, {v: ('no_activity', None, None) for v in VIEWS}
    where = f'ours {mk["day"]} {mk["m"]}'
    met, cl = account(fifo(L), L, [], mk['end_mark'], NAN, None, 0., where, devs, cnt)
    vd = abs(met['total'] - mk['value_c'] / 100.)
    devs['value_accounting'] = max(devs.get('value_accounting', 0.), vd)
    need(vd <= REL_TOL * max(1., abs(mk['value_c']) / 100.), f'{where}: the accounting total {met["total"]!r} differs from the recorded value / 100 '
                                                             f'({mk["value_c"] / 100.!r})')
    return rows, {v: ('ok', met, cl) for v in VIEWS}


# ------------------------------------------------------------------ theirs ------------------------------------------------------------------
def read_selected(pf, groups, cols, mids, addrs, cnt, kind, dict_cols):
    """Rows of the BOOK markets (and, with addrs, of the top-10 addresses) from the given row groups; a group whose mid statistics hold no
    BOOK market is never read (counted) -> arrow Table of plain columns, or None."""
    md = pf.metadata
    names = [md.row_group(0).column(k).path_in_schema for k in range(md.num_columns)] if md.num_row_groups else []
    jm = names.index('mid') if 'mid' in names else None
    um = np.array(sorted({int(x) for x in mids}), np.int64)
    parts = []
    for i in groups:
        if jm is not None:
            st = md.row_group(i).column(jm).statistics
            if st is not None and st.has_min_max:
                k = int(np.searchsorted(um, int(st.min), 'left'))
                if k >= len(um) or um[k] > int(st.max):
                    cnt[f'{kind}_groups_pruned'] = cnt.get(f'{kind}_groups_pruned', 0) + 1
                    continue
        t = pf.read_row_group(i, columns=list(cols))
        cnt[f'{kind}_groups_read'] = cnt.get(f'{kind}_groups_read', 0) + 1
        cnt[f'{kind}_rows_scanned'] = cnt.get(f'{kind}_rows_scanned', 0) + t.num_rows
        msk = np.isin(X._np(t.column('mid')).astype(np.int64), um)
        if addrs is not None:
            a = t.column('addr')
            X._spelling(a, addrs, kind)
            msk &= X._mask(a, addrs)
        if msk.any():
            f = t.filter(pa.array(msk))
            parts.append(pa.table({c: X._str(f.column(c)) for c in cols}))       # dictionary -> plain strings (dict_cols and any stored one)
        del t
    return pa.concat_tables(parts) if parts else None


def their_arrays(T, addrs, mday):
    """The top-10 BOOK rows -> numpy arrays, the sid rank and the address x market groups in econ's FIFO order; refuses on a value econ never
    writes on an ok row, or on a BOOK market's row under another day."""
    A = dict(n=0 if T is None else T.num_rows, groups={})
    if T is None:
        return A
    A.update(a=X._codes(T.column('addr'), addrs, 'fills_valued address'), role=X._codes(T.column('role'), ROLES, 'fills_valued role'))
    for k, c in (('mid', 'mid'), ('mki', 'mk_idx'), ('pxr', 'px_rank'), ('s', 's'), ('settle', 'settle_ms')):
        A[k] = X._np(T.column(c)).astype(np.int64)
    for c in ('q', 'p', 'fee_fill', 'mq', 'rq'):
        A[c] = X._np(T.column(c)).astype(np.float64)
    A['self'] = X._np(pc.fill_null(T.column('self_trade'), False)).astype(bool)
    A['ok'] = X._eq(T.column('status'), OK)
    A['status'] = np.asarray(X._np(T.column('status')), dtype=object)
    dd = np.asarray(X._np(T.column('day')), dtype=object)
    want = np.array([mday.get(int(x), '') for x in A['mid']], dtype=object)
    need(bool((dd == want).all()), "fills_valued: a BOOK market's row carries another day than the market's (refused)")
    ok = A['ok']
    lot = ok & ~A['self']
    need(bool(np.isin(A['s'][ok], (-1, 1)).all()), 'fills_valued: s outside {-1, +1} on an ok row')
    need(bool((np.isfinite(A['q']) & (A['q'] > 0))[ok].all()), 'fills_valued: q not a positive number on an ok row')
    need(bool((np.isfinite(A['p']) & (A['p'] > 0) & (A['p'] < 1))[ok].all()), 'fills_valued: p outside (0, 1) on an ok row')
    need(bool(np.isfinite(A['fee_fill'])[ok].all()), 'fills_valued: fee_fill missing on an ok row')
    need(bool((np.isfinite(A['mq']) & np.isfinite(A['rq']))[lot].all()), 'fills_valued: mq / rq missing on an ok lot row')
    need(bool(((A['mq'] == 0) & (A['rq'] == 0))[ok & A['self']].all()), 'fills_valued: a self-trade row with econ mq / rq other than 0')
    sid = T.column('sid').combine_chunks()
    u = pc.unique(sid)
    rk = np.empty(len(u), np.int64)
    rk[X._np(pc.sort_indices(u)).astype(np.int64)] = np.arange(len(u))
    A['sidr'] = rk[X._np(pc.index_in(sid, value_set=u)).astype(np.int64)]
    key = A['mid'] * KEY_MUL + A['a']
    o = np.lexsort((A['role'], A['mki'], A['pxr'], A['sidr'], A['settle'], key))
    ks = key[o]
    uk, st = np.unique(ks, return_index=True)
    en = np.r_[st[1:], len(ks)]
    A['groups'] = {int(k): o[a:b] for k, a, b in zip(uk, st, en)}
    A['status_counts'] = dict(collections.Counter(A['status'].tolist()))
    return A


def mm_table(T, addrs):
    """econ maker_market rows of the top 10 in the BOOK markets, summed over days per address x market (key mid x 16 + rank - 1)."""
    if T is None:
        return {}
    a, mid = X._codes(T.column('addr'), addrs, 'maker_market address'), X._np(T.column('mid')).astype(np.int64)
    f = lambda c: X._np(T.column(c))  # noqa: E731
    df = pd.DataFrame(dict(key=mid * KEY_MUL + a, n_ok=f('n_mk').astype(np.int64) + f('n_tk').astype(np.int64), n_excl=f('n_excl').astype(np.int64),
                           net_pos=f('net_pos').astype(float), fee_fill=f('fee_fill').astype(float)))
    g = df.groupby('key', sort=True).sum()
    return {int(k): dict(n_ok=int(r[0]), n_excl=int(r[1]), net_pos=float(r[2]), fee_fill=float(r[3]))
            for k, r in zip(g.index, g[['n_ok', 'n_excl', 'net_pos', 'fee_fill']].itertuples(index=False, name=None))}


def completeness(A, MM, pop, devs, cnt):
    """Per address x BOOK market: our read rows against econ maker_market (module rules: THEIRS); refuses on the first failure."""
    want = {m * KEY_MUL + a for _, m in pop for a in range(NTOP)}
    have = {k for k in A['groups'] if k in want}
    mmk = {k for k in MM if k in want}
    need(have == mmk, f'completeness: the address x market groups read ({len(have)}) differ from econ maker_market ({len(mmk)}): '
                      f'{sorted(have ^ mmk)[:5]} (key = mid x 16 + rank - 1)')
    for k in sorted(have):
        idx = A['groups'][k]
        ok = A['ok'][idx]
        lot = ok & ~A['self'][idx]
        mine = dict(n_ok=int(ok.sum()), n_excl=int((~ok).sum()), net_pos=float((A['s'][idx] * A['q'][idx])[lot].sum()),
                    fee_fill=float(A['fee_fill'][idx][ok].sum()))
        e_ = MM[k]
        need(mine['n_ok'] == e_['n_ok'] and mine['n_excl'] == e_['n_excl'],
             f'completeness: mid {k // KEY_MUL} rank {k % KEY_MUL + 1}: {mine["n_ok"]} ok / {mine["n_excl"]} excluded rows read vs maker_market '
             f'{e_["n_ok"]} / {e_["n_excl"]}')
        for c in ('net_pos', 'fee_fill'):
            dv = abs(mine[c] - e_[c])
            devs[f'mm_{c}'] = max(devs.get(f'mm_{c}', 0.), dv)
            need(dv <= REL_TOL * max(1., abs(e_[c])), f'completeness: mid {k // KEY_MUL} rank {k % KEY_MUL + 1}: {c} {mine[c]!r} vs maker_market {e_[c]!r}')
    cnt['completeness_groups'] = len(have)


def their_member(A, idx, mk, om, a, devs, cnt):
    """One top-10 address in one market (module rules: THEIRS, FIFO CHECK, VIEWS) -> (row counts, {view: (status, metrics, closes)})."""
    start, end, M = mk['start'], mk['end'], mk['end_mark']
    where = f'rank {a + 1} {mk["day"]} {mk["m"]}'
    idx = np.zeros(0, np.int64) if idx is None else idx
    ok, slf = A['ok'][idx] if len(idx) else np.zeros(0, bool), A['self'][idx] if len(idx) else np.zeros(0, bool)
    wh = side_of(A['settle'][idx], start, end) if len(idx) else np.zeros(0, np.int8)
    lot = ok & ~slf
    li = idx[lot]
    L = list(zip(A['settle'][li].astype(float).tolist(), A['s'][li].tolist(), A['q'][li].tolist(), A['p'][li].tolist(), A['role'][li].tolist(),
                 A['fee_fill'][li].tolist(), A['sidr'][li].tolist(), [WITHIN] * len(li))) if len(li) else []
    if L:                                           # the same FIFO over the whole recorded history reproduces econ's mq / rq per row
        fh = fifo(L)
        tol = SH_TOL_ABS + SH_TOL_REL * float(A['q'][li].sum())
        dv = np.maximum(np.abs(np.asarray(fh['matched']) - A['mq'][li]), np.abs(np.asarray(fh['rem']) - A['rq'][li]))
        devs['econ_fifo'] = max(devs.get('econ_fifo', 0.), float(dv.max()))
        need(bool((dv <= tol).all()), f'econ FIFO check: {where}: {int((dv > tol).sum())} of {len(L)} rows whose matched / residual shares differ '
                                      f'from econ mq / rq (max {float(dv.max()):.3g} sh)')
        cnt['econ_fifo_rows'] = cnt.get('econ_fifo_rows', 0) + len(L)
    lw = wh[lot]
    pre = [lg for lg, w in zip(L, lw.tolist()) if w == -1]
    inn = [lg for lg, w in zip(L, lw.tolist()) if w == 0]
    si = idx[ok & slf & (wh == 0)]
    S = list(zip(A['s'][si].tolist(), A['q'][si].tolist(), A['p'][si].tolist(), A['fee_fill'][si].tolist())) if len(si) else []
    rows = dict(n_rows_pre=len(pre), n_rows_in=len(inn), n_rows_post=int((lw == 1).sum()), n_self_in=len(S), n_excl_pre=int((~ok & (wh == -1)).sum()),
                n_excl_in=int((~ok & (wh == 0)).sum()), n_excl_post=int((~ok & (wh == 1)).sum()))
    for st_ in A['status'][idx][~ok].tolist():
        cnt[f'excluded_rows:{st_}'] = cnt.get(f'excluded_rows:{st_}', 0) + 1
    lots, c0, drop, first = [], 0., 0, NAN
    if pre:
        fp = fifo(pre)
        it_ = fp['items']
        lots = [(it_[i][LT], it_[i][LS], fp['rem'][i], om['mark'], it_[i][LR], 0., it_[i][LK], CARRIED) for i in fp['book']]
        if lots and sum(x[LQ] for x in lots) < DUST_SH:
            lots, drop = [], 1
        if lots:
            c0 = float(sum(lg[LS] * lg[LQ] for lg in pre))
            first = float(min(x[LT] for x in lots))
    rows.update(carried_pos_recorded=float(sum(x[LS] * x[LQ] for x in lots)), carried_lots_recorded=len(lots), carried_dust_dropped=drop,
                carried_first_ms=first)
    base = 'ok' if (rows['n_rows_in'] + rows['n_self_in']) > 0 else ('rows_excluded' if rows['n_excl_in'] > 0 else 'no_activity')
    out = {}
    for v in VIEWS:
        st = 'no_opening_mark' if (base == 'ok' and v == 'carry' and lots and not om['ok']) else base
        if st != 'ok':
            out[v] = (st, None, None)
            continue
        init = lots if v == 'carry' else []
        met, cl = account(fifo(inn, init), inn, S, M, om['mark'], c0 if (v == 'carry' and lots) else None, float(sum(x[LS] * x[LQ] for x in init)),
                          f'{where} {v}', devs, cnt)
        out[v] = ('ok', met, cl)
    # the member's in-window flow summed directly (no FIFO; the party-level recompute) and, without an ok row inside, its carried lots alone
    din = (float(sum(lg[LS] * lg[LQ] * (M - lg[LP]) for lg in inn)) - float(sum(lg[LF] for lg in inn))
           + float(sum(s * q * (M - p) for s, q, p, _ in S)) - float(sum(f for _, _, _, f in S)))
    co = None
    if base != 'ok' and lots and om['ok']:
        co = ('ok',) + account(fifo([], lots), [], [], M, om['mark'], c0, float(sum(x[LS] * x[LQ] for x in lots)), f'{where} carry_only', devs, cnt)
        cnt['member_carry_only'] = cnt.get('member_carry_only', 0) + 1
    return rows, out, dict(base=base, lots=lots, c0=c0 if lots else 0., din=din, carry_only=co)


def combine(members, v, mk, om, where, devs, cnt):
    """A party from its members' results (rows, views, extra) in one view (module rules: PARTIES, PARTY IDENTITY) -> dict(status, met, cl, rows,
    mstat). Active iff a member has an ok row inside; zero view = its active members; carry view = its active members plus every member with
    carried lots and no ok row inside (carry_only). The party total is checked against an independent party-level direct recompute."""
    rows = {k: (float(sum(mm[0][k] for mm in members)) if k == 'carried_pos_recorded' else int(sum(mm[0][k] for mm in members))) for k in ROW_KEYS}
    fs = [mm[0]['carried_first_ms'] for mm in members if math.isfinite(mm[0]['carried_first_ms'])]
    rows['carried_first_ms'] = float(min(fs)) if fs else NAN
    sts = collections.Counter()
    for mm in members:
        ex = mm[2]
        if ex['base'] == 'ok':
            sts[mm[1][v][0]] += 1
        elif v == 'carry' and ex['lots']:
            sts['carry_only' if om['ok'] else 'no_opening_mark'] += 1
        else:
            sts[ex['base']] += 1
    active = [mm for mm in members if mm[2]['base'] == 'ok']
    out = dict(status=None, met=None, cl=None, rows=rows, mstat=sts)
    if not active:
        out['status'] = 'rows_excluded' if any(mm[2]['base'] == 'rows_excluded' for mm in members) else 'no_activity'
        return out
    if sts.get('no_opening_mark'):
        out['status'] = 'no_opening_mark'
        return out
    acc = [mm[1][v] for mm in active]
    if v == 'carry' and not _MUT_EXCLUDE_CARRY_ONLY:
        acc += [mm[2]['carry_only'] for mm in members if mm[2]['base'] != 'ok' and mm[2]['lots']]
    met = {k: (int(sum(x[1][k] for x in acc)) if k.startswith('n_') else float(sum(x[1][k] for x in acc))) for k in MET_ADD}
    met['max_abs_pos'] = float(max(x[1]['max_abs_pos'] for x in acc))
    met['identity_dev'] = float(max(x[1]['identity_dev'] for x in acc))
    cl = {k: np.concatenate([x[2][k] for x in acc]) for k in EMPTY_CL}
    direct = float(sum(mm[2]['din'] for mm in active))             # independent of which members were accounted
    if v == 'carry' and any(mm[2]['lots'] for mm in members):
        direct += float(sum(mm[2]['c0'] for mm in members)) * (mk['end_mark'] - om['mark'])
    met['party_identity_dev'] = check_identity(met['total'], direct, f'party {where} {v}', devs, key='party_identity')
    cnt['party_identity_checks'] = cnt.get('party_identity_checks', 0) + 1
    out.update(status='ok', met=met, cl=cl)
    return out


# ------------------------------------------------------------------ markets and marks ------------------------------------------------------------------
def day_markets(path, d, mids, stat):
    """markets_<day> ws / we (seconds -> ms) and status of the day's BOOK markets."""
    need(Path(path).is_file(), f'markets: input missing: {Path(path).name}')
    _reg(stat, f'v2/markets_{d}.parquet', path)
    X.check_schema(path, tuple(MKT_SPEC), MKT_SPEC, 'markets')
    T = pq.read_table(path, columns=list(MKT_SPEC)).to_pandas()
    need(T['mid'].is_unique, f'markets {d}: market id not unique')
    T = T[T['mid'].isin(list(mids)).to_numpy()]
    return {int(r[0]): (int(round(float(r[1]) * 1000)), int(round(float(r[2]) * 1000)), str(r[3])) for r in
            T[['mid', 'ws', 'we', 'status']].itertuples(index=False, name=None)}


def day_frames(path, d, mids, stat, cnt):
    """pf_mt_econ.frames of the day's BOOK markets (tob_<day>, every row of the day checked), or None without a frame."""
    need(Path(path).is_file(), f'tob: input missing: {Path(path).name}')
    _reg(stat, f'v2/tob_{d}.parquet', path)
    X.check_schema(path, TOB_COLS, C.SCHEMAS['tob'], 'tob')
    pf = pq.ParquetFile(path, read_dictionary=sorted(TOB_DICT))
    T = read_selected(pf, range(pf.metadata.num_row_groups), TOB_COLS, mids, None, cnt, f'tob_{d}', TOB_DICT)
    if T is None or T.num_rows == 0:
        return None
    need(bool(pc.all(pc.equal(T.column('day'), d)).as_py()), f'tob {d}: a frame of another day')
    df = T.to_pandas()
    cnt[f'tob_{d}_frames'] = int(len(df))
    return E.frames(df)


def mark_at(Fr, m, x):
    """pf_mt_econ.mid_at for one market at floor(x) -> dict(mark, age, st, ok)."""
    mm, age, st = E.mid_at(Fr, np.array([m], np.int64), np.array([int(math.floor(x))], np.int64))
    return dict(mark=float(mm[0]), age=int(age[0]), st=E.MARK_ST[int(st[0])], ok=bool(int(st[0]) == 0 and np.isfinite(mm[0])))


# ------------------------------------------------------------------ readings ------------------------------------------------------------------
def reading_of(o, t, dT, dE):
    """The frozen reading of one comparison, the support rule aside (module rules: READING; first matching row)."""
    if not (math.isfinite(o) and math.isfinite(t)):
        return 'mixed_unsupported'
    if o < 0 and t > 0:
        return 'our_turnover_loses'
    if ((o >= 0 and t >= 0) or (o < 0 and t < 0)) and abs(dE) > abs(dT):
        return 'exposure_gap'
    if o < 0 and t < 0:
        return 'both_adverse'
    return 'mixed_unsupported'


def per_share(net, sh, n_closes):
    return net / sh if (n_closes > 0 and sh > 0) else NAN


def compare(ou, th):
    """The reading inputs of one comparison from two metric dicts (single market or pooled sums)."""
    o, t = per_share(ou['close_net'], ou['close_sh'], ou['n_closes']), per_share(th['close_net'], th['close_sh'], th['n_closes'])
    dT = ou['close_net'] - th['close_net']
    dE = (ou['carry'] + ou['resid']) - (th['carry'] + th['resid'])
    dF = th['fee_unclosed'] - ou['fee_unclosed']
    dS = (ou['self_gross'] - ou['self_fee']) - (th['self_gross'] - th['self_fee'])
    return dict(o_c_sh=100. * o, t_c_sh=100. * t, d_turnover=dT, d_exposure=dE, d_fee_unclosed=dF, d_self=dS, d_total=ou['total'] - th['total'],
                exposure_larger=bool(abs(dE) > abs(dT)), reading_raw=reading_of(o, t, dT, dE))


def readings(res, keys):
    """Per comparison party x view: both-active markets, per-market readings (support rule applied), counts and the pooled reading."""
    out = {}
    for p in COMPARE:
        for v in VIEWS:
            both = [k for k in keys if res[k]['ours'][v]['status'] == 'ok' and res[k][p][v]['status'] == 'ok']
            sup = len(both) >= MIN_MARKETS
            per = {}
            for k in both:
                c = compare(res[k]['ours'][v]['met'], res[k][p][v]['met'])
                c['reading'] = c['reading_raw'] if sup else 'mixed_unsupported'
                per[k] = c
            sums = {}
            for side in ('ours', p):
                z = {f: 0. for f in ('close_net', 'close_sh', 'n_closes', 'carry', 'resid', 'fee_unclosed', 'self_gross', 'self_fee', 'total')}
                for k in both:
                    for f in z:
                        z[f] += res[k][side][v]['met'][f]
                sums[side] = z
            pooled = compare(sums['ours'], sums[p])
            pooled['reading'] = pooled['reading_raw'] if sup else 'mixed_unsupported'
            out[(p, v)] = dict(both=both, supported=sup, per=per, pooled=pooled, sums=sums)
    return out


# ------------------------------------------------------------------ outputs ------------------------------------------------------------------
def per_market_frame(pop, mks, oms, eds, res, RD):
    rows = []
    for k in pop:
        mk, om, ed = mks[k], oms.get(k), eds.get(k)
        for p in PARTIES:
            for v in VIEWS:
                r = res[k][p][v]
                row = dict.fromkeys(PM_COLS, NAN)
                row.update(day=k[0], m=k[1], party=p, view=v, primary=v == PRIMARY, status=r['status'],
                           n_members=1 if p == 'ours' else len(MEMBERS[p]), market_status=mk['status'], market_reason=mk['reason'],
                           start_ms=mk['start'], end_ms=mk['end'], end_mark=mk['end_mark'])
                for s_ in MSTAT:
                    row[f'n_members_{s_}'] = int(r['mstat'].get(s_, 0))
                if om is not None:
                    row.update(open_mark=om['mark'], open_status=om['st'], open_age_ms=om['age'])
                if ed is not None:
                    row.update(tob_mid_end=ed['mark'], tob_end_status=ed['st'], end_mark_minus_tob_c=100. * (mk['end_mark'] - ed['mark']) if ed['ok'] else NAN)
                if p == 'ours':
                    row.update(value_recorded_c=mk['value_c'], value_exact=mk['value_exact'])
                if r['rows'] is not None:
                    row.update({c: r['rows'][c] for c in ROW_KEYS + ('carried_first_ms',)})
                if r['met'] is not None:
                    row.update({c: r['met'][c] for c in MET_ADD + ('identity_dev', 'max_abs_pos')})
                    row['party_identity_dev'] = r['met'].get('party_identity_dev', NAN)
                    row.update(close_net_c_sh=_cpc(r['met']['close_net'], r['met']['close_sh']) if r['met']['n_closes'] > 0 else NAN,
                               exposure=r['met']['carry'] + r['met']['resid'])
                    row.update(hold_stats(r['cl']['hold'], r['cl']['sh'], r['cl']['dust']))
                if p != 'ours':
                    rd = RD[(p, v)]
                    so, sp = res[k]['ours'][v]['status'], r['status']
                    if k in rd['per']:
                        c = rd['per'][k]
                        row.update(both_active=True, pair_status='both_active', **{f: c[f] for f in ('reading', 'reading_raw', 'o_c_sh', 't_c_sh', 'd_turnover',
                                                                                                         'd_exposure', 'd_fee_unclosed', 'd_self', 'd_total',
                                                                                                         'exposure_larger')})
                    else:
                        row.update(both_active=False, pair_status=f'not_both_active:ours_{so}|theirs_{sp}')
                rows.append(row)
    return pd.DataFrame(rows, columns=PM_COLS)


def closes_frame(res, keys):
    """closes_by_role rows: party x view x scope x entry kind x role pair (module rules: OUTPUTS)."""
    rows = []
    for v in VIEWS:
        okp = lambda p, k: res[k][p][v]['status'] == 'ok'  # noqa: E731
        scopes = {'ours': {'active': [k for k in keys if okp('ours', k)]}}
        for p in COMPARE:
            both = [k for k in keys if okp('ours', k) and okp(p, k)]
            scopes[p] = {'active': [k for k in keys if okp(p, k)], 'both_active': both}
            scopes['ours'][f'both_active_{p}'] = both
        for p in PARTIES:
            for sc, ks in scopes[p].items():
                cls = [res[k][p][v]['cl'] for k in ks]
                CL = {f: np.concatenate([c[f] for c in cls]) if cls else EMPTY_CL[f] for f in EMPTY_CL}
                for ek in KINDS + ('all',):
                    mk_ = np.ones(len(CL['sh']), bool) if ek == 'all' else CL['kind'] == KINDS.index(ek)
                    for rp in PAIRS + ('all',):
                        m = mk_ if rp == 'all' else mk_ & (CL['rp'] == PAIRS.index(rp))
                        sh, pnl, fee = float(CL['sh'][m].sum()), float(CL['pnl'][m].sum()), float(CL['fee'][m].sum())
                        r = dict(party=p, view=v, scope=sc, entry_kind=ek, role_pair=rp, n_markets=len(ks), n_closes=int((m & ~CL['dust']).sum()),
                                 n_dust=int((m & CL['dust']).sum()), sh=sh, pnl=pnl, fee=fee, net=pnl - fee, c_gross_sh=_cpc(pnl, sh), c_net_sh=_cpc(pnl - fee, sh))
                        r.update(hold_stats(CL['hold'][m], CL['sh'][m], CL['dust'][m]))
                        rows.append(r)
    return pd.DataFrame(rows, columns=CR_COLS)


def answer_obj(res, keys, RD, days, addrs):
    """answer.json (module rules: READING): statuses, reading counts with denominators, the pooled reading with its market mix, per-day sums."""
    statuses = {p: {v: dict(collections.Counter(res[k][p][v]['status'] for k in keys)) for v in VIEWS} for p in PARTIES}
    rd_out, by_day = {}, {}
    for p in COMPARE:
        rd_out[p], by_day[p] = {}, {}
        for v in VIEWS:
            R = RD[(p, v)]
            both = R['both']
            sh_o = float(sum(res[k]['ours'][v]['met']['close_sh'] for k in both))
            sh_t = float(sum(res[k][p][v]['met']['close_sh'] for k in both))
            counts, raw = {}, {}
            for r_ in READINGS:
                ks = [k for k in both if R['per'][k]['reading'] == r_]
                counts[r_] = dict(n=len(ks), n_markets_both=len(both), our_close_sh=float(sum(res[k]['ours'][v]['met']['close_sh'] for k in ks)),
                                  their_close_sh=float(sum(res[k][p][v]['met']['close_sh'] for k in ks)), markets=[f'{k[0]}:{k[1]}' for k in ks])
                raw[r_] = int(sum(1 for k in both if R['per'][k]['reading_raw'] == r_))
            pooled = dict(R['pooled'], our=R['sums']['ours'], theirs=R['sums'][p])
            rd_out[p][v] = dict(primary=v == PRIMARY, n_markets=len(keys), n_markets_both=len(both), supported=R['supported'], our_close_sh=sh_o,
                                their_close_sh=sh_t, counts=counts, counts_raw_before_support_rule=raw, pooled_secondary=pooled,
                                market_mix=dict(collections.Counter(k[0] for k in both)), markets_both=[f'{k[0]}:{k[1]}' for k in both])
            dd = {}
            for d in days:
                kd = [k for k in both if k[0] == d]
                dd[d] = dict(n_markets_both=len(kd), **{f'{side}_{f}': float(sum(res[k][s_][v]['met'][f] for k in kd))
                                                        for side, s_ in (('our', 'ours'), ('their', p))
                                                        for f in ('close_net', 'close_sh', 'carry', 'resid', 'fee_unclosed', 'total')})
            by_day[p][v] = dd
    n = len(days)
    return dict(question='where does the observed gap lie between our CURRENT_ASYNC_FAST10 ledger and the ex-ante top makers on the same BOOK '
                         'markets and intervals?', contract=CONTRACT, review=REVIEW, arm=ARM, primary_view=PRIMARY, views=list(VIEWS),
                parties={p: [dict(rank=i + 1, addr=addrs[i]) for i in MEMBERS[p]] for p in COMPARE}, days=list(days), n_markets=len(keys),
                population_by_day=dict(collections.Counter(k[0] for k in keys)), min_markets=MIN_MARKETS, rule=RULES['reading'],
                ci=f'none_G{n}_lt_5' if n < 5 else f'eligible_G{n}', units='USD totals; per-share columns in cents',
                statuses=statuses, readings=rd_out, by_day_secondary=by_day,
                descriptive='consumed days; descriptive only: not an avoided-loss estimate, an imitation claim, a validation or a profitability claim; '
                            'no model skill measured')


# ------------------------------------------------------------------ the build ------------------------------------------------------------------
def build(P=None, days=None, write=False, require_frozen=False, n_expected=None, guard_mib=C.GUARD_MIB, log=print):
    """The matched-gap build (module rules): preflight, ours, theirs (rows, completeness, FIFO check), marks, accounting, readings; write-once
    outputs when write, else the in-memory result (rehearsal / selftest)."""
    P = dict(default_paths(), **(P or {}))
    t0, c0, asof0 = time.time(), time.process_time(), _now()
    stages, tc, mem = {}, [time.process_time()], {}

    def stage(name):
        now = time.process_time()
        stages[name] = round(stages.get(name, 0.) + now - tc[0], 3)
        tc[0] = now
        mem[name] = E.guard(guard_mib, name)
        need(now - c0 < CPU_CAP_S, f'CPU self-guard: {now - c0:.0f} s >= {CPU_CAP_S:.0f} s at {name}')

    out = Path(P['out'])
    if write:
        need(not out.exists(), f'the matched-gap output exists (write-once): {out}')
        need(out.parent.is_dir(), f'the receipt directory is missing: {out.parent}')
    git = X._git(list(TOOL_FILES))
    if require_frozen:
        need(bool(git.get('tracked')) and bool(git.get('unmodified')), 'freeze: the tool, its selftest and the contract must be tracked and unmodified '
                                                                      'before the run (commit first)')
    man_p = Path(P['manifest'])
    need(man_p.is_file(), f'the BOOK manifest is missing: {man_p.name}')
    stat, cnt, devs = {}, {}, {}
    _reg(stat, 'manifest', man_p)
    man = json.loads(man_p.read_text())
    mdays = [str(x) for x in (man.get('days') or [])]
    need(len(mdays) >= 1 and len(set(mdays)) == len(mdays) and all(x in C.MEASURE_DAYS for x in mdays),
         'the BOOK manifest days must be distinct measurement days (pf_mt_common.MEASURE_DAYS; refused before any path is built)')
    days = tuple(sorted(mdays)) if days is None else tuple(sorted(str(x) for x in days))
    need(len(days) >= 1 and len(set(days)) == len(days) and all(x in mdays for x in days), 'days must be distinct days of the BOOK manifest')
    comp, parts = man.get('complete') or {}, man.get('parts') or {}
    need(all(d in comp for d in mdays) and all(d in (parts.get(pt) or {}) for pt in PARTS for d in mdays), 'the BOOK manifest lacks complete / parts')
    for d in mdays:
        need(sorted(int(x) for pt in PARTS for x in parts[pt][d]) == sorted(int(x) for x in comp[d]), f'the parts of {d} do not partition its complete markets')
    pop = [(d, m) for d in days for m in sorted(int(x) for x in comp[d])]
    need(len({m for _, m in pop}) == len(pop), 'a BOOK market listed twice')
    need(n_expected is None or len(pop) == n_expected, f'the population holds {len(pop)} markets, the contract {n_expected}')
    mday = {m: d for d, m in pop}
    man_sha = X._sha(man_p)
    stage('preflight')
    # ---- ours ----
    recs, acc, hashes = load_ours(P, man, man_sha, days, stat, cnt)
    need(set(recs) == set(pop), 'ours: the records are not exactly the BOOK markets of the days')
    mks = {}
    for k in pop:
        mk = our_market(k[0], k[1], *recs[k])
        if mk['status'] == OK:
            need(k in acc, f'ours {k}: no accepted arms.csv value')
            dv = abs(acc[k] - mk['value_c'])
            devs['accepted_value_c'] = max(devs.get('accepted_value_c', 0.), dv)
            need(dv <= VALUE_TOL_C, f'ours {k}: the record value {mk["value_c"]!r} differs from the accepted arms.csv value {acc[k]!r}')
        mks[k] = mk
    cnt['ours_no_interval'] = sum(1 for k in pop if mks[k]['status'] != OK)
    del recs
    stage('ours')
    # ---- identities, markets windows ----
    econ, root = Path(P['econ']), Path(P['root'])
    idp = _reg(stat, 'econ/identities.json', econ / 'identities.json')
    addrs = X.load_top10(idp)
    for d in days:
        mids = [m for dd, m in pop if dd == d]
        W = day_markets(root / f'markets_{d}.parquet', d, mids, stat)
        for m in mids:
            mk = mks[(d, m)]
            need(m in W, f'markets {d}: BOOK market {m} missing')
            ws, we, mst = W[m]
            mk['market_flags'] = mst
            cnt[f'market_status:{mst}'] = cnt.get(f'market_status:{mst}', 0) + 1
            if math.isfinite(mk['start']):
                need(float(ws) == mk['opening_ms'] and mk['start'] >= ws and mk['end'] <= we - E.END_GUARD_MS,
                     f'markets {d} {m}: our interval [{mk["start"]!r}, {mk["end"]!r}] (opening {mk["opening_ms"]!r}) is not inside ws {ws} / we - 2 s {we - E.END_GUARD_MS}')
    stage('markets')
    # ---- theirs: rows (one pass over the days' row groups), completeness ----
    fvp = _reg(stat, 'econ/fills_valued.parquet', econ / 'fills_valued.parquet')
    mmp = _reg(stat, 'econ/maker_market.parquet', econ / 'maker_market.parquet')
    X.check_schema(fvp, FV_COLS, E.FV, 'fills_valued')
    X.check_schema(mmp, MM_COLS, E.MM, 'maker_market')
    pf = pq.ParquetFile(fvp, read_dictionary=sorted(FV_DICT))
    fg, fv_info = X.day_groups(pf, days, 'fills_valued')
    T = read_selected(pf, sorted({i for d in days for i, _ in fg[d]}), FV_COLS, [m for _, m in pop], addrs, cnt, 'fills_valued', FV_DICT)
    A = their_arrays(T, addrs, mday)
    del T, pf
    E.trim()
    stage('fills_valued')
    mmf = pq.ParquetFile(mmp, read_dictionary=sorted(MM_DICT))
    mg, mm_info = X.day_groups(mmf, days, 'maker_market')
    Tm = read_selected(mmf, sorted({i for d in days for i, _ in mg[d]}), MM_COLS, [m for _, m in pop], addrs, cnt, 'maker_market', MM_DICT)
    MM = mm_table(Tm, addrs)
    del Tm, mmf
    completeness(A, MM, pop, devs, cnt)
    stage('maker_market')
    # ---- opening marks (and the END diagnostic) ----
    oms, eds = {}, {}
    for d in days:
        mids = [m for dd, m in pop if dd == d]
        Fr = day_frames(root / f'tob_{d}.parquet', d, mids, stat, cnt)
        for m in mids:
            mk = mks[(d, m)]
            if math.isfinite(mk['start']):
                oms[(d, m)] = mark_at(Fr, m, mk['start'])
                eds[(d, m)] = mark_at(Fr, m, mk['end'])
                cnt[f'open_status:{oms[(d, m)]["st"]}'] = cnt.get(f'open_status:{oms[(d, m)]["st"]}', 0) + 1
        del Fr
    E.trim()
    stage('tob')
    # ---- the accounting ----
    res = {}
    empty = dict(n_rows_pre=0, n_rows_in=0, n_rows_post=0, n_self_in=0, n_excl_pre=0, n_excl_in=0, n_excl_post=0, carried_pos_recorded=0.,
                 carried_lots_recorded=0, carried_dust_dropped=0, carried_first_ms=NAN)
    for k in pop:
        mk = mks[k]
        if mk['status'] != OK:
            res[k] = {p: {v: dict(status='no_interval', met=None, cl=None, rows=None, mstat=collections.Counter(no_interval=1 if p == 'ours' else len(MEMBERS[p])))
                          for v in VIEWS} for p in PARTIES}
            continue
        om = oms[k]
        orow, oview = our_member(mk, devs, cnt)
        res[k] = {'ours': {v: dict(status=oview[v][0], met=oview[v][1], cl=oview[v][2], rows=orow, mstat=collections.Counter([oview[v][0]]))
                           for v in VIEWS}}
        mem_res = []
        for a in range(NTOP):
            idx = A['groups'].get(k[1] * KEY_MUL + a)
            mem_res.append(their_member(A, idx, mk, om, a, devs, cnt) if idx is not None else
                           (dict(empty), {v: ('no_activity', None, None) for v in VIEWS}, dict(base='no_activity', lots=[], c0=0., din=0., carry_only=None)))
        for p in COMPARE:
            res[k][p] = {v: combine([mem_res[i] for i in MEMBERS[p]], v, mk, om, f'{p} {k[0]} {k[1]}', devs, cnt) for v in VIEWS}
        for p in PARTIES:                                   # CROSS-VIEW: carry - zero = carried_pos_recorded x (END - open)
            rc, rz = res[k][p]['carry'], res[k][p]['zero']
            if rc['status'] == OK and rz['status'] == OK:
                cp = float(rc['rows']['carried_pos_recorded'])
                check_cross_view(rc['met']['total'], rz['met']['total'], cp, mk['end_mark'], om['mark'], f'{p} {k[0]} {k[1]}', devs, cnt)
                need(abs(rc['met']['carried_pos_in'] - cp) <= REL_TOL * max(1., abs(cp)) and rz['met']['carried_pos_in'] == 0.,
                     f'carried position: {p} {k[0]} {k[1]}: {rc["met"]["carried_pos_in"]!r} entered the carry view, {cp!r} recorded')
                if abs(cp) > DUST_SH:
                    cnt['rows_carried_pos'] = cnt.get('rows_carried_pos', 0) + 1
                    cnt['rows_carried_pos_views_differ'] = cnt.get('rows_carried_pos_views_differ', 0) + int(rc['met']['total'] != rz['met']['total'])
                    cnt['rows_carried_pos_end_eq_open'] = cnt.get('rows_carried_pos_end_eq_open', 0) + int(mk['end_mark'] == om['mark'])
    stage('accounting')
    RD = readings(res, pop)
    PM = per_market_frame(pop, mks, oms, eds, res, RD)
    CR = closes_frame(res, pop)
    ANS = answer_obj(res, pop, RD, days, addrs)
    stage('tables')
    # ---- inputs unchanged since they were opened; hashes ----
    for k, (p, sz, mt) in stat.items():
        if k not in hashes:
            hashes[k] = dict(path=str(p), sha256=X._sha(p), bytes=p.stat().st_size)
    for k, (p, sz, mt) in stat.items():
        need((p.stat().st_size, p.stat().st_mtime_ns) == (sz, mt) and hashes[k]['bytes'] == sz, f'input {k} changed during the run (refused)')
    stage('hashes')
    checks = {k: dict(max_dev=v_, ok=True) for k, v_ in sorted(devs.items())}
    checks['value_exact'] = dict(markets=int(sum(1 for k in pop if mks[k]['value_exact'])), of=int(sum(1 for k in pop if mks[k]['status'] == OK)), ok=True)
    srcs = [HERE / 'matched_gap.py', HERE / 'matched_gap_selftest.py', HERE / 'closes.py', RESEARCH / 'pf_mt_common.py', RESEARCH / 'pf_mt_econ.py']
    rec = dict(version=VERSION, module='tools/matched_gap.py', contract=CONTRACT, review=REVIEW, run=C.RUN, out=str(out), fix='r1: carry view accounts '
               'every member\'s carried lots (carry_only); party identity and cross-view checks', arm=ARM, days=list(days), n_markets=len(pop),
               asof_start=asof0, addresses=[dict(rank=i + 1, addr=a, top3=i < 3) for i, a in enumerate(addrs)], parties={p: list(MEMBERS[p]) for p in COMPARE},
               inputs=hashes, sources={p.name: X._sha(p) for p in srcs if p.is_file()}, contract_sha256=X._sha(CONTRACT_FILE) if CONTRACT_FILE.is_file() else None,
               git=git, row_groups=dict(fills_valued=fv_info, maker_market=mm_info), counts=dict(sorted(cnt.items())),
               statuses=ANS['statuses'], fills_valued_status_counts=A.get('status_counts', {}), checks=checks,
               constants=dict(rel_tol=REL_TOL, sh_tol_abs=SH_TOL_ABS, sh_tol_rel=SH_TOL_REL, dust_sh=DUST_SH, pos_eps=POS_EPS, value_tol_c=VALUE_TOL_C,
                              min_markets=MIN_MARKETS, quants=list(QUANTS), cpu_cap_s=CPU_CAP_S, n_book=N_BOOK),
               rules=RULES, not_measured='no model skill, policy or profitability: an accounting of recorded and simulated fills on four consumed days',
               slot=os.environ.get('PF_SLOT'), argv=sys.argv)
    result = dict(per_market=PM, closes_by_role=CR, answer=ANS, receipt=rec, mks=mks, oms=oms, eds=eds, res=res, readings=RD)
    if not write:
        rec.update(cpu_s=dict(by_stage=stages, total=round(time.process_time() - c0, 3), cap=CPU_CAP_S), wall_s=round(time.time() - t0, 2),
                   memory=dict(guard_mib=guard_mib, vmhwm_mib=E.hwm(), by_stage=mem), asof_end=_now())
        return result
    tmp = out.parent / f'.{out.name}.partial_{time.strftime("%Y%m%dT%H%M%S", time.gmtime())}_{os.getpid()}'
    need(not tmp.exists(), 'the partial output directory exists')
    tmp.mkdir()
    try:
        PM.to_csv(tmp / 'per_market.csv', index=False)
        CR.to_csv(tmp / 'closes_by_role.csv', index=False)
        X._json(tmp / 'answer.json', ANS)
        outputs = {p.name: dict(sha256=X._sha(p), bytes=p.stat().st_size) for p in sorted(tmp.iterdir())}
        outputs['per_market.csv']['rows'], outputs['closes_by_role.csv']['rows'] = len(PM), len(CR)
        stage('write')
        rec.update(outputs=outputs, cpu_s=dict(by_stage=stages, total=round(time.process_time() - c0, 3), cap=CPU_CAP_S), wall_s=round(time.time() - t0, 2),
                   memory=dict(guard_mib=guard_mib, vmhwm_mib=E.guard(guard_mib, 'receipt'), by_stage=mem, process='one process'), asof_end=_now())
        X._json(tmp / 'receipt.json', rec)
        need(not out.exists(), f'the matched-gap output appeared during the run (write-once): {out}')
        os.rename(tmp, out)
        return result
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def rehearsal_summary(r):
    """What a no-write rehearsal may print: format facts, checks, counts, statuses, timings, memory - no P&L and no reading."""
    rec, mks = r['receipt'], r['mks']
    okk = [k for k in sorted(mks) if mks[k]['status'] == OK]
    facts = dict(markets=len(mks), interval_ok=len(okk), fills=int(sum(mks[k]['n_fills'] for k in okk)),
                 fills_at_END=int(sum(1 for k in okk for lg in mks[k]['legs'] if lg[LT] == mks[k]['end'])),
                 fills_at_start=int(sum(1 for k in okk for lg in mks[k]['legs'] if lg[LT] == mks[k]['start'])),
                 opening_wait_ms=[mks[k]['start'] - mks[k]['opening_ms'] for k in okk], interval_ms=[mks[k]['end'] - mks[k]['start'] for k in okk],
                 value_exact=[bool(mks[k]['value_exact']) for k in okk], market_flags=sorted({str(mks[k].get('market_flags')) for k in mks}),
                 open_status=[r['oms'][k]['st'] for k in okk], open_age_ms=[r['oms'][k]['age'] for k in okk],
                 end_tob_status=[r['eds'][k]['st'] for k in okk],
                 end_mark_vs_tob_abs_c=[round(abs(100. * (mks[k]['end_mark'] - r['eds'][k]['mark'])), 3) if r['eds'][k]['ok'] else None for k in okk])
    return dict(facts=facts, counts=rec['counts'], statuses=rec['statuses'], checks=rec['checks'], row_groups=rec['row_groups'], cpu_s=rec['cpu_s'],
                wall_s=rec['wall_s'], memory=rec['memory'], fills_valued_status_counts=rec['fills_valued_status_counts'],
                inputs={k: v['bytes'] for k, v in rec['inputs'].items()}, per_market_rows=len(r['per_market']), closes_by_role_rows=len(r['closes_by_role']))


if __name__ == '__main__':
    a = sys.argv[1:]
    need(C.RUN == 'v2', f'the matched-gap build reads the v2 roots: set MT_RUN=v2 (got {C.RUN!r})')
    if a[:1] == ['run'] and (len(a) == 1 or (len(a) == 3 and a[1] == '--out')):
        sub = a[2] if len(a) == 3 else 'matched_gap'
        need(bool(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', sub)) and '..' not in sub, f'run --out: {sub!r} is not a plain subdirectory name')
        r = build(dict(out=Path(C.RECEIPT) / sub), write=True, require_frozen=True, n_expected=N_BOOK)
        rc = r['receipt']
        print(json.dumps(X._clean(dict(outputs={k: v.get('rows') for k, v in rc['outputs'].items()}, checks=rc['checks'], statuses=rc['statuses'],
                                       vmhwm_mib=rc['memory']['vmhwm_mib'], cpu_s=rc['cpu_s'])), indent=1))
    elif len(a) == 2 and a[0] == 'rehearse':
        need(a[1] in C.MEASURE_DAYS, f'rehearse: {a[1]!r} is not a measurement day (refused before any path is built)')
        print(json.dumps(X._clean(rehearsal_summary(build(days=(a[1],), write=False))), indent=1, sort_keys=True))
    else:
        sys.exit('usage: MT_RUN=v2 matched_gap.py run [--out <subdir>] | MT_RUN=v2 matched_gap.py rehearse <day>')
