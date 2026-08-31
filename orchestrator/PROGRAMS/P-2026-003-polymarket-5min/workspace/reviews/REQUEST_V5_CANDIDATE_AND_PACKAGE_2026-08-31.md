# Coordinator review request — v5 candidate, then v5 deploy package — 2026-08-31

**Review tip:** `9efb932`. **Two verdicts requested, in this order.**

---

## (A) CANDIDATE — review first, it is small

`live/pm_research/collect_pm.py`, sha256
`39889848ad0f056852192fc00f72e1c64b230d4298588a21d144ee67311eb0a8`.

Your CODE/TEST RELEASE covered `7aa9520`; the candidate has changed, so it
needs a fresh verdict before the package review means anything.

**The change is two constants:** `APP_HEARTBEAT_INTERVAL_S` 10 → 3 and
`APP_HEARTBEAT_TIMEOUT_S` 10 → 3.

**Why.** My third audit angle found that v5 fixes the *contract* (RFC control
Pong → the venue-documented text `PING`/`PONG`) while reintroducing the
*detection lag* O1a existed to remove: worst-case dead-socket blindness was
interval + timeout = **20 s**, against the v4 keepalive's **6 s** after O1a.
Detection lag is stamped INSIDE the recorded gap duration (`gap_start` comes
from the last message received), and recorded gap duration is the quantity
P1/P2/P3 are denominated in. At the measured **68.8 disconnects/hr**, every
+1 s of lag adds ~68.8 s/hr, so 20 s carried a term of **~960 s/hr — eight
times the entire P1 bar of 120.** Every post-deploy day would have failed P1
on a *measurement artifact* and read as "v5 did not work" while the collector
was fine. The trade was stated in neither the runbook nor your review.

**USER ruled to clear it.** Blindness is now **6.0 s**, matching O1a exactly,
with the wrong-contract fix intact.

**Why tightening the interval is legitimate:** the documented ten-second
cadence is a MINIMUM liveness expectation on the client — sending more often
is unobjectionable, sending less is not.

**Evidence, measured not assumed:** a live shadow probe on the real BTC
channel at the new cadence returned **24 PINGs / 24 PONGs over 75 s, zero
disconnects**. Caveat recorded: it attached to an expired slug, so
`market_messages_n` was 1 — it proves the PING/PONG transport contract at 3 s,
not concurrent market flow.

**Tests — v5 heartbeat suite 12 → 21**, asserting properties rather than
numbers:
- blindness DERIVED from both constants and required to be no worse than v4's;
- the documented cadence asserted still honoured;
- the deadline required to clear the observed round-trip by ≥10×;
- two BEHAVIOURAL probes driving the real timeout path (shrinking the constant
  makes a silent socket time out; a PONG inside the deadline does not; the
  constant is asserted restored afterwards);
- plus the two gaps **both** our reviews had left uncovered: heartbeat
  task-leak across reconnects, and pongs-never-exceed-pings. **The leak test is
  falsified** — removing the cancel from the heartbeat's `finally` makes it
  fail, and the source was restored and hash-verified.

Other suites at this tip: collector selftest 17, legacy v4 10/10, day-bar
seam 7/7.

**Committed alongside, NO bar touched (rule 13):**
`DAY_BAR_V2_PREREGISTRATION.md` now carries a DECLARED PREMISE — the bars are
denominated in lost seconds, detection lag lands inside that quantity, so the
bars are **not cadence-independent** and any future cadence change reinterprets
all three. Also recorded there: at 68.8/hr even a 6 s lag leaves a ~413 s/hr
floor, so P1 is a joint constraint on (disconnect rate × detection lag) + real
outage and can only pass if v5 collapses the RATE — which is its thesis. P1 is
therefore close to a direct test of whether v5 works rather than an independent
gate.

---

## (B) DEPLOY PACKAGE — same tip

Since your round-4 filing I ran three further adversarial angles, at the USER's
instruction to clear everything my own audits could find before you start:

1. **the CLI executed AS A SEQUENCE** — never done before; only pure functions
   had been tested;
2. **a differential fuzz of both era consumers** — 17,729 ledgers, 735
   disagreements at peak;
3. **interruption/concurrency**, plus the v5 collector code itself.

**Thirty findings, all closed.** The ones worth your attention:

- **Three DEAD ENDS** where no command could run again on an append-only
  ledger: a half-landed recovery bundle (two rows, one stdout — a SIGINT
  between them lands row 1 alone) bricked every mode, **and the branch that
  would have repaired it had been DELETED as dead**, reading "the walk refuses
  it" as coverage when the walk refusing *is* the brick; `--abort-row` could
  emit a row older than a landed rollback; and the no-mode fallthrough printed
  argparse help to STDOUT, which the runbook redirects into the ledger.
- **A plain `transitioned` row RETURNING to the previous version bypassed the
  entire rollback evidence contract** — no stage, no restoration receipt, no
  `closes_boundary_utc`, no strictly-after check — while my walk reported
  `clob_v4` in force with `clob_v5` still open. DA's consumer caught it.
- My walk was **clob_v5-specific**, which would have stranded the *next*
  collector version permanently, long after this deploy is finished.

**Gate at this tip:** preflight suite 193, DA 147, mutation audit **0 survivors
over 117 in-scope sites** with all four controls firing, chain equivalence
**35/35 with both consumers agreeing** (16 fuzz repros ported in), and live
`--pre-arm` correctly REFUSING because the instant lapsed.

---

## Deliberately NOT done — please do not file these as findings

1. **`CAND_SHA` / `CAND_COMMIT` are stale on purpose**, pointing at the old
   candidate. They are re-pointed ONCE after your candidate verdict, rather
   than chased per edit.
2. **There is no ruled instant.** 07:00:00Z lapsed cleanly — nothing armed,
   nothing restarted, `--pre-arm` refusing by name. A new instant is a USER
   ruling that follows your verdicts.

Nothing is armed; live v4 pid 3687786 continuous. Please file under
`workspace/reviews/` and push.
