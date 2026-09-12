#!/usr/bin/env python3
"""RETIRED 2026-09-12 by REVIEW ruling — kept as provenance, do not rely on it.

WHY IT IS RETIRED, not improved: the discriminator this needs is UNAVAILABLE IN
PRINCIPLE from the prose. It cannot separate
    (a) "I have already routed this"   -- past reference, legitimate
    (b) "I am routing this"            -- the actual act-claim, the real failure
    (c) "this is that seat's to handle"-- an assignment, not a claim of action
so its 15 flags are an UPPER BOUND with an unknown false-positive rate, and at
least one (`da163`, "has ALREADY been routed to BE") is confirmed spurious.

TWO CONCRETE DEFECTS, both found by the reviewer, both instructive:
  * `getmtime` IS NOT A DISPATCH TIME. Any copy or checkout moves it, and the
    whole +/-10-minute window rests on that one field.
  * The report looks only FORWARD, so a claim honoured 11 minutes EARLIER
    prints as "NEVER dispatched".

AND IT FAILED THE TEST THIS PROGRAMME SET TONIGHT -- "could a reader tomorrow
find this without asking anyone?" The script landed in the repo while its input
population (803 dispatch files) sat in a scratch directory no reviewer could
reach, so the numbers it produced could be neither reproduced nor falsified.

THE RULE THAT REPLACES IT: **a claim of action names the act** -- "routed to DA
(DA 313)", "filed (R-939)", "landed (count 1 at the fetched ref)". Checkable when
written, with no prose parsing at all. And the exposure is the ACT CLASS -- an act
whose only trace is the sentence announcing it -- NOT a role.
"""

_ORIGINAL_DOCSTRING = """Did I do what I said I did? Scan my outbound dispatches for claims that a
thing was ROUTED/SENT to a seat, and check a dispatch to that seat actually
followed. Positive control required: it must catch the known miss."""
import os, re, sys, glob
SP = os.path.dirname(os.path.abspath(__file__))
SEATS = {"BE":"be","DA":"da","DE":"de","REV":"rev","MEM":"mem"}
CLAIM = re.compile(r"\b(?:routed|routing|dispatched|sent|goes|going|handed)\b[^.]{0,60}?\bto\s+(BE|DA|DE|REV|MEM)\b", re.I)
files=[]
for f in glob.glob(os.path.join(SP,"*.txt")):
    b=os.path.basename(f)
    m=re.match(r"(be|da|de|rev|mem)(\d+)", b)
    if m: files.append((os.path.getmtime(f), b, m.group(1), f))
files.sort()
WINDOW = 600  # 10 minutes either side
if not files: print("no dispatch files found"); sys.exit(2)
print(f"scanned {len(files)} dispatch files\n")
misses=[]
for mt, name, prefix, path in files:
    try: txt=open(path, encoding="utf-8", errors="replace").read()
    except Exception: continue
    for m in CLAIM.finditer(txt):
        target=SEATS[m.group(1).upper()]
        if target==prefix: continue            # telling a seat you routed to itself
        # A claim is HONOURED if a dispatch to that seat sits within WINDOW
        # seconds either side: shortly BEFORE (the claim describes what I just
        # did) or shortly AFTER (I said it and then did it). A dispatch 40
        # minutes later does NOT honour the claim -- that is the failure.
        near=[n for t,n,pp,_ in files if pp==target and abs(t-mt)<=WINDOW]
        if not near:
            gap=[t-mt for t,n,pp,_ in files if pp==target and t>mt]
            when=f"nearest later dispatch +{min(gap)/60:.0f} min" if gap else "NEVER dispatched"
            misses.append((name, m.group(1).upper(), m.group(0).strip()[:50], when))
print(f"CLAIMS WITH NO SUBSEQUENT DISPATCH TO THAT SEAT: {len(misses)}")
for name, seat, frag, when in misses: print(f"  {name}: claimed -> {seat}  ({frag!r})  {when}")
# positive control: the instrument must be able to fire
print(f"\nCONTROL: total claim-phrases matched across all files = "
      f"{sum(len(CLAIM.findall(open(p,encoding='utf-8',errors='replace').read())) for _,_,_,p in files)}"
      f"  (0 would mean the regex never fires and a clean result is meaningless)")
