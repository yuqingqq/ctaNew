# Taking in the three displayed markets — descriptive readout

USER: “it looks the key difference maybe they are taking from time to time?”

The saved own arm is passive-only; the recorded makers use both roles. On the displayed Oct26/3789484 interval, rank1 has90 taker fill records /223 total legs and1,030.48 taker shares /2,774.60 total shares (37.14%). Rank3 has7 /319 and205.96 /8,142.59 shares (2.53%). These are recorded fills, not counts of submitted decisions; rank1 includes two self-trade take legs. A taker record can be a partial execution. All three fixed markets/all ranks are retained in readout.json; inactive ranks are explicit.

Some taking reduces recorded exposure, some adds or reverses it. The local classification uses strict-earlier settlement-clock net flow and is descriptive only; match-window ambiguity, simultaneous events and client notices prevent calling it decision-time inventory. Do not promote its counts into a trigger.

The accepted same-market role comparison (report_matched_gap_interpretation_v2.md) already shows a passive-close gap: ours maker→maker -1.90c/sh versus rank1 +4.49,rank2 +4.45,rank3 +4.06. That does NOT isolate a passive strategy: earlier taking may affect subsequent passive quotes and the sample of completed closes. It only rules out explaining the accounting gap by role mix alone. The superseding report_interpretation_v3.md also withdraws the claim that negative taker exits are caused solely by spread/fees; a take may prevent greater later harm.

Working mechanism to investigate, not an established maker rule: use passive quoting for inventory acquisition/turnover, and selective taking for an inventory or value reason while retaining valuable resting orders. Taking can also initiate exposure; it is not synonymous with liquidation. No take/controller variant was tested here. The current inventory hysteresis defect must be corrected before a taking ablation can be interpreted cleanly.

Source: the already independently checked dashboard data.json.gz, same3 consumed markets and supported intervals. All ranks selected before readout; no model/null/policy test or new tape. run_light mt_dash_taker_readout charged0.13 CPU seconds; allow1s publication overhead. Prior packet upper62.46s ->63.59s, below approved180s. Existing source-checks cover role/quantity identities; read.py reconciles maker+taker share denominators. Research line upper5.26693h +.05reserve <5.35; task upper1.95623h +.05reserve <2.05.
