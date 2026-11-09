//! Read-only, bounded factual-core replay. No changed orders receive captured fills.
//! Usage: native-root bounded-run fresh-output scope.json run-name
use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use flate2::{read::MultiGzDecoder, write::GzEncoder, Compression};
use pf_engine::core::CoreInput;
use pf_engine::ladder::core::{BasisMode, Emitted, GuardModels, LadderCore};
use pf_engine::ladder::Request;
use pf_strategy::ladder::{desired_ladder, LadderConfig, MinuteLine};
use serde_json::{json, Value};

fn lines(p: &Path) -> impl Iterator<Item=std::io::Result<String>> {
    BufReader::new(MultiGzDecoder::new(File::open(p).unwrap())).lines()
}
fn output(p: &Path) -> GzEncoder<File> {
    GzEncoder::new(OpenOptions::new().write(true).create_new(true).open(p).unwrap(), Compression::fast())
}
fn emit(w: &mut impl Write, v: &Value) {
    serde_json::to_writer(&mut *w,v).unwrap();w.write_all(b"\n").unwrap();
}
fn output_time(v: &Emitted) -> f64 {
    match v { Emitted::Decision{t,..} | Emitted::Request(Request::Place{t,..}|Request::Cancel{t,..}) => *t }
}
fn check_output(actual: &Emitted, expected: &Emitted) -> Result<(),String> {
    if actual==expected { Ok(()) } else { Err(format!("native output differs: actual={actual:?}, expected={expected:?}")) }
}
fn basis(l: &Value,params: &Value) -> BasisMode {
    let mode: BasisMode=l.get("basis_mode").cloned().map(serde_json::from_value).transpose().unwrap().unwrap_or_default();
    let pm: BasisMode=params.get("basis_mode").cloned().map(serde_json::from_value).transpose().unwrap().unwrap_or_default();
    assert_eq!(mode,pm,"header/params basis mismatch");
    match l["version"].as_str().unwrap() {
        "pm_ladder_v3"=>assert_eq!(mode,BasisMode::LegacyHourlyCarryV1),
        "pm_ladder_v3_basis_v1"=>assert_eq!(mode,BasisMode::ContinuousV1),
        _=>panic!("unknown version"),
    }
    mode
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_output_positive_and_bad_size_refusal() {
        let p=|size|Emitted::Request(Request::Place{market:7,id:1,lane:0,tick:50,size,t:1000.});
        assert!(check_output(&p(5.),&serde_json::from_str(&serde_json::to_string(&p(5.)).unwrap()).unwrap()).is_ok());
        assert!(check_output(&p(5.),&p(25.)).is_err());
    }
    #[test]
    fn versions_bind_basis_mode() {
        assert_eq!(basis(&json!({"version":"pm_ladder_v3"}),&json!({})),BasisMode::LegacyHourlyCarryV1);
        assert_eq!(basis(&json!({"version":"pm_ladder_v3_basis_v1","basis_mode":"continuous_v1"}),&json!({"basis_mode":"continuous_v1"})),BasisMode::ContinuousV1);
        assert!(std::panic::catch_unwind(||basis(&json!({"version":"pm_ladder_v3_basis_v1","basis_mode":"continuous_v1"}),&json!({}))).is_err());
    }
}

fn main() {
    let args:Vec<_>=std::env::args().collect();assert_eq!(args.len(),6);
    let root=Path::new(&args[1]);let run=Path::new(&args[2]);let out=Path::new(&args[3]);
    let scope:Value=serde_json::from_str(&fs::read_to_string(&args[4]).unwrap()).unwrap();
    let windows:BTreeMap<u64,f64>=scope["windows"].as_array().unwrap().iter()
        .filter(|w|w["run"]==args[5]).map(|w|(w["market"].as_u64().unwrap(),w["ws"].as_f64().unwrap()*1000.)).collect();
    assert!(!windows.is_empty());fs::create_dir(out).unwrap();
    let mut input=lines(&run.join("inputs.jsonl.gz"));
    let header:Value=serde_json::from_str(&input.next().unwrap().unwrap()).unwrap();
    assert_eq!(header["protocol"],"PFMM_LADDER_JOURNAL_V3");let l=&header["ladder"];
    let params:Value=serde_json::from_str(&fs::read_to_string(root.join(l["params"]["path"].as_str().unwrap())).unwrap()).unwrap();
    let g=&params["guard"];let ml=&params["minute_lines"];
    let lm:BTreeMap<String,[f64;4]>=BTreeMap::from([
        ("90".into(),serde_json::from_value(ml["pull_and_ladder_line_90"].clone()).unwrap()),
        ("95".into(),serde_json::from_value(ml["size_line_95"].clone()).unwrap())]);
    let cfg=LadderConfig::from_arm(l["policy"].as_str().unwrap(),l["rules"].as_array().unwrap().iter().map(|v|v.as_str().unwrap()),l["size"].as_str().unwrap(),&lm).unwrap();
    let model=|name:&str|pf_model::Model::parse(&fs::read_to_string(root.join(l[name]["path"].as_str().unwrap())).unwrap()).unwrap();
    let models=GuardModels{bid:model("guard_bid"),ask:model("guard_ask"),threshold:[g["threshold_c60"]["bid"].as_f64().unwrap(),g["threshold_c60"]["ask"].as_f64().unwrap()]};
    let mode=basis(l,&params);let mut core=LadderCore::new_with_basis_mode(cfg.clone(),models,mode);
    let mut logged=lines(&run.join("outputs.jsonl.gz"));let mut states=output(&out.join("states.jsonl.gz"));
    let mut books=output(&out.join("books.jsonl.gz"));let mut held=BTreeMap::new();
    let(mut inputs,mut compared,mut count_d,mut nbook)=(0u64,0u64,0u64,0u64);
    let mut last_t=f64::NEG_INFINITY;let mut per_market:BTreeMap<u64,u64>=BTreeMap::new();
    for line in input {
        let i:CoreInput=serde_json::from_str(&line.unwrap()).unwrap();inputs+=1;
        if let Some(t)=i.time(){assert!(t>=last_t);last_t=t;}
        if let CoreInput::Frame{market,known_ms,bids,asks,..}=&i {
            if let Some(ws)=windows.get(market) {
                if *known_ms>=*ws-1000. && *known_ms<*ws+65000. {
                    emit(&mut books,&json!({"market":market,"t":known_ms,"input_index":inputs,"bids":bids,"asks":asks}));nbook+=1;
                }
            }
        }
        let step=core.apply(&i).unwrap();
        for actual in &step.log {
            let expected:Emitted=serde_json::from_str(&logged.next().expect("missing expected output").unwrap()).unwrap();
            check_output(actual,&expected).unwrap_or_else(|e|panic!("output {compared}, input {inputs}: {e}"));compared+=1;
        }
        for (market,d) in step.decisions {
            let Some(ws)=windows.get(&market) else {continue};
            if d.t<*ws || d.t>=*ws+65000. {continue;}
            let x=d.inputs;let is_held=held.get(&market).copied().unwrap_or(false);
            let mut pure=desired_ladder(&cfg,&x).lanes;if is_held{pure=[(None,0.);4];}
            assert_eq!(pure,d.decision.lanes,"conditional quote parity");
            let mut relaxed=BTreeMap::new();
            for delta in [0.005,0.01,0.02,0.04] {
                let mut alt=cfg.clone();alt.pull=cfg.pull.map(|v|MinuteLine(v.0.map(|z|z-delta)));
                alt.rung_line=MinuteLine(cfg.rung_line.0.map(|z|z-delta));
                let mut q=desired_ladder(&alt,&x).lanes;if is_held{q=[(None,0.);4];}
                relaxed.insert(format!("{delta}"),q);
            }
            let book=x.book.map(|b|json!({"bids":b.bids.map(|z|(z.tick,z.size)),"asks":b.asks.map(|z|(z.tick,z.size))}));
            emit(&mut states,&json!({"market":market,"t":d.t,"input_index":inputs,"rel_s":x.rel_s,
                "v":x.v,"binance":x.binance,"net":x.net,"live":x.live,"rung_cur":x.rung_cur,
                "held":is_held,"book":book,"lanes":d.decision.lanes,"relaxed":relaxed,
                "withheld":d.decision.withheld.map(|v|v.map(|z|format!("{z:?}"))),"no_cover":d.decision.no_cover}));
            count_d+=1;*per_market.entry(market).or_default()+=1;
        }
        if let CoreInput::Hold{market,held:h,..}=i{held.insert(market,h);}
        if inputs%2_000_000==0{eprintln!("inputs={inputs}, outputs={compared}, exported={count_d}");}
    }
    // A prefix stops before the next captured input. Only its final unprocessed
    // scheduled outputs may remain, at/after last input time and outside scope.
    let scope_end=windows.values().map(|v|v+65000.).max_by(f64::total_cmp).unwrap();
    let mut tail=0u64;let mut tail_min=None;
    for line in logged {
        let v:Emitted=serde_json::from_str(&line.unwrap()).unwrap();let t=output_time(&v);
        assert!(t>=last_t && t>=scope_end,"unmatched output inside verified scope");
        tail_min=Some(tail_min.map_or(t,|old:f64|old.min(t)));tail+=1;
    }
    assert!(compared>0 && per_market.len()==windows.len());assert!(last_t>=scope_end);
    states.finish().unwrap();books.finish().unwrap();
    let report=json!({"run":args[5],"inputs":inputs,"outputs_matched":compared,"unapplied_boundary_tail_outputs":tail,
        "tail_min_t":tail_min,"last_applied_input_time":last_t,"scope_end":scope_end,"exported_decisions":count_d,
        "book_frames":nbook,"market_counts":per_market,"basis_mode":mode,"outputs_exact_for_applied_inputs":true,
        "pure_quote_function_exact":true,"scope":"factual input replay; relaxed quotes conditional on captured state; no simulated fills or changed OMS"});
    let mut f=OpenOptions::new().write(true).create_new(true).open(out.join("replay.json")).unwrap();emit(&mut f,&report);
    println!("{report}");
}
