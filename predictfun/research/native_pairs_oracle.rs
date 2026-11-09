//! Differential check of generated LZ research traces against pinned native A2.
//! Reads synthetic inputs only; never opens an exchange connection.
use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use pf_engine::ladder::{LadderWindow, Request, SizeLayering};
use pf_strategy::ladder::LadderConfig;
use pf_strategy::Want;
use serde_json::{json,Value};

fn n(v:&Value)->f64 {v.as_f64().unwrap()}
fn cfg(minimum:Option<f64>,rpm:f64)->LadderConfig {
    let lines=BTreeMap::from([("90".to_string(),[0.0492,0.0623,0.0789,0.0917]),("95".to_string(),[0.0698,0.0841,0.1009,0.1094])]);
    let mut c=LadderConfig::from_arm("MICRO",["G4","G2","G6","G5","G3","G1","PM90","GS","LD2","XNA","XV","S60"],"AM:5:25:95",&lines).unwrap();
    c.min_usd=minimum;c.venue_rpm=rpm;c
}
fn normalized(r:Request)->Value {
    match r {
        Request::Place{id,lane,tick,size,t,..}=>json!({"kind":"place","t":t,"id":id,"lane":lane,"tick":tick,"size":size}),
        Request::Cancel{id,lane,t,..}=>json!({"kind":"cancel","t":t,"id":id,"lane":lane}),
    }
}
fn near(actual:&Value,expected:&Value)->bool {
    match (actual,expected) {
        (Value::Number(a),Value::Number(b))=>(a.as_f64().unwrap()-b.as_f64().unwrap()).abs()<=1e-8,
        (Value::Array(a),Value::Array(b))=>a.len()==b.len()&&a.iter().zip(b).all(|(x,y)|near(x,y)),
        (Value::Object(a),Value::Object(b))=>a.len()==b.len()&&a.iter().all(|(k,x)|b.get(k).is_some_and(|y|near(x,y))),
        _=>actual==expected,
    }
}
fn check(case:&Value)->Result<Value,String> {
    let mut w=LadderWindow::new(cfg(case["minimum"].as_f64(),n(&case["rpm"])),1,0.).with_size_layering(SizeLayering::IncrementalV1);
    w.diagnostic_clear_events();
    let(mut actual,mut expected)=(Vec::new(),Vec::new());
    let mut scheduled:Vec<(f64,bool,u64)>=vec![];
    let mut due_decide=None;
    let(mut decisions,mut fills,mut accepts,mut rejects,mut retries)=(0,0,0,0,0);
    for e in case["events"].as_array().unwrap() {
        let t=n(&e["t"]);let kind=e["kind"].as_str().unwrap();
        if due_decide.is_some_and(|at|at<t) {return Err(format!("{} missing full-fill decision at {due_decide:?}",case["name"]));}
        scheduled.sort_by(|a,b|a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
        while scheduled.first().is_some_and(|(at,_,_)|*at<=t) {
            let(_,answer,id)=scheduled.remove(0);if answer {w.answered(id);}else{w.cancel_done(id);}
        }
        let mut out=vec![];
        match kind {
            "D"=>{
                let want:[Want;4]=serde_json::from_value(e["logical"].clone()).unwrap();
                w.diagnostic_wants(t,&want,&mut out);decisions+=1;due_decide=None;
                let(total,filled)=w.diagnostic_pairs();
                for (label,a,b) in [("total",json!(total),&e["total"]),("filled",json!(filled),&e["filled"])] {
                    if !near(&a,b){return Err(format!("{} D {t} pair {label}: native={a} python={b}",case["name"]));}
                }
                let intents:Vec<_>=(0..8).map(|m|w.intent(m).unwrap()).collect();
                if !near(&json!(intents),&e["physical"]){return Err(format!("{} D {t} physical wants differ",case["name"]));}
            },
            "retry"=>{w.run_retries(t,&mut out);retries+=1;},
            "place"=>{
                let id=e["id"].as_u64().unwrap();scheduled.push((n(&e["answer"]),true,id));
                expected.push(json!({"kind":"place","t":t,"id":id,"lane":e["lane"],"tick":e["tick"],"size":e["size"]}));
            },
            "cancel"=>{
                let id=e["id"].as_u64().unwrap();scheduled.push((n(&e["end"]),false,id));
                expected.push(json!({"kind":"cancel","t":t,"id":id,"lane":e["lane"]}));
            },
            "accepted"=>{w.accepted(e["id"].as_u64().unwrap());accepts+=1;},
            "rejected"=>{w.rejected(e["id"].as_u64().unwrap());rejects+=1;},
            "fill"=>{
                w.filled(e["id"].as_u64().unwrap(),n(&e["qty"]),t);fills+=1;
                if w.diagnostic_fill_decide(t){due_decide=Some(t);}
                w.diagnostic_retry_only();
            },
            other=>return Err(format!("unknown synthetic event {other}")),
        }
        actual.extend(out.into_iter().map(normalized));
    }
    if due_decide.is_some(){return Err("missing final full-fill decision".into());}
    if !near(&json!(actual),&json!(expected)) {
        let mismatch=actual.iter().zip(&expected).position(|(a,b)|!near(a,b)).unwrap_or(actual.len().min(expected.len()));
        return Err(format!("{} request mismatch {mismatch} native={:?} python={:?}; lengths {} vs{}",case["name"],actual.get(mismatch),expected.get(mismatch),actual.len(),expected.len()));
    }
    Ok(json!({"name":case["name"],"events":case["events"].as_array().unwrap().len(),"requests":actual.len(),"decisions":decisions,"fills":fills,"accepts":accepts,"rejects":rejects,"retry_events":retries,
        "native_throttled":w.counts().throttled,"native_topup_throttled":w.counts().throttled_topup,"native_topup_withheld":w.counts().topup_withheld,
        "known_position":w.known_position(),"requests_and_pair_state_match":true}))
}
fn main() {
    let args:Vec<_>=std::env::args().collect();assert_eq!(args.len(),3);
    let input=BufReader::new(File::open(&args[1]).unwrap());
    let mut dest=OpenOptions::new().write(true).create_new(true).open(&args[2]).unwrap();
    let mut count=0;
    for line in input.lines(){let case:Value=serde_json::from_str(&line.unwrap()).unwrap();let result=check(&case).unwrap();serde_json::to_writer(&mut dest,&result).unwrap();dest.write_all(b"\n").unwrap();count+=1;}
    assert!(count>0);eprintln!("verified {count} synthetic traces");
}

#[cfg(test)]
mod tests {
    use super::*;
    fn basic()->Value {
        let physical=json!([[50,5.],[null,0.],[null,0.],[null,0.],[null,0.],[null,0.],[null,0.],[null,0.]]);
        json!({"name":"positive","minimum":1.,"rpm":500.,"events":[
            {"kind":"D","t":0.,"logical":[[50,5.],[null,0.],[null,0.],[null,0.]],"physical":physical,"total":[[50,5.],null,null,null],"filled":[0.,0.,0.,0.]},
            {"kind":"place","t":0.,"id":0,"lane":0,"tick":50,"size":5.,"answer":44.},
            {"kind":"accepted","t":40.,"id":0}]})
    }
    #[test] fn exact_positive_changed_request_and_pair_refusal() {
        let good=basic();assert!(check(&good).is_ok());
        let mut bad=good.clone();bad["events"][1]["size"]=json!(25.);assert!(check(&bad).is_err());
        let mut bad=good.clone();bad["events"][0]["filled"][0]=json!(1.);assert!(check(&bad).is_err());
    }
}
