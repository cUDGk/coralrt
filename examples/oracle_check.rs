//! Compare coralrt with the outputs recorded from pycoral (tests/oracle/dump_pycoral.py): every output byte must match,
//! then time each model like the recording did.
//!
//!   cargo run --release --example oracle_check -- <test_data dir> <oracle dir>
use coralrt::{Interpreter, Options};
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let (data, oracle) = (Path::new(&args[1]), Path::new(&args[2]));
    let mut interps: HashMap<String, Interpreter> = HashMap::new();
    let (mut cases, mut failed) = (0, 0);
    for line in std::fs::read_to_string(oracle.join("manifest.txt"))?.lines() {
        // case, model, outputs[, inputs]: one input is <case>.in.bin, several are <case>.in<k>.bin
        let f: Vec<&str> = line.split('\t').collect();
        let (case, model, n) = (f[0], f[1], f[2].parse::<usize>()?);
        let n_in: Option<usize> = f.get(3).map(|s| s.parse()).transpose()?;
        if !interps.contains_key(model) {
            interps.insert(model.to_string(), Interpreter::from_file(data.join(model), &Options::default())?);
        }
        let it = interps.get_mut(model).unwrap();
        match n_in {
            None => it.set_input(0, &std::fs::read(oracle.join(format!("{case}.in.bin")))?)?,
            Some(n_in) => {
                for k in 0..n_in {
                    it.set_input(k, &std::fs::read(oracle.join(format!("{case}.in{k}.bin")))?)?;
                }
            }
        }
        it.invoke()?;
        // TFLite_Detection_PostProcess (fast NMS) leaves the slots past the detection count as they were: compare the count
        let dpp = it.op_names().last() == Some(&"TFLite_Detection_PostProcess");
        let count = if dpp { f32::from_le_bytes(it.output(3)?[..4].try_into()?) as usize } else { 0 };
        cases += 1;
        for k in 0..n {
            let want = std::fs::read(oracle.join(format!("{case}.out{k}.bin")))?;
            let got = it.output(k)?;
            let used = if dpp && k < 3 { count * if k == 0 { 16 } else { 4 } } else { want.len() };
            let (w, g) = (&want[..used.min(want.len())], &got[..used.min(got.len())]);
            if w != g || want.len() != got.len() {
                let diff = w.iter().zip(g).filter(|(a, b)| a != b).count();
                let first = w.iter().zip(g).position(|(a, b)| a != b);
                println!("MISMATCH {case} output {k}: {diff} of {} bytes differ, first at {first:?} (sizes {} vs {})", w.len(), want.len(), got.len());
                failed += 1;
            }
        }
    }
    println!("{cases} cases, {failed} mismatching outputs");
    // recorded only for the whole models (dump_pycoral.py)
    let timing = std::fs::read_to_string(oracle.join("timing.txt")).unwrap_or_default();
    for line in timing.lines() {
        let f: Vec<&str> = line.split('\t').collect();
        let it = interps.get_mut(f[0]).unwrap();
        let mut ts: Vec<f64> = (0..100)
            .map(|_| {
                let st = Instant::now();
                it.invoke().unwrap();
                st.elapsed().as_secs_f64() * 1e3
            })
            .collect();
        ts.sort_by(f64::total_cmp);
        println!("{}: coralrt {:.2} ms, pycoral {} ms", f[0], ts[50], f[1]);
    }
    if failed > 0 {
        std::process::exit(1);
    }
    Ok(())
}
