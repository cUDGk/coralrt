//! Compare coralrt with tflite_runtime 2.5 on the op cases recorded by tests/ops/official.py: the output shapes and
//! types must match, and every byte (or, for cases with a tolerance, every float within it).
//!
//!   cargo run --release --example op_check -- <out dir> [case name prefix ...]
use coralrt::{Interpreter, Options, TType};
use std::path::Path;

/// "2x3:float32;:int32" -> [([2, 3], "float32"), ([], "int32")]
fn descs(s: &str) -> Vec<(Vec<usize>, String)> {
    s.split(';')
        .filter(|d| !d.is_empty())
        .map(|d| {
            let (shape, dtype) = d.split_once(':').unwrap();
            (shape.split('x').filter(|x| !x.is_empty()).map(|x| x.parse().unwrap()).collect(), dtype.to_string())
        })
        .collect()
}

fn type_name(t: TType) -> &'static str {
    match t {
        TType::F32 => "float32",
        TType::F16 => "float16",
        TType::F64 => "float64",
        TType::I8 => "int8",
        TType::U8 => "uint8",
        TType::I16 => "int16",
        TType::I32 => "int32",
        TType::I64 => "int64",
        TType::U32 => "uint32",
        TType::U64 => "uint64",
        TType::Bool => "bool",
        TType::Str => "string",
        TType::C64 => "complex64",
        TType::C128 => "complex128",
        _ => "other",
    }
}

/// the strings of a buffer in TFLite's string layout
fn strings(d: &[u8]) -> Vec<&[u8]> {
    let rd = |i: usize| i32::from_le_bytes(d[4 * i..4 * i + 4].try_into().unwrap()) as usize;
    (0..rd(0)).map(|k| &d[rd(1 + k)..rd(2 + k)]).collect()
}

/// floats of the type, as f64 (complex: real and imaginary parts in turn)
fn floats(d: &[u8], dtype: &str) -> Option<Vec<f64>> {
    Some(match dtype {
        "float32" | "complex64" => d.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap()) as f64).collect(),
        "float64" | "complex128" => d.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect(),
        "float16" => d.chunks_exact(2).map(|c| half(u16::from_le_bytes([c[0], c[1]])) as f64).collect(),
        _ => return None,
    })
}

fn half(h: u16) -> f32 {
    let (s, e, m) = ((h >> 15) as u32, ((h >> 10) & 0x1f) as i32, (h & 0x3ff) as f32);
    let v = match e {
        0 => m * 2f32.powi(-24),
        31 => if m == 0.0 { f32::INFINITY } else { f32::NAN },
        e => (1.0 + m / 1024.0) * 2f32.powi(e - 15),
    };
    if s == 1 { -v } else { v }
}

/// a description of how `got` differs from `want`, or None when it is within the case's tolerance
fn compare(want: &[u8], got: &[u8], dtype: &str, tol: f64) -> Option<String> {
    if want == got {
        return None;
    }
    if tol > 0.0 {
        if let (Some(w), Some(g)) = (floats(want, dtype), floats(got, dtype)) {
            if w.len() == g.len() {
                let bad: Vec<usize> = (0..w.len()).filter(|&i| !(w[i] == g[i] || (w[i].is_nan() && g[i].is_nan()) || (w[i] - g[i]).abs() <= tol * w[i].abs().max(1.0))).collect();
                if bad.is_empty() {
                    return None;
                }
                let i = bad[0];
                return Some(format!("{} of {} values off, first [{i}] {} vs {}", bad.len(), w.len(), g[i], w[i]));
            }
        }
    }
    let n = want.len().min(got.len());
    let diff = (0..n).filter(|&i| want[i] != got[i]).count();
    Some(format!("{diff} of {} bytes differ, first at {:?} (sizes {} vs {})", want.len(), (0..n).find(|&i| want[i] != got[i]), got.len(), want.len()))
}

fn run(dir: &Path, name: &str, tol: f64, invokes: usize, ins: &[(Vec<usize>, String)], outs: &[(Vec<usize>, String)]) -> coralrt::Result<Vec<String>> {
    let oracle = dir.join("oracle");
    let mut it = Interpreter::from_file(dir.join("models").join(format!("{name}.tflite")), &Options::default())?;
    for (k, (shape, _)) in ins.iter().enumerate() {
        let t = it.inputs()[k];
        if it.tensor(t).shape != *shape {
            it.resize_input(k, shape)?;
        }
    }
    it.allocate_tensors()?;
    let mut problems = Vec::new();
    for inv in 0..invokes {
        for (k, (shape, dtype)) in ins.iter().enumerate() {
            let data = std::fs::read(oracle.join(format!("{name}.{inv}.in{k}.bin"))).map_err(|e| coralrt::Error(e.to_string()))?;
            if dtype == "string" {
                let t = it.inputs()[k];
                it.set_strings(t, &strings(&data), Some(shape))?;
            } else {
                it.set_input(k, &data)?;
            }
        }
        it.invoke()?;
        for (k, (shape, dtype)) in outs.iter().enumerate() {
            let want = std::fs::read(oracle.join(format!("{name}.{inv}.out{k}.bin"))).map_err(|e| coralrt::Error(e.to_string()))?;
            let t = it.tensor(*it.outputs().get(k).ok_or_else(|| coralrt::Error(format!("no output {k}")))?);
            if t.shape != *shape || type_name(t.dtype) != dtype {
                problems.push(format!("output {k} is {}{:?}, tflite {dtype}{shape:?}", type_name(t.dtype), t.shape));
                continue;
            }
            if let Some(p) = compare(&want, t.data(), dtype, tol) {
                problems.push(format!("invoke {inv} output {k}: {p}"));
            }
        }
    }
    Ok(problems)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = Path::new(&args[1]);
    let prefixes = &args[2..];
    let manifest = std::fs::read_to_string(dir.join("oracle").join("manifest.txt")).expect("manifest.txt");
    let (mut good, mut bad, mut skipped, mut refused, mut not_refused) = (0, 0, 0, 0, 0);
    for line in manifest.lines() {
        let f: Vec<&str> = line.split('\t').collect();
        let (name, status) = (f[0], f[1]);
        if !prefixes.is_empty() && !prefixes.iter().any(|p| name.starts_with(p.as_str())) {
            continue;
        }
        let (tol, invokes): (f64, usize) = (f[2].parse().unwrap(), f[3].parse().unwrap());
        let (ins, outs) = (descs(f[4]), descs(f[5]));
        match status {
            "load_error" => skipped += 1,
            "run_error" => match run(dir, name, tol, invokes, &ins, &outs) {
                Err(_) => refused += 1,
                Ok(_) => {
                    println!("RAN    {name}: tflite 2.5 refuses it ({})", f.get(6).unwrap_or(&""));
                    not_refused += 1;
                }
            },
            _ => match run(dir, name, tol, invokes, &ins, &outs) {
                Ok(p) if p.is_empty() => good += 1,
                Ok(p) => {
                    println!("DIFF   {name}: {}", p.join("; "));
                    bad += 1;
                }
                Err(e) => {
                    println!("ERROR  {name}: {e}");
                    bad += 1;
                }
            },
        }
    }
    println!("{good} match, {bad} differ, {refused} refused like tflite, {not_refused} run where tflite refuses, {skipped} unreadable by tflite 2.5");
    if bad > 0 {
        std::process::exit(1);
    }
}
