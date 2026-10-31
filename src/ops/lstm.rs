// From TensorFlow Lite 2.5 (Copyright The TensorFlow Authors, Apache License 2.0), translated and changed; see NOTICE.
//! LSTM (lstm.cc), UNIDIRECTIONAL_SEQUENCE_LSTM and BIDIRECTIONAL_SEQUENCE_LSTM, each time step as lstm_eval.cc
//! takes it: float and hybrid (8-bit weights; the input, aux input, state and hidden rows quantized per batch row)
//! through the portable tensor_utils in their order, so float matches exactly too; int8 with an int16 cell in
//! integers and gemmlowp fixed point (8x8_16, five intermediates), or LSTM's 8x8_8 variant (twelve intermediates)
//! with its float layer norm and activations. LSTM's 5-input basic kernel is optimized_ops::LstmCell: float through
//! the float GEMM, or uint8 through the 8-bit GEMM into a fixed-point cell. The states are inputs updated in place.

use super::fixedpoint::{inv_sqrt_multiplier_exp, rounding_divide_by_pot, I16};
use super::tensor_utils::*;
use super::util::*;
use super::{eigen, Io, Kernel};
use crate::interp::Env;
use crate::tensor::{SparseDim, Sparsity, TType, Tensor};
use crate::tflite::{Act, OpSpec};
use crate::Result;

pub fn create(s: &OpSpec) -> Result<Option<Box<dyn Kernel>>> {
    let flag = |i: usize, d: u8| opt(s, i, d) != 0;
    let params = |asymmetric: usize| -> Result<Params> {
        Ok(Params { act: opt_act(s, 0)?, cell_clip: opt(s, 1, 0f32), proj_clip: opt(s, 2, 0f32), asymmetric: flag(asymmetric, 0) })
    };
    Ok(Some(match s.code {
        // LSTMOptions.kernel_type BASIC
        16 if opt(s, 3, 0u8) == 1 => Box::new(Basic),
        16 => Box::new(Lstm::new(Kind::Single, params(4)?)),
        44 => Box::new(Lstm::new(Kind::Sequence { time_major: flag(3, 0) }, params(4)?)),
        52 => Box::new(Lstm::new(Kind::Bidirectional { time_major: flag(4, 1), merge: flag(3, 0) }, params(5)?)),
        _ => return Ok(None),
    }))
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Single,
    Sequence { time_major: bool },
    Bidirectional { time_major: bool, merge: bool },
}

#[derive(Clone, Copy)]
struct Params {
    act: Act,
    cell_clip: f32,
    proj_clip: f32,
    asymmetric: bool,
}

/// how the cell computes, from the input's and the weights' types and the number of intermediates
enum Mode {
    Float,
    Hybrid,
    Int16(Box<Int16>),
    Int8(Box<Int8>),
}

/// where one direction's tensors sit among the op's inputs; usize::MAX for none. Gates in the order input,
/// forget, cell, output.
struct Slots {
    x: [usize; 4],
    h: [usize; 4],
    peep: [usize; 4],
    bias: [usize; 4],
    proj: usize,
    proj_bias: usize,
    h_state: usize,
    c_state: usize,
    ln: [usize; 4],
    aux: [usize; 4],
}

const NONE: usize = usize::MAX;

fn slots(kind: Kind, dir: usize) -> Slots {
    let bidi = matches!(kind, Kind::Bidirectional { .. });
    let o = 17 * dir;
    Slots {
        x: [1 + o, 2 + o, 3 + o, 4 + o],
        h: [5 + o, 6 + o, 7 + o, 8 + o],
        peep: [9 + o, 10 + o, NONE, 11 + o],
        bias: [12 + o, 13 + o, 14 + o, 15 + o],
        proj: 16 + o,
        proj_bias: 17 + o,
        h_state: if bidi { 35 + 2 * dir } else { 18 },
        c_state: if bidi { 36 + 2 * dir } else { 19 },
        ln: if bidi { [NONE; 4] } else { [20, 21, 22, 23] },
        aux: if bidi { [40 + 4 * dir, 41 + 4 * dir, 42 + 4 * dir, 43 + 4 * dir] } else { [NONE; 4] },
    }
}

fn opt_in<'a>(io: &Io<'a>, k: usize) -> Option<&'a Tensor> {
    if k == NONE { None } else { io.opt(k) }
}

struct Lstm {
    kind: Kind,
    p: Params,
    mode: Mode,
    /// hybrid, asymmetric: each direction's weight row sums, worked out on the first step after prepare
    sums: [Option<Sums>; 2],
    /// hybrid: the input scaling factors as the last step left them. 2.5 quantizes the projection into the same
    /// buffer, and a sparse recurrent matrix is scaled by it (not by the state's factors).
    input_sf: Vec<f32>,
    /// 8x8_8: the quantized cell and projection clips, which 2.5 only works out when the cell scale is 2^-15
    clips8: (i16, i8),
}

impl Lstm {
    fn new(kind: Kind, p: Params) -> Lstm {
        Lstm { kind, p, mode: Mode::Float, sums: [None, None], input_sf: Vec::new(), clips8: (0, 0) }
    }
}

// *** weights ***

/// a weight matrix as the float and hybrid steps read it: float, or 8-bit bytes as int8 with the tensor's scale,
/// and for a sparse one its stored blocks and the ledger of 1x16 blocks (lstm.cc copy_ledger)
struct Mat {
    f: Vec<f32>,
    q: Vec<i32>,
    scale: f32,
    ledger: Option<Vec<u8>>,
}

impl Mat {
    fn of(t: &Tensor) -> Result<Mat> {
        let ledger = t.sparsity.as_ref().map(ledger).transpose()?;
        Ok(if t.dtype == TType::F32 {
            Mat { f: t.as_slice::<f32>().to_vec(), q: Vec::new(), scale: 1.0, ledger }
        } else {
            Mat { f: Vec::new(), q: t.data().iter().map(|&b| b as i8 as i32).collect(), scale: t.scale(), ledger }
        })
    }
}

/// copy_ledger: per row the number of nonzero blocks, then their indices, each in a byte
fn ledger(sp: &Sparsity) -> Result<Vec<u8>> {
    let Some(SparseDim::Sparse(segments, indices)) = sp.dims.get(1) else {
        return err("the sparse weights need a sparse second dimension");
    };
    let mut l = Vec::new();
    for w in segments.windows(2) {
        let (a, b) = (w[0] as usize, w[1] as usize);
        l.push(u8::try_from(b - a).or_else(|_| err("more than 255 blocks in a row"))?);
        for &i in &indices[a..b] {
            l.push(u8::try_from(i).or_else(|_| err("a block index over 255"))?);
        }
    }
    Ok(l)
}

/// one direction's tensors for the float and hybrid steps
struct Dir {
    x: [Option<Mat>; 4],
    aux: [Option<Mat>; 4],
    h: [Option<Mat>; 4],
    proj: Option<Mat>,
    /// peephole weights in float (hybrid: VectorScalarMultiply, the scale times each int8 value)
    peep: [Option<Vec<f32>>; 4],
    ln: [Option<Vec<f32>>; 4],
    bias: [Option<Vec<f32>>; 4],
    proj_bias: Option<Vec<f32>>,
    n_input: usize,
    n_aux: usize,
    n_cell: usize,
    n_output: usize,
}

fn float_dir(io: &Io, sl: &Slots, aux: bool) -> Result<Dir> {
    let m = |k: usize| opt_in(io, k).map(Mat::of).transpose();
    let f = |k: usize| opt_in(io, k).map(|t| t.as_slice::<f32>().to_vec());
    let peep = |k: usize| {
        opt_in(io, k).map(|t| if t.dtype == TType::F32 { t.as_slice::<f32>().to_vec() } else { t.data().iter().map(|&b| t.scale() * b as i8 as f32).collect() })
    };
    let a = |g: usize| if aux { m(sl.aux[g]) } else { Ok(None) };
    let (x, h) = (io.input(sl.x[1])?, io.input(sl.h[1])?);
    Ok(Dir {
        x: [m(sl.x[0])?, m(sl.x[1])?, m(sl.x[2])?, m(sl.x[3])?],
        aux: [a(0)?, a(1)?, a(2)?, a(3)?],
        h: [m(sl.h[0])?, m(sl.h[1])?, m(sl.h[2])?, m(sl.h[3])?],
        proj: m(sl.proj)?,
        peep: sl.peep.map(peep),
        ln: sl.ln.map(f),
        bias: sl.bias.map(f),
        proj_bias: f(sl.proj_bias),
        n_input: x.shape[1],
        n_aux: if aux { opt_in(io, sl.aux[1]).map_or(0, |t| t.shape[1]) } else { 0 },
        n_cell: x.shape[0],
        n_output: h.shape[1],
    })
}

/// the hybrid row sums of each matrix (ComputeRowSums)
struct Sums {
    x: [Vec<i32>; 4],
    aux: [Vec<i32>; 4],
    h: [Vec<i32>; 4],
    proj: Vec<i32>,
}

impl Sums {
    fn of(d: &Dir) -> Sums {
        let rs = |m: &Option<Mat>, cols: usize| m.as_ref().map_or(Vec::new(), |m| row_sums(&m.q, cols));
        Sums {
            x: [0, 1, 2, 3].map(|g| rs(&d.x[g], d.n_input)),
            aux: [0, 1, 2, 3].map(|g| rs(&d.aux[g], d.n_aux)),
            h: [0, 1, 2, 3].map(|g| rs(&d.h[g], d.n_output)),
            proj: rs(&d.proj, d.n_cell),
        }
    }
}

// *** float and hybrid steps ***

/// CwiseClipping: std::max(std::min(clip, x), -clip)
fn clip_f(v: &mut [f32], clip: f32) {
    for x in v {
        let m = if *x < clip { *x } else { clip };
        *x = if m < -clip { -clip } else { m };
    }
}

/// MeanStddevNormalization, row by row
fn mean_stddev_norm(v: &mut [f32], n: usize) {
    for row in v.chunks_mut(n) {
        let mean = row.iter().fold(0.0f32, |s, &x| s + x) / n as f32;
        let variance = row.iter().fold(0.0f32, |s, &x| s + (x - mean) * (x - mean)) / n as f32;
        let inv = 1.0f32 / (variance + 1e-8f32).sqrt();
        row.iter_mut().for_each(|x| *x = (*x - mean) * inv);
    }
}

/// ApplyActivationToVector for a gate: the fused activation, or the sigmoid of the input, forget and output gates
fn activate(v: &mut [f32], act: Option<Act>) {
    match act {
        Some(a) => apply_activation(v, a),
        None => v.iter_mut().for_each(|x| *x = eigen::logistic(*x)),
    }
}

/// a gate before its matmuls: the bias, or zeros when layer norm adds the bias later
fn gate_start(d: &Dir, g: usize, nb: usize) -> Vec<f32> {
    match (&d.ln[g], &d.bias[g]) {
        (None, Some(b)) => b.repeat(nb),
        _ => vec![0.0; d.n_cell * nb],
    }
}

/// a gate after its matmuls: peephole, layer norm (then the bias), activation
fn gate_finish(d: &Dir, g: usize, c: &[f32], act: Option<Act>, out: &mut [f32]) {
    let n = d.n_cell;
    if let Some(p) = &d.peep[g] {
        for (o, c) in out.chunks_mut(n).zip(c.chunks(n)) {
            for i in 0..n {
                o[i] += p[i] * c[i];
            }
        }
    }
    if let (Some(l), Some(b)) = (&d.ln[g], &d.bias[g]) {
        mean_stddev_norm(out, n);
        for o in out.chunks_mut(n) {
            for i in 0..n {
                o[i] *= l[i];
            }
            for i in 0..n {
                o[i] += b[i];
            }
        }
    }
    activate(out, act);
}

/// float MatrixBatchVectorMultiplyAccumulate over nb rows
fn acc_f(m: &Mat, rows: usize, cols: usize, v: &[f32], nb: usize, out: &mut [f32]) {
    for b in 0..nb {
        mbvma_f32(&m.f, rows, cols, &v[b * cols..], &mut out[b * rows..(b + 1) * rows]);
    }
}

/// hybrid MatrixBatchVectorMultiplyAccumulate over nb quantized rows, scaled by their factors times the weights'
/// scale; less the zero points times the row sums when asymmetric. A sparse matrix takes the ledger's 1x16 blocks.
#[allow(clippy::too_many_arguments)]
fn acc_h(m: &Mat, rows: usize, cols: usize, q: &[i32], sf: &[f32], offset: Option<(&[i32], &[i32])>, nb: usize, out: &mut [f32]) {
    if let Some(l) = &m.ledger {
        for b in 0..nb {
            let scale = m.scale * sf[b];
            let (mut at, mut li) = (0usize, 0usize);
            for r in 0..rows {
                let mut dot = 0i32;
                let blocks = l[li] as usize;
                li += 1;
                for _ in 0..blocks {
                    let start = b * cols + l[li] as usize * 16;
                    li += 1;
                    for c in 0..16 {
                        dot = dot.wrapping_add(m.q[at] * q[start + c]);
                        at += 1;
                    }
                }
                out[b * rows + r] += dot as f32 * scale;
            }
        }
        return;
    }
    for b in 0..nb {
        let offset = offset.map(|(sums, zp)| (sums, zp[b]));
        mbvma_i8(&m.q, rows, cols, &q[b * cols..], sf[b] * m.scale, offset, &mut out[b * rows..(b + 1) * rows]);
    }
}

/// UpdateLstmCellFloat: cell = forget * cell + input * cell gate (CIFG: 1 - forget), clipped
fn update_cell(c: &mut [f32], ig: Option<&[f32]>, fg: &mut [f32], cg: &[f32], clip: f32) {
    for k in 0..c.len() {
        c[k] *= fg[k];
    }
    match ig {
        Some(ig) => (0..c.len()).for_each(|k| c[k] += cg[k] * ig[k]),
        None => (0..c.len()).for_each(|k| {
            fg[k] = 1.0 - fg[k];
            c[k] += cg[k] * fg[k];
        }),
    }
    if clip > 0.0 {
        clip_f(c, clip);
    }
}

/// the hidden values before the projection: the output gate times the activated cell. With no activation 2.5 leaves
/// its scratch, which holds the cell gate, in place of the activated cell.
fn hidden(p: &Params, c: &[f32], og: &[f32], cg: &mut [f32]) {
    if p.act != Act::None {
        cg.copy_from_slice(c);
        apply_activation(cg, p.act);
    }
    for k in 0..cg.len() {
        cg[k] *= og[k];
    }
}

/// the state from the hidden values: the projection (bias first) clipped, or the hidden values as they are
fn project_start(d: &Dir, nb: usize, h: &mut [f32]) {
    match &d.proj_bias {
        Some(b) => h.copy_from_slice(&b.repeat(nb)),
        None => h.fill(0.0),
    }
}

/// LstmStepFloat
fn step_float(d: &Dir, p: &Params, x: &[f32], aux: Option<&[f32]>, h: &mut [f32], c: &mut [f32], nb: usize) {
    let (x_zero, aux_zero) = (is_zero(x), aux.is_none_or(is_zero));
    let gate = |g: usize, c: &[f32], h: &[f32], act: Option<Act>| {
        let mut out = gate_start(d, g, nb);
        if !x_zero {
            acc_f(d.x[g].as_ref().unwrap(), d.n_cell, d.n_input, x, nb, &mut out);
        }
        if let (false, Some(a)) = (aux_zero, aux) {
            acc_f(d.aux[g].as_ref().unwrap(), d.n_cell, d.n_aux, a, nb, &mut out);
        }
        acc_f(d.h[g].as_ref().unwrap(), d.n_cell, d.n_output, h, nb, &mut out);
        gate_finish(d, g, c, act, &mut out);
        out
    };
    let ig = d.x[0].as_ref().map(|_| gate(0, c, h, None));
    let mut fg = gate(1, c, h, None);
    let mut cg = gate(2, c, h, Some(p.act));
    update_cell(c, ig.as_deref(), &mut fg, &cg, p.cell_clip);
    let og = gate(3, c, h, None);
    hidden(p, c, &og, &mut cg);
    match &d.proj {
        Some(m) => {
            project_start(d, nb, h);
            acc_f(m, d.n_output, d.n_cell, &cg, nb, h);
            if p.proj_clip > 0.0 {
                clip_f(h, p.proj_clip);
            }
        }
        None => h.copy_from_slice(&cg[..h.len()]),
    }
}

/// LstmStepHybrid
#[allow(clippy::too_many_arguments)]
fn step_hybrid(d: &Dir, p: &Params, sums: Option<&Sums>, input_sf: &mut [f32], x: &[f32], aux: Option<&[f32]>, h: &mut [f32], c: &mut [f32], nb: usize) {
    let (x_zero, aux_zero, h_zero) = (is_zero(x), aux.is_none_or(is_zero), is_zero(h));
    let quantize = |v: &[f32], n: usize| batch_quantize(v, n, p.asymmetric);
    let (qx, x_zp) = if x_zero {
        (Vec::new(), Vec::new())
    } else {
        let (q, sf, zp) = quantize(x, d.n_input);
        input_sf[..nb].copy_from_slice(&sf);
        (q, zp)
    };
    let qa = aux.filter(|_| !aux_zero).map(|a| quantize(a, d.n_aux));
    let qh = (!h_zero).then(|| quantize(h, d.n_output));
    let sf_in: &[f32] = input_sf;
    let gate = |g: usize, c: &[f32], act: Option<Act>| {
        let mut out = gate_start(d, g, nb);
        let nc = d.n_cell;
        if !x_zero {
            acc_h(d.x[g].as_ref().unwrap(), nc, d.n_input, &qx, sf_in, sums.map(|s| (&s.x[g][..], &x_zp[..])), nb, &mut out);
        }
        if let Some((q, sf, zp)) = &qa {
            acc_h(d.aux[g].as_ref().unwrap(), nc, d.n_aux, q, sf, sums.map(|s| (&s.aux[g][..], &zp[..])), nb, &mut out);
        }
        if let Some((q, sf, zp)) = &qh {
            let m = d.h[g].as_ref().unwrap();
            // 2.5 scales a sparse recurrent matrix by the input's factors
            let sf = if m.ledger.is_some() { sf_in } else { sf };
            acc_h(m, nc, d.n_output, q, sf, sums.map(|s| (&s.h[g][..], &zp[..])), nb, &mut out);
        }
        gate_finish(d, g, c, act, &mut out);
        out
    };
    let ig = d.x[0].as_ref().map(|_| gate(0, c, None));
    let mut fg = gate(1, c, None);
    let mut cg = gate(2, c, Some(p.act));
    update_cell(c, ig.as_deref(), &mut fg, &cg, p.cell_clip);
    let og = gate(3, c, None);
    hidden(p, c, &og, &mut cg);
    match &d.proj {
        Some(m) => {
            project_start(d, nb, h);
            if !is_zero(&cg) {
                // the projection's rows quantized into the input's scaling factors (and zero points)
                let (q, sf, zp) = quantize(&cg, d.n_cell);
                acc_h(m, d.n_output, d.n_cell, &q, &sf, sums.map(|s| (&s.proj[..], &zp[..])), nb, h);
                for b in 0..nb {
                    input_sf[b] = if m.ledger.is_some() { sf[b] } else { sf[b] * m.scale };
                }
            }
            if p.proj_clip > 0.0 {
                clip_f(h, p.proj_clip);
            }
        }
        None => h.copy_from_slice(&cg[..h.len()]),
    }
}

// *** integer steps ***

/// one direction's tensors for the integer steps, as int32 values
struct IntDir {
    x: [Option<Vec<i32>>; 4],
    h: [Option<Vec<i32>>; 4],
    peep: [Option<Vec<i32>>; 4],
    ln: [Option<Vec<i32>>; 4],
    bias: [Option<Vec<i32>>; 4],
    proj: Option<Vec<i32>>,
    proj_bias: Option<Vec<i32>>,
    n_input: usize,
    n_cell: usize,
    n_output: usize,
}

/// int8 weights (uint8 bytes read as int8), int16 and int32 values
fn int_values(t: &Tensor) -> Vec<i32> {
    match t.dtype {
        TType::I16 => t.as_slice::<i16>().iter().map(|&v| v as i32).collect(),
        TType::I32 => t.as_slice::<i32>().to_vec(),
        _ => t.data().iter().map(|&b| b as i8 as i32).collect(),
    }
}

fn int_dir(io: &Io, sl: &Slots) -> Result<IntDir> {
    let v = |k: usize| opt_in(io, k).map(int_values);
    let (x, h) = (io.input(sl.x[1])?, io.input(sl.h[1])?);
    Ok(IntDir {
        x: sl.x.map(v),
        h: sl.h.map(v),
        peep: sl.peep.map(v),
        ln: sl.ln.map(v),
        bias: sl.bias.map(v),
        proj: v(sl.proj),
        proj_bias: v(sl.proj_bias),
        n_input: x.shape[1],
        n_cell: x.shape[0],
        n_output: h.shape[1],
    })
}

/// IntegerLstmParameter for 8x8_16
struct Int16 {
    x_scale: [(i32, i32); 4],
    h_scale: [(i32, i32); 4],
    c_scale: [(i32, i32); 4],
    ln_scale: [(i32, i32); 4],
    guard: [i32; 4],
    /// the gate biases (none with layer norm) with the input's (recurrent: the state's) zero point times the weights'
    /// row sums folded in, and the projection bias with the hidden zero point's
    x_bias: [Vec<i32>; 4],
    h_bias: [Vec<i32>; 4],
    proj_bias: Vec<i32>,
    hidden: (i32, i32),
    hidden_zp: i32,
    proj: (i32, i32),
    h_zp: i32,
    cell_scale: i32,
    clips: (i16, i8),
}

/// IntegerLstmParameter for 8x8_8
struct Int8 {
    x_scale: [(i32, i32); 4],
    h_scale: [(i32, i32); 4],
    ln_scale: [(i32, i32); 4],
    proj: (i32, i32),
    /// intermediate_scale_a then intermediate_scale_b, laid out as in the struct: 2.5 reads a[11], which is b[3]
    ab: [i32; 16],
    zp: [i32; 12],
    x_zp: i32,
    h_zp: i32,
}

/// MatrixBatchVectorMultiplyAccumulate for the integer LSTM: bias plus the dot product, requantized, plus the zero
/// point and what the output holds, saturated to [lo, hi]
#[allow(clippy::too_many_arguments)]
fn acc_int(x: &[i32], bias: &[i32], w: &[i32], (m, s): (i32, i32), cols: usize, zp: i32, lo: i32, hi: i32, out: &mut [i32]) {
    let rows = bias.len();
    for (b, o) in out.chunks_mut(rows).enumerate() {
        for r in 0..rows {
            let dot = (0..cols).fold(bias[r], |a, c| a.wrapping_add(x[b * cols + c] * w[r * cols + c]));
            o[r] = mul_by_quantized_multiplier(dot, m, s).wrapping_add(zp).wrapping_add(o[r]).clamp(lo, hi);
        }
    }
}

/// ApplyLayerNorm: integer mean and variance (exact for power-of-two sizes), the inverse square root as a multiplier,
/// then the weights, the bias and the layer norm scale
fn layer_norm_int(v: &mut [i32], w: &[i32], bias: &[i32], (a, s): (i32, i32), guard: i32, n: usize) {
    for row in v.chunks_mut(n) {
        let sum: i64 = row.iter().map(|&x| x as i64).sum();
        let sum_sq: i64 = row.iter().map(|&x| (x * x) as i64).sum();
        let mean = (sum * 1024 / n as i64) as i32;
        let temp = (1i32 << 20) / n as i32;
        let variance = sum_sq * temp as i64 - mean as i64 * mean as i64;
        let mut variance2 = (variance / (1 << 20)) as i32;
        if variance2 < 1 {
            variance2 = guard;
        }
        let (ia, ib) = inv_sqrt_multiplier_exp(variance2, -1);
        for j in 0..n {
            let rescaled = mul_by_quantized_multiplier((1024 * row[j]).wrapping_sub(mean), ia, ib);
            let val3 = rescaled.wrapping_mul(w[j]).wrapping_add(bias[j]) as i64;
            let val4 = ((if val3 > 0 { val3 + 512 } else { val3 - 512 }) / 1024) as i32;
            row[j] = mul_by_quantized_multiplier(val4, a, s + 12).clamp(-32768, 32767);
        }
    }
}

/// UpdateLstmCellInteger: cell = forget * cell >> 15, plus input (CIFG: 32767 - forget) * cell gate rounded by
/// 30 + the cell scale's exponent, saturated, clipped
fn update_cell_int(c: &mut [i32], ig: Option<&[i32]>, fg: &[i32], cg: &[i32], cell_scale: i32, clip: i16) {
    for k in 0..c.len() {
        let kept = rounding_divide_by_pot(fg[k] * c[k], 15) as i16 as i32;
        let i = ig.map_or((32767 - fg[k]) as i16 as i32, |ig| ig[k]);
        let added = rounding_divide_by_pot(i * cg[k], 30 + cell_scale) as i16 as i32;
        c[k] = (kept + added).clamp(-32768, 32767);
        if clip > 0 {
            c[k] = c[k].min(clip as i32).max(-(clip as i32));
        }
    }
}

/// LstmStepInteger8x8_16
fn step_i16(d: &IntDir, q: &Int16, x: &[i32], h: &mut [i32], c: &mut [i32], nb: usize) {
    let (ni, nc, no) = (d.n_input, d.n_cell, d.n_output);
    let gate = |g: usize, c: &[i32], sigmoid: bool| {
        let mut out = vec![0i32; nb * nc];
        acc_int(x, &q.x_bias[g], d.x[g].as_ref().unwrap(), q.x_scale[g], ni, 0, -32768, 32767, &mut out);
        acc_int(h, &q.h_bias[g], d.h[g].as_ref().unwrap(), q.h_scale[g], no, 0, -32768, 32767, &mut out);
        if let Some(w) = &d.peep[g] {
            // 2.5 walks the peephole by the output size, not the cell size
            let (m, s) = q.c_scale[g];
            for b in 0..nb {
                for v in 0..no {
                    let k = b * no + v;
                    out[k] = mul_by_quantized_multiplier(w[v] * c[k], m, s).wrapping_add(out[k]).clamp(-32768, 32767);
                }
            }
        }
        if let (Some(l), Some(b)) = (&d.ln[g], &d.bias[g]) {
            layer_norm_int(&mut out, l, b, q.ln_scale[g], q.guard[g], nc);
        }
        for v in out.iter_mut() {
            *v = if sigmoid { I16.logistic(*v as i64, 3) } else { I16.tanh(*v as i64, 3) } as i32;
        }
        out
    };
    let ig = d.x[0].as_ref().map(|_| gate(0, c, true));
    let fg = gate(1, c, true);
    let cg = gate(2, c, false);
    update_cell_int(c, ig.as_deref(), &fg, &cg, q.cell_scale, q.clips.0);
    let og = gate(3, c, true);
    // CalculateLstmOutputInteger8x8_16: the hidden values take the hidden zero point away (not add it)
    let hid: Vec<i32> = c
        .iter()
        .zip(&og)
        .map(|(&c, &o)| {
            let t = I16.tanh(c as i64, 15 + q.cell_scale) as i32;
            mul_by_quantized_multiplier(o * t, q.hidden.0, q.hidden.1).wrapping_sub(q.hidden_zp).clamp(-128, 127)
        })
        .collect();
    match &d.proj {
        Some(w) => {
            h.fill(0);
            acc_int(&hid, &q.proj_bias, w, q.proj, nc, q.h_zp, -128, 127, h);
            if q.clips.1 > 0 {
                let clip = q.clips.1 as i32;
                h.iter_mut().for_each(|v| *v = (*v).min(clip).max(-clip));
            }
        }
        None => h.copy_from_slice(&hid[..h.len()]),
    }
}

/// ApplyLayerNormFloat (8x8_8): mean and variance in float, the result in Q3.12
fn layer_norm_float(v: &mut [i32], w: &[i32], (a, b): (i32, i32), bias: &[i32], n: usize) {
    let scale = (a as f64 * 2f64.powf((b - 31) as f64)) as f32;
    let bias_scale = (2f64.powi(-10) * scale as f64) as f32;
    for row in v.chunks_mut(n) {
        let (mut sum, mut sum_sq) = (0f32, 0f32);
        for &x in row.iter() {
            sum += x as f32;
            sum_sq += x as f32 * x as f32;
        }
        let mean = sum / n as f32;
        let variance = sum_sq / n as f32 - mean * mean;
        let inv = if variance == 0.0 { (1.0f64 / 1e-8f64.sqrt()) as f32 } else { 1.0f32 / variance.sqrt() };
        for i in 0..n {
            let normalized = (row[i] as f32 - mean) * inv;
            let weighted = normalized * w[i] as f32 * scale + bias[i] as f32 * bias_scale;
            row[i] = round_i32(weighted as f64 * 4096.0).clamp(-32768, 32767);
        }
    }
}

/// ApplySigmoidFloat / ApplyTanhFloat (8x8_8): float functions on the int16 values times 2^integer_bits, back to Q0.15
fn float_fn(v: &mut [i32], integer_bits: i32, f: impl Fn(f32) -> f32) {
    for x in v {
        let y = f((*x as f64 * 2f64.powi(integer_bits)) as f32);
        *x = (cvtt(y as f64 * 32768.0, false) as i32).clamp(-32768, 32767);
    }
}

/// LstmStepInteger8x8_8. It always updates the cell as CIFG and leaves the input gate and the peepholes out.
fn step_i8(d: &IntDir, q: &Int8, clips: (i16, i8), x: &[i32], h: &mut [i32], c: &mut [i32], nb: usize) {
    let (ni, nc, no) = (d.n_input, d.n_cell, d.n_output);
    let ab = |k: usize| (q.ab[k], q.ab[8 + k]);
    // MatrixBatchVectorMultiply into int8 with the gate's zero point (an int8_t parameter)
    let mul = |v: &[i32], v_zp: i32, w: &[i32], (m, s): (i32, i32), cols: usize, zp: i32| -> Vec<i32> {
        (0..nb * nc)
            .map(|k| {
                let (b, r) = (k / nc, k % nc);
                let dot = (0..cols).fold(0i32, |a, c| a.wrapping_add((v[b * cols + c] - v_zp) * w[r * cols + c]));
                mul_by_quantized_multiplier(dot, m, s).wrapping_add(zp as i8 as i32).clamp(-128, 127)
            })
            .collect()
    };
    let gate = |g: usize, x_sum: (i32, i32), x_zp: usize, h_sum: (i32, i32), h_zp: usize, sigmoid: bool| {
        let s0 = mul(x, q.x_zp, d.x[g].as_ref().unwrap(), q.x_scale[g], ni, q.zp[x_zp]);
        let s1 = mul(h, q.h_zp, d.h[g].as_ref().unwrap(), q.h_scale[g], no, q.zp[h_zp]);
        // TwoGateSaturatingAdd
        let (z0, z1) = (q.zp[x_zp] as i8 as i32, q.zp[h_zp] as i8 as i32);
        let mut out: Vec<i32> = s0
            .iter()
            .zip(&s1)
            .map(|(&a, &b)| {
                let xs = mul_by_quantized_multiplier(a - z0, x_sum.0, x_sum.1);
                let hs = mul_by_quantized_multiplier(b - z1, h_sum.0, h_sum.1);
                hs.wrapping_add(xs).clamp(-32768, 32767)
            })
            .collect();
        layer_norm_float(&mut out, d.ln[g].as_ref().unwrap(), q.ln_scale[g], d.bias[g].as_ref().unwrap(), nc);
        if sigmoid {
            float_fn(&mut out, -12, |v| 1.0f32 / (1.0f32 + (-v).exp()));
        } else {
            float_fn(&mut out, -12, f32::tanh);
        }
        out
    };
    let fg = gate(1, ab(2), 4, ab(3), 5, true);
    let cg = gate(2, ab(4), 7, ab(5), 8, false);
    update_cell_int(c, None, &fg, &cg, -15, clips.0);
    let og = gate(3, ab(6), 10, (q.ab[11], q.ab[8 + 7]), 7, true);
    // CalculateLstmOutputInteger8x8_8
    let mut t = c.to_vec();
    float_fn(&mut t, -15, f32::tanh);
    for (t, &o) in t.iter_mut().zip(&og) {
        *t = rounding_divide_by_pot(o * *t, 15) as i16 as i32;
    }
    let (w, bias) = (d.proj.as_ref().unwrap(), d.proj_bias.as_ref().unwrap());
    for b in 0..nb {
        for r in 0..no {
            let acc = (0..nc).fold(bias[r] as i64, |a, k| a + (t[b * nc + k] * w[r * nc + k]) as i64);
            h[b * no + r] = mul_by_quantized_multiplier_i64(acc, q.proj.0, q.proj.1).wrapping_add(q.h_zp).clamp(-128, 127);
        }
    }
    if clips.1 > 0 {
        let clip = clips.1 as i32;
        h.iter_mut().for_each(|v| *v = (*v).min(clip).max(-clip));
    }
}

// *** walking the sequence ***

/// The steps of one direction as lstm_eval's Eval* take them: time-major steps every batch row at once,
/// batch-major one row at a time. `step(input offset, output offset, batch row, rows)`; the aux input is read at the
/// input's offset, and the state rows at b * the output's last dimension, as 2.5 does.
#[allow(clippy::too_many_arguments)]
fn walk(time_major: bool, forward: bool, max_time: usize, n_batch: usize, n_input: usize, leading: usize, offset: usize, mut step: impl FnMut(usize, usize, usize, usize) -> Result<()>) -> Result<()> {
    let at = |t: usize| if forward { t } else { max_time - 1 - t };
    if time_major {
        for t in (0..max_time).map(at) {
            step(t * n_batch * n_input, t * n_batch * leading + offset, 0, n_batch)?;
        }
    } else {
        for b in 0..n_batch {
            for t in (0..max_time).map(at) {
                let k = b * max_time + t;
                step(k * n_input, k * leading + offset, b, 1)?;
            }
        }
    }
    Ok(())
}

impl Lstm {
    fn bidi(&self) -> bool {
        matches!(self.kind, Kind::Bidirectional { .. })
    }

    fn time_major(&self) -> bool {
        match self.kind {
            Kind::Single => true,
            Kind::Sequence { time_major } => time_major,
            // 2.5 runs the hybrid passes as time-major whatever the option says
            Kind::Bidirectional { time_major, .. } => time_major || matches!(self.mode, Mode::Hybrid),
        }
    }

    fn merge(&self) -> bool {
        matches!(self.kind, Kind::Bidirectional { merge: true, .. })
    }

    /// the input of a direction and its aux input: the backward pass reads the aux input as its input when there
    /// are no aux weights (stacking without cross links)
    fn inputs<'a>(&self, io: &Io<'a>, dir: usize) -> Result<(&'a Tensor, Option<&'a Tensor>)> {
        let input = io.input(0)?;
        if !self.bidi() {
            return Ok((input, None));
        }
        let aux = io.opt(39);
        if io.opt(41).is_some() {
            Ok((input, aux))
        } else if dir == 1 && aux.is_some() {
            Ok((aux.unwrap(), None))
        } else {
            Ok((input, None))
        }
    }

    fn run(&mut self, io: &mut Io, dir: usize) -> Result<()> {
        let sl = slots(self.kind, dir);
        let (input, aux) = self.inputs(io, dir)?;
        let time_major = self.time_major();
        let shape = &input.shape;
        let (max_time, n_batch) = match (shape.len(), time_major) {
            (2, _) => (1, shape[0]),
            (_, true) => (shape[0], shape[1]),
            _ => (shape[1], shape[0]),
        };
        let ni = shape[shape.len() - 1];
        let out_k = if dir == 1 && !self.merge() { 1 } else { 0 };
        let leading = *io.outs[out_k].shape.last().unwrap();
        let offset = if dir == 1 && self.merge() { io.input(slots(self.kind, 0).h[3])?.shape[1] } else { 0 };
        if let Mode::Float | Mode::Hybrid = self.mode {
            let hybrid = matches!(self.mode, Mode::Hybrid);
            let d = float_dir(io, &sl, aux.is_some())?;
            let (nc, no, na) = (d.n_cell, d.n_output, d.n_aux);
            let x = input.as_slice::<f32>();
            let a = aux.map(|t| t.as_slice::<f32>());
            let mut hs = io.var(sl.h_state)?.as_slice::<f32>().to_vec();
            let mut cs = io.var(sl.c_state)?.as_slice::<f32>().to_vec();
            if hybrid && self.p.asymmetric && self.sums[dir].is_none() {
                self.sums[dir] = Some(Sums::of(&d));
            }
            if self.input_sf.len() < n_batch {
                self.input_sf.resize(n_batch, 0.0);
            }
            let (p, sums, input_sf) = (&self.p, self.sums[dir].as_ref(), &mut self.input_sf);
            let out = io.outs[out_k].as_mut_slice::<f32>();
            walk(time_major, dir == 0, max_time, n_batch, ni, leading, offset, |xo, oo, b, nb| {
                let x = &x[xo..xo + nb * ni];
                let aux = match a {
                    Some(a) => Some(a.get(xo..xo + nb * na).map_or_else(|| err("2.5 reads the aux input past its end"), Ok)?),
                    None => None,
                };
                let Some(h) = hs.get_mut(b * leading..b * leading + nb * no) else {
                    return err("2.5 reads the output state past its end");
                };
                let c = &mut cs[b * nc..(b + nb) * nc];
                if hybrid {
                    step_hybrid(&d, p, sums, input_sf, x, aux, h, c, nb);
                } else {
                    step_float(&d, p, x, aux, h, c, nb);
                }
                for r in 0..nb {
                    out[oo + r * leading..oo + r * leading + no].copy_from_slice(&h[r * no..(r + 1) * no]);
                }
                Ok(())
            })?;
            io.var(sl.h_state)?.as_mut_slice::<f32>().copy_from_slice(&hs);
            io.var(sl.c_state)?.as_mut_slice::<f32>().copy_from_slice(&cs);
            return Ok(());
        }
        let d = int_dir(io, &sl)?;
        let (nc, no) = (d.n_cell, d.n_output);
        let x = int_values(input);
        let mut hs = int_values(io.var(sl.h_state)?);
        let mut cs = int_values(io.var(sl.c_state)?);
        let (mode, clips8) = (&self.mode, self.clips8);
        let out = io.outs[0].as_mut_slice::<i8>();
        walk(time_major, true, max_time, n_batch, ni, leading, 0, |xo, oo, b, nb| {
            let x = &x[xo..xo + nb * ni];
            let h = &mut hs[b * no..(b + nb) * no];
            let c = &mut cs[b * nc..(b + nb) * nc];
            match mode {
                Mode::Int16(q) => step_i16(&d, q, x, h, c, nb),
                Mode::Int8(q) => step_i8(&d, q, clips8, x, h, c, nb),
                _ => unreachable!(),
            }
            // the integer steps copy the state rows out contiguously
            for (o, &v) in out[oo..oo + nb * no].iter_mut().zip(h.iter()) {
                *o = v as i8;
            }
            Ok(())
        })?;
        io.var(sl.h_state)?.as_mut_slice::<i8>().iter_mut().zip(&hs).for_each(|(o, &v)| *o = v as i8);
        io.var(sl.c_state)?.as_mut_slice::<i16>().iter_mut().zip(&cs).for_each(|(o, &v)| *o = v as i16);
        Ok(())
    }
}

// *** prepare ***

/// CheckInputTensorDimensions for one direction: weights, peepholes, biases, projection, layer norm and aux
/// weights against the sizes and types, the CIFG, peephole and projection groups all or none. (n_cell, n_output).
fn check_dir(io: &Io, sl: &Slots, n_input: usize, n_aux: Option<usize>, ln: bool, integer: bool) -> Result<(usize, usize)> {
    let get = |k: usize| opt_in(io, k);
    let (xo, ho) = (io.input(sl.x[3])?, io.input(sl.h[3])?);
    if xo.shape.len() != 2 || xo.shape[1] != n_input || ho.shape.len() != 2 || ho.shape[0] != xo.shape[0] {
        return err("the input-to-output and recurrent-to-output weights do not fit the input");
    }
    let (nc, no) = (xo.shape[0], ho.shape[1]);
    let wt = io.input(sl.x[1])?.dtype;
    if !matches!(wt, TType::F32 | TType::I8 | TType::U8) {
        return err(format!("{wt:?} weights are not supported"));
    }
    let cifg = get(sl.x[0]).is_none();
    let fits = |t: Option<&Tensor>, shape: &[usize], dtype: TType, required: bool| t.map_or(!required, |t| t.shape == shape && t.dtype == dtype);
    let (bias_t, ln_t, peep_t) = if integer { (TType::I32, TType::I16, TType::I16) } else { (TType::F32, TType::F32, wt) };
    let mut ok = get(sl.h[0]).is_none() == cifg;
    for g in 0..4 {
        ok &= fits(get(sl.x[g]), &[nc, n_input], wt, g != 0) && fits(get(sl.h[g]), &[nc, no], wt, g != 0);
        ok &= fits(get(sl.bias[g]), &[nc], bias_t, g != 0) && (g != 0 || get(sl.bias[0]).is_none() == cifg);
        ok &= fits(get(sl.peep[g]), &[nc], peep_t, false);
        if ln {
            ok &= fits(get(sl.ln[g]), &[nc], ln_t, g != 0) && (g != 0 || get(sl.ln[0]).is_none() == cifg);
        }
        if let Some(na) = n_aux {
            ok &= fits(get(sl.aux[g]), &[nc, na], wt, g != 0 || !cifg);
        }
    }
    let peeps = [0, 1, 3].map(|g| get(sl.peep[g]).is_some());
    ok &= peeps == [false; 3] || ((peeps[0] || cifg) && peeps[1] && peeps[2]);
    ok &= fits(get(sl.proj), &[no, nc], wt, false) && fits(get(sl.proj_bias), &[no], bias_t, false);
    ok &= get(sl.proj).is_some() || get(sl.proj_bias).is_none();
    if !ok {
        return err("the LSTM weights, biases, peepholes, projection or layer norm do not fit together");
    }
    Ok((nc, no))
}

/// a sparse hybrid matrix as 2.5's ledger can take it: dense rows of 1x16 blocks, indices and counts in a byte
fn check_sparse(t: &Tensor) -> Result<()> {
    let Some(sp) = &t.sparsity else { return Ok(()) };
    let (rows, cols) = (t.shape[0], t.shape[1]);
    let ok = match &sp.dims[..] {
        [SparseDim::Dense(r), SparseDim::Sparse(seg, idx), SparseDim::Dense(16)] => {
            *r == rows
                && cols % 16 == 0
                && sp.traversal_order == [0, 1, 2]
                && sp.block_map == [1]
                && seg.len() == rows + 1
                && seg.windows(2).all(|w| w[0] <= w[1] && w[1] - w[0] <= 255)
                && idx.len() == seg[rows] as usize
                && idx.iter().all(|&i| (0..=255).contains(&i) && (i as usize + 1) * 16 <= cols)
                && t.data().len() == idx.len() * 16
        }
        _ => false,
    };
    if ok { Ok(()) } else { err("the sparse LSTM weights are not in 1x16 blocks the ledger can hold") }
}

/// static_cast<int16_t> / <int8_t> of std::min(std::max(clip / scale, lo), hi), when the clip is set
fn quantized_clip(clip: f32, scale: f32, lo: f32, hi: f32) -> i32 {
    let v = clip / scale;
    let v = if v < lo { lo } else { v };
    let v = if hi < v { hi } else { v };
    cvtt(v as f64, false) as i32
}

impl Lstm {
    /// PopulateQuantizedLstmParams8x8_16 and PopulatePrecomputedZPTimesWeightsWithBias
    fn int16_params(&self, io: &mut Io, sl: &Slots, ln: bool) -> Result<Int16> {
        let (c_scale, (os, h_zp)) = (io.var(sl.c_state)?.scale(), {
            let h = io.var(sl.h_state)?;
            (h.scale(), h.zero_point())
        });
        let out_scale = io.outs[0].scale();
        let input = io.input(0)?;
        if io.inter.len() != 5 {
            return err("the 8x8_16 integer LSTM needs 5 intermediates");
        }
        let mut is = [2f32.powi(-12); 5];
        let mut iz = [0i32; 5];
        for (i, t) in io.inter.iter().enumerate().filter(|&(i, _)| ln || i == 4) {
            (is[i], iz[i]) = (t.scale(), t.zero_point());
        }
        let (cell_scale, pot) = checked_log2(c_scale);
        if !pot || !(-15..=-9).contains(&cell_scale) {
            return err("the integer LSTM's cell scale must be a power of two from 2^-15 to 2^-9");
        }
        let get = |k: usize| opt_in(io, k);
        let scale = |k: usize| get(k).map_or(1.0, |t| t.scale());
        let q = |x: f32| quantize_multiplier(x as f64);
        let in_s = input.scale();
        let present = |k: usize, v: f32| if get(k).is_some() { q(v) } else { q(1.0) };
        let x_scale = [0, 1, 2, 3].map(|g| present(sl.x[g], scale(sl.x[g]) * in_s / is[g]));
        let h_scale = [0, 1, 2, 3].map(|g| present(sl.h[g], scale(sl.h[g]) * os / is[g]));
        let c_scale_m = [0, 1, 2, 3].map(|g| present(sl.peep[g], (2f64.powi(cell_scale) * scale(sl.peep[g]) as f64 / is[g] as f64) as f32));
        let ln_f = [0, 1, 2, 3].map(|g| scale(sl.ln[g]));
        let row_sums_times = |w: Option<&Tensor>, bias: Option<&Tensor>, zp: i32| -> Vec<i32> {
            let Some(w) = w else { return Vec::new() };
            let (rows, cols) = (w.shape[0], w.shape[1]);
            let mut out = bias.map_or(vec![0; rows], |b| b.as_slice::<i32>().to_vec());
            let wv = int_values(w);
            for r in 0..rows {
                let sum = wv[r * cols..(r + 1) * cols].iter().fold(0i32, |a, &v| a.wrapping_add(v));
                out[r] = out[r].wrapping_add(sum.wrapping_mul(zp));
            }
            out
        };
        let in_zp = input.zero_point();
        Ok(Int16 {
            x_scale,
            h_scale,
            c_scale: c_scale_m,
            ln_scale: ln_f.map(q),
            guard: ln_f.map(|s| (cvtt((10000.0f32 * s) as f64, false) as i32).max(1)),
            x_bias: [0, 1, 2, 3].map(|g| row_sums_times(get(sl.x[g]), if ln { None } else { get(sl.bias[g]) }, -in_zp)),
            h_bias: [0, 1, 2, 3].map(|g| row_sums_times(get(sl.h[g]), None, -h_zp)),
            proj_bias: row_sums_times(get(sl.proj), get(sl.proj_bias), iz[4]),
            hidden: q((2f64.powi(-15) / is[4] as f64 * 2f64.powi(-15)) as f32),
            hidden_zp: iz[4],
            proj: q(scale(sl.proj) * is[4] / os),
            h_zp,
            cell_scale,
            clips: (
                if self.p.cell_clip > 0.0 { quantized_clip(self.p.cell_clip, c_scale, -32768.0, 32767.0) as i16 } else { 0 },
                if self.p.proj_clip > 0.0 { quantized_clip(self.p.proj_clip, out_scale, -128.0, 127.0) as i8 } else { 0 },
            ),
        })
    }

    /// PopulateQuantizedLstmParams8x8_8. Its check of the cell scale returns before the clips, which 2.5 ignores:
    /// with another cell scale the clips stay as they were.
    fn int8_params(&mut self, io: &mut Io, sl: &Slots) -> Result<Int8> {
        let c_scale = io.var(sl.c_state)?.scale();
        let (os, h_zp) = {
            let h = io.var(sl.h_state)?;
            (h.scale(), h.zero_point())
        };
        let out_scale = io.outs[0].scale();
        let input = io.input(0)?;
        let get = |k: usize| opt_in(io, k);
        let needed = [sl.proj, sl.proj_bias, sl.ln[1], sl.ln[2], sl.ln[3]];
        if io.inter.len() != 12 || needed.iter().any(|&k| get(k).is_none()) {
            return err("the 8x8_8 integer LSTM needs 12 intermediates, layer norm and a projection with a bias");
        }
        let s: Vec<f32> = io.inter.iter().map(|t| t.scale()).collect();
        let zp: Vec<i32> = io.inter.iter().map(|t| t.zero_point()).collect();
        let scale = |k: usize| get(k).map_or(1.0, |t| t.scale());
        let q = |x: f32| quantize_multiplier(x as f64);
        let in_s = input.scale();
        let (xs, hs) = ([1, 4, 7, 10], [2, 5, 8, 11]);
        let x_scale = [0, 1, 2, 3].map(|g| if get(sl.x[g]).is_some() { q(scale(sl.x[g]) * in_s / s[xs[g]]) } else { q(1.0) });
        let h_scale = [0, 1, 2, 3].map(|g| if get(sl.h[g]).is_some() { q(scale(sl.h[g]) * os / s[hs[g]]) } else { q(1.0) });
        let mut ab = [0i32; 16];
        for (k, (i, j)) in [(1, 0), (2, 0), (4, 3), (5, 3), (7, 6), (8, 6), (10, 9), (11, 9)].into_iter().enumerate() {
            (ab[k], ab[8 + k]) = q(s[i] / s[j]);
        }
        if c_scale as f64 == 1.0 / 32768.0 {
            let clip = self.p.cell_clip;
            self.clips8 = (
                if clip > 0.0 && clip < 1.0 { quantized_clip(clip, c_scale, -32768.0, 32767.0) as i16 } else { 0 },
                if self.p.proj_clip > 0.0 { quantized_clip(self.p.proj_clip, out_scale, -128.0, 127.0) as i8 } else { 0 },
            );
        }
        Ok(Int8 {
            x_scale,
            h_scale,
            ln_scale: [0, 1, 2, 3].map(|g| q(scale(sl.ln[g]))),
            proj: q((scale(sl.proj) as f64 * 2f64.powi(-15) / os as f64) as f32),
            ab,
            zp: std::array::from_fn(|i| zp[i]),
            x_zp: input.zero_point(),
            h_zp,
        })
    }
}

impl Kernel for Lstm {
    fn prepare(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        let bidi = self.bidi();
        let n_in = io.n_in();
        if bidi && n_in != 48 || !bidi && n_in != 20 && n_in != 24 {
            return err(format!("{n_in} inputs: the LSTM full kernel takes 20 or 24, the bidirectional one 48"));
        }
        if io.outs.len() != if bidi && !self.merge() { 2 } else { 1 } {
            return err("wrong number of outputs");
        }
        if !(self.p.cell_clip >= 0.0 && self.p.proj_clip >= 0.0) {
            return err("the clips must not be negative");
        }
        let input = io.input(0)?;
        let rank = if self.kind == Kind::Single { 2 } else { 3 };
        if input.shape.len() != rank {
            return err(format!("the input must have {rank} dimensions"));
        }
        let ln = !bidi && opt_in(io, 21).is_some();
        if !bidi && !ln && [20, 22, 23].iter().any(|&k| opt_in(io, k).is_some()) {
            return err("layer norm coefficients without the forget gate's");
        }
        let integer = input.dtype == TType::I8;
        let declared_major = match self.kind {
            Kind::Single => true,
            Kind::Sequence { time_major } | Kind::Bidirectional { time_major, .. } => time_major,
        };
        let shape = input.shape.clone();
        let n_batch = if rank == 2 || !declared_major { shape[0] } else { shape[1] };
        let n_input = shape[rank - 1];
        // aux input (bidirectional): with aux weights it feeds both passes; without, the backward pass reads it
        let use_aux = bidi && opt_in(io, 41).is_some();
        let aux = if bidi { io.opt(39) } else { None };
        if let Some(a) = aux {
            let ok = a.dtype == TType::F32 && if use_aux { a.shape.len() == 3 && a.shape[..2] == shape[..2] } else { a.shape == shape };
            if !ok {
                return err("the aux input does not fit the input");
            }
        } else if use_aux {
            return err("aux weights without an aux input");
        }
        let n_aux = use_aux.then(|| aux.unwrap().shape[2]);
        let sl = slots(self.kind, 0);
        let (nc, no) = check_dir(io, &sl, n_input, n_aux, ln, integer)?;
        if bidi && check_dir(io, &slots(self.kind, 1), n_input, n_aux, false, false)? != (nc, no) {
            return err("2.5 checks each direction against both directions' sizes: they must be the same");
        }
        let dirs = if bidi { 2 } else { 1 };
        let wt = io.input(sl.x[3])?.dtype;
        for dir in 0..dirs {
            let s = slots(self.kind, dir);
            if io.var(s.h_state)?.elements() != n_batch * no || io.var(s.c_state)?.elements() != n_batch * nc {
                return err("the state tensors do not fit the batch and the sizes");
            }
        }
        // the cell's computation
        let mode = match (input.dtype, wt) {
            (TType::F32, TType::F32) => 0,
            (TType::F32, TType::I8 | TType::U8) => 1,
            (TType::I8, TType::I8 | TType::U8) if !bidi => 2,
            (i, w) => return err(format!("{i:?} input with {w:?} weights is not supported")),
        };
        let states = [io.var(sl.h_state)?.dtype, io.var(sl.c_state)?.dtype];
        let types_ok = if mode < 2 {
            states == [TType::F32; 2] && io.outs.iter().all(|o| o.dtype == TType::F32)
        } else {
            states == [TType::I8, TType::I16] && io.outs[0].dtype == TType::I8
        };
        if !types_ok {
            return err("the states and outputs must be float32 (int8 and int16 states, int8 output for int8)");
        }
        let has_proj = opt_in(io, sl.proj).is_some();
        if !has_proj && no > nc {
            return err("2.5 copies the output state from the cell size: without a projection it must not be larger");
        }
        // sparse weights: only LSTM's symmetric hybrid kernel takes them, through ledgers, and needs a projection
        let mats: Vec<usize> = (0..dirs).flat_map(|d| {
            let s = slots(self.kind, d);
            s.x.into_iter().chain(s.h).chain(s.aux).chain([s.proj])
        }).collect();
        let sparse_op = io.input(sl.x[3])?.sparsity.is_some();
        for &k in &mats {
            if let Some(t) = opt_in(io, k).filter(|t| t.sparsity.is_some()) {
                if !(sparse_op && mode == 1 && self.kind == Kind::Single && !self.p.asymmetric && has_proj) {
                    return err("sparse LSTM weights are only supported by LSTM's symmetric hybrid kernel with a projection");
                }
                check_sparse(t)?;
            }
        }
        if sparse_op && (mode != 1 || self.kind != Kind::Single || self.p.asymmetric || !has_proj) {
            return err("sparse LSTM weights are only supported by LSTM's symmetric hybrid kernel with a projection");
        }
        if mode == 0 && bidi && self.merge() && !declared_major && n_batch > 1 {
            return err("2.5 steps a merged batch-major pass past its output state");
        }
        if mode == 1 && bidi && !declared_major && shape[1] > shape[0] {
            return err("2.5 runs this hybrid pass as time-major, past its state for a longer time than batch");
        }
        self.mode = match mode {
            0 => Mode::Float,
            1 => Mode::Hybrid,
            _ => {
                if opt_in(io, sl.peep[3]).is_some() && no > nc {
                    return err("2.5 walks the integer peephole past the cell for an output larger than the cell");
                }
                match (io.inter.len(), self.kind) {
                    (5, _) => Mode::Int16(Box::new(self.int16_params(io, &sl, ln)?)),
                    (12, Kind::Single) => Mode::Int8(Box::new(self.int8_params(io, &sl)?)),
                    (n, _) => return err(format!("{n} intermediates: the integer LSTM takes 5 (or 12 for LSTM)")),
                }
            }
        };
        self.sums = [None, None];
        self.input_sf.clear();
        // outputs
        let last = if self.merge() { 2 * no } else { no };
        let mut out_shape = if self.kind == Kind::Single { vec![n_batch, no] } else { shape.clone() };
        *out_shape.last_mut().unwrap() = last;
        io.outs[0].resize(out_shape);
        if bidi && !self.merge() {
            let mut s = shape;
            *s.last_mut().unwrap() = no;
            io.outs[1].resize(s);
        }
        Ok(())
    }

    fn eval(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        for dir in 0..if self.bidi() { 2 } else { 1 } {
            self.run(io, dir)?;
        }
        Ok(())
    }

    fn mutated_inputs(&self) -> &[usize] {
        if self.bidi() { &[35, 36, 37, 38] } else { &[18, 19] }
    }
}

// *** basic kernel ***

/// LSTM's 5-input basic kernel (optimized_ops::LstmCell): the input and the previous activation concatenated, one
/// fully connected layer into the gates (input, new input, forget, output), then the cell. It writes the activation,
/// the state, the concatenation and the gates' pre-activations, and copies the first two back into the previous
/// activation and state inputs.
struct Basic;

impl Kernel for Basic {
    fn prepare(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        if io.n_in() != 5 || io.outs.len() != 4 {
            return err("the basic LSTM kernel takes 5 inputs and 4 outputs");
        }
        let (pa, ps) = (io.var(1)?.shape.clone(), io.var(4)?.shape.clone());
        let (i, w, b) = (io.input(0)?, io.input(2)?, io.input(3)?);
        let ok = i.shape.len() == 2 && pa.len() == 2 && pa[0] == i.shape[0] && ps == pa && w.shape == [4 * pa[1], i.shape[1] + pa[1]] && b.shape == [4 * pa[1]];
        if !ok {
            return err("the basic LSTM's input, weights, bias and states do not fit together");
        }
        let (batches, depth, total) = (i.shape[0], pa[1], i.shape[1] + pa[1]);
        io.outs[0].resize(pa);
        io.outs[1].resize(ps);
        io.outs[2].resize(vec![batches, total]);
        io.outs[3].resize(vec![batches, 4 * depth]);
        Ok(())
    }

    fn eval(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        let (i, w, b) = (io.input(0)?, io.input(2)?, io.input(3)?);
        let pa = io.var(1)?.clone();
        let ps = io.var(4)?.clone();
        let types = [i.dtype, pa.dtype, w.dtype, b.dtype, ps.dtype, io.outs[1].dtype, io.outs[0].dtype, io.outs[2].dtype, io.outs[3].dtype];
        let (batches, ni, d) = (i.shape[0], i.shape[1], pa.shape[1]);
        let total = ni + d;
        if types == [TType::F32; 9] {
            let (x, a) = (i.as_slice::<f32>(), pa.as_slice::<f32>());
            let concat: Vec<f32> = (0..batches).flat_map(|r| x[r * ni..(r + 1) * ni].iter().chain(&a[r * d..(r + 1) * d]).copied()).collect();
            let (wv, bv) = (w.as_slice::<f32>(), b.as_slice::<f32>());
            // the float GEMM (Eigen) sums in its own order: compared within a tolerance
            let gates: Vec<f32> = (0..batches * 4 * d)
                .map(|k| {
                    let (r, u) = (k / (4 * d), k % (4 * d));
                    let dot: f64 = (0..total).map(|j| wv[u * total + j] as f64 * concat[r * total + j] as f64).sum();
                    (dot + bv[u] as f64) as f32
                })
                .collect();
            let prev = ps.as_slice::<f32>();
            let (mut state, mut activ) = (vec![0f32; batches * d], vec![0f32; batches * d]);
            for r in 0..batches {
                for c in 0..d {
                    let g = |k: usize| gates[r * 4 * d + k * d + c];
                    let s = eigen::logistic(g(0)) * eigen::tanh(g(1)) + eigen::logistic(g(2)) * prev[r * d + c];
                    state[r * d + c] = s;
                    activ[r * d + c] = eigen::logistic(g(3)) * eigen::tanh(s);
                }
            }
            io.outs[0].as_mut_slice::<f32>().copy_from_slice(&activ);
            io.outs[1].as_mut_slice::<f32>().copy_from_slice(&state);
            io.outs[2].as_mut_slice::<f32>().copy_from_slice(&concat);
            io.outs[3].as_mut_slice::<f32>().copy_from_slice(&gates);
        } else if types == [TType::U8, TType::U8, TType::U8, TType::I32, TType::I16, TType::I16, TType::U8, TType::U8, TType::I16] {
            let (log2, pot) = checked_log2(io.outs[1].scale());
            if !pot {
                return err("The internal state of a LSTM cell must have a power-of-two scale.");
            }
            if 15 + log2 != 4 {
                return err("The only case of quantized LstmCell currently supported is with StateIntegerBits==4");
            }
            let (m, shift) = quantize_multiplier((4096.0f32 * b.scale()) as f64);
            let wzp = w.zero_point();
            let (x, a) = (i.data(), pa.data());
            let concat: Vec<u8> = (0..batches).flat_map(|r| x[r * ni..(r + 1) * ni].iter().chain(&a[r * d..(r + 1) * d]).copied()).collect();
            let (wv, bv) = (w.data(), b.as_slice::<i32>());
            // the 8-bit GEMM into int16 (the input's zero point is 128)
            let gates: Vec<i32> = (0..batches * 4 * d)
                .map(|k| {
                    let (r, u) = (k / (4 * d), k % (4 * d));
                    let dot = (0..total).fold(0i32, |s, j| s.wrapping_add((wv[u * total + j] as i32 - wzp) * (concat[r * total + j] as i32 - 128)));
                    gemm_requantize(dot.wrapping_add(bv[u]), m, shift, true).clamp(-32768, 32767)
                })
                .collect();
            let prev = ps.as_slice::<i16>();
            let f = I16;
            let (mut state, mut activ) = (vec![0i16; batches * d], vec![0u8; batches * d]);
            for r in 0..batches {
                for c in 0..d {
                    let g = |k: usize| gates[r * 4 * d + k * d + c] as i64;
                    // F3 gates; F0 products; the state in Q4.11
                    let input_times_modulation = f.mul(f.logistic(g(0), 3), f.tanh(g(1), 3));
                    let prev_times_forget = f.mul(f.logistic(g(2), 3), prev[r * d + c] as i64);
                    let s = (f.rescale(input_times_modulation, 0, 4) + prev_times_forget).clamp(-32768, 32767);
                    let out = f.mul(f.logistic(g(3), 3), f.tanh(f.rescale(s, 4, 3), 3));
                    state[r * d + c] = s as i16;
                    activ[r * d + c] = (128 + f.rdbp(out, 8).clamp(-128, 127)) as u8;
                }
            }
            io.outs[0].as_mut_slice::<u8>().copy_from_slice(&activ);
            io.outs[1].as_mut_slice::<i16>().copy_from_slice(&state);
            io.outs[2].as_mut_slice::<u8>().copy_from_slice(&concat);
            io.outs[3].as_mut_slice::<i16>().iter_mut().zip(&gates).for_each(|(o, &v)| *o = v as i16);
        } else {
            return err("Unsupported combination of data types for LstmCell");
        }
        let (activ, state) = (io.outs[0].data().to_vec(), io.outs[1].data().to_vec());
        io.var(1)?.data_mut().copy_from_slice(&activ);
        io.var(4)?.data_mut().copy_from_slice(&state);
        Ok(())
    }

    fn mutated_inputs(&self) -> &[usize] {
        &[1, 4]
    }
}
