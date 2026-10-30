//! FULLY_CONNECTED (fully_connected.cc), the paths 2.5 takes by type: float through Eigen's GEMM (here summed in
//! double: the order of Eigen's blocked product depends on the CPU's cache sizes, so float matches within
//! rounding), quantized through ruy / gemmlowp (exact integers, then MultiplyByQuantizedMultiplier), hybrid (float
//! input, 8-bit weights) through portable_tensor_utils, and the shuffled 4x16 uint8 weights.

use super::eigen;
use super::util::*;
use super::{Io, Kernel};
use crate::interp::Env;
use crate::tensor::{TType, Tensor};
use crate::tflite::{Act, OpSpec};
use crate::Result;

pub fn create(s: &OpSpec) -> Result<Option<Box<dyn Kernel>>> {
    Ok(Some(match s.code {
        9 => Box::new(FullyConnected {
            act: opt_act(s, 0)?,
            shuffled: match opt::<i8>(s, 1, 0) {
                0 => false,
                1 => true,
                f => return err(format!("weights format {f} is not supported")),
            },
            keep_num_dims: opt(s, 2, 0u8) != 0,
            asymmetric: opt(s, 3, 0u8) != 0,
            multiplier: (0, 0),
            act_q: (0, 0),
            row_sums: None,
        }),
        126 => Box::new(BatchMatMul {
            adj_x: opt(s, 0, 0u8) != 0,
            adj_y: opt(s, 1, 0u8) != 0,
            asymmetric: opt(s, 2, 0u8) != 0,
            multiplier: (0, 0),
            row_sums: None,
        }),
        _ => return Ok(None),
    }))
}

/// batch_matmul.cc: lhs [.., M, K] (or [.., K, M] with adj_x) times rhs [.., K, N] (or [.., N, K] with adj_y),
/// batch dimensions broadcast. 2.5 runs it as GEMMs of the rhs transposed to N x K against the lhs as M x K.
struct BatchMatMul {
    adj_x: bool,
    adj_y: bool,
    asymmetric: bool,
    multiplier: (i32, i32),
    /// hybrid: the N x K rows' sums, worked out on the first run after each prepare as 2.5 does
    row_sums: Option<Vec<i32>>,
}

impl Kernel for BatchMatMul {
    fn prepare(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        let (a, b) = (io.input(0)?, io.input(1)?);
        let o = &io.outs[0];
        let ok_type = |t: TType| matches!(t, TType::F32 | TType::I8 | TType::I16);
        if !ok_type(a.dtype) || !ok_type(b.dtype) || !((a.dtype == TType::F32 && b.dtype == TType::I8) || a.dtype == b.dtype) {
            return err(format!("{:?} x {:?} is not supported", a.dtype, b.dtype));
        }
        if !(2..=4).contains(&a.shape.len()) || !(2..=4).contains(&b.shape.len()) {
            return err("BATCH_MATMUL takes 2 to 4 dimensions");
        }
        if matches!(a.dtype, TType::I8 | TType::I16) {
            self.multiplier = quantize_multiplier((a.scale() * b.scale()) as f64 / o.scale() as f64);
        }
        if a.dtype == TType::I16 && (a.zero_point() != 0 || b.zero_point() != 0 || o.zero_point() != 0) {
            return err("int16 needs zero points 0");
        }
        let r = a.shape.len().max(b.shape.len());
        let (ea, eb) = (extend(&a.shape, r), extend(&b.shape, r));
        let mut shape = Vec::with_capacity(r);
        for k in 0..r - 2 {
            if ea[k] != eb[k] && ea[k] != 1 && eb[k] != 1 {
                return err("the batch dimensions do not broadcast");
            }
            shape.push(if ea[k] != eb[k] && ea[k] == 1 { eb[k] } else { ea[k] });
        }
        let (ka, kb) = (if self.adj_x { ea[r - 2] } else { ea[r - 1] }, if self.adj_y { eb[r - 1] } else { eb[r - 2] });
        if ka != kb {
            return err("the depths do not match");
        }
        shape.push(if self.adj_x { ea[r - 1] } else { ea[r - 2] });
        shape.push(if self.adj_y { eb[r - 2] } else { eb[r - 1] });
        self.row_sums = None;
        io.outs[0].resize(shape);
        Ok(())
    }

    fn eval(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        let (a, b) = (io.input(0)?, io.input(1)?);
        let o = &mut io.outs[0];
        let r = o.shape.len();
        let (m, n) = (o.shape[r - 2], o.shape[r - 1]);
        let k = if self.adj_x { a.shape[a.shape.len() - 2] } else { a.shape[a.shape.len() - 1] };
        // GEMM operands: the lhs as M x K rows, the rhs as N x K rows, each batch contiguous
        let rows_of = |t: &Tensor, swap: bool| -> Result<Vec<f64>> {
            let v = f64s(t)?;
            let s = &t.shape;
            let (p, q) = (s[s.len() - 2], s[s.len() - 1]);
            Ok(if swap { v.chunks(p * q).flat_map(|mat| (0..q).flat_map(move |j| (0..p).map(move |i| mat[i * q + j]))).collect() } else { v })
        };
        let (lhs, rhs) = (rows_of(a, self.adj_x)?, rows_of(b, !self.adj_y)?);
        // 5-D extended shapes as 2.5 passes them: the rhs as [.., N, K] (GEMM lhs), the lhs as [.., K, M] (GEMM rhs)
        let batch = |s: &[usize]| extend(&s[..s.len() - 2], 3);
        let (bl, bg) = (batch(&b.shape), batch(&a.shape));
        let (sl, sg) = ([bl[0], bl[1], bl[2], n, k], [bg[0], bg[1], bg[2], k, m]);
        let extent = |s: &[usize; 5], x: usize| if s[x] == 1 { 0 } else { s[x + 1..].iter().product() };
        let dims: Vec<usize> = (0..3).map(|x| if sl[x] == sg[x] || sl[x] != 1 { sl[x] } else { sg[x] }).collect();
        let mut batches = Vec::new();
        for b0 in 0..dims[0] {
            for b1 in 0..dims[1] {
                for b2 in 0..dims[2] {
                    batches.push([b0, b1, b2]);
                }
            }
        }
        let offset = |s: &[usize; 5], bb: &[usize; 3]| (0..3).map(|x| bb[x] * extent(s, x)).sum::<usize>();
        let mut res = vec![0.0f64; dims.iter().product::<usize>() * m * n];
        match (a.dtype, b.dtype) {
            (TType::F32, TType::I8) => {
                // EvalHybrid: the lhs rows quantized, int32 GEMM, offsets removed in float and truncated back
                let rows: Vec<f32> = lhs.iter().map(|&v| v as f32).collect();
                let mut quant = Vec::with_capacity(rows.len());
                let (mut scales, mut offs) = (Vec::new(), Vec::new());
                for row in rows.chunks(k.max(1)) {
                    let (q, s, zp) = quantize_row(row, self.asymmetric);
                    quant.extend(q);
                    scales.push(s * b.scale());
                    offs.push(zp);
                }
                let w: Vec<i32> = rhs.iter().map(|&v| v as i32).collect();
                if self.row_sums.is_none() {
                    let count = sl[1] * sl[2] * n;
                    self.row_sums = Some(w.chunks(k.max(1)).take(count).map(|r| r.iter().sum()).collect());
                }
                let rs = self.row_sums.as_ref().unwrap();
                // 2.5 steps through the scales and row sums by M (N) per batch dimension, not by their products
                let step = |s: &[usize; 5], len: usize, bb: &[usize; 3]| (0..3).map(|x| if extent(s, x) == 0 { 0 } else { bb[x] * len }).sum::<usize>();
                for (bi, bb) in batches.iter().enumerate() {
                    let (lo, go) = (offset(&sl, bb), offset(&sg, bb));
                    let (io_, wo) = (step(&sg, m, bb), step(&sl, n, bb));
                    for j in 0..m {
                        for i in 0..n {
                            let dot = (0..k).fold(0i32, |s, d| s.wrapping_add(w[lo + i * k + d] * quant[go + j * k + d]));
                            let acc = cvtt((dot as f32 - rs[wo + i] as f32 * offs[io_ + j] as f32) as f64, false) as i32;
                            res[bi * m * n + j * n + i] = (0.0f32 + scales[io_ + j] * acc as f32) as f64;
                        }
                    }
                }
            }
            (TType::F32, _) => {
                for (bi, bb) in batches.iter().enumerate() {
                    let (lo, go) = (offset(&sl, bb), offset(&sg, bb));
                    for j in 0..m {
                        for i in 0..n {
                            let dot: f64 = (0..k).map(|d| (rhs[lo + i * k + d] as f32 as f64) * (lhs[go + j * k + d] as f32 as f64)).sum();
                            res[bi * m * n + j * n + i] = dot as f32 as f64;
                        }
                    }
                }
            }
            (t, _) => {
                let (mult, shift) = self.multiplier;
                let (za, zb, zo) = (a.zero_point() as i64, b.zero_point() as i64, o.zero_point());
                let (lo_q, hi_q) = type_range(t);
                for (bi, bb) in batches.iter().enumerate() {
                    let (lo, go) = (offset(&sl, bb), offset(&sg, bb));
                    for j in 0..m {
                        for i in 0..n {
                            let prod = |d: usize| (rhs[lo + i * k + d] as i64 - zb) * (lhs[go + j * k + d] as i64 - za);
                            let v = if t == TType::I16 {
                                mul_by_quantized_multiplier_i64((0..k).map(prod).sum(), mult, shift)
                            } else {
                                gemm_requantize((0..k).fold(0i32, |s, d| s.wrapping_add(prod(d) as i32)), mult, shift, false)
                            };
                            res[bi * m * n + j * n + i] = v.wrapping_add(zo).clamp(lo_q, hi_q) as f64;
                        }
                    }
                }
            }
        }
        put_f64s(o, &res)
    }
}

/// a shape with 1s in front up to rank r
fn extend(s: &[usize], r: usize) -> Vec<usize> {
    let mut v = vec![1; r.saturating_sub(s.len())];
    v.extend_from_slice(s);
    v
}

struct FullyConnected {
    act: Act,
    shuffled: bool,
    keep_num_dims: bool,
    /// hybrid: quantize each input row with its own zero point
    asymmetric: bool,
    multiplier: (i32, i32),
    act_q: (i32, i32),
    /// hybrid, asymmetric: the weights' row sums, worked out on the first run as 2.5 does
    row_sums: Option<Vec<i32>>,
}

impl Kernel for FullyConnected {
    fn prepare(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        let (i, f, bias) = (io.input(0)?, io.input(1)?, io.opt(2));
        let o = &io.outs[0];
        if io.outs.len() != if self.shuffled { 2 } else { 1 } {
            return err("FULLY_CONNECTED has one output (two with shuffled weights)");
        }
        let quantized = matches!(f.dtype, TType::U8 | TType::I8);
        let hybrid = quantized && i.dtype == TType::F32;
        let bias_int = bias.is_none_or(|b| matches!(b.dtype, TType::I32 | TType::I64));
        let bias_float = bias.is_none_or(|b| b.dtype == TType::F32);
        let types_ok = if !quantized {
            i.dtype == TType::F32 && o.dtype == TType::F32 && f.dtype == TType::F32 && bias_float
        } else if self.shuffled {
            i.dtype == TType::U8 && f.dtype == TType::U8 && o.dtype == TType::I16 && bias_int
        } else if hybrid {
            o.dtype == TType::F32 && bias_float
        } else {
            matches!(i.dtype, TType::U8 | TType::I8 | TType::I16) && matches!(o.dtype, TType::U8 | TType::I8 | TType::I16) && bias_int
        };
        if !types_ok {
            return err(format!("input {:?}, weights {:?}, output {:?} is not supported", i.dtype, f.dtype, o.dtype));
        }
        if f.sparsity.is_some() {
            return err("sparse weights are not supported yet");
        }
        if !hybrid && !matches!(self.act, Act::None | Act::Relu | Act::ReluN1To1 | Act::Relu6) {
            return err(format!("activation {:?} is not supported", self.act));
        }
        if f.shape.len() != 2 || f.shape[1] == 0 {
            return err("the weights must be 2-D");
        }
        let (units, depth) = (f.shape[0], f.shape[1]);
        let batches = i.elements() / depth;
        if bias.is_some_and(|b| b.elements() != units) {
            return err("the bias needs one value per output unit");
        }
        if matches!(i.dtype, TType::U8 | TType::I8 | TType::I16) {
            // GetQuantizedConvolutionMultipler: the bias scale near input scale * weight scale
            if let Some(b) = bias {
                let product = i.scale() as f64 * f.scale() as f64;
                if (product - b.scale() as f64).abs() / o.scale() as f64 > 0.02 {
                    return err("the bias scale does not match input scale * weight scale");
                }
            }
            let (m, shift) = quantize_multiplier((i.scale() * f.scale()) as f64 / o.scale() as f64);
            self.multiplier = (m, shift);
            self.act_q = act_range_q(self.act, o);
        }
        if i.dtype == TType::I16 && o.dtype == TType::I16 && (i.zero_point() != 0 || o.zero_point() != 0) {
            return err("int16 needs zero points 0");
        }
        // 2.5 sums the rows again after each prepare, and only then
        self.row_sums = None;
        let shape = if self.keep_num_dims {
            if i.shape.last() != Some(&depth) {
                return err("keep_num_dims needs the input's last dimension to be the weights' depth");
            }
            let mut s = i.shape.clone();
            *s.last_mut().unwrap() = units;
            s
        } else {
            vec![batches, units]
        };
        io.outs[0].resize(shape);
        Ok(())
    }

    fn eval(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        let (i, f, bias) = (io.input(0)?, io.input(1)?, io.opt(2));
        let (units, depth) = (f.shape[0], f.shape[1]);
        let batches = io.outs[0].elements() / units.max(1);
        if f.dtype == TType::F32 {
            let (lo, hi) = act_range_f(self.act);
            let (x, w) = (i.as_slice::<f32>(), f.as_slice::<f32>());
            let b = bias.map(|t| t.as_slice::<f32>());
            let out = io.outs[0].as_mut_slice::<f32>();
            for bb in 0..batches {
                let xr = &x[bb * depth..(bb + 1) * depth];
                for u in 0..units {
                    let dot = w[u * depth..(u + 1) * depth].iter().zip(xr).map(|(&a, &v)| a as f64 * v as f64).sum::<f64>() as f32;
                    out[bb * units + u] = clamp_f(dot + b.map_or(0.0, |b| b[u]), lo, hi);
                }
            }
            return Ok(());
        }
        if self.shuffled {
            return shuffled(i, f, bias, io.outs.as_mut_slice(), self.multiplier, batches, units, depth);
        }
        if i.dtype == TType::F32 {
            return self.hybrid(i, f, bias, &mut io.outs[0], batches, units, depth);
        }
        let o = &mut io.outs[0];
        let (m, shift) = self.multiplier;
        let (lo, hi) = self.act_q;
        let (xo, wo, oo) = (-i.zero_point() as i64, -f.zero_point() as i64, o.zero_point());
        let mut res = Vec::with_capacity(batches * units);
        if i.dtype == TType::I16 {
            // reference_integer_ops::FullyConnected for int16: int64 sums, int64 bias, no input offset
            let (x, w) = (ints(i)?, ints(f)?);
            let b = match bias {
                Some(t) if t.dtype == TType::I64 => Some(t.as_slice::<i64>()),
                Some(_) => return err("int16 FULLY_CONNECTED needs an int64 bias"),
                None => None,
            };
            for bb in 0..batches {
                for u in 0..units {
                    let mut acc: i64 = (0..depth).map(|d| ((w[u * depth + d] + wo) * x[bb * depth + d]) as i32 as i64).sum();
                    acc += b.map_or(0, |b| b[u]);
                    res.push(mul_by_quantized_multiplier_i64(acc, m, shift).clamp(lo, hi) as i64);
                }
            }
        } else {
            // the uint8 -> int16 kernel reads int8 tensors as their bytes, keeps the zero points as uint8 (-2 becomes
            // 254) and leaves out the output's
            let as_bytes = o.dtype == TType::I16;
            let vals = |t: &Tensor| -> Result<Vec<i64>> { if as_bytes { Ok(t.data().iter().map(|&b| b as i64).collect()) } else { ints(t) } };
            let (x, w) = (vals(i)?, vals(f)?);
            let (xo, wo, oo) = if as_bytes { (-(i.zero_point() as u8 as i64), -(f.zero_point() as u8 as i64), 0) } else { (xo, wo, oo) };
            let b = bias.map(|t| ints(t)).transpose()?;
            for bb in 0..batches {
                for u in 0..units {
                    let mut acc = 0i32;
                    for d in 0..depth {
                        acc = acc.wrapping_add(((w[u * depth + d] + wo) * (x[bb * depth + d] + xo)) as i32);
                    }
                    acc = acc.wrapping_add(b.as_ref().map_or(0, |b| b[u] as i32));
                    res.push(gemm_requantize(acc, m, shift, as_bytes || i.dtype == TType::U8).wrapping_add(oo).clamp(lo, hi) as i64);
                }
            }
        }
        put_ints(o, &res)
    }
}

impl FullyConnected {
    /// EvalHybridDense: each input row quantized to int8 on its own scale (portable_tensor_utils), integer dot
    /// products with the weights' bytes as int8, scaled back into the float output that starts from the bias
    #[allow(clippy::too_many_arguments)]
    fn hybrid(&mut self, i: &Tensor, f: &Tensor, bias: Option<&Tensor>, o: &mut Tensor, batches: usize, units: usize, depth: usize) -> Result<()> {
        let x = i.as_slice::<f32>();
        let w: Vec<i32> = f.data().iter().map(|&b| b as i8 as i32).collect();
        let out = o.as_mut_slice::<f32>();
        for bb in 0..batches {
            for u in 0..units {
                out[bb * units + u] = bias.map_or(0.0, |b| b.as_slice::<f32>()[u]);
            }
        }
        if !x.iter().all(|&v| v == 0.0) {
            if self.asymmetric && self.row_sums.is_none() {
                self.row_sums = Some(w.chunks(depth).map(|r| r.iter().sum()).collect());
            }
            for bb in 0..batches {
                let (q, scale, offset) = quantize_row(&x[bb * depth..(bb + 1) * depth], self.asymmetric);
                let scale = scale * f.scale();
                for u in 0..units {
                    let mut dot = w[u * depth..(u + 1) * depth].iter().zip(&q).fold(0i32, |s, (&a, &b)| s.wrapping_add(a * b));
                    if let Some(rs) = &self.row_sums {
                        dot = dot.wrapping_sub(rs[u].wrapping_mul(offset));
                    }
                    out[bb * units + u] += dot as f32 * scale;
                }
            }
        }
        apply_activation(out, self.act);
        Ok(())
    }
}

/// SymmetricQuantizeFloats / AsymmetricQuantizeFloats (portable_tensor_utils.cc): (values, scale, zero point)
fn quantize_row(v: &[f32], asymmetric: bool) -> (Vec<i32>, f32, i32) {
    // std::minmax_element: the first smallest, the last largest
    let min = v.iter().copied().fold(v[0], |m, x| if x < m { x } else { m });
    let max = v.iter().copied().fold(v[0], |m, x| if x < m { m } else { x });
    if !asymmetric {
        let (a, b) = (min.abs(), max.abs());
        let range = if a < b { b } else { a };
        if range == 0.0 {
            return (vec![0; v.len()], 1.0, 0);
        }
        let inv = 127.0 / range;
        let q = v.iter().map(|&x| (cvtt((x * inv).round() as f64, false) as i32).clamp(-127, 127)).collect();
        return (q, range / 127.0, 0);
    }
    let (rmin, rmax) = ((min as f64).min(0.0), (max as f64).max(0.0));
    if rmin == rmax {
        return (vec![0; v.len()], 1.0, 0);
    }
    let scale = (rmax - rmin) / 255.0;
    let from_min = -128.0 - rmin / scale;
    let from_max = 127.0 - rmax / scale;
    let zp = if 128.0 + (rmin / scale).abs() < 127.0 + (rmax / scale).abs() { from_min } else { from_max };
    let zp = if zp <= -128.0 {
        -128
    } else if zp >= 127.0 {
        127
    } else {
        cvtt(zp.round(), false) as i8 as i32
    };
    let sf = scale as f32;
    let inv = (1.0 / sf as f64) as f32;
    let q = v.iter().map(|&x| (cvtt((zp as f32 + x * inv).round() as f64, false) as i32).clamp(-128, 127)).collect();
    (q, sf, zp)
}

/// tensor_utils::ApplyActivationToVector
fn apply_activation(v: &mut [f32], act: Act) {
    // std::max(a, std::min(x, b)) on NaN: min gives NaN, max gives a
    let clamp = |x: f32, a: f32, b: f32| {
        let m = if b < x { b } else { x };
        if a < m { m } else { a }
    };
    match act {
        Act::None => {}
        Act::Relu => v.iter_mut().for_each(|x| *x = if 0.0 < *x { *x } else { 0.0 }),
        Act::ReluN1To1 => v.iter_mut().for_each(|x| *x = clamp(*x, -1.0, 1.0)),
        Act::Relu6 => v.iter_mut().for_each(|x| *x = clamp(*x, 0.0, 6.0)),
        Act::Tanh => v.iter_mut().for_each(|x| *x = eigen::tanh(*x)),
        Act::SignBit => v.iter_mut().for_each(|x| *x = x.is_sign_negative() as u8 as f32),
    }
}

/// ShuffledFullyConnected (no NEON): uint8 input with its sign bit flipped (zero point 128) into the workspace
/// output, weights in 4x16 blocks read as int8, int16 output; 1 or 4 batches only
#[allow(clippy::too_many_arguments)]
fn shuffled(i: &Tensor, f: &Tensor, bias: Option<&Tensor>, outs: &mut [Tensor], (m, shift): (i32, i32), batches: usize, units: usize, depth: usize) -> Result<()> {
    let bias = match bias {
        Some(b) if b.dtype == TType::I32 => b.as_slice::<i32>(),
        _ => return err("shuffled weights need an int32 bias"),
    };
    if !(batches == 1 || batches == 4) || depth % 16 != 0 || units % 4 != 0 {
        return err("shuffled weights take 1 or 4 batches, a depth that is a multiple of 16 and units in fours");
    }
    let x = i.data();
    let mut ws = vec![0u8; batches * depth];
    if batches == 1 {
        ws.iter_mut().zip(x).for_each(|(d, &v)| *d = v ^ 0x80);
    } else {
        // per 16 columns, the 4 batches one after another
        let mut at = 0;
        for c in (0..depth).step_by(16) {
            for bb in 0..4 {
                for j in 0..16 {
                    ws[at] = x[bb * depth + c + j] ^ 0x80;
                    at += 1;
                }
            }
        }
    }
    let w = f.data();
    let mut res = vec![0i64; batches * units];
    for c in (0..units).step_by(4) {
        for r in 0..4 {
            for bb in 0..batches {
                let mut acc = 0i32;
                for d in (0..depth).step_by(16) {
                    let block = (c / 4) * 4 * depth + (d / 16) * 64 + r * 16;
                    let input = if batches == 1 { d } else { (d / 16) * 64 + bb * 16 };
                    for j in 0..16 {
                        acc = acc.wrapping_add(w[block + j] as i8 as i32 * ws[input + j] as i8 as i32);
                    }
                }
                let v = mul_by_quantized_multiplier(acc.wrapping_add(bias[c + r]), m, shift);
                res[bb * units + c + r] = v.clamp(-32768, 32767) as i64;
            }
        }
    }
    put_ints(&mut outs[0], &res)?;
    let wsd = outs[1].data_mut();
    let n = wsd.len().min(ws.len());
    wsd[..n].copy_from_slice(&ws[..n]);
    Ok(())
}
