// From TensorFlow Lite 2.5 (Copyright The TensorFlow Authors, Apache License 2.0), translated and changed; see NOTICE.
//! Normalizations along the last axis: SOFTMAX and LOG_SOFTMAX (activations.cc), L2_NORMALIZATION (l2norm.cc) and
//! LOCAL_RESPONSE_NORMALIZATION (local_response_norm.cc), each the way the kernel 2.5's resolver picks computes it.

use super::eigen;
use super::fixedpoint as fp;
use super::util::*;
use super::{Io, Kernel};
use crate::interp::Env;
use crate::tensor::{TType, Tensor};
use crate::tflite::{Act, OpSpec};
use crate::Result;

#[derive(Clone, Copy)]
enum Op {
    Softmax { beta: f32 },
    LogSoftmax,
    L2Norm,
    Lrn { radius: i32, bias: f32, alpha: f32, beta: f32 },
}

pub fn create(s: &OpSpec) -> Result<Option<Box<dyn Kernel>>> {
    let op = match s.code {
        25 => Op::Softmax { beta: opt(s, 0, 0.0) },
        50 => Op::LogSoftmax,
        11 => {
            if opt_act(s, 0)? != Act::None {
                return err("L2_NORMALIZATION takes no fused activation");
            }
            Op::L2Norm
        }
        13 => Op::Lrn { radius: opt(s, 0, 0), bias: opt(s, 1, 0.0), alpha: opt(s, 2, 0.0), beta: opt(s, 3, 0.0) },
        _ => return Ok(None),
    };
    Ok(Some(Box::new(Norm { op, table: Vec::new(), int16: None })))
}

struct Norm {
    op: Op,
    /// 8-bit SOFTMAX / LOG_SOFTMAX: PopulateSoftmaxLookupTable's table, exp(-input_scale * beta * (255 - k)) at k,
    /// here for k from -128 (stored at k + 128): 2.5's LOG_SOFTMAX on int8 indexes below the table's start when a
    /// row spans more than 127 and reads whatever memory is there; this reads what the table would have held
    table: Vec<f32>,
    int16: Option<Int16Softmax>,
}

/// SoftmaxPrepare for int16: the exp and 1 / (1 + x) tables, and the multiplier that maps [-10, 0] onto the exp table
struct Int16Softmax {
    exp_lut: Vec<i16>,
    recip_lut: Vec<i16>,
    multiplier: i32,
    shift: i32,
}

impl Kernel for Norm {
    fn prepare(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        let i = io.input(0)?;
        let o = &mut io.outs[0];
        let (it, ot, rank) = (i.dtype, o.dtype, i.shape.len());
        let types_ok = match self.op {
            Op::Softmax { .. } if ot == TType::I16 => matches!(it, TType::I8 | TType::U8 | TType::I16),
            Op::L2Norm => it == ot && matches!(ot, TType::F32 | TType::U8 | TType::I8),
            Op::Lrn { .. } => it == ot && ot == TType::F32,
            _ => it == ot,
        };
        if !types_ok {
            return err(format!("input {it:?}, output {ot:?}"));
        }
        match self.op {
            Op::Softmax { beta } => {
                if rank == 0 {
                    return err("SOFTMAX needs at least one dimension");
                }
                if matches!(it, TType::U8 | TType::I8) {
                    self.table = exp_table(i.scale(), beta);
                }
                if it == TType::I16 {
                    if i.zero_point() != 0 || o.zero_point() != 0 {
                        return err("int16 SOFTMAX needs zero points 0");
                    }
                    let (multiplier, shift) = quantize_multiplier((i.scale() * beta) as f64 / (10.0 / 65535.0));
                    let (exp_lut, recip_lut) = (gen_lut(f64::exp, -10.0, 0.0), gen_lut(|x| 1.0 / (1.0 + x), 0.0, 1.0));
                    self.int16 = Some(Int16Softmax { exp_lut, recip_lut, multiplier, shift });
                }
            }
            Op::LogSoftmax => {
                if rank == 0 {
                    return err("LOG_SOFTMAX needs at least one dimension");
                }
                if matches!(it, TType::U8 | TType::I8) {
                    let zp = if it == TType::U8 { 255 } else { 127 };
                    if o.scale() != 16.0 / 256.0 || o.zero_point() != zp {
                        return err(format!("LOG_SOFTMAX needs an output scale of 16/256 and zero point {zp}"));
                    }
                    self.table = exp_table(i.scale(), 1.0);
                }
            }
            Op::L2Norm => {
                if rank == 0 || rank > 4 {
                    return err("L2_NORMALIZATION takes 1 to 4 dimensions");
                }
                if ot != TType::F32 && (o.scale() != 1.0 / 128.0 || o.zero_point() != if ot == TType::U8 { 128 } else { 0 }) {
                    return err("L2_NORMALIZATION needs an output scale of 1/128 and zero point 128 (uint8) or 0 (int8)");
                }
            }
            Op::Lrn { radius, .. } => {
                if rank != 4 {
                    return err("LOCAL_RESPONSE_NORMALIZATION takes 4 dimensions");
                }
                // 2.5 sizes a buffer with depth + 2 * radius
                if radius < 0 {
                    return err("the radius must not be negative");
                }
            }
        }
        same_shape(i, o);
        Ok(())
    }

    fn eval(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        let i = io.input(0)?;
        let o = &mut io.outs[0];
        let depth = i.shape[i.shape.len() - 1];
        if depth == 0 {
            return Ok(());
        }
        match (self.op, i.dtype) {
            (Op::Softmax { beta }, TType::F32) => softmax_f32(i.as_slice(), o.as_mut_slice(), depth, beta),
            (Op::Softmax { .. }, TType::U8 | TType::I8) => softmax_q(&self.table, i, o, depth)?,
            (Op::Softmax { .. }, TType::I16) => {
                if i.shape.len() > 4 {
                    return err("int16 SOFTMAX takes 1 to 4 dimensions");
                }
                softmax_i16(self.int16.as_ref().unwrap(), i.as_slice(), o.as_mut_slice(), depth)
            }
            (Op::LogSoftmax, TType::F32) => log_softmax_f32(i.as_slice(), o.as_mut_slice(), depth),
            (Op::LogSoftmax, TType::U8 | TType::I8) => log_softmax_q(&self.table, i, o, depth)?,
            (Op::L2Norm, TType::F32) => l2norm_f32(i.as_slice(), o.as_mut_slice(), depth),
            (Op::L2Norm, TType::U8 | TType::I8) => l2norm_q(i, o, depth)?,
            (Op::Lrn { radius, bias, alpha, beta }, TType::F32) => {
                lrn(i.as_slice(), o.as_mut_slice(), depth, radius as usize, bias, alpha, beta)
            }
            (_, t) => return err(format!("{t:?} is not supported")),
        }
        Ok(())
    }
}

fn exp_table(input_scale: f32, beta: f32) -> Vec<f32> {
    let scale = -input_scale * beta;
    (-128..=255).map(|k: i32| (scale * (255 - k) as f32).exp()).collect()
}

/// optimized_ops::Softmax (SoftmaxImpl, one thread): (x - max) * beta, Eigen's exp over the whole buffer, then each
/// row times 1 / its Eigen redux sum (each row summed from its own alignment)
fn softmax_f32(x: &[f32], o: &mut [f32], depth: usize, beta: f32) {
    for (row, out) in x.chunks(depth).zip(o.chunks_mut(depth)) {
        let max = row.iter().copied().fold(row[0], f32::max);
        out.iter_mut().zip(row).for_each(|(d, &v)| *d = (v - max) * beta);
    }
    eigen::map_array(o, eigen::exp, f32::exp);
    for (r, out) in o.chunks_mut(depth).enumerate() {
        let inv = 1.0 / eigen::sum(out, r * depth);
        out.iter_mut().for_each(|d| *d *= inv);
    }
}

/// optimized_ops::Softmax for 8-bit input: exp from the table at the row's max, summed in float, each probability
/// over sum * output scale rounded (uint8 output by adding 0.5 and truncating, without the zero point, as the x86
/// build does)
fn softmax_q(table: &[f32], i: &Tensor, o: &mut Tensor, depth: usize) -> Result<()> {
    let x = ints(i)?;
    let (scale, zp, u8_out) = (o.scale(), o.zero_point(), o.dtype == TType::U8);
    let (lo, hi) = type_range(o.dtype);
    let mut out = Vec::with_capacity(x.len());
    for row in x.chunks(depth) {
        let max = *row.iter().max().unwrap();
        let e = |v: i64| table[(255 - max + v + 128) as usize];
        let sum = row.iter().fold(0.0f32, |s, &v| s + e(v));
        let inv = 1.0 / (sum * scale);
        for &v in row {
            let p = e(v) * inv;
            let q = if u8_out { cvtt((p + 0.5) as f64, false) as i32 } else { (cvtt(p.round() as f64, false) as i32).wrapping_add(zp) };
            out.push(q.clamp(lo, hi) as i64);
        }
    }
    put_ints(o, &out)
}

/// reference_ops::SoftmaxInt16: exp of the scaled differences from a table, their sum's reciprocal from another
fn softmax_i16(p: &Int16Softmax, x: &[i16], o: &mut [i16], depth: usize) {
    for (row, out) in x.chunks(depth).zip(o.chunks_mut(depth)) {
        let max = *row.iter().max().unwrap() as i32;
        let mut sum = 0i32;
        for (d, &v) in out.iter_mut().zip(row) {
            let scaled = mul_by_quantized_multiplier(v as i32 - max, p.multiplier, p.shift);
            *d = lut_lookup(scaled.wrapping_add(32767).clamp(-32768, 32767) as i16, &p.exp_lut);
            sum = sum.wrapping_add(*d as i32);
        }
        let headroom_plus_one = (sum as u32).leading_zeros() as i32;
        let shifted = ((((sum as i64) << (headroom_plus_one - 1)) + (1 << 13)) >> 14) as i32;
        let sym = shifted.wrapping_sub((1 << 15) + (1 << 16)).clamp(-32768, 32767) as i16;
        let reciprocal = lut_lookup(sym, &p.recip_lut) as i64;
        let right_shift = 31 - headroom_plus_one;
        let round = 1i64 << (right_shift - 1);
        for d in out.iter_mut() {
            *d = (((*d as i64 * reciprocal + round) >> right_shift) as i32).clamp(0, 32767) as i16;
        }
    }
}

/// gen_lut (common.h): f at 512 points of [min, max] in Q0.15, each moved by half its error at the midpoint to the
/// next, then f(max) for the last slope
fn gen_lut(f: impl Fn(f64) -> f64, min: f64, max: f64) -> Vec<i16> {
    const NUM: usize = 513;
    let step = (max - min) / (NUM - 1) as f64;
    let half_step = step / 2.0;
    let q = |v: f64| v.clamp(-32768.0, 32767.0) as i16;
    let mut t: Vec<i16> = (0..NUM - 1)
        .map(|i| {
            let i = i as f64;
            let sample = (f(min + i * step) * 32768.0).round();
            let interp = ((f(min + (i + 1.0) * step) * 32768.0 + (f(min + i * step) * 32768.0).round()) / 2.0).round();
            let mid = (f(min + i * step + half_step) * 32768.0).round();
            q(sample - ((interp - mid) / 2.0).round())
        })
        .collect();
    t.push(q((f(max) * 32768.0).round()));
    t
}

/// generic_int16_table_lookup: the entry at the top 9 bits, interpolated toward the next by the low 7
fn lut_lookup(v: i16, lut: &[i16]) -> i16 {
    let index = (256 + (v >> 7) as i32) as usize;
    let slope = lut[index + 1].wrapping_sub(lut[index]) as i32;
    (lut[index] as i32 + ((slope * (v & 0x7f) as i32 + 64) >> 7)) as i16
}

/// optimized_ops::LogSoftmax: per row x - max - log(sum of exp(x - max)), the sum an Eigen redux over the expression
/// (Eigen's exp on its packets of 4 from the row's start, std::exp after them)
fn log_softmax_f32(x: &[f32], o: &mut [f32], depth: usize) {
    let mut e = vec![0.0f32; depth];
    for (row, out) in x.chunks(depth).zip(o.chunks_mut(depth)) {
        let max = row.iter().copied().fold(row[0], f32::max);
        e.iter_mut().zip(row).for_each(|(d, &v)| *d = v - max);
        eigen::map_array(&mut e, eigen::exp, f32::exp);
        let log_sum = eigen::sum(&e, 0).ln();
        out.iter_mut().zip(row).for_each(|(d, &v)| *d = v - max - log_sum);
    }
}

/// optimized_ops::LogSoftmax for 8-bit input: exp from the table, the log in float, rounded half to even. On int8 the
/// table is read from 127 - max instead of 255 - max, and the 128 it is off by is added back through the max.
fn log_softmax_q(table: &[f32], i: &Tensor, o: &mut Tensor, depth: usize) -> Result<()> {
    let x = ints(i)?;
    let (is, os, zp) = (i.scale(), o.scale(), o.zero_point());
    let (lo, hi) = type_range(o.dtype);
    let (top, zp_offset) = if i.dtype == TType::I8 { (127, 128) } else { (255, 0) };
    let scale = is / os;
    let mut out = Vec::with_capacity(x.len());
    for row in x.chunks(depth) {
        let max = *row.iter().max().unwrap();
        let sum = row.iter().fold(0.0f32, |s, &v| s + table[(top - max + v + 128) as usize]);
        let precomputed = (is * (max + zp_offset) as f32 + sum.ln()) / os;
        for &v in row {
            let log_prob = scale * v as f32 - precomputed;
            out.push((cvtt((log_prob.round_ties_even() + zp as f32) as f64, false) as i32).clamp(lo, hi) as i64);
        }
    }
    put_ints(o, &out)
}

/// optimized_ops::L2Normalization: x / max(sqrt(sum of squares in order), 1e-6)
fn l2norm_f32(x: &[f32], o: &mut [f32], depth: usize) {
    for (row, out) in x.chunks(depth).zip(o.chunks_mut(depth)) {
        let norm = row.iter().fold(0.0f32, |s, &v| s + v * v).sqrt();
        let norm = if norm < 1e-6 { 1e-6 } else { norm };
        out.iter_mut().zip(row).for_each(|(d, &v)| *d = v / norm);
    }
}

/// L2Normalization for uint8 (optimized_ops) and int8 (reference_integer_ops): the integer sum of squares to a
/// fixed-point 1 / sqrt, applied to each difference in Q7
fn l2norm_q(i: &Tensor, o: &mut Tensor, depth: usize) -> Result<()> {
    let x = ints(i)?;
    let zp = i.zero_point() as i64;
    let mut out = Vec::with_capacity(x.len());
    for row in x.chunks(depth) {
        let squares = row.iter().fold(0i32, |s, &v| s.wrapping_add(((v - zp) * (v - zp)) as i32));
        let (m, shift) = fp::inv_sqrt_multiplier_exp(squares, -1);
        for &v in row {
            let diff = (v - zp) as i32;
            out.push(if o.dtype == TType::U8 {
                (128 + mul_smaller_than_one_exp(128 * diff, m, shift)).clamp(0, 255)
            } else {
                mul_by_quantized_multiplier(diff, m, shift + 7).clamp(-128, 127)
            } as i64);
        }
    }
    put_ints(o, &out)
}

/// optimized_ops::LocalResponseNormalization: per pixel a running window sum of alpha * x^2 (each square added when
/// it enters and subtracted when it leaves, in 2.5's order), then x * (bias + window)^-beta over the whole buffer:
/// x * (1 / v) for beta 1, x * (1 / sqrt(v)) with Eigen's packet sqrt for beta 0.5, powf otherwise (the SSE2 build
/// has no packet pow)
fn lrn(x: &[f32], o: &mut [f32], depth: usize, radius: usize, bias: f32, alpha: f32, beta: f32) {
    let mut padded = vec![0.0f32; depth + 2 * radius];
    for (col, out) in x.chunks(depth).zip(o.chunks_mut(depth)) {
        padded[radius..radius + depth].iter_mut().zip(col).for_each(|(p, &v)| *p = v * v * alpha);
        let mut acc = padded[..2 * radius].iter().fold(0.0f32, |s, &p| s + p);
        for k in 0..depth {
            acc += padded[k + 2 * radius];
            out[k] = bias + acc;
            acc -= padded[k];
        }
    }
    if beta == 1.0 {
        o.iter_mut().zip(x).for_each(|(d, &v)| *d = v * (1.0 / *d));
    } else if beta == 0.5 {
        eigen::map_array(o, eigen::psqrt, f32::sqrt);
        o.iter_mut().zip(x).for_each(|(d, &v)| *d = v * (1.0 / *d));
    } else {
        o.iter_mut().zip(x).for_each(|(d, &v)| *d = v * d.powf(-beta));
    }
}
