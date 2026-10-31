// From TensorFlow Lite 2.5 (Copyright The TensorFlow Authors, Apache License 2.0), translated and changed; see NOTICE.
//! Elementwise ops on two broadcast inputs (add.cc, sub.cc, mul.cc, div.cc, floor_div.cc, floor_mod.cc, pow.cc,
//! squared_difference.cc, maximum_minimum.cc, comparisons.cc, logical.cc, activations.cc PRELU).

use super::fixedpoint::{self as fp, I16};
use super::util::*;
use super::{Io, Kernel};
use crate::interp::Env;
use crate::tensor::{broadcast_shape, Elem, TType, Tensor};
use crate::tflite::{Act, OpSpec};
use crate::Result;

#[derive(Clone, Copy, PartialEq, Debug)]
enum Op {
    Add,
    Sub,
    Mul,
    Div,
    FloorDiv,
    FloorMod,
    Pow,
    SquaredDifference,
    Maximum,
    Minimum,
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    And,
    Or,
    Prelu,
}

pub fn create(s: &OpSpec) -> Result<Option<Box<dyn Kernel>>> {
    let (op, act, pot) = match s.code {
        0 => (Op::Add, opt_act(s, 0)?, opt(s, 1, 1u8) != 0),
        41 => (Op::Sub, opt_act(s, 0)?, opt(s, 1, 1u8) != 0),
        18 => (Op::Mul, opt_act(s, 0)?, false),
        42 => (Op::Div, opt_act(s, 0)?, false),
        90 => (Op::FloorDiv, Act::None, false),
        95 => (Op::FloorMod, Act::None, false),
        78 => (Op::Pow, Act::None, false),
        99 => (Op::SquaredDifference, Act::None, false),
        55 => (Op::Maximum, Act::None, false),
        57 => (Op::Minimum, Act::None, false),
        71 => (Op::Equal, Act::None, false),
        72 => (Op::NotEqual, Act::None, false),
        58 => (Op::Less, Act::None, false),
        63 => (Op::LessEqual, Act::None, false),
        61 => (Op::Greater, Act::None, false),
        62 => (Op::GreaterEqual, Act::None, false),
        86 => (Op::And, Act::None, false),
        84 => (Op::Or, Act::None, false),
        54 => (Op::Prelu, Act::None, false),
        _ => return Ok(None),
    };
    Ok(Some(Box::new(Binary { op, act, pot_scale_int16: pot })))
}

struct Binary {
    op: Op,
    act: Act,
    /// AddOptions / SubOptions pot_scale_int16: int16 with power-of-two scales takes the gemmlowp path
    pot_scale_int16: bool,
}

impl Kernel for Binary {
    fn prepare(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        let (a, b) = (io.input(0)?, io.input(1)?);
        let comparison = matches!(self.op, Op::Equal | Op::NotEqual | Op::Less | Op::LessEqual | Op::Greater | Op::GreaterEqual);
        if a.dtype != b.dtype {
            return err(format!("input types differ: {:?} and {:?}", a.dtype, b.dtype));
        }
        if !comparison && io.outs[0].dtype != a.dtype && !(self.op == Op::Mul && a.dtype == TType::I16) {
            return err(format!("input {:?}, output {:?}", a.dtype, io.outs[0].dtype));
        }
        let shape = broadcast_shape(&a.shape, &b.shape)?;
        io.outs[0].resize(shape);
        Ok(())
    }

    fn eval(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        let (a, b) = (io.input(0)?, io.input(1)?);
        eval(self, a, b, &mut io.outs[0])
    }
}

/// o[i] = f(a[ia], b[ib]) over the broadcast
fn bmap<A: Elem, B: Elem, O: Elem>(a: &Tensor, b: &Tensor, o: &mut Tensor, f: impl Fn(A, B) -> O) -> Result<()> {
    let so = o.shape.clone();
    let (xa, xb) = (a.as_slice::<A>(), b.as_slice::<B>());
    let v = o.as_mut_slice::<O>();
    broadcast(&so, &a.shape, &b.shape, |i, ia, ib| v[i] = f(xa[ia], xb[ib]))
}

/// the elements of a quantized tensor (uint8, int8, int16) as int32
fn qvals(t: &Tensor) -> Result<Vec<i32>> {
    Ok(match t.dtype {
        TType::U8 => t.as_slice::<u8>().iter().map(|&x| x as i32).collect(),
        TType::I8 => t.as_slice::<i8>().iter().map(|&x| x as i32).collect(),
        TType::I16 => t.as_slice::<i16>().iter().map(|&x| x as i32).collect(),
        d => return err(format!("{d:?} is not a quantized type here")),
    })
}

/// store int32 values into a quantized tensor (already clamped)
fn put_q(o: &mut Tensor, v: &[i32]) -> Result<()> {
    match o.dtype {
        TType::U8 => o.as_mut_slice::<u8>().iter_mut().zip(v).for_each(|(d, &x)| *d = x as u8),
        TType::I8 => o.as_mut_slice::<i8>().iter_mut().zip(v).for_each(|(d, &x)| *d = x as i8),
        TType::I16 => o.as_mut_slice::<i16>().iter_mut().zip(v).for_each(|(d, &x)| *d = x as i16),
        d => return err(format!("{d:?} is not a quantized type here")),
    }
    Ok(())
}

/// out[i] = f(a[ia], b[ib]) on quantized values as int32
fn qmap(a: &Tensor, b: &Tensor, o: &mut Tensor, f: impl Fn(i32, i32) -> i32) -> Result<()> {
    let (xa, xb) = (qvals(a)?, qvals(b)?);
    let mut v = vec![0; o.elements()];
    let so = o.shape.clone();
    broadcast(&so, &a.shape, &b.shape, |i, ia, ib| v[i] = f(xa[ia], xb[ib]))?;
    put_q(o, &v)
}

fn int_act_range(act: Act) -> (i64, i64) {
    let (lo, hi) = act_range_f(act);
    (if act == Act::None { i64::MIN } else { lo as i64 }, if hi == f32::MAX { i64::MAX } else { hi as i64 })
}

/// the rank-4 limit of the *4DSlow broadcasts TFLite 2.5 uses for these
fn check_rank4(a: &Tensor, b: &Tensor) -> Result<()> {
    if a.shape != b.shape && a.shape.len().max(b.shape.len()) > 4 {
        return err("broadcasting above 4 dimensions is not supported");
    }
    Ok(())
}

fn eval(k: &Binary, a: &Tensor, b: &Tensor, o: &mut Tensor) -> Result<()> {
    let (t, op) = (a.dtype, k.op);
    let so = o.shape.clone();
    match (op, t) {
        // *** float ***
        (Op::Add | Op::Sub | Op::Mul | Op::Div, TType::F32) => {
            let (lo, hi) = act_range_f(k.act);
            bmap(a, b, o, |x: f32, y: f32| {
                let r = match op {
                    Op::Add => x + y,
                    Op::Sub => x - y,
                    Op::Mul => x * y,
                    _ => x / y,
                };
                clamp_f(r, lo, hi)
            })?;
        }
        (Op::FloorDiv, TType::F32) => bmap(a, b, o, |x: f32, y: f32| (x / y).floor())?,
        (Op::FloorMod, TType::F32) => bmap(a, b, o, |x: f32, y: f32| {
            // floor_mod.h: fmod, then moved to the divisor's sign
            let m = x % y;
            if m != 0.0 && ((y < 0.0) != (m < 0.0)) { m + y } else { m }
        })?,
        (Op::Pow, TType::F32) => bmap(a, b, o, f32::powf)?,
        (Op::SquaredDifference, TType::F32) => bmap(a, b, o, |x: f32, y: f32| (x - y) * (x - y))?,
        (Op::Prelu, TType::F32) => bmap(a, b, o, |x: f32, al: f32| if x >= 0.0 { x } else { x * al })?,
        // *** integers ***
        (Op::Add | Op::Sub | Op::Mul, TType::I32 | TType::I64) => {
            let (lo, hi) = int_act_range(k.act);
            let (xa, xb) = (ints(a)?, ints(b)?);
            let mut v = vec![0; o.elements()];
            broadcast(&so, &a.shape, &b.shape, |i, ia, ib| {
                let (x, y) = (xa[ia], xb[ib]);
                let r = match op {
                    Op::Add => x.wrapping_add(y),
                    Op::Sub => x.wrapping_sub(y),
                    _ => x.wrapping_mul(y),
                };
                // int32 arithmetic wraps at 32 bits before the activation clamp
                let r = if t == TType::I32 { r as i32 as i64 } else { r };
                v[i] = r.clamp(lo, hi);
            })?;
            put_ints(o, &v)?;
        }
        (Op::Div, TType::I32) => {
            let (lo, hi) = int_act_range(k.act);
            if b.as_slice::<i32>().contains(&0) {
                return err("division by 0");
            }
            bmap(a, b, o, |x: i32, y: i32| (x.wrapping_div(y) as i64).clamp(lo, hi) as i32)?;
        }
        (Op::FloorDiv, TType::I32) => {
            if b.as_slice::<i32>().contains(&0) {
                return err("division by 0");
            }
            // reference_ops::FloorDiv: std::floor of the double quotient
            bmap(a, b, o, |x: i32, y: i32| (x as f64 / y as f64).floor() as i32)?;
        }
        (Op::FloorMod, TType::I32 | TType::I64) => {
            let (xa, xb) = (ints(a)?, ints(b)?);
            if xb.contains(&0) {
                return err("division by 0");
            }
            let mut v = vec![0; o.elements()];
            broadcast(&so, &a.shape, &b.shape, |i, ia, ib| {
                let (x, y) = (xa[ia], xb[ib]);
                let m = if t == TType::I32 { (x as i32).wrapping_rem(y as i32) as i64 } else { x.wrapping_rem(y) };
                v[i] = if m != 0 && ((y < 0) != (m < 0)) { m + y } else { m };
            })?;
            put_ints(o, &v)?;
        }
        (Op::Pow, TType::I32) => {
            if b.as_slice::<i32>().iter().any(|&e| e < 0) {
                return err("POW does not support negative value for int32.");
            }
            // std::pow on ints is the double pow; static_cast back as x86 does it
            bmap(a, b, o, |x: i32, y: i32| cvtt((x as f64).powf(y as f64), false) as i32)?;
        }
        (Op::SquaredDifference, TType::I32) => bmap(a, b, o, |x: i32, y: i32| x.wrapping_sub(y).wrapping_mul(x.wrapping_sub(y)))?,
        (Op::SquaredDifference, TType::I8) => squared_difference_q(a, b, o)?,
        (Op::Maximum | Op::Minimum, _) => {
            let max = op == Op::Maximum;
            match t {
                TType::F32 => bmap(a, b, o, |x: f32, y: f32| if max { if x > y { x } else { y } } else if x < y { x } else { y })?,
                TType::I64 => bmap(a, b, o, |x: i64, y: i64| if max { x.max(y) } else { x.min(y) })?,
                TType::I32 | TType::U8 | TType::I8 | TType::I16 => {
                    if t == TType::I32 {
                        bmap(a, b, o, |x: i32, y: i32| if max { x.max(y) } else { x.min(y) })?
                    } else {
                        qmap(a, b, o, |x, y| if max { x.max(y) } else { x.min(y) })?
                    }
                }
                d => return err(format!("{d:?} is not supported")),
            }
        }
        // *** comparisons ***
        (Op::Equal | Op::NotEqual | Op::Less | Op::LessEqual | Op::Greater | Op::GreaterEqual, _) => compare(op, a, b, o)?,
        (Op::And | Op::Or, TType::Bool) => {
            check_rank4(a, b)?;
            bmap(a, b, o, |x: u8, y: u8| if op == Op::And { (x != 0 && y != 0) as u8 } else { (x != 0 || y != 0) as u8 })?
        }
        // *** quantized ***
        (Op::Add | Op::Sub, TType::U8 | TType::I8) => add_sub_q(k, a, b, o, 20)?,
        (Op::Add | Op::Sub, TType::I16) => {
            let pot_scales = [a.scale(), b.scale(), o.scale()].map(|s| match checked_log2(s) {
                (l, true) => Some(l),
                _ => None,
            });
            if a.zero_point() != 0 || b.zero_point() != 0 || o.zero_point() != 0 {
                return err("int16 needs zero points 0");
            }
            match (k.pot_scale_int16, pot_scales) {
                (true, [Some(la), Some(lb), Some(lo)]) => add_sub_int16_pot(k, a, b, o, la - lo, lb - lo)?,
                _ => {
                    check_rank4(a, b)?;
                    add_sub_q(k, a, b, o, 15)?
                }
            }
        }
        (Op::Mul, TType::U8 | TType::I8 | TType::I16) if o.dtype == t => {
            if t == TType::I16 && (a.zero_point() != 0 || b.zero_point() != 0 || o.zero_point() != 0) {
                return err("int16 needs zero points 0");
            }
            if t == TType::I16 {
                check_rank4(a, b)?;
            }
            // mul.cc Prepare: the product of the float scales, in float, then widened
            let (m, shift) = quantize_multiplier((a.scale() * b.scale() / o.scale()) as f64);
            let (off1, off2, offo) = (-a.zero_point(), -b.zero_point(), o.zero_point());
            let (lo, hi) = act_range_q(k.act, o);
            if t == TType::I16 {
                qmap(a, b, o, |x, y| (offo + mul_by_quantized_multiplier((off1 + x).wrapping_mul(off2 + y), m, shift)).clamp(lo, hi))?;
            } else {
                let (xa, xb) = (byte_table(t, |x| off1 + x), byte_table(t, |y| off2 + y));
                let (da, db, od) = (a.data(), b.data(), o.data_mut());
                broadcast(&so, &a.shape, &b.shape, |i, ia, ib| {
                    let (x, y) = (xa[da[ia] as usize], xb[db[ib] as usize]);
                    od[i] = (offo + mul_by_quantized_multiplier(x.wrapping_mul(y), m, shift)).clamp(lo, hi) as u8;
                })?;
            }
        }
        (Op::Mul, TType::I16) => {
            // int16 x int16 -> 8 bits (reference_integer_ops::Mul): F0 product, divided by 2^8
            if a.shape != b.shape {
                return err("int16 x int16 to 8 bits does not broadcast");
            }
            let (lo, hi) = act_range_q(k.act, o);
            let zp = o.zero_point();
            let (xa, xb) = (a.as_slice::<i16>(), b.as_slice::<i16>());
            let v: Vec<i32> = xa
                .iter()
                .zip(xb)
                .map(|(&x, &y)| {
                    let r = I16.rdbp(I16.mul(x as i64, y as i64), 8);
                    zp + (r.clamp((lo - zp) as i64, (hi - zp) as i64) as i32)
                })
                .collect();
            put_q(o, &v)?;
        }
        (Op::Div, TType::U8) => {
            // reference_ops::Div (div.h): the divisor's reciprocal in fixed point, then the output's multiplier
            // (div.cc Prepare works it out in float, then widens it)
            let (m, shift) = quantize_multiplier((a.scale() / (b.scale() * o.scale())) as f64);
            let (off1, off2, offo) = (-a.zero_point(), -b.zero_point(), o.zero_point());
            let (lo, hi) = act_range_q(k.act, o);
            let (da, db, od) = (a.data(), b.data(), o.data_mut());
            broadcast(&so, &a.shape, &b.shape, |i, ia, ib| {
                let (x, y) = (off1 + da[ia] as i32, off2 + db[ib] as i32);
                let (inv, recip_shift) = if y > 0 {
                    fp::reciprocal(y, 31)
                } else {
                    let (r, s) = fp::reciprocal(y.wrapping_neg(), 31);
                    (r.wrapping_neg(), s)
                };
                let headroom = fp::clsb(x);
                let unscaled = srdhm(x.wrapping_mul(1i32.wrapping_shl(headroom as u32)), inv);
                let total_shift = shift - recip_shift - headroom;
                let v = offo.wrapping_add(fp::rounding_divide_by_pot(srdhm(unscaled, m), -total_shift));
                od[i] = v.clamp(lo, hi) as u8;
            })?;
        }
        (Op::Prelu, TType::U8 | TType::I8) => {
            // PreluPrepare: both multipliers in float, then widened
            let (m1, s1) = quantize_multiplier((a.scale() / o.scale()) as f64);
            let (m2, s2) = quantize_multiplier((a.scale() * b.scale() / o.scale()) as f64);
            let (ioff, aoff, ooff) = (-a.zero_point(), -b.zero_point(), o.zero_point());
            let (tlo, thi) = type_range(o.dtype);
            qmap(a, b, o, |x, al| {
                let input = ioff + x;
                let v = if input >= 0 { mul_by_quantized_multiplier(input, m1, s1) } else { mul_by_quantized_multiplier(input * (aoff + al), m2, s2) };
                (v + ooff).clamp(tlo, thi)
            })?;
        }
        (op, t) => return err(format!("{op:?} on {t:?} is not supported")),
    }
    Ok(())
}

/// add.cc / sub.cc for 8 bits (left shift 20) and int16 with general scales (15): both inputs rescaled to twice the
/// larger scale, then the output's multiplier
fn add_sub_q(k: &Binary, a: &Tensor, b: &Tensor, o: &mut Tensor, left_shift: i32) -> Result<()> {
    let (sca, scb, sco) = (a.scale(), b.scale(), o.scale());
    let twice_max = (2.0 * if sca < scb { scb } else { sca }) as f64;
    let (m1, s1) = quantize_multiplier_smaller_than_one(sca as f64 / twice_max)?;
    let (mut m2, s2) = quantize_multiplier_smaller_than_one(scb as f64 / twice_max)?;
    let (mo, s_o) = quantize_multiplier_smaller_than_one(twice_max / (((1 << left_shift) as f32 * sco) as f64))?;
    if k.op == Op::Sub {
        m2 = -m2;
    }
    let (off1, off2, offo) = (-a.zero_point(), -b.zero_point(), o.zero_point());
    let (lo, hi) = act_range_q(k.act, o);
    let scale = |x: i32, off: i32, m: i32, s: i32| mul_smaller_than_one_exp((off + x).wrapping_mul(1 << left_shift), m, s);
    if a.dtype.size() == 1 {
        // each input's rescaled value depends on its own byte alone
        let xa = byte_table(a.dtype, |x| scale(x, off1, m1, s1));
        let xb = byte_table(b.dtype, |y| scale(y, off2, m2, s2));
        let so = o.shape.clone();
        let (da, db, od) = (a.data(), b.data(), o.data_mut());
        broadcast(&so, &a.shape, &b.shape, |i, ia, ib| {
            let sum = xa[da[ia] as usize] + xb[db[ib] as usize];
            od[i] = (mul_smaller_than_one_exp(sum, mo, s_o) + offo).clamp(lo, hi) as u8;
        })
    } else {
        qmap(a, b, o, |x, y| (mul_smaller_than_one_exp(scale(x, off1, m1, s1) + scale(y, off2, m2, s2), mo, s_o) + offo).clamp(lo, hi))
    }
}

/// int16 ADD / SUB with power-of-two scales (reference_ops::Add / Sub16): one input divided down to the other's scale
/// in gemmlowp, then a saturating add / subtract
fn add_sub_int16_pot(k: &Binary, a: &Tensor, b: &Tensor, o: &mut Tensor, shift1: i32, shift2: i32) -> Result<()> {
    if !(shift1 == 0 || shift2 == 0) || shift1 > 0 || shift2 > 0 {
        return err("int16 power-of-two scales: one input must have the output's scale, the other a smaller one");
    }
    if k.op == Op::Add && a.shape != b.shape {
        return err("int16 ADD with power-of-two scales does not broadcast");
    }
    let (lo, hi) = act_range_q(k.act, o);
    let sat = |v: i64| v.clamp(-32768, 32767);
    qmap(a, b, o, |x, y| {
        let (x, y) = (x as i64, y as i64);
        let r = match (k.op, shift1 == 0) {
            (Op::Add, true) => sat(x + I16.rdbp(y, -shift2)),
            (Op::Add, false) => sat(I16.rdbp(x, -shift1) + y),
            (_, true) => sat(x - I16.rdbp(y, -shift2)),
            (_, false) => sat(I16.rdbp(x, -shift1) - y),
        };
        (r as i32).clamp(lo, hi)
    })
}

/// squared_difference.cc for int8: both inputs rescaled (left shift 7), the difference squared, then rescaled
fn squared_difference_q(a: &Tensor, b: &Tensor, o: &mut Tensor) -> Result<()> {
    let (sca, scb) = (a.scale(), b.scale());
    let twice_max = (2.0 * if sca < scb { scb } else { sca }) as f64;
    let (m1, s1) = quantize_multiplier_smaller_than_one(sca as f64 / twice_max)?;
    let (m2, s2) = quantize_multiplier_smaller_than_one(scb as f64 / twice_max)?;
    let (mo, so) = quantize_multiplier_smaller_than_one((twice_max * twice_max) / ((1 << 14) as f64 * o.scale() as f64))?;
    let (off1, off2, offo) = (-a.zero_point(), -b.zero_point(), o.zero_point());
    qmap(a, b, o, |x, y| {
        let sx = mul_smaller_than_one_exp((off1 + x) * (1 << 7), m1, s1);
        let sy = mul_smaller_than_one_exp((off2 + y) * (1 << 7), m2, s2);
        let d = sx.wrapping_sub(sy);
        (mul_smaller_than_one_exp(d.wrapping_mul(d), mo, so) + offo).clamp(-128, 127)
    })
}

/// comparisons.cc: numbers as they are, 8-bit values rescaled to a common scale first, strings by their bytes
fn compare(op: Op, a: &Tensor, b: &Tensor, o: &mut Tensor) -> Result<()> {
    if o.dtype != TType::Bool {
        return err("the output must be bool");
    }
    check_rank4(a, b)?;
    fn cmp<T: PartialOrd>(op: Op, x: T, y: T) -> u8 {
        (match op {
            Op::Equal => x == y,
            Op::NotEqual => x != y,
            Op::Less => x < y,
            Op::LessEqual => x <= y,
            Op::Greater => x > y,
            _ => x >= y,
        }) as u8
    }
    match a.dtype {
        TType::F32 => bmap(a, b, o, |x: f32, y: f32| cmp(op, x, y)),
        TType::I32 => bmap(a, b, o, |x: i32, y: i32| cmp(op, x, y)),
        TType::I64 => bmap(a, b, o, |x: i64, y: i64| cmp(op, x, y)),
        TType::Bool if matches!(op, Op::Equal | Op::NotEqual) => bmap(a, b, o, |x: u8, y: u8| cmp(op, x != 0, y != 0)),
        TType::U8 | TType::I8 => {
            // ComparisonWithScaling: offset, shift left by 8, QuantizeMultiplierSmallerThanOneExp(scale)
            let (m1, s1) = quantize_multiplier_smaller_than_one(a.scale() as f64)?;
            let (m2, s2) = quantize_multiplier_smaller_than_one(b.scale() as f64)?;
            let (off1, off2) = (-a.zero_point(), -b.zero_point());
            let xa = byte_table(a.dtype, |x| mul_smaller_than_one_exp((off1 + x) * (1 << 8), m1, s1));
            let xb = byte_table(b.dtype, |y| mul_smaller_than_one_exp((off2 + y) * (1 << 8), m2, s2));
            let so = o.shape.clone();
            let (da, db, od) = (a.data(), b.data(), o.data_mut());
            broadcast(&so, &a.shape, &b.shape, |i, ia, ib| od[i] = cmp(op, xa[da[ia] as usize], xb[db[ib] as usize]))
        }
        TType::Str if matches!(op, Op::Equal | Op::NotEqual) => {
            let (sa, sb) = (a.strings()?, b.strings()?);
            let so = o.shape.clone();
            let od = o.data_mut();
            broadcast(&so, &a.shape, &b.shape, |i, ia, ib| od[i] = cmp(op, sa[ia], sb[ib]))
        }
        d => err(format!("{op:?} on {d:?} is not supported")),
    }
}
