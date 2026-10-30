//! Elementwise ops on two broadcast inputs.

use super::fixedpoint as fp;
use super::util::*;
use super::{Io, Kernel};
use crate::interp::Env;
use crate::tensor::{broadcast_shape, TType};
use crate::tflite::{Act, OpSpec};
use crate::Result;

pub fn create(s: &OpSpec) -> Result<Option<Box<dyn Kernel>>> {
    Ok(Some(match s.code {
        0 => Box::new(Arith { op: ArithOp::Add, act: opt_act(s, 0)? }),
        41 => Box::new(Arith { op: ArithOp::Sub, act: opt_act(s, 0)? }),
        18 => Box::new(Arith { op: ArithOp::Mul, act: opt_act(s, 0)? }),
        42 => Box::new(Arith { op: ArithOp::Div, act: opt_act(s, 0)? }),
        90 => Box::new(FloorDiv),
        _ => return Ok(None),
    }))
}

/// output shape: the inputs' broadcast shape
fn prepare_broadcast(io: &mut Io) -> Result<()> {
    let shape = broadcast_shape(&io.input(0)?.shape, &io.input(1)?.shape)?;
    io.outs[0].resize(shape);
    Ok(())
}

#[derive(Clone, Copy, PartialEq)]
enum ArithOp {
    Add,
    Sub,
    Mul,
    Div,
}

struct Arith {
    op: ArithOp,
    act: Act,
}

impl Kernel for Arith {
    fn prepare(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        prepare_broadcast(io)
    }

    fn eval(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        let (a, b) = (io.input(0)?, io.input(1)?);
        let o = &mut io.outs[0];
        let (ta, tb, to) = (a.dtype, b.dtype, o.dtype);
        if ta != tb || ta != to {
            return err(format!("mixed types {ta:?} {tb:?} -> {to:?}"));
        }
        let (so, sa, sb) = (o.shape.clone(), &a.shape[..], &b.shape[..]);
        match (to, self.op) {
            (TType::U8 | TType::I8, ArithOp::Add | ArithOp::Sub) => {
                // add.cc / sub.cc Prepare: both inputs rescaled to twice the larger scale, shifted left by 20
                let (sca, scb, sco) = (a.scale(), b.scale(), o.scale());
                let left_shift = 20;
                let twice_max = (2.0 * if sca < scb { scb } else { sca }) as f64;
                let (m1, s1) = quantize_multiplier_smaller_than_one(sca as f64 / twice_max)?;
                let (mut m2, s2) = quantize_multiplier_smaller_than_one(scb as f64 / twice_max)?;
                let (mo, s_o) = quantize_multiplier_smaller_than_one(twice_max / (((1 << left_shift) as f32 * sco) as f64))?;
                if self.op == ArithOp::Sub {
                    m2 = -m2;
                }
                let (off1, off2, offo) = (-a.zero_point(), -b.zero_point(), o.zero_point());
                let (lo, hi) = act_range_q(self.act, o);
                // each input's rescaled value depends on its own byte alone
                let xa = byte_table(ta, |x| mul_smaller_than_one_exp((off1 + x) * (1 << left_shift), m1, s1));
                let xb = byte_table(tb, |y| mul_smaller_than_one_exp((off2 + y) * (1 << left_shift), m2, s2));
                let (da, db, od) = (a.data(), b.data(), o.data_mut());
                broadcast(&so, sa, sb, |i, ia, ib| {
                    let sum = xa[da[ia] as usize] + xb[db[ib] as usize];
                    od[i] = (mul_smaller_than_one_exp(sum, mo, s_o) + offo).clamp(lo, hi) as u8;
                })?;
            }
            (TType::U8 | TType::I8, ArithOp::Mul) => {
                // mul.cc Prepare: the product of the float scales, in float, then widened
                let (m, shift) = quantize_multiplier((a.scale() * b.scale() / o.scale()) as f64);
                let (off1, off2, offo) = (-a.zero_point(), -b.zero_point(), o.zero_point());
                let (lo, hi) = act_range_q(self.act, o);
                let (xa, xb) = (byte_table(ta, |x| off1 + x), byte_table(tb, |y| off2 + y));
                let (da, db, od) = (a.data(), b.data(), o.data_mut());
                broadcast(&so, sa, sb, |i, ia, ib| {
                    let (x, y) = (xa[da[ia] as usize], xb[db[ib] as usize]);
                    od[i] = (offo + mul_by_quantized_multiplier(x.wrapping_mul(y), m, shift)).clamp(lo, hi) as u8;
                })?;
            }
            (TType::F32, op) => {
                let (lo, hi) = act_range_f(self.act);
                let (xa, xb) = (a.as_slice::<f32>(), b.as_slice::<f32>());
                let v = o.as_mut_slice::<f32>();
                broadcast(&so, sa, sb, |i, ia, ib| {
                    let (x, y) = (xa[ia], xb[ib]);
                    let r = match op {
                        ArithOp::Add => x + y,
                        ArithOp::Sub => x - y,
                        ArithOp::Mul => x * y,
                        ArithOp::Div => x / y,
                    };
                    v[i] = clamp_f(r, lo, hi);
                })?;
            }
            (TType::U8, ArithOp::Div) => {
                // reference_ops::Div (div.h): the divisor's reciprocal in fixed point, then the output's multiplier
                // (div.cc Prepare works it out in float, then widens it)
                let (m, shift) = quantize_multiplier((a.scale() / (b.scale() * o.scale())) as f64);
                let (off1, off2, offo) = (-a.zero_point(), -b.zero_point(), o.zero_point());
                let (lo, hi) = act_range_q(self.act, o);
                let (da, db, od) = (a.data(), b.data(), o.data_mut());
                broadcast(&so, sa, sb, |i, ia, ib| {
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
            (TType::I32, ArithOp::Div) => {
                let (lo, hi) = act_range_f(self.act);
                let (lo, hi) = (if self.act == Act::None { i32::MIN } else { lo as i32 }, if hi == f32::MAX { i32::MAX } else { hi as i32 });
                let (xa, xb) = (a.as_slice::<i32>(), b.as_slice::<i32>());
                if xb.contains(&0) {
                    return err("division by 0");
                }
                let v = o.as_mut_slice::<i32>();
                broadcast(&so, sa, sb, |i, ia, ib| v[i] = xa[ia].wrapping_div(xb[ib]).clamp(lo, hi))?;
            }
            (TType::I32 | TType::I64, op) if op != ArithOp::Div => {
                let (lo, hi) = act_range_f(self.act);
                let (lo, hi) = (if self.act == Act::None { i64::MIN } else { lo as i64 }, if hi == f32::MAX { i64::MAX } else { hi as i64 });
                let (xa, xb) = (ints(a)?, ints(b)?);
                let mut v = vec![0; o.elements()];
                broadcast(&so, sa, sb, |i, ia, ib| {
                    let (x, y) = (xa[ia], xb[ib]);
                    let r = match op {
                        ArithOp::Add => x.wrapping_add(y),
                        ArithOp::Sub => x.wrapping_sub(y),
                        _ => x.wrapping_mul(y),
                    };
                    // int32 arithmetic wraps at 32 bits before the activation clamp
                    let r = if to == TType::I32 { r as i32 as i64 } else { r };
                    v[i] = r.clamp(lo, hi);
                })?;
                put_ints(o, &v)?;
            }
            (t, _) => return err(format!("{t:?} is not supported")),
        }
        Ok(())
    }
}

struct FloorDiv;

impl Kernel for FloorDiv {
    fn prepare(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        prepare_broadcast(io)
    }

    fn eval(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        let (a, b) = (io.input(0)?, io.input(1)?);
        let o = &mut io.outs[0];
        let (so, sa, sb) = (o.shape.clone(), &a.shape[..], &b.shape[..]);
        match o.dtype {
            TType::F32 => {
                let (xa, xb) = (a.as_slice::<f32>(), b.as_slice::<f32>());
                let v = o.as_mut_slice::<f32>();
                broadcast(&so, sa, sb, |i, ia, ib| v[i] = (xa[ia] / xb[ib]).floor())?;
            }
            TType::I32 => {
                let (xa, xb) = (ints(a)?, ints(b)?);
                if xb.contains(&0) {
                    return err("division by 0");
                }
                let mut v = vec![0; o.elements()];
                // reference_ops::FloorDiv: std::floor of the double quotient
                broadcast(&so, sa, sb, |i, ia, ib| v[i] = (xa[ia] as f64 / xb[ib] as f64).floor() as i64)?;
                put_ints(o, &v)?;
            }
            t => return err(format!("{t:?} is not supported")),
        }
        Ok(())
    }
}
