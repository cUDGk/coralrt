//! Changes of element type: QUANTIZE, DEQUANTIZE, CAST; and FAKE_QUANT, float rounded to a quantized grid.

use super::util::*;
use super::{Io, Kernel};
use crate::interp::Env;
use crate::tensor::{TType, Tensor};
use crate::tflite::OpSpec;
use crate::Result;

pub fn create(s: &OpSpec) -> Result<Option<Box<dyn Kernel>>> {
    Ok(Some(match s.code {
        114 => Box::new(Elementwise(quantize)),
        6 => Box::new(Elementwise(dequantize)),
        53 => Box::new(Elementwise(cast)),
        80 => {
            if opt(s, 3, 0u8) != 0 {
                return err("narrow_range FAKE_QUANT is not supported at run time");
            }
            Box::new(FakeQuant { min: opt(s, 0, 0.0), max: opt(s, 1, 0.0), num_bits: opt(s, 2, 0) })
        }
        _ => return Ok(None),
    }))
}

/// NumericVerify (numeric_verify.cc, a custom op of the BuiltinOpResolver): the quantized input dequantized, minus
/// the float reference. With log_if_failed and a tolerance of 0.1 or more, a difference over tolerance * scale is an
/// error. A constant input is worked out once and the output then kept, whatever the reference.
pub struct NumericVerify {
    tolerance: f32,
    log_if_failed: bool,
    done: bool,
}

impl NumericVerify {
    pub fn new(custom: &[u8]) -> NumericVerify {
        let get = |k: &str| (!custom.is_empty()).then(|| crate::flex::root(custom).get(k)).flatten();
        NumericVerify { tolerance: get("tolerance").map_or(0.0, |r| r.as_f64() as f32), log_if_failed: get("log_if_failed").is_some_and(|r| r.as_bool()), done: false }
    }
}

impl Kernel for NumericVerify {
    fn prepare(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        if io.n_in() != 2 || io.outs.len() != 1 {
            return err("NumericVerify takes 2 inputs and 1 output");
        }
        let (i, r) = (io.input(0)?, io.input(1)?);
        if !matches!(i.dtype, TType::U8 | TType::I8 | TType::I16 | TType::F16) || r.dtype != TType::F32 || r.elements() < i.elements() {
            return err("NumericVerify takes a quantized (or float16) input and a float32 reference as large");
        }
        // 2.5 makes the output float32 whatever the model says
        io.outs[0].dtype = TType::F32;
        same_shape(i, &mut io.outs[0]);
        Ok(())
    }

    fn eval(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        let (i, r) = (io.input(0)?, io.input(1)?);
        let constant = i.alloc == crate::tensor::Alloc::Constant;
        if constant && self.done {
            return Ok(());
        }
        let mut deq = Tensor { dtype: TType::F32, ..Tensor::default() };
        deq.resize(i.shape.clone());
        dequantize(i, &mut deq)?;
        self.done = constant;
        let out = io.outs[0].as_mut_slice::<f32>();
        let max_diff = self.tolerance * i.scale();
        for (k, (o, (&d, &x))) in out.iter_mut().zip(deq.as_slice::<f32>().iter().zip(r.as_slice::<f32>())).enumerate() {
            *o = d - x;
            if self.log_if_failed && self.tolerance >= 0.1 && o.abs() > max_diff {
                return err(format!("Mismatch: {x} is quantized to element {k} as {d}, off by {} > {max_diff} (tolerance)", o.abs()));
            }
        }
        Ok(())
    }
}

/// fake_quant.cc (reference_ops::FakeQuant): the range nudged so 0 is on the grid (NudgeQuantizationRange), then
/// each value clamped to it and rounded to the grid (FakeQuantizeArray), all in float
struct FakeQuant {
    min: f32,
    max: f32,
    num_bits: i32,
}

impl Kernel for FakeQuant {
    fn prepare(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        if io.input(0)?.dtype != TType::F32 {
            return err("FAKE_QUANT takes float32");
        }
        same_shape(io.input(0)?, &mut io.outs[0]);
        Ok(())
    }

    fn eval(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        let quant_max = ((1i64 << self.num_bits) - 1) as i32;
        let (qmin, qmax) = (0.0f32, quant_max as f32);
        let scale = (self.max - self.min) / (qmax - qmin);
        let zero_point_from_min = qmin - self.min / scale;
        // the zero point is a uint16_t
        let zp = if zero_point_from_min < qmin {
            0
        } else if zero_point_from_min > qmax {
            quant_max as u16
        } else {
            cvtt(zero_point_from_min.round() as f64, false) as u16
        } as f32;
        let (nmin, nmax) = ((qmin - zp) * scale, (qmax - zp) * scale);
        let inv_scale = 1.0 / scale;
        let i = io.input(0)?;
        for (d, &x) in io.outs[0].as_mut_slice::<f32>().iter_mut().zip(i.as_slice::<f32>()) {
            // std::min(nmax, std::max(nmin, x)): NaN becomes nmin
            let c = if nmin < x { x } else { nmin };
            let c = if c < nmax { c } else { nmax };
            *d = ((c - nmin) * inv_scale).round() * scale + nmin;
        }
        Ok(())
    }
}

/// an op from one tensor to one of the same shape, element by element (1-byte inputs go through a 256-entry table)
struct Elementwise(fn(&Tensor, &mut Tensor) -> Result<()>);

impl Kernel for Elementwise {
    fn prepare(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        same_shape(io.input(0)?, &mut io.outs[0]);
        Ok(())
    }
    fn eval(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        by_table(io.input(0)?, &mut io.outs[0], self.0)
    }
}

fn quantize(i: &Tensor, o: &mut Tensor) -> Result<()> {
    let (lo, hi) = type_range(o.dtype);
    match (i.dtype, o.dtype) {
        (TType::F32, TType::U8 | TType::I8 | TType::I16) => {
            // optimized_ops::AffineQuantize: divides by the scale as a double
            let (scale, zp) = (o.scale() as f64, o.zero_point());
            let v: Vec<i64> = i.as_slice::<f32>().iter().map(|&x| round_i32(x as f64 / scale).wrapping_add(zp).clamp(lo, hi) as i64).collect();
            put_ints(o, &v)?;
        }
        (TType::U8 | TType::I8, TType::U8 | TType::I8) | (TType::I16, TType::I8 | TType::I16 | TType::I32) => {
            if i.dtype == TType::I16 && o.dtype == TType::I16 && (i.zero_point() != 0 || o.zero_point() != 0) {
                return err("int16 -> int16 needs zero points 0");
            }
            // Requantize with QuantizeMultiplier(input scale / output scale)
            let (m, shift) = quantize_multiplier(i.scale() as f64 / o.scale() as f64);
            let (izp, ozp) = (i.zero_point(), o.zero_point());
            let v: Vec<i64> = ints(i)?.iter().map(|&x| mul_by_quantized_multiplier(x as i32 - izp, m, shift).wrapping_add(ozp).clamp(lo, hi) as i64).collect();
            put_ints(o, &v)?;
        }
        (a, b) => return err(format!("{a:?} -> {b:?} is not supported")),
    }
    Ok(())
}

fn dequantize(i: &Tensor, o: &mut Tensor) -> Result<()> {
    if i.dtype == TType::F16 && o.dtype == TType::F32 {
        for (d, &h) in o.as_mut_slice::<f32>().iter_mut().zip(i.as_slice::<u16>()) {
            *d = f16_to_f32(h);
        }
        return Ok(());
    }
    if !matches!(i.dtype, TType::U8 | TType::I8 | TType::I16) || o.dtype != TType::F32 {
        return err(format!("{:?} -> {:?} is not supported", i.dtype, o.dtype));
    }
    let (scale, zp) = (i.scale() as f64, i.zero_point() as i64);
    let v: Vec<f32> = ints(i)?.into_iter().map(|x| (scale * (x - zp) as f64) as f32).collect();
    o.as_mut_slice::<f32>().copy_from_slice(&v);
    Ok(())
}


/// cast.cc: static_cast of every element; a complex value casts its real part, a cast to complex has no imaginary one
fn cast(i: &Tensor, o: &mut Tensor) -> Result<()> {
    if i.dtype == TType::C64 && o.dtype == TType::C64 {
        o.as_mut_slice::<f32>().copy_from_slice(i.as_slice::<f32>());
        return Ok(());
    }
    let float_in = matches!(i.dtype, TType::F32 | TType::F64 | TType::F16 | TType::C64);
    let reals: Vec<f64> = match i.dtype {
        TType::C64 => i.as_slice::<f32>().iter().step_by(2).map(|&x| x as f64).collect(),
        _ if float_in => f64s(i)?,
        _ => Vec::new(),
    };
    match o.dtype {
        TType::C64 => {
            let v = if float_in { reals } else { ints(i)?.into_iter().map(|x| x as f32 as f64).collect() };
            for (d, x) in o.as_mut_slice::<f32>().chunks_exact_mut(2).zip(v) {
                d[0] = x as f32;
                d[1] = 0.0;
            }
            Ok(())
        }
        TType::F32 | TType::F64 | TType::F16 | TType::Bool if float_in => put_f64s(o, &reals),
        t if float_in => put_ints(o, &reals.iter().map(|&x| cvtt(x, matches!(t, TType::I64 | TType::U64))).collect::<Vec<_>>()),
        _ => put_ints(o, &ints(i)?),
    }
}
