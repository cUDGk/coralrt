//! Changes of element type: QUANTIZE, DEQUANTIZE, CAST.

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
        _ => return Ok(None),
    }))
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
        (TType::F32, TType::U8 | TType::I8) => {
            // optimized_ops::AffineQuantize: divides by the scale as a double
            let (scale, zp) = (o.scale() as f64, o.zero_point());
            for (d, &x) in o.data_mut().iter_mut().zip(i.as_slice::<f32>()) {
                *d = round_i32(x as f64 / scale).wrapping_add(zp).clamp(lo, hi) as u8;
            }
        }
        (TType::U8 | TType::I8, TType::U8 | TType::I8) => {
            // Requantize with QuantizeMultiplier(input scale / output scale)
            let (m, shift) = quantize_multiplier(i.scale() as f64 / o.scale() as f64);
            let (izp, ozp) = (i.zero_point(), o.zero_point());
            let v = byte_table(i.dtype, |x| (mul_by_quantized_multiplier(x - izp, m, shift) + ozp).clamp(lo, hi));
            o.data_mut().iter_mut().zip(i.data()).for_each(|(d, &b)| *d = v[b as usize] as u8);
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
