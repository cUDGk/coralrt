//! Elementwise ops on one input.

use super::util::*;
use super::{Io, Kernel};
use crate::interp::Env;
use crate::tflite::OpSpec;
use crate::Result;

pub fn create(s: &OpSpec) -> Result<Option<Box<dyn Kernel>>> {
    Ok(Some(match s.code {
        75 => Box::new(FloatFn(f32::sqrt)),
        _ => return Ok(None),
    }))
}

/// a float32 function of each element
struct FloatFn(fn(f32) -> f32);

impl Kernel for FloatFn {
    fn prepare(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        same_shape(io.input(0)?, &mut io.outs[0]);
        Ok(())
    }
    fn eval(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        let i = io.input(0)?;
        let o = io.outs[0].as_mut_slice::<f32>();
        for (d, &x) in o.iter_mut().zip(i.as_slice::<f32>()) {
            *d = (self.0)(x);
        }
        Ok(())
    }
}
