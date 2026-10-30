//! Reductions along axes.

use super::util::*;
use super::{Io, Kernel};
use crate::interp::Env;
use crate::tensor::{Alloc, TType};
use crate::tflite::OpSpec;
use crate::Result;

pub fn create(s: &OpSpec) -> Result<Option<Box<dyn Kernel>>> {
    Ok(Some(match s.code {
        56 => Box::new(ArgMinMax { max: true }),
        79 => Box::new(ArgMinMax { max: false }),
        _ => return Ok(None),
    }))
}

/// arg_min_max.cc: the first index of the largest (smallest) value along one axis
struct ArgMinMax {
    max: bool,
}

impl ArgMinMax {
    fn axis(io: &Io) -> Result<usize> {
        let a = *ints(io.input(1)?)?.first().ok_or_else(|| crate::Error("no axis".into()))?;
        axis_of(a, io.input(0)?.shape.len())
    }
    fn resize(io: &mut Io) -> Result<()> {
        let ax = Self::axis(io)?;
        let mut shape = io.input(0)?.shape.clone();
        shape.remove(ax);
        io.outs[0].resize(shape);
        Ok(())
    }
}

impl Kernel for ArgMinMax {
    fn prepare(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        if io.input(1)?.alloc == Alloc::Constant {
            Self::resize(io)
        } else {
            io.outs[0].alloc = Alloc::Dynamic;
            Ok(())
        }
    }

    fn eval(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        if io.outs[0].alloc == Alloc::Dynamic {
            Self::resize(io)?;
        }
        let i = io.input(0)?;
        let ax = Self::axis(io)?;
        let (outer, len, inner) = (i.shape[..ax].iter().product::<usize>(), i.shape[ax], i.shape[ax + 1..].iter().product::<usize>());
        let best = match i.dtype {
            TType::F32 | TType::F64 => along(&f64s(i)?, outer, len, inner, self.max),
            _ => along(&ints(i)?, outer, len, inner, self.max),
        };
        put_ints(&mut io.outs[0], &best)
    }
}

/// for each (outer, inner) position, the first index along the axis of the largest (smallest) value: std::greater /
/// std::less, so ties keep the earlier one
fn along<T: PartialOrd>(v: &[T], outer: usize, len: usize, inner: usize, max: bool) -> Vec<i64> {
    let mut res = Vec::with_capacity(outer * inner);
    for out in 0..outer {
        for inn in 0..inner {
            let at = |k: usize| (out * len + k) * inner + inn;
            let mut best = 0;
            for k in 1..len {
                let better = if max { v[at(k)] > v[at(best)] } else { v[at(k)] < v[at(best)] };
                if better {
                    best = k;
                }
            }
            res.push(best as i64);
        }
    }
    res
}
