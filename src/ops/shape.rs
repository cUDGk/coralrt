// From TensorFlow Lite 2.5 (Copyright The TensorFlow Authors, Apache License 2.0), translated and changed; see NOTICE.
//! Ops that move elements around without arithmetic (apart from requantizing CONCATENATION).

use super::util::*;
use super::{Io, Kernel};
use crate::interp::Env;
use crate::tensor::{Alloc, TType, Tensor};
use crate::tflite::{Act, OpSpec};
use crate::Result;

pub fn create(s: &OpSpec) -> Result<Option<Box<dyn Kernel>>> {
    Ok(Some(match s.code {
        22 => Box::new(Reshape { params: opt_vec(s, 0) }),
        83 => Box::new(Pack { axis: opt(s, 1, 0) }),
        88 => Box::new(Unpack { axis: opt(s, 1, 0) }),
        2 => Box::new(Concat { axis: opt(s, 0, 0), act: opt_act(s, 1)? }),
        107 => Box::new(GatherNd),
        _ => return Ok(None),
    }))
}

/// reshape.cc: the shape from a 1-D int32 shape input, else from the options; one -1 takes what is left
struct Reshape {
    params: Vec<i32>,
}

impl Reshape {
    fn shape(&self, io: &Io) -> Result<Vec<usize>> {
        let mut dims: Vec<i64> = match io.opt(1) {
            Some(s) if s.shape.len() == 1 && s.dtype == TType::I32 => s.as_slice::<i32>()[..s.shape[0]].iter().map(|&d| d as i64).collect(),
            // legacy models use [0] for a scalar
            _ if self.params == [0] => Vec::new(),
            _ => self.params.iter().map(|&d| d as i64).collect(),
        };
        let n_in = io.input(0)?.elements() as i64;
        let known: i64 = dims.iter().filter(|&&d| d != -1).product();
        match dims.iter().filter(|&&d| d == -1).count() {
            0 => {}
            1 => *dims.iter_mut().find(|d| **d == -1).unwrap() = if known == 0 { 0 } else { n_in / known },
            _ => return err("more than one -1 in the shape"),
        }
        if dims.iter().any(|&d| d < 0) || dims.iter().product::<i64>() != n_in {
            return err(format!("cannot reshape {} elements to {dims:?}", n_in));
        }
        Ok(dims.into_iter().map(|d| d as usize).collect())
    }
}

impl Kernel for Reshape {
    fn prepare(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        // strings are always sized while running; so is a shape that comes from a non-constant tensor
        if io.outs[0].dtype == TType::Str || (io.n_in() == 2 && io.opt(1).is_some_and(|s| s.alloc != Alloc::Constant)) {
            io.outs[0].alloc = Alloc::Dynamic;
            return Ok(());
        }
        let shape = self.shape(io)?;
        io.outs[0].resize(shape);
        Ok(())
    }

    fn eval(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        let shape = self.shape(io)?;
        let i = io.input(0)?;
        let o = &mut io.outs[0];
        if o.dtype == TType::Str {
            o.buf = i.buf.clone();
            o.shape = shape;
            return Ok(());
        }
        o.resize(shape);
        let n = o.data().len();
        o.data_mut().copy_from_slice(&i.data()[..n]);
        Ok(())
    }
}

/// split `shape` at `axis` into (outer elements, elements from axis on)
fn outer_inner(shape: &[usize], axis: usize) -> (usize, usize) {
    (shape[..axis].iter().product(), shape[axis..].iter().product())
}

struct Pack {
    axis: i32,
}

impl Kernel for Pack {
    fn prepare(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        let s = &io.input(0)?.shape;
        let ax = axis_of(self.axis as i64, s.len() + 1)?;
        let mut shape = s.clone();
        shape.insert(ax, io.n_in());
        io.outs[0].resize(shape);
        Ok(())
    }

    fn eval(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        let o = &mut io.outs[0];
        let es = o.dtype.size();
        let ax = axis_of(self.axis as i64, o.shape.len())?;
        let i0 = io.ins[0].unwrap();
        let (outer, _) = outer_inner(&i0.shape, ax);
        let copy = i0.elements() / outer.max(1) * es;
        let n = io.ins.len();
        for k in 0..outer {
            for (m, i) in io.ins.iter().enumerate() {
                let d = (k * n + m) * copy;
                o.data_mut()[d..d + copy].copy_from_slice(&i.unwrap().data()[k * copy..(k + 1) * copy]);
            }
        }
        Ok(())
    }
}

struct Unpack {
    axis: i32,
}

impl Kernel for Unpack {
    fn prepare(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        let s = &io.input(0)?.shape;
        let ax = axis_of(self.axis as i64, s.len())?;
        let mut shape = s.clone();
        let n = shape.remove(ax);
        if n != io.outs.len() {
            return err(format!("{n} slices for {} outputs", io.outs.len()));
        }
        for o in io.outs.iter_mut() {
            o.resize(shape.clone());
        }
        Ok(())
    }

    fn eval(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        let i = io.input(0)?;
        let es = i.dtype.size();
        let ax = axis_of(self.axis as i64, i.shape.len())?;
        let outer: usize = i.shape[..ax].iter().product();
        let n = io.outs.len();
        let copy = io.outs[0].elements() / outer.max(1) * es;
        for k in 0..outer {
            for (m, o) in io.outs.iter_mut().enumerate() {
                let s = (k * n + m) * copy;
                o.data_mut()[k * copy..(k + 1) * copy].copy_from_slice(&i.data()[s..s + copy]);
            }
        }
        Ok(())
    }
}

struct Concat {
    axis: i32,
    act: Act,
}

impl Kernel for Concat {
    fn prepare(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        let i0 = io.input(0)?;
        let ax = axis_of(self.axis as i64, i0.shape.len())?;
        let mut shape = i0.shape.clone();
        shape[ax] = 0;
        for i in io.ins.iter() {
            let i = i.ok_or_else(|| crate::Error("missing input".into()))?;
            if i.shape.len() != shape.len() {
                return err("inputs of different ranks");
            }
            shape[ax] += i.shape[ax];
        }
        io.outs[0].resize(shape);
        Ok(())
    }

    fn eval(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        if self.act != Act::None {
            return err("a fused activation is not supported");
        }
        let o = &mut io.outs[0];
        let es = o.dtype.size();
        let ax = axis_of(self.axis as i64, o.shape.len())?;
        let outer: usize = o.shape[..ax].iter().product();
        let base: usize = o.shape[ax + 1..].iter().product();
        let mut d = 0;
        for k in 0..outer {
            for i in io.ins.iter().map(|i| i.unwrap()) {
                let n = i.shape[ax] * base;
                let src = &i.data()[k * n * es..(k + 1) * n * es];
                let rescale = o.dtype == TType::U8 && (i.zero_point() != o.zero_point() || i.scale() != o.scale());
                if rescale {
                    // ConcatenationWithScaling (uint8 only; int8 inputs share the output's parameters)
                    let scale = i.scale() * (1.0 / o.scale());
                    let bias = -(i.zero_point() as f32) * scale;
                    let zp = o.zero_point();
                    for (j, &q) in src.iter().enumerate() {
                        let v = (q as f32 * scale + bias).round() as i32 + zp;
                        o.data_mut()[d + j] = v.clamp(0, 255) as u8;
                    }
                } else {
                    o.data_mut()[d..d + n * es].copy_from_slice(src);
                }
                d += n * es;
            }
        }
        Ok(())
    }
}

struct GatherNd;

impl GatherNd {
    fn check(p: &Tensor, idx: &Tensor) -> Result<usize> {
        let nd = *idx.shape.last().ok_or_else(|| crate::Error("indices without dimensions".into()))?;
        if nd > p.shape.len() {
            return err(format!("{nd} index dimensions for params of rank {}", p.shape.len()));
        }
        Ok(nd)
    }
}

impl Kernel for GatherNd {
    fn prepare(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        let (p, idx) = (io.input(0)?, io.input(1)?);
        let nd = Self::check(p, idx)?;
        let shape = [&idx.shape[..idx.shape.len() - 1], &p.shape[nd..]].concat();
        io.outs[0].resize(shape);
        Ok(())
    }

    fn eval(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        let (p, idx) = (io.input(0)?, io.input(1)?);
        let o = &mut io.outs[0];
        let nd = Self::check(p, idx)?;
        let pshape = &p.shape;
        let slices: usize = idx.shape[..idx.shape.len() - 1].iter().product();
        let slice: usize = pshape[nd..].iter().product::<usize>();
        let mut dims_to_count = vec![0; nd];
        let mut remain: usize = pshape.iter().product();
        for (j, c) in dims_to_count.iter_mut().enumerate() {
            *c = remain / pshape[j].max(1);
            remain = *c;
        }
        let indices = ints(idx)?;
        // the first element of each slice
        let mut starts = Vec::with_capacity(slices);
        for s in 0..slices {
            let mut from = 0;
            for j in 0..nd {
                let v = indices[s * nd + j];
                if v < 0 || v as usize >= pshape[j] {
                    return err(format!("index {v} out of range [0, {})", pshape[j]));
                }
                from += v as usize * dims_to_count[j];
            }
            starts.push(from);
        }
        if p.dtype == TType::Str {
            let src = p.strings()?;
            let picked: Vec<&[u8]> = starts.iter().flat_map(|&s| src[s..s + slice].iter().copied()).collect();
            let shape = o.shape.clone();
            o.set_strings(&picked, shape);
            return Ok(());
        }
        let (es, d) = (p.dtype.size(), p.data());
        for (s, &from) in starts.iter().enumerate() {
            o.data_mut()[s * slice * es..(s + 1) * slice * es].copy_from_slice(&d[from * es..(from + slice) * es]);
        }
        Ok(())
    }
}
