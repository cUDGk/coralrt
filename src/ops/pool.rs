//! Pooling over height and width (pooling.cc): AVERAGE_POOL_2D, MAX_POOL_2D, L2_POOL_2D, the way 2.5's optimized
//! kernels compute them. Every window is the filter clipped to the input; padding never counts.

use super::util::*;
use super::{Io, Kernel};
use crate::interp::Env;
use crate::tensor::TType;
use crate::tflite::{Act, OpSpec};
use crate::Result;

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Avg,
    Max,
    L2,
}

pub fn create(s: &OpSpec) -> Result<Option<Box<dyn Kernel>>> {
    let kind = match s.code {
        1 => Kind::Avg,
        17 => Kind::Max,
        12 => Kind::L2,
        _ => return Ok(None),
    };
    Ok(Some(Box::new(Pool {
        kind,
        padding: opt_padding(s, 0)?,
        stride: (opt(s, 2, 0), opt(s, 1, 0)),
        filter: (opt(s, 4, 0), opt(s, 3, 0)),
        act: opt_act(s, 5)?,
        pad: (0, 0),
    })))
}

/// (height, width) pairs
struct Pool {
    kind: Kind,
    padding: Padding,
    stride: (i32, i32),
    filter: (i32, i32),
    act: Act,
    pad: (i32, i32),
}

impl Kernel for Pool {
    fn prepare(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        let i = io.input(0)?;
        let o = &mut io.outs[0];
        if i.shape.len() != 4 || i.dtype != o.dtype {
            return err("pooling takes a 4-D input and an output of its type");
        }
        if self.stride.0 <= 0 || self.stride.1 <= 0 {
            return err("pooling strides must be positive");
        }
        if self.kind == Kind::L2 && i.dtype != TType::F32 {
            return err("L2_POOL_2D takes float32");
        }
        let (oh, ph, _) = out_size_and_pad(self.padding, i.shape[1] as i32, self.filter.0, self.stride.0, 1);
        let (ow, pw, _) = out_size_and_pad(self.padding, i.shape[2] as i32, self.filter.1, self.stride.1, 1);
        if oh < 0 || ow < 0 {
            return err("the filter is larger than the input");
        }
        self.pad = (ph, pw);
        o.resize(vec![i.shape[0], oh as usize, ow as usize, i.shape[3]]);
        Ok(())
    }

    fn eval(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        let i = io.input(0)?;
        let o = &mut io.outs[0];
        let windows = self.windows(&i.shape, &o.shape);
        let depth = i.shape[3];
        match (self.kind, i.dtype) {
            (Kind::Avg | Kind::L2, TType::F32) => {
                let (lo, hi) = act_range_f(self.act);
                let (x, v) = (i.as_slice::<f32>(), o.as_mut_slice::<f32>());
                for (out, cells) in v.chunks_mut(depth).zip(&windows) {
                    // 2.5 adds each input pixel into the outputs it falls in, in input order, from 0
                    out.fill(0.0);
                    for &p in cells {
                        for (d, &a) in out.iter_mut().zip(&x[p * depth..(p + 1) * depth]) {
                            *d += if self.kind == Kind::L2 { a * a } else { a };
                        }
                    }
                    let count = cells.len() as f32;
                    for d in out.iter_mut() {
                        // L2: times 1 / count, then sqrt (Eigen's scalar path: the row-wise product has no packets)
                        let r = if self.kind == Kind::L2 { (*d * (1.0 / count)).sqrt() } else { *d / count };
                        *d = clamp_f(r, lo, hi);
                    }
                }
            }
            (Kind::Max, TType::F32) => {
                let (lo, hi) = act_range_f(self.act);
                let (x, v) = (i.as_slice::<f32>(), o.as_mut_slice::<f32>());
                for (out, cells) in v.chunks_mut(depth).zip(&windows) {
                    out.fill(f32::MIN);
                    for &p in cells {
                        for (d, &a) in out.iter_mut().zip(&x[p * depth..(p + 1) * depth]) {
                            if *d < a {
                                *d = a;
                            }
                        }
                    }
                    out.iter_mut().for_each(|d| *d = clamp_f(*d, lo, hi));
                }
            }
            (_, TType::U8 | TType::I8 | TType::I16) => {
                let (lo, hi) = act_range_q(self.act, o);
                let x = ints(i)?;
                let mut res = Vec::with_capacity(windows.len() * depth);
                for cells in &windows {
                    if cells.is_empty() {
                        return err("a pooling window holds no input");
                    }
                    let n = cells.len() as i64;
                    for c in 0..depth {
                        let vals = cells.iter().map(|&p| x[p * depth + c]);
                        let r = if self.kind == Kind::Max {
                            // uint8 starts from 0, int8 from the activation minimum, int16 from its lowest
                            let start = match i.dtype {
                                TType::U8 => 0,
                                TType::I8 => lo as i64,
                                _ => i16::MIN as i64,
                            };
                            vals.fold(start, i64::max)
                        } else {
                            let acc: i64 = vals.sum();
                            // to the nearest, halves away from zero (uint8: always up), as int16
                            (if acc > 0 || i.dtype == TType::U8 { (acc + n / 2) / n } else { (acc - n / 2) / n }) as i16 as i64
                        };
                        res.push(r.clamp(lo as i64, hi as i64));
                    }
                }
                put_ints(o, &res)?;
            }
            (_, t) => return err(format!("{t:?} is not supported")),
        }
        Ok(())
    }
}

impl Pool {
    /// for each output pixel (batch, y, x in order), the flat input pixels of its window in raster order
    fn windows(&self, is: &[usize], os: &[usize]) -> Vec<Vec<usize>> {
        let (ih, iw) = (is[1] as i32, is[2] as i32);
        let mut all = Vec::with_capacity(os[0] * os[1] * os[2]);
        for b in 0..os[0] {
            for oy in 0..os[1] as i32 {
                for ox in 0..os[2] as i32 {
                    let (y0, x0) = (oy * self.stride.0 - self.pad.0, ox * self.stride.1 - self.pad.1);
                    let ys = (-y0).max(0)..self.filter.0.min(ih - y0);
                    let xs = (-x0).max(0)..self.filter.1.min(iw - x0);
                    let mut cells = Vec::new();
                    for fy in ys {
                        for fx in xs.clone() {
                            cells.push((b * is[1] + (y0 + fy) as usize) * is[2] + (x0 + fx) as usize);
                        }
                    }
                    all.push(cells);
                }
            }
        }
        all
    }
}
