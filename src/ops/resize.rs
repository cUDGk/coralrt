//! RESIZE_BILINEAR and RESIZE_NEAREST_NEIGHBOR (resize_bilinear.cc, resize_nearest_neighbor.cc) on NHWC images, each
//! through the path 2.5 takes for its type and sizes.

use super::util::*;
use super::{Io, Kernel};
use crate::interp::Env;
use crate::tensor::{Alloc, TType};
use crate::tflite::OpSpec;
use crate::Result;

pub fn create(s: &OpSpec) -> Result<Option<Box<dyn Kernel>>> {
    Ok(Some(Box::new(match s.code {
        23 => Resize { bilinear: true, align: opt(s, 2, 0u8) != 0, half: opt(s, 3, 0u8) != 0 },
        97 => Resize { bilinear: false, align: opt(s, 0, 0u8) != 0, half: opt(s, 1, 0u8) != 0 },
        _ => return Ok(None),
    })))
}

struct Resize {
    bilinear: bool,
    align: bool,
    half: bool,
}

impl Resize {
    fn resize(&self, io: &mut Io) -> Result<()> {
        let size = ints(io.input(1)?)?;
        if size.len() < 2 || (self.bilinear && (size[0] <= 0 || size[1] <= 0)) || size[0] < 0 || size[1] < 0 {
            return err(format!("bad output size {size:?}"));
        }
        let s = io.input(0)?.shape.clone();
        io.outs[0].resize(vec![s[0], size[0] as usize, size[1] as usize, s[3]]);
        Ok(())
    }
}

impl Kernel for Resize {
    fn prepare(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        let (i, size) = (io.input(0)?, io.input(1)?);
        if i.shape.len() != 4 || size.shape.len() != 1 || size.dtype != TType::I32 || (!self.bilinear && size.shape[0] != 2) {
            return err("resizing takes a 4-D input and an int32 size of 2");
        }
        // 2.5 gives the output the input's type
        io.outs[0].dtype = i.dtype;
        if size.alloc != Alloc::Constant {
            io.outs[0].alloc = Alloc::Dynamic;
            return Ok(());
        }
        if self.half && self.align && self.bilinear {
            return err("If half_pixel_centers is True, align_corners must be False.");
        }
        self.resize(io)
    }

    fn eval(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        if io.outs[0].alloc == Alloc::Dynamic {
            self.resize(io)?;
        }
        let i = io.input(0)?;
        let o = &mut io.outs[0];
        let [b, ih, iw, d] = [i.shape[0], i.shape[1], i.shape[2], i.shape[3]];
        let (oh, ow) = (o.shape[1], o.shape[2]);
        let (align, half) = (self.align, self.half);
        if !self.bilinear {
            if !matches!(i.dtype, TType::F32 | TType::U8 | TType::I8 | TType::I16) {
                return err(format!("{:?} is not supported", i.dtype));
            }
            let fast = i.dtype == TType::U8 && !align && !half;
            let ys: Vec<usize> = (0..oh).map(|y| nearest(y, ih, oh, align, half, fast)).collect();
            let xs: Vec<usize> = (0..ow).map(|x| nearest(x, iw, ow, align, half, fast)).collect();
            let es = d * i.dtype.size();
            let (src, dst) = (i.data(), o.data_mut());
            let mut at = 0;
            for bb in 0..b {
                for &y in &ys {
                    for &x in &xs {
                        let from = ((bb * ih + y) * iw + x) * es;
                        dst[at..at + es].copy_from_slice(&src[from..from + es]);
                        at += es;
                    }
                }
            }
            return Ok(());
        }
        match i.dtype {
            TType::F32 if !align && !half && oh == 2 * ih && ow == 2 * iw => {
                resize_2x2(i.as_slice(), o.as_mut_slice(), b, ih, iw, d)
            }
            TType::U8 if !align && half && d % 8 == 0 && oh >= 8 && ow >= 8 && oh == 8 * ih && ow == 8 * iw => {
                resize_888(i.as_slice(), o.as_mut_slice(), b, ih, iw, d)
            }
            TType::F32 | TType::U8 => {
                let scale = |n: usize, m: usize| if align && m > 1 { (n - 1) as f32 / (m - 1) as f32 } else { n as f32 / m as f32 };
                let (hs, ws) = (scale(ih, oh), scale(iw, ow));
                let x = f64s(i)?;
                let mut res = Vec::with_capacity(b * oh * ow * d);
                for bb in 0..b {
                    for y in 0..oh {
                        let (fy, y0, y1) = interp(y, hs, half, ih);
                        for xx in 0..ow {
                            let (fx, x0, x1) = interp(xx, ws, half, iw);
                            let (dy, dx) = (fy - y0 as f32, fx - x0 as f32);
                            let w = [(1.0 - dy) * (1.0 - dx), (1.0 - dy) * dx, dy * (1.0 - dx), dy * dx];
                            let at = |yy: usize, xx: usize| ((bb * ih + yy) * iw + xx) * d;
                            let corners = [at(y0, x0), at(y0, x1), at(y1, x0), at(y1, x1)];
                            for c in 0..d {
                                let v = |k: usize| x[corners[k] + c] as f32 * w[k];
                                res.push(if i.dtype == TType::F32 {
                                    // ResizeBilinearGeneric: from 0, the four corners added in turn
                                    (((0.0 + v(0)) + v(1)) + v(2)) + v(3)
                                } else {
                                    // ResizeBilinearGenericSmallChannel: one sum, + 0.5, truncated
                                    cvtt((v(0) + v(1) + v(2) + v(3) + 0.5) as f64, false) as u8 as f32
                                } as f64);
                            }
                        }
                    }
                }
                put_f64s(o, &res)?;
            }
            TType::I8 | TType::I16 => {
                let scale10 = |n: usize, m: usize| {
                    let (n, m) = if align && m > 1 { (n as i32 - 1, m as i32 - 1) } else { (n as i32, m as i32) };
                    ((1 << 10) * n + m / 2) / m
                };
                let (hs, ws) = (scale10(ih, oh), scale10(iw, ow));
                let x = ints(i)?;
                let mut res = Vec::with_capacity(b * oh * ow * d);
                for bb in 0..b {
                    for y in 0..oh {
                        let (fy, y0, y1) = interp_int(y as i32, hs, half, ih as i32);
                        for xx in 0..ow {
                            let (fx, x0, x1) = interp_int(xx as i32, ws, half, iw as i32);
                            let (dy, dx) = ((fy - 1024 * y0) as i64, (fx - 1024 * x0) as i64);
                            let at = |yy: i32, xx: i32| ((bb * ih + yy as usize) * iw + xx as usize) * d;
                            for c in 0..d {
                                let v20 = x[at(y0, x0) + c] * (1024 - dy) * (1024 - dx)
                                    + x[at(y1, x0) + c] * dy * (1024 - dx)
                                    + x[at(y0, x1) + c] * (1024 - dy) * dx
                                    + x[at(y1, x1) + c] * dy * dx;
                                let round = if v20 > 0 { 1 << 19 } else { -(1 << 19) };
                                res.push((v20 + round) / (1 << 20));
                            }
                        }
                    }
                }
                put_ints(o, &res)?;
            }
            t => return err(format!("{t:?} is not supported")),
        }
        Ok(())
    }
}

/// GetNearestNeighbor; `fast`: the uint8 kernel's 16.16 fixed point instead (no corners, no half pixels)
fn nearest(v: usize, n: usize, m: usize, align: bool, half: bool, fast: bool) -> usize {
    if fast {
        let scale = ((n as i32) << 16) / m as i32 + 1;
        return (((v as i32).wrapping_mul(scale) >> 16) as usize).min(n - 1);
    }
    let scale = if align && m > 1 { (n - 1) as f32 / (m - 1) as f32 } else { n as f32 / m as f32 };
    let p = (v as f32 + if half { 0.5 } else { 0.0 }) * scale;
    let r = (cvtt(if align { p.round() } else { p.floor() } as f64, false) as i32).min(n as i32 - 1);
    (if half { r.max(0) } else { r }) as usize
}

/// ComputeInterpolationValues: the source coordinate and the pixels below and above it
fn interp(v: usize, scale: f32, half: bool, n: usize) -> (f32, usize, usize) {
    let s = if half { (v as f32 + 0.5) * scale - 0.5 } else { v as f32 * scale };
    let lo = (cvtt(s.floor() as f64, false) as i32).max(0);
    let hi = (cvtt(s.ceil() as f64, false) as i32).min(n as i32 - 1);
    (s, lo as usize, hi as usize)
}

/// ComputeInterpolationValuesInteger: the same in 1/1024 steps
fn interp_int(v: i32, scale10: i32, half: bool, n: i32) -> (i32, i32, i32) {
    let s = if half { v * scale10 + scale10 / 2 - (1 << 9) } else { v * scale10 };
    (s, (s / 1024).max(0), ((s + 1023) / 1024).min(n - 1))
}

/// ResizeBilinear2x2 (float, exactly twice the size): halves of neighbor sums
fn resize_2x2(x: &[f32], o: &mut [f32], b: usize, ih: usize, iw: usize, d: usize) {
    let ow = 2 * iw;
    for bb in 0..b {
        for y0 in 0..ih {
            for x0 in 0..iw {
                let (x1, y1) = ((x0 + 1).min(iw - 1), (y0 + 1).min(ih - 1));
                for c in 0..d {
                    let at = |y: usize, xx: usize| x[((bb * ih + y) * iw + xx) * d + c];
                    let (a, r, u, ru) = (at(y0, x0), at(y0, x1), at(y1, x0), at(y1, x1));
                    let out = |y: usize, xx: usize| ((bb * 2 * ih + y) * ow + xx) * d + c;
                    let down = (a + u) / 2.0;
                    o[out(2 * y0, 2 * x0)] = a;
                    o[out(2 * y0, 2 * x0 + 1)] = (a + r) / 2.0;
                    o[out(2 * y0 + 1, 2 * x0)] = down;
                    o[out(2 * y0 + 1, 2 * x0 + 1)] = (down + (r + ru) / 2.0) / 2.0;
                }
            }
        }
    }
}

/// ResizeBilinear888Uint8 (the build without NEON): 8x with half-pixel centers, stepped along each row in 8.8 fixed
/// point (0.5 added up front for rounding), with 2.5's uint16 / int16 wraparound
fn resize_888(x: &[u8], o: &mut [u8], b: usize, h: usize, w: usize, d: usize) {
    let (oh, ow) = (8 * h, 8 * w);
    for bb in 0..b {
        let inp = |r: usize, j: usize, c: usize| x[((bb * h + r) * w + j) * d + c] as i32;
        let mut put = |row: usize, col: usize, c: usize, acc: u16| o[((bb * oh + row) * ow + col) * d + c] = (acc >> 8) as u8;
        for c in 0..d {
            // top and bottom margins: the first and last input rows, stretched along x over 4 output rows each
            for (r, rows) in [(0, 0..4), (h - 1, oh - 4..oh)] {
                let mut acc = ((inp(r, 0, c) << 8) + 128) as u16;
                let mut cols = vec![acc; 4];
                for j in 0..w - 1 {
                    let wd = ((inp(r, j + 1, c) - inp(r, j, c)) as u16) << 4;
                    acc = acc.wrapping_add(wd);
                    cols.push(acc);
                    for _ in 1..8 {
                        acc = acc.wrapping_add(wd << 1);
                        cols.push(acc);
                    }
                    acc = acc.wrapping_add(wd);
                }
                cols.extend([acc; 4]);
                for row in rows {
                    for (col, &a) in cols.iter().enumerate() {
                        put(row, col, c, a);
                    }
                }
            }
            for k in 0..h - 1 {
                let row0 = 4 + 8 * k;
                let hd = (((inp(k + 1, 0, c) - inp(k, 0, c)) as u16) << 4) as i16;
                let hd2 = ((hd as i32) << 1) as i16;
                let mut acc = [0u16; 8];
                acc[0] = (((inp(k, 0, c) << 8) + 128) as u16).wrapping_add(hd as u16);
                for r in 1..8 {
                    acc[r] = acc[r - 1].wrapping_add(hd2 as u16);
                }
                for (r, &a) in acc.iter().enumerate() {
                    (0..4).for_each(|col| put(row0 + r, col, c, a));
                }
                for j in 0..w - 1 {
                    let wd = (((inp(k, j + 1, c) - inp(k, j, c)) as u16) << 4) as i16;
                    let hwd = ((inp(k + 1, j + 1, c) - inp(k, j + 1, c) - inp(k + 1, j, c) + inp(k, j, c)) as u16) as i16;
                    let hwd2 = ((hwd as i32) << 1) as i16;
                    let mut incr_base = (wd as i32 + hwd as i32) as u16;
                    let mut incr = [0i16; 8];
                    for r in 0..8 {
                        if r > 0 {
                            incr_base = incr_base.wrapping_add(hwd2 as u16);
                        }
                        acc[r] = acc[r].wrapping_add(incr_base);
                        put(row0 + r, 8 * j + 4, c, acc[r]);
                        incr[r] = ((incr_base as i32) << 1) as i16;
                    }
                    for p in 1..8 {
                        for r in 0..8 {
                            acc[r] = acc[r].wrapping_add(incr[r] as u16);
                            put(row0 + r, 8 * j + 4 + p, c, acc[r]);
                        }
                    }
                    for r in 0..8 {
                        acc[r] = acc[r].wrapping_add((incr[r] / 2) as u16);
                    }
                }
                for (r, &a) in acc.iter().enumerate() {
                    (0..4).for_each(|p| put(row0 + r, 8 * (w - 1) + 4 + p, c, a));
                }
            }
        }
    }
}
