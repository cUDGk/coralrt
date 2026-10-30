//! Convolutions (conv.cc): CONV_2D the way 2.5's x86 build runs it. Float goes through Eigen's convolution, whose
//! summation order depends on its blocking, so it is summed in double here and matches within rounding. The
//! quantized paths are im2col + GEMM (exact integer sums; padding contributes nothing) with the GEMM's
//! requantization, int16 is the reference kernel, and the hybrid paths quantize the float input per batch.

use super::util::*;
use super::{Io, Kernel};
use crate::interp::Env;
use crate::tensor::{TType, Tensor};
use crate::tflite::{Act, OpSpec};
use crate::Result;

pub fn create(s: &OpSpec) -> Result<Option<Box<dyn Kernel>>> {
    Ok(Some(match s.code {
        3 => Box::new(Conv2D {
            padding: opt_padding(s, 0)?,
            stride: (opt(s, 2, 0), opt(s, 1, 0)),
            act: opt_act(s, 3)?,
            dilation: (opt(s, 5, 1), opt(s, 4, 1)),
            geom: Geom::default(),
            quant: Quant::default(),
        }),
        4 => Box::new(DepthwiseConv2D {
            padding: opt_padding(s, 0)?,
            stride: (opt(s, 2, 0), opt(s, 1, 0)),
            depth_multiplier: opt(s, 3, 0),
            act: opt_act(s, 4)?,
            dilation: (opt(s, 6, 1), opt(s, 5, 1)),
            geom: Geom::default(),
            quant: Quant::default(),
        }),
        _ => return Ok(None),
    }))
}

/// sizes of a 2-D convolution over NHWC, (height, width) pairs
#[derive(Default, Clone)]
struct Geom {
    batches: usize,
    input: (usize, usize),
    in_ch: usize,
    output: (usize, usize),
    out_ch: usize,
    filter: (usize, usize),
    stride: (i32, i32),
    dilation: (i32, i32),
    pad: (i32, i32),
}

impl Geom {
    /// for each output pixel (batch, y, x in order), its filter taps inside the image as (input pixel, tap)
    fn taps(&self) -> Vec<Vec<(usize, usize)>> {
        let (ih, iw) = (self.input.0 as i32, self.input.1 as i32);
        let mut all = Vec::with_capacity(self.batches * self.output.0 * self.output.1);
        for b in 0..self.batches {
            for oy in 0..self.output.0 as i32 {
                for ox in 0..self.output.1 as i32 {
                    let mut taps = Vec::new();
                    for fy in 0..self.filter.0 as i32 {
                        let y = oy * self.stride.0 - self.pad.0 + self.dilation.0 * fy;
                        for fx in 0..self.filter.1 as i32 {
                            let x = ox * self.stride.1 - self.pad.1 + self.dilation.1 * fx;
                            if y >= 0 && y < ih && x >= 0 && x < iw {
                                let pixel = (b * self.input.0 + y as usize) * self.input.1 + x as usize;
                                taps.push((pixel, fy as usize * self.filter.1 + fx as usize));
                            }
                        }
                    }
                    all.push(taps);
                }
            }
        }
        all
    }

    /// acc = mac(acc, input index, filter index) over every tap and input channel, per (output pixel, channel)
    fn dots<A: Copy>(&self, zero: A, mut mac: impl FnMut(A, usize, usize) -> A) -> Vec<A> {
        let (ic, kk) = (self.in_ch, self.filter.0 * self.filter.1);
        let mut res = Vec::with_capacity(self.batches * self.output.0 * self.output.1 * self.out_ch);
        for taps in self.taps() {
            for oc in 0..self.out_ch {
                let mut acc = zero;
                for &(pixel, tap) in &taps {
                    for c in 0..ic {
                        acc = mac(acc, pixel * ic + c, (oc * kk + tap) * ic + c);
                    }
                }
                res.push(acc);
            }
        }
        res
    }
}

/// the quantized parameters worked out in prepare (PopulateConvolutionQuantizationParams)
#[derive(Default, Clone)]
struct Quant {
    /// per output channel (one value repeated for a per-tensor filter)
    multipliers: Vec<(i32, i32)>,
    act: (i32, i32),
}

struct Conv2D {
    padding: Padding,
    stride: (i32, i32),
    act: Act,
    dilation: (i32, i32),
    geom: Geom,
    quant: Quant,
}

impl Kernel for Conv2D {
    fn prepare(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        let (i, f) = (io.input(0)?, io.input(1)?);
        let bias = io.opt(2).ok_or_else(|| crate::Error("CONV_2D needs a bias".into()))?;
        let o = &io.outs[0];
        if i.shape.len() != 4 || f.shape.len() != 4 || i.shape[3] != f.shape[3] {
            return err("CONV_2D takes a 4-D input and a 4-D filter of the same depth");
        }
        if !matches!(i.dtype, TType::F32 | TType::U8 | TType::I8 | TType::I16) || o.dtype != i.dtype {
            return err(format!("input {:?}, output {:?} is not supported", i.dtype, o.dtype));
        }
        if i.dtype == TType::I16 && (i.zero_point() != 0 || o.zero_point() != 0) {
            return err("int16 needs zero points 0");
        }
        let bias_ok = match i.dtype {
            TType::U8 | TType::I8 => bias.dtype == TType::I32 && bias.zero_point() == 0,
            TType::I16 => bias.dtype == TType::I64 && bias.zero_point() == 0,
            t => bias.dtype == t,
        };
        if !bias_ok || bias.elements() != f.shape[0] {
            return err("the bias must have the input's accumulator type and one value per output channel");
        }
        let (oc, fh, fw) = (f.shape[0], f.shape[1], f.shape[2]);
        let (oh, ph, _) = out_size_and_pad(self.padding, i.shape[1] as i32, fh as i32, self.stride.0, self.dilation.0);
        let (ow, pw, _) = out_size_and_pad(self.padding, i.shape[2] as i32, fw as i32, self.stride.1, self.dilation.1);
        if oh < 0 || ow < 0 {
            return err("the filter is larger than the input");
        }
        self.geom = Geom {
            batches: i.shape[0],
            input: (i.shape[1], i.shape[2]),
            in_ch: i.shape[3],
            output: (oh as usize, ow as usize),
            out_ch: oc,
            filter: (fh, fw),
            stride: self.stride,
            dilation: self.dilation,
            pad: (ph, pw),
        };
        if i.dtype != TType::F32 {
            self.quant = conv_quant(i, f, Some(bias), o, self.act, oc)?;
        }
        io.outs[0].resize(vec![i.shape[0], oh as usize, ow as usize, oc]);
        Ok(())
    }

    fn eval(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        let (i, f, bias) = (io.input(0)?, io.input(1)?, io.input(2)?);
        let o = &mut io.outs[0];
        let g = &self.geom;
        let oc = g.out_ch;
        match (i.dtype, f.dtype) {
            (TType::F32, TType::F32) => {
                let (x, w, b) = (i.as_slice::<f32>(), f.as_slice::<f32>(), bias.as_slice::<f32>());
                let (lo, hi) = act_range_f(self.act);
                let acc = g.dots(0.0f64, |a, xi, wi| a + x[xi] as f64 * w[wi] as f64);
                let v = o.as_mut_slice::<f32>();
                for (k, a) in acc.into_iter().enumerate() {
                    v[k] = clamp_f(a as f32 + b[k % oc], lo, hi);
                }
            }
            (TType::F32, TType::U8 | TType::I8) => self.hybrid(i, f, bias, o)?,
            (TType::I16, _) => {
                let (x, w, b) = (ints(i)?, ints(f)?, bias.as_slice::<i64>());
                let (lo, hi) = self.quant.act;
                let acc = g.dots(0i64, |a, xi, wi| a + (x[xi] * w[wi]) as i32 as i64);
                let res: Vec<i64> = acc
                    .into_iter()
                    .enumerate()
                    .map(|(k, a)| {
                        let (m, s) = self.quant.multipliers[k % oc];
                        mul_by_quantized_multiplier_i64(a + b[k % oc], m, s).clamp(lo, hi) as i64
                    })
                    .collect();
                put_ints(o, &res)?;
            }
            (t, _) => {
                // uint8 subtracts both zero points; int8 filters are symmetric (their zero point is not used)
                let (x, w, b) = (ints(i)?, ints(f)?, bias.as_slice::<i32>());
                let (xz, wz) = (i.zero_point() as i64, if t == TType::U8 { f.zero_point() as i64 } else { 0 });
                let (oz, (lo, hi)) = (o.zero_point(), self.quant.act);
                let acc = g.dots(0i32, |a, xi, wi| a.wrapping_add(((w[wi] - wz) * (x[xi] - xz)) as i32));
                let res: Vec<i64> = acc
                    .into_iter()
                    .enumerate()
                    .map(|(k, a)| {
                        let (m, s) = self.quant.multipliers[k % oc];
                        gemm_requantize(a.wrapping_add(b[k % oc]), m, s, t == TType::U8).wrapping_add(oz).clamp(lo, hi) as i64
                    })
                    .collect();
                put_ints(o, &res)?;
            }
        }
        Ok(())
    }
}

impl Conv2D {
    /// EvalHybrid / EvalHybridPerChannel: float input, 8-bit filter
    fn hybrid(&self, i: &Tensor, f: &Tensor, bias: &Tensor, o: &mut Tensor) -> Result<()> {
        let (lo, hi) = act_range_f(self.act);
        let b = bias.as_slice::<f32>();
        let w: Vec<i32> = f.data().iter().map(|&v| v as i8 as i32).collect();
        let x = i.as_slice::<f32>();
        let per_batch = x.len() / self.geom.batches.max(1);
        let per_channel = f.dtype == TType::I8 && f.scales.len() > 1 && f.scales.iter().any(|&s| s != f.scales[0]);
        let oc = self.geom.out_ch;
        let mut q = Vec::with_capacity(x.len());
        let (mut scales, mut zps) = (Vec::new(), Vec::new());
        for row in x.chunks(per_batch.max(1)) {
            let (v, s, zp) = super::dense::quantize_row(row, per_channel);
            q.extend(v);
            scales.push(s);
            zps.push(zp);
        }
        let v = o.as_mut_slice::<f32>();
        if per_channel {
            // HybridConvPerChannel: 2.5 runs it without dilation; padding holds the zero point, so it cancels
            let g = Geom { dilation: (1, 1), ..self.geom.clone() };
            let acc = g.dots(0i32, |a, xi, wi| a.wrapping_add(w[wi] * (q[xi] - zps[xi / per_batch])));
            let pixels = acc.len() / oc.max(1);
            for (k, a) in acc.into_iter().enumerate() {
                let batch = (k / oc) / (pixels / g.batches.max(1)).max(1);
                let r = a as f32 * (f.scales[k % oc] * scales[batch]) + b[k % oc];
                // cwiseMin(max) then cwiseMax(min)
                let r = if hi < r { hi } else { r };
                v[k] = if r < lo { lo } else { r };
            }
        } else {
            let acc = self.geom.dots(0i32, |a, xi, wi| a.wrapping_add(w[wi] * q[xi]));
            let pixels_per_batch = (acc.len() / oc.max(1)) / self.geom.batches.max(1);
            for (k, a) in acc.into_iter().enumerate() {
                let scale = scales[(k / oc) / pixels_per_batch.max(1)] * f.scale();
                v[k] = clamp_f((0.0 + a as f32 * scale) + b[k % oc], lo, hi);
            }
        }
        Ok(())
    }
}

/// depthwise_conv.cc: each output channel from one input channel (channel / depth multiplier). 2.5's x86 kernels
/// keep a running sum per output from its start value (the bias, or 0 for hybrid), taps in filter order; the 8-bit
/// ones requantize with the reference MultiplyByQuantizedMultiplier, so everything here matches exactly, float too.
struct DepthwiseConv2D {
    padding: Padding,
    stride: (i32, i32),
    /// from the options: only the int16 and hybrid kernels use it; the others divide the channel counts
    depth_multiplier: i32,
    act: Act,
    dilation: (i32, i32),
    geom: Geom,
    quant: Quant,
}

impl Kernel for DepthwiseConv2D {
    fn prepare(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        let (i, f, bias) = (io.input(0)?, io.input(1)?, io.opt(2));
        let o = &io.outs[0];
        // with a third input listed, 2.5 reads it as the bias, so an absent one (-1) fails
        if io.n_in() == 3 && bias.is_none() {
            return err("the bias input is missing");
        }
        if i.shape.len() != 4 || f.shape.len() != 4 || f.shape[0] != 1 {
            return err("DEPTHWISE_CONV_2D takes a 4-D input and a 1 x H x W x C filter");
        }
        let hybrid = i.dtype == TType::F32 && f.dtype == TType::I8;
        if !matches!(i.dtype, TType::F32 | TType::U8 | TType::I8 | TType::I16) || o.dtype != i.dtype || !(hybrid || f.dtype == i.dtype || i.dtype == TType::I16) {
            return err(format!("input {:?}, filter {:?}, output {:?} is not supported", i.dtype, f.dtype, o.dtype));
        }
        if i.dtype == TType::I16 && (i.zero_point() != 0 || o.zero_point() != 0) {
            return err("int16 needs zero points 0");
        }
        let oc = f.shape[3];
        if let Some(b) = bias {
            let ok = match i.dtype {
                TType::U8 | TType::I8 => b.dtype == TType::I32 && b.zero_point() == 0,
                TType::I16 => b.dtype == TType::I64 && b.zero_point() == 0,
                t => b.dtype == t,
            };
            if !ok || b.shape.len() != 1 || b.shape[0] != oc {
                return err("the bias must have the input's accumulator type and one value per output channel");
            }
        }
        let (fh, fw) = (f.shape[1], f.shape[2]);
        let (oh, ph, _) = out_size_and_pad(self.padding, i.shape[1] as i32, fh as i32, self.stride.0, self.dilation.0);
        let (ow, pw, _) = out_size_and_pad(self.padding, i.shape[2] as i32, fw as i32, self.stride.1, self.dilation.1);
        if oh < 0 || ow < 0 {
            return err("the filter is larger than the input");
        }
        self.geom = Geom {
            batches: i.shape[0],
            input: (i.shape[1], i.shape[2]),
            in_ch: i.shape[3],
            output: (oh as usize, ow as usize),
            out_ch: oc,
            filter: (fh, fw),
            stride: self.stride,
            dilation: self.dilation,
            pad: (ph, pw),
        };
        if i.dtype != TType::F32 {
            self.quant = conv_quant(i, f, bias, o, self.act, oc)?;
        }
        if hybrid && f.scales.len() != f.shape[(f.quantized_dimension as usize).min(3)] {
            return err("a hybrid filter needs one scale per entry of its quantized dimension");
        }
        io.outs[0].resize(vec![i.shape[0], oh as usize, ow as usize, oc]);
        Ok(())
    }

    fn eval(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        let (i, f, bias) = (io.input(0)?, io.input(1)?, io.opt(2));
        let o = &mut io.outs[0];
        let g = &self.geom;
        let (ic, oc) = (g.in_ch, g.out_ch);
        let hybrid = i.dtype == TType::F32 && f.dtype == TType::I8;
        let dm = if hybrid || i.dtype == TType::I16 {
            self.depth_multiplier.max(1) as usize
        } else {
            if ic == 0 || oc % ic != 0 {
                return err("the filter's channels must be a multiple of the input's");
            }
            oc / ic
        };
        let taps = g.taps();
        // acc = mac(acc, input index, filter index) over the taps, per (output pixel, channel)
        let dots = |init: &dyn Fn(usize) -> i64, mac: &dyn Fn(i64, usize, usize) -> i64| -> Vec<i64> {
            let mut res = Vec::with_capacity(taps.len() * oc);
            for t in &taps {
                for c in 0..oc {
                    res.push(t.iter().fold(init(c), |a, &(pixel, tap)| mac(a, pixel * ic + c / dm, tap * oc + c)));
                }
            }
            res
        };
        match (i.dtype, f.dtype) {
            (TType::F32, TType::F32) => {
                let (x, w) = (i.as_slice::<f32>(), f.as_slice::<f32>());
                let b = bias.map(|t| t.as_slice::<f32>());
                let (lo, hi) = act_range_f(self.act);
                let v = o.as_mut_slice::<f32>();
                for (p, t) in taps.iter().enumerate() {
                    for c in 0..oc {
                        let acc = t.iter().fold(b.map_or(0.0, |b| b[c]), |a, &(pixel, tap)| a + w[tap * oc + c] * x[pixel * ic + c / dm]);
                        // std::max(lo, std::min(hi, acc)): NaN becomes hi
                        let m = if acc < hi { acc } else { hi };
                        v[p * oc + c] = if lo < m { m } else { lo };
                    }
                }
            }
            (TType::F32, _) => {
                let x = i.as_slice::<f32>();
                let w: Vec<i64> = ints(f)?;
                let per_batch = x.len() / g.batches.max(1);
                let (mut q, mut scales, mut zps) = (Vec::with_capacity(x.len()), Vec::new(), Vec::new());
                for row in x.chunks(per_batch.max(1)) {
                    let (v, s, zp) = super::dense::quantize_row(row, true);
                    q.extend(v.into_iter().map(|v| v as i64));
                    scales.push(s);
                    zps.push(zp as i64);
                }
                let acc = dots(&|_| 0, &|a, xi, wi| (a as i32).wrapping_add((w[wi] * (q[xi] - zps[xi / per_batch])) as i32) as i64);
                let b = bias.map(|t| t.as_slice::<f32>());
                let (lo, hi) = act_range_f(self.act);
                let pixels = g.output.0 * g.output.1;
                let v = o.as_mut_slice::<f32>();
                for (k, a) in acc.into_iter().enumerate() {
                    let c = k % oc;
                    let r = (a as i32 as f32 * scales[k / oc / pixels.max(1)]) * f.scales[c] + b.map_or(0.0, |b| b[c]);
                    v[k] = clamp_f(r, lo, hi);
                }
            }
            (TType::I16, _) => {
                let (x, w) = (ints(i)?, ints(f)?);
                let b = bias.map(|t| t.as_slice::<i64>());
                let acc = dots(&|_| 0, &|a, xi, wi| a + w[wi] * x[xi]);
                let (lo, hi) = self.quant.act;
                let res: Vec<i64> = acc
                    .into_iter()
                    .enumerate()
                    .map(|(k, a)| {
                        let (m, s) = self.quant.multipliers[k % oc];
                        mul_by_quantized_multiplier_i64(a + b.map_or(0, |b| b[k % oc]), m, s).clamp(lo, hi) as i64
                    })
                    .collect();
                put_ints(o, &res)?;
            }
            (t, _) => {
                let (x, w) = (ints(i)?, ints(f)?);
                let b = bias.map(|t| t.as_slice::<i32>());
                let (xz, wz) = (i.zero_point() as i64, if t == TType::U8 { f.zero_point() as i64 } else { 0 });
                let acc = dots(&|c| b.map_or(0, |b| b[c]) as i64, &|a, xi, wi| (a as i32).wrapping_add(((w[wi] - wz) * (x[xi] - xz)) as i32) as i64);
                let (oz, (lo, hi)) = (o.zero_point(), self.quant.act);
                let res: Vec<i64> = acc
                    .into_iter()
                    .enumerate()
                    .map(|(k, a)| {
                        let (m, s) = self.quant.multipliers[k % oc];
                        mul_by_quantized_multiplier(a as i32, m, s).wrapping_add(oz).clamp(lo, hi) as i64
                    })
                    .collect();
                put_ints(o, &res)?;
            }
        }
        Ok(())
    }
}

/// PopulateConvolutionQuantizationParams: uint8 one multiplier from the float product of the scales (with the
/// bias scale check), int8 / int16 one per output channel from double products; the activation range
fn conv_quant(i: &Tensor, f: &Tensor, bias: Option<&Tensor>, o: &Tensor, act: Act, channels: usize) -> Result<Quant> {
    let per_channel = f.scales.len() > 1;
    if per_channel && (!matches!(i.dtype, TType::I8 | TType::I16) || f.dtype != TType::I8 || f.scales.len() != channels) {
        return err("per-channel filters need int8 or int16 input, int8 weights and one scale per output channel");
    }
    let multipliers = if i.dtype == TType::U8 {
        if let Some(b) = bias {
            if (i.scale() as f64 * f.scale() as f64 - b.scale() as f64).abs() / o.scale() as f64 > 0.02 {
                return err("the bias scale does not match input scale * filter scale");
            }
        }
        vec![quantize_multiplier((i.scale() * f.scale()) as f64 / o.scale() as f64); channels]
    } else {
        (0..channels)
            .map(|c| quantize_multiplier(i.scale() as f64 * f.scales[if per_channel { c } else { 0 }] as f64 / o.scale() as f64))
            .collect()
    };
    Ok(Quant { multipliers, act: act_range_q(act, o) })
}
