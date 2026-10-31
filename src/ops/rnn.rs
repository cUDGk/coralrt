// From TensorFlow Lite 2.5 (Copyright The TensorFlow Authors, Apache License 2.0), translated and changed; see NOTICE.
//! Simple recurrent cells (basic_rnn.cc, unidirectional_sequence_rnn.cc, bidirectional_sequence_rnn.cc): every time
//! step is kernel_utils::RnnBatchStep, in float or hybrid (8-bit weights, the float input and state quantized per
//! batch row). The sums run in the portable tensor_utils' order, so float matches exactly too. The hidden states
//! are variable inputs, updated in place.

use super::tensor_utils::*;
use super::util::*;
use super::{Io, Kernel};
use crate::interp::Env;
use crate::tensor::{TType, Tensor};
use crate::tflite::{Act, OpSpec};
use crate::Result;

pub fn create(s: &OpSpec) -> Result<Option<Box<dyn Kernel>>> {
    let flag = |i: usize| opt(s, i, 0u8) != 0;
    Ok(Some(match s.code {
        24 => Box::new(Rnn { kind: Kind::Basic, act: opt_act(s, 0)?, asymmetric: flag(1), sums: [None, None] }),
        35 => Box::new(Rnn { kind: Kind::Sequence { time_major: flag(0) }, act: opt_act(s, 1)?, asymmetric: flag(2), sums: [None, None] }),
        46 => Box::new(Rnn { kind: Kind::Bidirectional { time_major: flag(0), merge: flag(2) }, act: opt_act(s, 1)?, asymmetric: flag(3), sums: [None, None] }),
        27 => Box::new(Svdf { rank: opt(s, 0, 0), act: opt_act(s, 1)?, asymmetric: flag(2), scales: [(0, 0); 2], sums: None, time: None }),
        _ => return Ok(None),
    }))
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Basic,
    Sequence { time_major: bool },
    Bidirectional { time_major: bool, merge: bool },
}

struct Rnn {
    kind: Kind,
    act: Act,
    asymmetric: bool,
    /// hybrid, asymmetric: the row sums of each direction's input, aux and recurrent weights, worked out on the
    /// first step after each prepare as 2.5 does
    sums: [Option<RowSums>; 2],
}

#[derive(Clone)]
struct RowSums {
    input: Vec<i32>,
    aux: Vec<i32>,
    recurrent: Vec<i32>,
}

/// weights as float, or as int8 (8-bit bytes read as int8) with their scale
enum Weights {
    F32(Vec<f32>),
    I8(Vec<i32>, f32),
}

impl Weights {
    fn of(t: &Tensor) -> Weights {
        if t.dtype == TType::F32 { Weights::F32(t.as_slice::<f32>().to_vec()) } else { Weights::I8(t.data().iter().map(|&b| b as i8 as i32).collect(), t.scale()) }
    }
}

/// one direction: input, optional aux and recurrent weights, bias, sizes
struct Cell {
    input: Weights,
    aux: Option<Weights>,
    recurrent: Weights,
    bias: Vec<f32>,
    units: usize,
    input_size: usize,
    aux_size: usize,
}

impl Rnn {
    /// RnnBatchStep over `batch` rows: x and aux hold the rows back to back, h the hidden state (updated), each
    /// output row starts `stride` after the previous
    #[allow(clippy::too_many_arguments)]
    fn step(&mut self, dir: usize, cell: &Cell, x: &[f32], aux: Option<&[f32]>, h: &mut [f32], out: &mut [f32], stride: usize, batch: usize) {
        let (u, is, as_) = (cell.units, cell.input_size, cell.aux_size);
        for k in 0..batch {
            out[k * stride..k * stride + u].copy_from_slice(&cell.bias);
        }
        match (&cell.input, &cell.recurrent) {
            (Weights::F32(wi), Weights::F32(wr)) => {
                for k in 0..batch {
                    mbvma_f32(wi, u, is, &x[k * is..], &mut out[k * stride..k * stride + u]);
                }
                if let (Some(Weights::F32(wa)), Some(a)) = (&cell.aux, aux) {
                    if as_ > 0 {
                        for k in 0..batch {
                            mbvma_f32(wa, u, as_, &a[k * as_..], &mut out[k * stride..k * stride + u]);
                        }
                    }
                }
                for k in 0..batch {
                    mbvma_f32(wr, u, u, &h[k * u..], &mut out[k * stride..k * stride + u]);
                }
            }
            (Weights::I8(wi, si), Weights::I8(wr, sr)) => {
                if self.asymmetric && self.sums[dir].is_none() {
                    let aux = match &cell.aux {
                        Some(Weights::I8(wa, _)) if aux.is_some() => row_sums(wa, as_),
                        _ => Vec::new(),
                    };
                    self.sums[dir] = Some(RowSums { input: row_sums(wi, is), aux, recurrent: row_sums(wr, u) });
                }
                let sums = self.sums[dir].clone();
                // each part: skipped when its whole batch is zero; quantized per row; sums of int8 products
                let mut part = |w: &[i32], scale: f32, v: &[f32], n: usize, rs: Option<&Vec<i32>>| {
                    if is_zero(&v[..batch * n]) {
                        return;
                    }
                    let (q, sc, zp) = batch_quantize(&v[..batch * n], n, self.asymmetric);
                    for k in 0..batch {
                        let offset = rs.filter(|_| self.asymmetric).map(|r| (r.as_slice(), zp[k]));
                        mbvma_i8(w, u, n, &q[k * n..], sc[k] * scale, offset, &mut out[k * stride..k * stride + u]);
                    }
                };
                part(wi, *si, x, is, sums.as_ref().map(|s| &s.input));
                if let (Some(Weights::I8(wa, sa)), Some(a)) = (&cell.aux, aux) {
                    part(wa, *sa, a, as_, sums.as_ref().map(|s| &s.aux));
                }
                part(wr, *sr, h, u, sums.as_ref().map(|s| &s.recurrent));
            }
            _ => {}
        }
        for k in 0..batch {
            let o = &mut out[k * stride..k * stride + u];
            apply_activation(o, self.act);
            h[k * u..(k + 1) * u].copy_from_slice(o);
        }
    }

    /// the cell of the weights at inputs w, r, b (and aux weights at a)
    fn cell(io: &Io, w: usize, r: usize, b: usize, a: Option<usize>, aux_size: usize) -> Result<Cell> {
        let (wt, rt, bt) = (io.input(w)?, io.input(r)?, io.input(b)?);
        Ok(Cell {
            input: Weights::of(wt),
            aux: a.and_then(|k| io.opt(k)).map(Weights::of),
            recurrent: Weights::of(rt),
            bias: bt.as_slice::<f32>().to_vec(),
            units: wt.shape[0],
            input_size: wt.shape[1],
            aux_size,
        })
    }
}

impl Kernel for Rnn {
    fn prepare(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        self.sums = [None, None];
        let i = io.input(0)?;
        // the hidden states are taken out of the inputs (variables): their shapes first
        let states: Vec<Vec<usize>> = self.mutated_inputs().to_vec().iter().map(|&k| io.var(k).map(|t| t.shape.clone())).collect::<Result<_>>()?;
        let check = |w: usize, r: usize, b: usize, h: usize, batch: usize, input_size: usize| -> Result<usize> {
            let (wt, rt, bt) = (io.input(w)?, io.input(r)?, io.input(b)?);
            let hs = &states[if h == 8 { 1 } else { 0 }];
            let u = wt.shape[0];
            let ok = wt.shape.len() == 2
                && wt.shape[1] == input_size
                && bt.shape.first() == Some(&u)
                && rt.shape == [u, u]
                && wt.dtype == rt.dtype
                && matches!(wt.dtype, TType::F32 | TType::U8 | TType::I8)
                && bt.dtype == TType::F32
                && *hs == [batch, u];
            if !ok {
                return err("the RNN weights, bias and state do not fit together");
            }
            Ok(u)
        };
        if i.dtype != TType::F32 {
            return err("RNN takes float32 input");
        }
        match self.kind {
            Kind::Basic => {
                if i.shape.len() != 2 {
                    return err("RNN takes [batch, input]");
                }
                let u = check(1, 2, 3, 4, i.shape[0], i.shape[1])?;
                io.outs[0].resize(vec![i.shape[0], u]);
            }
            Kind::Sequence { time_major } => {
                if i.shape.len() != 3 {
                    return err("UNIDIRECTIONAL_SEQUENCE_RNN takes a 3-D input");
                }
                let batch = if time_major { i.shape[1] } else { i.shape[0] };
                let u = check(1, 2, 3, 4, batch, i.shape[2])?;
                io.outs[0].resize(vec![i.shape[0], i.shape[1], u]);
            }
            Kind::Bidirectional { time_major, merge } => {
                if i.shape.len() != 3 || io.outs.len() != if merge { 1 } else { 2 } {
                    return err("BIDIRECTIONAL_SEQUENCE_RNN takes a 3-D input and has one output (two unless merged)");
                }
                let batch = if time_major { i.shape[1] } else { i.shape[0] };
                let fu = check(1, 2, 3, 4, batch, i.shape[2])?;
                let bu = check(5, 6, 7, 8, batch, i.shape[2])?;
                let s = |n: usize| vec![i.shape[0], i.shape[1], n];
                io.outs[0].resize(s(if merge { fu + bu } else { fu }));
                if !merge {
                    // 2.5 shapes the backward output [batch, time, units] even when time major
                    let t = if time_major { i.shape[0] } else { i.shape[1] };
                    io.outs[1].resize(vec![batch, t, bu]);
                }
            }
        }
        Ok(())
    }

    fn eval(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        let x = io.input(0)?.as_slice::<f32>().to_vec();
        let shape = io.input(0)?.shape.clone();
        match self.kind {
            Kind::Basic => {
                let cell = Self::cell(io, 1, 2, 3, None, 0)?;
                let mut h = io.var(4)?.as_slice::<f32>().to_vec();
                let mut out = vec![0.0f32; shape[0] * cell.units];
                self.step(0, &cell, &x, None, &mut h, &mut out, cell.units, shape[0]);
                io.var(4)?.as_mut_slice::<f32>().copy_from_slice(&h);
                io.outs[0].as_mut_slice::<f32>().copy_from_slice(&out);
            }
            Kind::Sequence { time_major } => {
                let cell = Self::cell(io, 1, 2, 3, None, 0)?;
                let (u, is) = (cell.units, cell.input_size);
                let (b, t) = if time_major { (shape[1], shape[0]) } else { (shape[0], shape[1]) };
                let mut h = io.var(4)?.as_slice::<f32>().to_vec();
                let mut out = vec![0.0f32; b * t * u];
                if time_major {
                    for s in 0..t {
                        self.step(0, &cell, &x[s * is * b..], None, &mut h, &mut out[s * u * b..], u, b);
                    }
                } else {
                    for bb in 0..b {
                        for s in 0..t {
                            let at = bb * u * t + s * u;
                            self.step(0, &cell, &x[bb * is * t + s * is..], None, &mut h[bb * u..(bb + 1) * u], &mut out[at..], u, 1);
                        }
                    }
                }
                io.var(4)?.as_mut_slice::<f32>().copy_from_slice(&h);
                io.outs[0].as_mut_slice::<f32>().copy_from_slice(&out);
            }
            Kind::Bidirectional { time_major, merge } => self.bidirectional(io, &x, &shape, time_major, merge)?,
        }
        Ok(())
    }

    fn mutated_inputs(&self) -> &[usize] {
        if matches!(self.kind, Kind::Bidirectional { .. }) { &[4, 8] } else { &[4] }
    }
}

impl Rnn {
    /// bidirectional_sequence_rnn.cc's EvalFloat / EvalHybrid, with their pointer arithmetic as written: the
    /// batch-major backward pass reads `input` (never the second input), aux rows step by the input size in some
    /// paths, and hybrid time-major runs the whole sequence max_time times over
    fn bidirectional(&mut self, io: &mut Io, x: &[f32], shape: &[usize], time_major: bool, merge: bool) -> Result<()> {
        let aux_t = io.opt(9).map(|t| (t.as_slice::<f32>().to_vec(), t.shape[2]));
        let use_aux = io.opt(10).is_some();
        // non-stacking mode: the aux input is the backward pass's input
        let (bw_x, aux) = match aux_t {
            Some((a, _)) if !use_aux => (a, None),
            other => (x.to_vec(), other),
        };
        let aux_size = aux.as_ref().map_or(0, |a| a.1);
        let fw = Self::cell(io, 1, 2, 3, Some(10), aux_size)?;
        let bw = Self::cell(io, 5, 6, 7, Some(11), aux_size)?;
        let hybrid = matches!(fw.input, Weights::I8(..));
        let (fu, bu, is) = (fw.units, bw.units, fw.input_size);
        let (b, t) = if time_major { (shape[1], shape[0]) } else { (shape[0], shape[1]) };
        let (fstep, bstep) = if merge { (fu + bu, fu + bu) } else { (fu, bu) };
        let mut fh = io.var(4)?.as_slice::<f32>().to_vec();
        let mut bh = io.var(8)?.as_slice::<f32>().to_vec();
        let mut fo = vec![0.0f32; b * t * fstep];
        let mut bo = vec![0.0f32; if merge { 0 } else { b * t * bu }];
        let a = aux.as_ref().map(|(v, _)| v.as_slice());
        // the aux rows are found with the input's size in places: with another size 2.5 reads past the aux input
        if a.is_some_and(|v| v.len() < b * t * is.max(aux_size)) {
            return err("an aux input of another size than the input is read past its end by 2.5");
        }
        if time_major {
            for _ in 0..if hybrid { t } else { 1 } {
                for s in 0..t {
                    let ap = a.map(|v| &v[s * is * b..]);
                    self.step(0, &fw, &x[s * is * b..], ap, &mut fh, &mut fo[s * fstep * b..], fstep, b);
                }
                for s in (0..t).rev() {
                    let ap = a.map(|v| &v[s * is * b..]);
                    let (target, base) = if merge { (&mut fo, fu) } else { (&mut bo, 0) };
                    self.step(1, &bw, &bw_x[s * is * b..], ap, &mut bh, &mut target[base + s * bstep * b..], bstep, b);
                }
            }
        } else {
            // the aux rows step by the aux size in float, by the input size in hybrid
            let astep = if hybrid { is } else { aux_size };
            for bb in 0..b {
                for s in 0..t {
                    let ap = a.map(|v| &v[bb * astep * t + s * astep..]);
                    self.step(0, &fw, &x[bb * is * t + s * is..], ap, &mut fh[bb * fu..(bb + 1) * fu], &mut fo[bb * fstep * t + s * fstep..], fstep, 1);
                }
                for s in (0..t).rev() {
                    let ap = a.map(|v| &v[bb * astep * t + s * astep..]);
                    let (target, base) = if merge { (&mut fo, bb * bstep * t + fu) } else { (&mut bo, bb * bstep * t) };
                    self.step(1, &bw, &x[bb * is * t + s * is..], ap, &mut bh[bb * bu..(bb + 1) * bu], &mut target[base + s * bstep..], bstep, 1);
                }
            }
        }
        io.var(4)?.as_mut_slice::<f32>().copy_from_slice(&fh);
        io.var(8)?.as_mut_slice::<f32>().copy_from_slice(&bh);
        io.outs[0].as_mut_slice::<f32>().copy_from_slice(&fo);
        if !merge {
            io.outs[1].as_mut_slice::<f32>().copy_from_slice(&bo);
        }
        Ok(())
    }
}

/// SVDF (svdf.cc, reference_ops::Eval*SVDF): the input through the feature weights into the newest slot of each
/// filter's memory (the state shifted by one first, flat across filters), then the memory through the time
/// weights, summed over the rank, plus bias. Float and hybrid in float; int8 with an int16 state.
struct Svdf {
    rank: i32,
    act: Act,
    asymmetric: bool,
    /// int8: the feature and time multipliers, from scale products worked out in float
    scales: [(i32, i32); 2],
    /// hybrid, asymmetric: the feature weights' row sums, once after each prepare
    sums: Option<Vec<i32>>,
    /// hybrid: the time weights dequantized on the first eval and kept for good (2.5 never redoes it)
    time: Option<Vec<f32>>,
}

impl Kernel for Svdf {
    fn prepare(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        let state = io.var(4)?.shape.clone();
        let (i, wf, wt, bias) = (io.input(0)?, io.input(1)?, io.input(2)?, io.opt(3));
        if !matches!(i.dtype, TType::F32 | TType::I8) || i.shape.len() != 2 || wf.shape.len() != 2 || wt.shape.len() != 2 {
            return err("SVDF takes a float32 or int8 [batch, input] and 2-D weights");
        }
        let filters = wf.shape[0];
        if self.rank <= 0 || filters % self.rank as usize != 0 {
            return err("the rank must divide the number of filters");
        }
        let (batch, units, memory) = (i.shape[0], filters / self.rank as usize, wt.shape[1]);
        let ok = i.shape[1] == wf.shape[1] && wt.shape[0] == filters && bias.is_none_or(|b| b.shape.first() == Some(&units)) && state == [batch, memory * filters];
        if !ok {
            return err("the SVDF weights, bias and state do not fit together");
        }
        // 2.5 picks the path by the feature weights' type and reads the rest as that path's types without checking
        let st = io.var(4)?.dtype;
        let b = bias.map(|b| b.dtype);
        let fits = match (i.dtype, wf.dtype) {
            (TType::F32, TType::F32) => wt.dtype == TType::F32 && st == TType::F32 && b.is_none_or(|b| b == TType::F32),
            (TType::F32, TType::I8 | TType::U8) => matches!(wt.dtype, TType::I8 | TType::U8) && st == TType::F32 && b.is_none_or(|b| b == TType::F32),
            (TType::I8, TType::I8 | TType::U8) => wt.dtype == TType::I16 && st == TType::I16 && b.is_none_or(|b| b == TType::I32) && io.outs[0].dtype == TType::I8,
            _ => false,
        };
        if !fits {
            return err("SVDF supports float32, hybrid (float32 with int8 weights) and int8 with int16 time weights and state");
        }
        if i.dtype == TType::I8 {
            let st = io.var(4)?.scale();
            let (i, wf, wt) = (io.input(0)?, io.input(1)?, io.input(2)?);
            let o = io.outs[0].scale();
            self.scales = [quantize_multiplier((i.scale() * wf.scale() / st) as f64), quantize_multiplier((st * wt.scale() / o) as f64)];
        }
        self.sums = None;
        io.outs[0].resize(vec![batch, units]);
        Ok(())
    }

    fn eval(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        let (i, wf, wt, bias) = (io.input(0)?, io.input(1)?, io.input(2)?, io.opt(3));
        let (batch, isz, filters, memory) = (i.shape[0], i.shape[1], wf.shape[0], wt.shape[1]);
        let (rank, units) = (self.rank as usize, filters / self.rank as usize);
        if i.dtype == TType::I8 {
            if self.act != Act::Relu {
                return err("int8 SVDF needs a RELU activation");
            }
            // uint8 feature weights are read as int8
            let w: Vec<i64> = wf.data().iter().map(|&v| v as i8 as i64).collect();
            let (x, tw) = (ints(i)?, ints(wt)?);
            let b = bias.map(|t| t.as_slice::<i32>().to_vec());
            let (izp, ozp) = (i.zero_point() as i64, io.outs[0].zero_point());
            let [(m1, s1), (m2, s2)] = self.scales;
            let st = io.var(4)?.as_mut_slice::<i16>();
            st.copy_within(1.., 0);
            for bb in 0..batch {
                for r in 0..filters {
                    let dot = (0..isz).fold(0i32, |a, c| a.wrapping_add((w[r * isz + c] * (x[bb * isz + c] - izp)) as i32));
                    st[(bb * filters + r) * memory + memory - 1] = mul_by_quantized_multiplier(dot, m1, s1).clamp(-32768, 32767) as i16;
                }
            }
            let scratch: Vec<i32> = (0..batch * filters)
                .map(|k| {
                    let f = k % filters;
                    (0..memory).fold(0i32, |a, m| a.wrapping_add(tw[f * memory + m] as i32 * st[k * memory + m] as i32))
                })
                .collect();
            let out: Vec<i64> = (0..batch * units)
                .map(|o| {
                    let mut v = scratch[o * rank..(o + 1) * rank].iter().fold(0i32, |a, &s| a.wrapping_add(s));
                    v = v.wrapping_add(b.as_ref().map_or(0, |b| b[o % units]));
                    mul_by_quantized_multiplier(v, m2, s2).wrapping_add(ozp).clamp(-128, 127) as i64
                })
                .collect();
            return put_ints(&mut io.outs[0], &out);
        }
        let x = i.as_slice::<f32>();
        // the time weights in float (hybrid: their int8 bytes times their scale)
        let tw: Vec<f32> = if wt.dtype == TType::F32 {
            wt.as_slice::<f32>().to_vec()
        } else {
            self.time.get_or_insert_with(|| wt.data().iter().map(|&v| v as i8 as f32 * wt.scale()).collect()).clone()
        };
        let mut scratch = vec![0.0f32; batch * filters];
        if wf.dtype == TType::F32 {
            let w = wf.as_slice::<f32>();
            for bb in 0..batch {
                mbvma_f32(w, filters, isz, &x[bb * isz..], &mut scratch[bb * filters..(bb + 1) * filters]);
            }
        } else if !is_zero(&x[..batch * isz]) {
            let w: Vec<i32> = wf.data().iter().map(|&v| v as i8 as i32).collect();
            if self.asymmetric && self.sums.is_none() {
                self.sums = Some(row_sums(&w, isz));
            }
            let (q, sc, zp) = batch_quantize(&x[..batch * isz], isz, self.asymmetric);
            for bb in 0..batch {
                let offset = self.sums.as_ref().filter(|_| self.asymmetric).map(|r| (r.as_slice(), zp[bb]));
                mbvma_i8(&w, filters, isz, &q[bb * isz..], sc[bb] * wf.scale(), offset, &mut scratch[bb * filters..(bb + 1) * filters]);
            }
        }
        let b = bias.map(|t| t.as_slice::<f32>().to_vec());
        let st = io.var(4)?.as_mut_slice::<f32>();
        st.copy_within(1.., 0);
        for (k, &v) in scratch.iter().enumerate() {
            st[k * memory + memory - 1] = v;
        }
        // ApplyTimeWeightsBiasAndActivation
        for (k, s) in scratch.iter_mut().enumerate() {
            let f = k % filters;
            *s = (0..memory).fold(0.0f32, |a, m| a + tw[f * memory + m] * st[k * memory + m]);
        }
        let out = io.outs[0].as_mut_slice::<f32>();
        for (o, d) in out.iter_mut().enumerate() {
            *d = scratch[o * rank..(o + 1) * rank].iter().fold(0.0f32, |a, &s| a + s);
            if let Some(b) = &b {
                *d += b[o % units];
            }
        }
        apply_activation(out, self.act);
        Ok(())
    }

    fn mutated_inputs(&self) -> &[usize] {
        &[4]
    }
}
