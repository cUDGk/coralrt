// From TensorFlow Lite 2.5 (Copyright The TensorFlow Authors, Apache License 2.0), translated and changed; see NOTICE.
//! Reductions and scans along axes (reduce.cc, arg_min_max.cc, cumsum.cc, segment_sum.cc, topk_v2.cc).

use super::util::*;
use super::{Io, Kernel};
use crate::interp::Env;
use crate::tensor::{Alloc, Elem, TType, Tensor};
use crate::tflite::OpSpec;
use crate::Result;

pub fn create(s: &OpSpec) -> Result<Option<Box<dyn Kernel>>> {
    let keep_dims = opt(s, 0, 0u8) != 0;
    Ok(Some(match s.code {
        56 => Box::new(ArgMinMax { max: true }),
        79 => Box::new(ArgMinMax { max: false }),
        40 => Box::new(Reduce { op: RedOp::Mean, keep_dims }),
        74 => Box::new(Reduce { op: RedOp::Sum, keep_dims }),
        81 => Box::new(Reduce { op: RedOp::Prod, keep_dims }),
        82 => Box::new(Reduce { op: RedOp::Max, keep_dims }),
        89 => Box::new(Reduce { op: RedOp::Min, keep_dims }),
        91 => Box::new(Reduce { op: RedOp::Any, keep_dims }),
        128 => Box::new(CumSum { exclusive: opt(s, 0, 0u8) != 0, reverse: opt(s, 1, 0u8) != 0 }),
        125 => Box::new(SegmentSum),
        48 => Box::new(TopK),
        _ => return Ok(None),
    }))
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum RedOp {
    Mean,
    Sum,
    Prod,
    Max,
    Min,
    Any,
}

struct Reduce {
    op: RedOp,
    keep_dims: bool,
}

/// the axes as given (for the output shape) and resolved: non-negative, without repeats (ResolveAxis)
fn axes(io: &Io) -> Result<(Vec<i64>, Vec<usize>)> {
    let rank = io.input(0)?.shape.len();
    let given = index_values(io.input(1)?)?;
    let mut resolved: Vec<usize> = Vec::new();
    for &a in &given {
        let r = if a < 0 { a + rank as i64 } else { a };
        if r < 0 || r as usize >= rank.max(1) {
            return err(format!("axis {a} out of range for rank {rank}"));
        }
        if !resolved.contains(&(r as usize)) {
            resolved.push(r as usize);
        }
    }
    Ok((given, resolved))
}

impl Reduce {
    /// ResizeOutputTensor (reduce.cc)
    fn out_shape(&self, io: &Io) -> Result<Vec<usize>> {
        let s = &io.input(0)?.shape;
        if s.is_empty() {
            return Ok(vec![]);
        }
        let (given, _) = axes(io)?;
        let is_axis = |d: usize| given.iter().any(|&a| a == d as i64 || a + s.len() as i64 == d as i64);
        Ok(if self.keep_dims {
            (0..s.len()).map(|d| if is_axis(d) { 1 } else { s[d] }).collect()
        } else {
            (0..s.len()).filter(|&d| !is_axis(d)).map(|d| s[d]).collect()
        })
    }
}

/// for every input element in row-major order, the output element it reduces into (ReducedOutputOffset)
fn out_offsets(shape: &[usize], reduced: &[usize]) -> Vec<usize> {
    let kept: Vec<usize> = (0..shape.len()).filter(|d| !reduced.contains(d)).collect();
    let mut st = vec![1; kept.len()];
    for k in (0..kept.len().saturating_sub(1)).rev() {
        st[k] = st[k + 1] * shape[kept[k + 1]];
    }
    let mut v = Vec::with_capacity(shape.iter().product());
    super::movement::each_index(shape, |_, idx| v.push(kept.iter().zip(&st).map(|(&d, &s)| idx[d] * s).sum()));
    v
}

/// ReduceGeneric: init, then reducer(current, in) over the input in order
fn reduce_generic<T: Elem>(i: &Tensor, o: &mut Tensor, offs: &[usize], init: T, f: impl Fn(T, T) -> T) {
    let v = o.as_mut_slice::<T>();
    v.fill(init);
    for (&x, &k) in i.as_slice::<T>().iter().zip(offs) {
        v[k] = f(v[k], x);
    }
}

/// sums per output element in input order (ReduceSumImpl) as i64 / f64-free f32
fn sums_i64(vals: &[i64], offs: &[usize], n_out: usize) -> Vec<i64> {
    let mut s = vec![0i64; n_out];
    for (&x, &k) in vals.iter().zip(offs) {
        s[k] = s[k].wrapping_add(x);
    }
    s
}

impl Kernel for Reduce {
    fn prepare(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        if io.input(1)?.alloc != Alloc::Constant {
            io.outs[0].alloc = Alloc::Dynamic;
            return Ok(());
        }
        let s = self.out_shape(io)?;
        io.outs[0].resize(s);
        Ok(())
    }

    fn eval(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        let s = self.out_shape(io)?;
        let (given, reduced) = axes(io)?;
        let i = io.input(0)?;
        let o = &mut io.outs[0];
        o.resize(s);
        if i.shape.contains(&0) {
            return Ok(());
        }
        let offs = out_offsets(&i.shape, &reduced);
        let n_out = o.elements();
        let count: usize = reduced.iter().map(|&d| i.shape[d]).product();
        let t = i.dtype;
        let quantized = matches!(t, TType::U8 | TType::I8 | TType::I16);
        let same_q = i.scale() == o.scale() && i.zero_point() == o.zero_point();
        // MEAN over height and width of a 4-D tensor, keeping dims: kernels of their own
        let hw = self.op == RedOp::Mean && self.keep_dims && i.shape.len() == 4 && reduced.len() == 2 && reduced.contains(&1) && reduced.contains(&2);
        match (self.op, t) {
            (RedOp::Mean, TType::F32) if hw => {
                let (b, h, w, c) = (i.shape[0], i.shape[1], i.shape[2], i.shape[3]);
                let (x, v) = (i.as_slice::<f32>(), o.as_mut_slice::<f32>());
                for ob in 0..b {
                    for d in 0..c {
                        let mut value = 0.0f32;
                        for ih in 0..h {
                            for iw in 0..w {
                                value += x[((ob * h + ih) * w + iw) * c + d];
                            }
                        }
                        v[ob * c + d] = value / (w * h) as f32;
                    }
                }
            }
            (RedOp::Mean, TType::F32) if given == [i.shape.len() as i64 - 1] => {
                // MeanGeneral: Eigen's column sums over the last dimension (only for the axis written as rank - 1)
                let n = i.shape[i.shape.len() - 1];
                let (x, v) = (i.as_slice::<f32>(), o.as_mut_slice::<f32>());
                for (r, d) in v.iter_mut().enumerate() {
                    *d = super::eigen::sum(&x[r * n..(r + 1) * n], r * n) / n as f32;
                }
            }
            (RedOp::Mean, TType::F32) => {
                let mut sums = vec![0.0f32; n_out];
                for (&x, &k) in i.as_slice::<f32>().iter().zip(&offs) {
                    sums[k] += x;
                }
                for (d, s) in o.as_mut_slice::<f32>().iter_mut().zip(&sums) {
                    *d = s / count as f32;
                }
            }
            (RedOp::Mean, TType::I32 | TType::I64) => {
                let sums = sums_i64(&ints(i)?, &offs, n_out);
                put_ints(o, &sums.iter().map(|&s| if count > 0 { s / count as i64 } else { 0 }).collect::<Vec<_>>())?;
            }
            (RedOp::Mean, TType::U8 | TType::I8) if hw => {
                // optimized_ops::Mean / optimized_integer_ops::Mean: bias and scale in float
                let (b, h, w, c) = (i.shape[0], i.shape[1], i.shape[2], i.shape[3]);
                let n = (w * h) as f32;
                let bias = o.zero_point() - (i.zero_point() as f32 * i.scale() / o.scale()) as i32;
                let (m, shift) = quantize_multiplier((i.scale() / (n * o.scale())) as f64);
                let (lo, hi) = type_range(t);
                let x = ints(i)?;
                let mut out = vec![0i64; n_out];
                for ob in 0..b {
                    for d in 0..c {
                        let mut acc = 0i32;
                        for k in 0..h * w {
                            acc += x[(ob * h * w + k) * c + d] as i32;
                        }
                        out[ob * c + d] = (mul_by_quantized_multiplier(acc, m, shift) + bias).clamp(lo, hi) as i64;
                    }
                }
                put_ints(o, &out)?;
            }
            (RedOp::Mean, TType::I16) if hw => {
                // reference_integer_ops::Mean with PrepareMeanOrSum's multiplier (in double)
                let (b, h, w, c) = (i.shape[0], i.shape[1], i.shape[2], i.shape[3]);
                let n = (w * h) as i32;
                let (m, shift) = quantize_multiplier(i.scale() as f64 / o.scale() as f64);
                let (izp, ozp) = (i.zero_point(), o.zero_point());
                let x = ints(i)?;
                let mut out = vec![0i64; n_out];
                for ob in 0..b {
                    for d in 0..c {
                        let mut acc = 0i32;
                        for k in 0..h * w {
                            acc += x[(ob * h * w + k) * c + d] as i32 - izp;
                        }
                        acc = mul_by_quantized_multiplier(acc, m, shift);
                        acc = if acc > 0 { (acc + n / 2) / n } else { (acc - n / 2) / n };
                        out[ob * c + d] = (acc + ozp).clamp(-32768, 32767) as i64;
                    }
                }
                put_ints(o, &out)?;
            }
            (RedOp::Mean, _) if quantized && same_q => {
                let sums = sums_i64(&ints(i)?, &offs, n_out);
                put_ints(o, &sums.iter().map(|&s| (s as i32 / count as i32) as i64).collect::<Vec<_>>())?;
            }
            (RedOp::Mean | RedOp::Sum, _) if quantized && (self.op == RedOp::Mean || (!same_q && t != TType::I16)) => {
                // QuantizedMeanOrSum
                let sums = sums_i64(&ints(i)?, &offs, n_out);
                let scale = i.scale() / o.scale();
                let (izp, ozp) = (i.zero_point(), o.zero_point());
                let (lo, hi) = type_range(t);
                let out: Vec<i64> = if self.op == RedOp::Sum {
                    let bias = -izp as f32 * scale * count as f32;
                    sums.iter().map(|&s| (((s as i32 as f32) * scale + bias).round() as i32 + ozp) as i64).collect()
                } else {
                    let bias = -izp as f32 * scale;
                    sums.iter()
                        .map(|&s| {
                            let mean = s as i32 as f32 / count as f32;
                            let r = ((mean * scale + bias).round() + ozp as f32).min(hi as f32).max(lo as f32);
                            r as i64
                        })
                        .collect()
                };
                put_ints(o, &out)?;
            }
            (RedOp::Sum | RedOp::Prod | RedOp::Max | RedOp::Min, _) if quantized && !same_q => {
                return err("input and output quantization must match");
            }
            (RedOp::Sum | RedOp::Prod | RedOp::Max | RedOp::Min, _) => {
                let op = self.op;
                macro_rules! generic {
                    ($t:ty, $add:expr, $mul:expr, $lowest:expr, $highest:expr) => {
                        match op {
                            RedOp::Sum => reduce_generic::<$t>(i, o, &offs, Default::default(), $add),
                            RedOp::Prod => reduce_generic::<$t>(i, o, &offs, 1 as $t, $mul),
                            RedOp::Max => reduce_generic::<$t>(i, o, &offs, $lowest, |c, x| if x > c { x } else { c }),
                            _ => reduce_generic::<$t>(i, o, &offs, $highest, |c, x| if x < c { x } else { c }),
                        }
                    };
                }
                match t {
                    TType::F32 => generic!(f32, |c, x| x + c, |c, x| x * c, f32::MIN, f32::MAX),
                    TType::I32 => generic!(i32, |c: i32, x| x.wrapping_add(c), |c: i32, x| x.wrapping_mul(c), i32::MIN, i32::MAX),
                    TType::I64 => generic!(i64, |c: i64, x| x.wrapping_add(c), |c: i64, x| x.wrapping_mul(c), i64::MIN, i64::MAX),
                    TType::U8 => generic!(u8, |c: u8, x| x.wrapping_add(c), |c: u8, x| x.wrapping_mul(c), u8::MIN, u8::MAX),
                    TType::I8 => generic!(i8, |c: i8, x| x.wrapping_add(c), |c: i8, x| x.wrapping_mul(c), i8::MIN, i8::MAX),
                    TType::I16 => generic!(i16, |c: i16, x| x.wrapping_add(c), |c: i16, x| x.wrapping_mul(c), i16::MIN, i16::MAX),
                    d => return err(format!("{d:?} is not supported")),
                }
            }
            (RedOp::Any, TType::Bool) => reduce_generic::<u8>(i, o, &offs, 0, |c, x| (c != 0 || x != 0) as u8),
            (op, d) => return err(format!("{op:?} on {d:?} is not supported")),
        }
        Ok(())
    }
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
        let (outer, len, inner) = around(&i.shape, Self::axis(io)?);
        let best = match i.dtype {
            TType::F32 | TType::F64 => along(&f64s(i)?, outer, len, inner, self.max),
            _ => along(&ints(i)?, outer, len, inner, self.max),
        };
        put_ints(&mut io.outs[0], &best)
    }
}

/// (elements before, along, after) the axis
fn around(shape: &[usize], ax: usize) -> (usize, usize, usize) {
    (shape[..ax].iter().product(), shape[ax], shape[ax + 1..].iter().product())
}

/// cumsum.cc (Eigen's scan, one thread): along the axis from 0 in order, from the end when reversed; exclusive
/// writes the sum before adding the element
struct CumSum {
    exclusive: bool,
    reverse: bool,
}

impl Kernel for CumSum {
    fn prepare(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        let (i, a) = (io.input(0)?, io.input(1)?);
        if !matches!(i.dtype, TType::I32 | TType::I64 | TType::F32) || a.dtype != TType::I32 || ints(a)?.len() != 1 || i.shape.is_empty() {
            return err("CUMSUM takes int32, int64 or float32 of rank 1 or more, and one int32 axis");
        }
        same_shape(i, &mut io.outs[0]);
        Ok(())
    }

    fn eval(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        let i = io.input(0)?;
        let (outer, len, inner) = around(&i.shape, axis_of(ints(io.input(1)?)?[0], i.shape.len())?);
        let (ex, rev, o) = (self.exclusive, self.reverse, &mut io.outs[0]);
        match i.dtype {
            TType::F32 => scan(i.as_slice::<f32>(), o.as_mut_slice(), outer, len, inner, ex, rev, |a, x| a + x),
            TType::I32 => scan(i.as_slice::<i32>(), o.as_mut_slice(), outer, len, inner, ex, rev, i32::wrapping_add),
            _ => scan(i.as_slice::<i64>(), o.as_mut_slice(), outer, len, inner, ex, rev, i64::wrapping_add),
        }
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn scan<T: Copy + Default>(x: &[T], o: &mut [T], outer: usize, len: usize, inner: usize, exclusive: bool, reverse: bool, add: impl Fn(T, T) -> T) {
    for out in 0..outer {
        for inn in 0..inner {
            let mut acc = T::default();
            for step in 0..len {
                let at = (out * len + if reverse { len - 1 - step } else { step }) * inner + inn;
                if exclusive {
                    o[at] = acc;
                    acc = add(acc, x[at]);
                } else {
                    acc = add(acc, x[at]);
                    o[at] = acc;
                }
            }
        }
    }
}

/// segment_sum.cc: each row of the data added, in order from zeros, into the output row of its segment id (the ids
/// sorted, from 0, in steps of 0 or 1)
struct SegmentSum;

impl SegmentSum {
    fn resize(io: &mut Io) -> Result<()> {
        let (d, ids_t) = (io.input(0)?, io.input(1)?);
        let ids = ints(ids_t)?;
        let n = ids_t.shape.first().copied().unwrap_or(0);
        if d.shape.first() != Some(&n) {
            return err("SEGMENT_SUM needs one segment id per row of the data");
        }
        let ok = ids[..n].iter().enumerate().all(|(k, &id)| if k == 0 { id == 0 } else { id - ids[k - 1] == 0 || id - ids[k - 1] == 1 });
        if !ok {
            return err("segment ids must start at 0 and go up in steps of 0 or 1");
        }
        let mut s = d.shape.clone();
        s[0] = if n == 0 { 0 } else { ids[n - 1] as usize + 1 };
        io.outs[0].resize(s);
        Ok(())
    }
}

impl Kernel for SegmentSum {
    fn prepare(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        let (d, ids) = (io.input(0)?, io.input(1)?);
        if !matches!(d.dtype, TType::I32 | TType::F32) || ids.dtype != TType::I32 {
            return err("SEGMENT_SUM takes int32 or float32 data and int32 segment ids");
        }
        if d.alloc == Alloc::Constant && ids.alloc == Alloc::Constant {
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
        let (d, ids) = (io.input(0)?, ints(io.input(1)?)?);
        let row: usize = d.shape[1..].iter().product();
        let o = &mut io.outs[0];
        o.data_mut().fill(0);
        for (r, &id) in ids.iter().take(d.shape[0]).enumerate() {
            let (src, dst) = (r * row..(r + 1) * row, id as usize * row..(id as usize + 1) * row);
            if d.dtype == TType::F32 {
                o.as_mut_slice::<f32>()[dst].iter_mut().zip(&d.as_slice::<f32>()[src]).for_each(|(a, &x)| *a += x);
            } else {
                o.as_mut_slice::<i32>()[dst].iter_mut().zip(&d.as_slice::<i32>()[src]).for_each(|(a, &x)| *a = a.wrapping_add(x));
            }
        }
        Ok(())
    }
}

/// topk_v2.cc: per row (last axis) the k largest values and their indices, larger first, the earlier index first
/// among equals
struct TopK;

impl TopK {
    fn resize(io: &mut Io) -> Result<()> {
        let k = ints(io.input(1)?)?;
        let mut s = io.input(0)?.shape.clone();
        match (k.as_slice(), s.last_mut()) {
            ([k], Some(last)) if *k >= 0 && *k as usize <= *last => *last = *k as usize,
            _ => return err("TOPK_V2 needs one k from 0 to the size of the last dimension"),
        }
        io.outs[0].resize(s.clone());
        io.outs[1].resize(s);
        Ok(())
    }
}

impl Kernel for TopK {
    fn prepare(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        let (i, k) = (io.input(0)?, io.input(1)?);
        if i.dtype != io.outs[0].dtype || k.dtype != TType::I32 {
            return err("TOPK_V2 needs values of the input's type and an int32 k");
        }
        // 2.5 sets the index output's type itself
        io.outs[1].dtype = TType::I32;
        if k.alloc == Alloc::Constant {
            Self::resize(io)
        } else {
            io.outs[0].alloc = Alloc::Dynamic;
            io.outs[1].alloc = Alloc::Dynamic;
            Ok(())
        }
    }

    fn eval(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        if io.outs[0].alloc == Alloc::Dynamic {
            Self::resize(io)?;
        }
        let i = io.input(0)?;
        let row = i.shape[i.shape.len() - 1];
        let k = io.outs[0].shape[i.shape.len() - 1];
        let picks = match i.dtype {
            TType::F32 => top_k(i.as_slice::<f32>(), row, k),
            TType::U8 | TType::I8 | TType::I32 | TType::I64 => top_k(&ints(i)?, row, k),
            t => return err(format!("{t:?} is not supported by TOPK_V2")),
        };
        let es = i.dtype.size();
        let (src, vals) = (i.data(), io.outs[0].data_mut());
        for (d, &p) in vals.chunks_exact_mut(es).zip(&picks) {
            d.copy_from_slice(&src[p * es..(p + 1) * es]);
        }
        put_ints(&mut io.outs[1], &picks.iter().map(|&p| (p % row.max(1)) as i64).collect::<Vec<_>>())
    }
}

/// the flat positions of each row's k largest, in TopContainer's order
fn top_k<T: PartialOrd>(v: &[T], row: usize, k: usize) -> Vec<usize> {
    let mut picks = Vec::with_capacity(v.len() / row.max(1) * k);
    for (r, vals) in v.chunks(row.max(1)).enumerate() {
        let mut idx: Vec<usize> = (0..vals.len()).collect();
        // a stable sort: equal values keep the earlier index first
        idx.sort_by(|&a, &b| vals[b].partial_cmp(&vals[a]).unwrap_or(std::cmp::Ordering::Equal));
        picks.extend(idx[..k].iter().map(|&c| r * row + c));
    }
    picks
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
