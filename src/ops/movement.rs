//! Ops that move elements without arithmetic: each says what shape its output has and which input element every
//! output element comes from (or that it is a fill value). Any element type, strings included where TFLite has them.
//! An output whose shape depends on input values that are not constant is sized while the op runs.

use super::util::*;
use super::{Io, Kernel};
use crate::interp::Env;
use crate::tensor::{Alloc, TType, Tensor};
use crate::tflite::OpSpec;
use crate::{Error, Result};

pub fn create(s: &OpSpec) -> Result<Option<Box<dyn Kernel>>> {
    let k: Box<dyn Movement> = match s.code {
        43 => Box::new(Squeeze { dims: opt_vec(s, 0) }),
        70 => Box::new(ExpandDims),
        39 => Box::new(Transpose),
        65 => Box::new(Slice),
        45 => Box::new(StridedSlice {
            begin_mask: opt(s, 0, 0),
            end_mask: opt(s, 1, 0),
            ellipsis_mask: opt(s, 2, 0),
            new_axis_mask: opt(s, 3, 0),
            shrink_axis_mask: opt(s, 4, 0),
        }),
        34 => Box::new(Pad { v2: false }),
        60 => Box::new(Pad { v2: true }),
        100 => Box::new(MirrorPad { symmetric: opt::<i8>(s, 0, 0) == 1 }),
        69 => Box::new(Tile),
        130 => Box::new(BroadcastTo),
        36 => Box::new(Gather { axis: opt(s, 0, 0), batch_dims: opt(s, 1, 0) }),
        105 => Box::new(ReverseV2),
        112 => Box::new(ReverseSequence { seq_dim: opt(s, 0, 0), batch_dim: opt(s, 1, 0) }),
        26 => Box::new(SpaceToDepth { block: opt(s, 0, 0) }),
        5 => Box::new(DepthToSpace { block: opt(s, 0, 0) }),
        38 => Box::new(SpaceToBatchNd),
        37 => Box::new(BatchToSpaceNd),
        113 => Box::new(MatrixDiag),
        94 => Box::new(Fill),
        _ => return Ok(None),
    };
    Ok(Some(Box::new(Mv(k))))
}

/// row-major strides
pub fn strides(shape: &[usize]) -> Vec<usize> {
    let mut s = vec![1; shape.len()];
    for k in (0..shape.len().saturating_sub(1)).rev() {
        s[k] = s[k + 1] * shape[k + 1];
    }
    s
}

/// f(flat index, multi-index) for every element of `shape`, in row-major order
pub fn each_index(shape: &[usize], mut f: impl FnMut(usize, &[usize])) {
    let n: usize = shape.iter().product();
    let mut idx = vec![0; shape.len()];
    for flat in 0..n {
        f(flat, &idx);
        for d in (0..shape.len()).rev() {
            idx[d] += 1;
            if idx[d] < shape[d] {
                break;
            }
            idx[d] = 0;
        }
    }
}

/// out[k] = input[src[k]], or the fill value for None (zeros when there is none)
pub fn take(i: &Tensor, o: &mut Tensor, src: &[Option<usize>], fill: Option<&[u8]>) -> Result<()> {
    if i.dtype == TType::Str {
        let s = i.strings()?;
        let picked: Vec<&[u8]> = src.iter().map(|k| k.map_or(fill.unwrap_or(&[]), |k| s[k])).collect();
        let shape = o.shape.clone();
        o.set_strings(&picked, shape);
        return Ok(());
    }
    let es = i.dtype.size();
    let zero = vec![0u8; es];
    let fill = fill.unwrap_or(&zero);
    let (d, od) = (i.data(), o.data_mut());
    for (k, s) in src.iter().enumerate() {
        let from = match s {
            Some(s) => &d[s * es..(s + 1) * es],
            None => fill,
        };
        od[k * es..(k + 1) * es].copy_from_slice(from);
    }
    Ok(())
}

/// the values of an integer input (shape tensors, indices, axes)
fn vals(io: &Io, k: usize) -> Result<Vec<i64>> {
    index_values(io.input(k)?)
}

trait Movement: Send {
    /// inputs whose values (not just shapes) decide the output shape
    fn shape_inputs(&self) -> &'static [usize] {
        &[]
    }
    fn shape(&self, io: &Io) -> Result<Vec<usize>>;
    /// for every output element, the input element it copies (None: the fill value)
    fn sources(&self, io: &Io, out: &[usize]) -> Result<Vec<Option<usize>>>;
    /// the raw bytes of the fill value
    fn fill(&self, _io: &Io) -> Result<Option<Vec<u8>>> {
        Ok(None)
    }
    /// which input the elements come from
    fn data_input(&self) -> usize {
        0
    }
}

struct Mv(Box<dyn Movement>);

impl Kernel for Mv {
    fn prepare(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        let dynamic = io.outs[0].dtype == TType::Str || self.0.shape_inputs().iter().any(|&k| io.opt(k).is_some_and(|t| t.alloc != Alloc::Constant));
        if dynamic {
            io.outs[0].alloc = Alloc::Dynamic;
            return Ok(());
        }
        let shape = self.0.shape(io)?;
        io.outs[0].resize(shape);
        Ok(())
    }

    fn eval(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        let shape = self.0.shape(io)?;
        let src = self.0.sources(io, &shape)?;
        let fill = self.0.fill(io)?;
        let i = io.input(self.0.data_input())?;
        let o = &mut io.outs[0];
        if o.dtype != i.dtype {
            return err(format!("input {:?}, output {:?}", i.dtype, o.dtype));
        }
        o.resize(shape);
        take(i, o, &src, fill.as_deref())
    }
}

/// every input element in order (the shape changes only)
fn identity(io: &Io) -> Result<Vec<Option<usize>>> {
    Ok((0..io.input(0)?.elements()).map(Some).collect())
}

struct Squeeze {
    dims: Vec<i32>,
}

impl Movement for Squeeze {
    fn shape(&self, io: &Io) -> Result<Vec<usize>> {
        let s = &io.input(0)?.shape;
        if self.dims.is_empty() {
            return Ok(s.iter().copied().filter(|&d| d != 1).collect());
        }
        let mut drop = vec![false; s.len()];
        for &d in &self.dims {
            let a = axis_of(d as i64, s.len())?;
            if s[a] != 1 {
                return err(format!("cannot squeeze dimension {a} of size {}", s[a]));
            }
            drop[a] = true;
        }
        Ok(s.iter().zip(&drop).filter(|p| !*p.1).map(|p| *p.0).collect())
    }
    fn sources(&self, io: &Io, _: &[usize]) -> Result<Vec<Option<usize>>> {
        identity(io)
    }
}

struct ExpandDims;

impl Movement for ExpandDims {
    fn shape_inputs(&self) -> &'static [usize] {
        &[1]
    }
    fn shape(&self, io: &Io) -> Result<Vec<usize>> {
        let mut s = io.input(0)?.shape.clone();
        let a = *vals(io, 1)?.first().ok_or_else(|| Error("no axis".into()))?;
        let r = s.len() as i64;
        let a = if a < 0 { a + r + 1 } else { a };
        if !(0..=r).contains(&a) {
            return err(format!("axis out of range for rank {r}"));
        }
        s.insert(a as usize, 1);
        Ok(s)
    }
    fn sources(&self, io: &Io, _: &[usize]) -> Result<Vec<Option<usize>>> {
        identity(io)
    }
}

struct Transpose;

impl Transpose {
    fn perm(io: &Io) -> Result<Vec<usize>> {
        let r = io.input(0)?.shape.len();
        let p = vals(io, 1)?;
        if p.len() != r {
            return err(format!("permutation of {} dimensions for rank {r}", p.len()));
        }
        let p: Vec<usize> = p.iter().map(|&x| axis_of(x, r)).collect::<Result<_>>()?;
        let mut seen = vec![false; r];
        for &x in &p {
            if std::mem::replace(&mut seen[x], true) {
                return err("the permutation repeats an axis");
            }
        }
        Ok(p)
    }
}

impl Movement for Transpose {
    fn shape_inputs(&self) -> &'static [usize] {
        &[1]
    }
    fn shape(&self, io: &Io) -> Result<Vec<usize>> {
        let s = &io.input(0)?.shape;
        Ok(Self::perm(io)?.iter().map(|&p| s[p]).collect())
    }
    fn sources(&self, io: &Io, out: &[usize]) -> Result<Vec<Option<usize>>> {
        let (st, p) = (strides(&io.input(0)?.shape), Self::perm(io)?);
        let mut v = vec![None; out.iter().product()];
        each_index(out, |k, idx| v[k] = Some(idx.iter().zip(&p).map(|(&i, &a)| i * st[a]).sum()));
        Ok(v)
    }
}

struct Slice;

impl Slice {
    fn bounds(io: &Io) -> Result<Vec<(usize, usize)>> {
        let s = &io.input(0)?.shape;
        let (b, z) = (vals(io, 1)?, vals(io, 2)?);
        if b.len() != s.len() || z.len() != s.len() {
            return err("begin and size must have one value per dimension");
        }
        (0..s.len())
            .map(|k| {
                let size = if z[k] == -1 { s[k] as i64 - b[k] } else { z[k] };
                if b[k] < 0 || size < 0 || b[k] + size > s[k] as i64 {
                    return err(format!("slice [{}, +{}) outside dimension {k} of size {}", b[k], z[k], s[k]));
                }
                Ok((b[k] as usize, size as usize))
            })
            .collect()
    }
}

impl Movement for Slice {
    fn shape_inputs(&self) -> &'static [usize] {
        &[1, 2]
    }
    fn shape(&self, io: &Io) -> Result<Vec<usize>> {
        Ok(Self::bounds(io)?.iter().map(|b| b.1).collect())
    }
    fn sources(&self, io: &Io, out: &[usize]) -> Result<Vec<Option<usize>>> {
        let (st, b) = (strides(&io.input(0)?.shape), Self::bounds(io)?);
        let mut v = vec![None; out.iter().product()];
        each_index(out, |k, idx| v[k] = Some(idx.iter().enumerate().map(|(d, &i)| (i + b[d].0) * st[d]).sum()));
        Ok(v)
    }
}

struct StridedSlice {
    begin_mask: i32,
    end_mask: i32,
    ellipsis_mask: i32,
    new_axis_mask: i32,
    shrink_axis_mask: i32,
}

impl StridedSlice {
    /// per axis (start, stride, count), and whether it is shrunk (strided_slice_logic.h, strided_slice.cc)
    fn axes(&self, io: &Io) -> Result<Vec<(i64, i64, usize, bool)>> {
        if self.ellipsis_mask != 0 || self.new_axis_mask != 0 {
            return err("ellipsis_mask and new_axis_mask are not implemented (as in TFLite 2.5)");
        }
        let s = &io.input(0)?.shape;
        let (b, e, st) = (vals(io, 1)?, vals(io, 2)?, vals(io, 3)?);
        if b.len() > s.len() || e.len() != b.len() || st.len() != b.len() {
            return err("begin, end and strides must have the same length, at most the rank");
        }
        let (mut begin_mask, mut end_mask) = (self.begin_mask, self.end_mask);
        let mut out = Vec::new();
        for axis in 0..s.len() {
            let (start, stop, stride) = if axis < b.len() { (b[axis] as i32 as i64, e[axis] as i32 as i64, st[axis] as i32 as i64) } else {
                begin_mask |= 1 << axis;
                end_mask |= 1 << axis;
                (0, 0, 1)
            };
            if stride == 0 {
                return err("stride value has to be non-zero");
            }
            let size = s[axis] as i64;
            let shrink = self.shrink_axis_mask & (1 << axis) != 0;
            let clampi = |v: i64| if stride > 0 { v.clamp(0, size) } else { v.clamp(-1, size - 1) };
            let begin = if size == 0 {
                0
            } else {
                let mut v = if begin_mask & (1 << axis) != 0 { if stride > 0 { i32::MIN as i64 } else { i32::MAX as i64 } } else { start };
                if v < 0 {
                    v += size;
                }
                clampi(v)
            };
            let end = if size == 0 {
                0
            } else if shrink {
                begin + 1
            } else {
                let mut v = if end_mask & (1 << axis) != 0 { if stride > 0 { i32::MAX as i64 } else { i32::MIN as i64 } } else { stop };
                if v < 0 {
                    v += size;
                }
                clampi(v)
            };
            // std::ceil((end - begin) / static_cast<float>(stride))
            let count = (((end - begin) as i32) as f32 / stride as f32).ceil().max(0.0) as usize;
            out.push((begin, stride, if shrink { 1 } else { count }, shrink));
        }
        Ok(out)
    }
}

impl Movement for StridedSlice {
    fn shape_inputs(&self) -> &'static [usize] {
        &[1, 2, 3]
    }
    fn shape(&self, io: &Io) -> Result<Vec<usize>> {
        Ok(self.axes(io)?.iter().filter(|a| !a.3).map(|a| a.2).collect())
    }
    fn sources(&self, io: &Io, _: &[usize]) -> Result<Vec<Option<usize>>> {
        let st = strides(&io.input(0)?.shape);
        let axes = self.axes(io)?;
        let full: Vec<usize> = axes.iter().map(|a| a.2).collect();
        let mut v = vec![None; full.iter().product()];
        each_index(&full, |k, idx| v[k] = Some(idx.iter().zip(&axes).zip(&st).map(|((&j, a), &s)| (a.0 + j as i64 * a.1) as usize * s).sum()));
        Ok(v)
    }
}

/// [rank, 2] paddings as (before, after) per dimension
fn paddings(io: &Io, k: usize, rank: usize) -> Result<Vec<(i64, i64)>> {
    let p = vals(io, k)?;
    if p.len() != 2 * rank {
        return err(format!("paddings must be [{rank}, 2]"));
    }
    if p.iter().any(|&x| x < 0) {
        return err("negative padding");
    }
    Ok(p.chunks(2).map(|c| (c[0], c[1])).collect())
}

struct Pad {
    v2: bool,
}

impl Movement for Pad {
    fn shape_inputs(&self) -> &'static [usize] {
        &[1]
    }
    fn shape(&self, io: &Io) -> Result<Vec<usize>> {
        let s = &io.input(0)?.shape;
        Ok(paddings(io, 1, s.len())?.iter().zip(s).map(|(p, &d)| d + p.0 as usize + p.1 as usize).collect())
    }
    fn sources(&self, io: &Io, out: &[usize]) -> Result<Vec<Option<usize>>> {
        let s = &io.input(0)?.shape;
        let (st, p) = (strides(s), paddings(io, 1, s.len())?);
        let mut v = vec![None; out.iter().product()];
        each_index(out, |k, idx| {
            let mut at = 0;
            for (d, &j) in idx.iter().enumerate() {
                let i = j as i64 - p[d].0;
                if i < 0 || i >= s[d] as i64 {
                    return;
                }
                at += i as usize * st[d];
            }
            v[k] = Some(at);
        });
        Ok(v)
    }
    fn fill(&self, io: &Io) -> Result<Option<Vec<u8>>> {
        let i = io.input(0)?;
        if self.v2 {
            if let Some(c) = io.opt(2) {
                return Ok(Some(c.data()[..i.dtype.size()].to_vec()));
            }
        }
        // quantized: the output's zero point
        if i.quantized() && matches!(i.dtype, TType::U8 | TType::I8 | TType::I16) {
            let zp = i.zero_point();
            return Ok(Some(match i.dtype {
                TType::I16 => (zp as i16).to_le_bytes().to_vec(),
                _ => vec![zp as u8],
            }));
        }
        Ok(None)
    }
}

struct MirrorPad {
    symmetric: bool,
}

impl Movement for MirrorPad {
    fn shape_inputs(&self) -> &'static [usize] {
        &[1]
    }
    fn shape(&self, io: &Io) -> Result<Vec<usize>> {
        let s = &io.input(0)?.shape;
        let p = paddings(io, 1, s.len())?;
        for (q, &d) in p.iter().zip(s) {
            let limit = if self.symmetric { d as i64 } else { d as i64 - 1 };
            if q.0 > limit || q.1 > limit {
                return err("mirror padding larger than the dimension allows");
            }
        }
        Ok(p.iter().zip(s).map(|(q, &d)| d + q.0 as usize + q.1 as usize).collect())
    }
    fn sources(&self, io: &Io, out: &[usize]) -> Result<Vec<Option<usize>>> {
        let s = &io.input(0)?.shape;
        let (st, p) = (strides(s), paddings(io, 1, s.len())?);
        let sym = self.symmetric as i64;
        let mut v = vec![None; out.iter().product()];
        each_index(out, |k, idx| {
            let at = idx
                .iter()
                .enumerate()
                .map(|(d, &j)| {
                    let (n, i) = (s[d] as i64, j as i64 - p[d].0);
                    let m = if i < 0 { -i - sym } else if i >= n { 2 * n - 2 + sym - i } else { i };
                    m as usize * st[d]
                })
                .sum();
            v[k] = Some(at);
        });
        Ok(v)
    }
}

struct Tile;

impl Movement for Tile {
    fn shape_inputs(&self) -> &'static [usize] {
        &[1]
    }
    fn shape(&self, io: &Io) -> Result<Vec<usize>> {
        let s = &io.input(0)?.shape;
        let m = vals(io, 1)?;
        if m.len() != s.len() || m.iter().any(|&x| x < 0) {
            return err("multiples must be non-negative, one per dimension");
        }
        Ok(s.iter().zip(&m).map(|(&d, &x)| d * x as usize).collect())
    }
    fn sources(&self, io: &Io, out: &[usize]) -> Result<Vec<Option<usize>>> {
        let s = &io.input(0)?.shape;
        let st = strides(s);
        let mut v = vec![None; out.iter().product()];
        each_index(out, |k, idx| v[k] = Some(idx.iter().enumerate().map(|(d, &j)| (j % s[d]) * st[d]).sum()));
        Ok(v)
    }
}

struct BroadcastTo;

impl Movement for BroadcastTo {
    fn shape_inputs(&self) -> &'static [usize] {
        &[1]
    }
    fn shape(&self, io: &Io) -> Result<Vec<usize>> {
        let s = &io.input(0)?.shape;
        let target: Vec<usize> = vals(io, 1)?.iter().map(|&x| x.max(0) as usize).collect();
        if target.len() < s.len() || target.len() > 8 {
            return err("the shape must have at least the input's rank and at most 8 dimensions");
        }
        let off = target.len() - s.len();
        for (k, &d) in s.iter().enumerate() {
            if d != 1 && d != target[off + k] {
                return err(format!("cannot broadcast {s:?} to {target:?}"));
            }
        }
        Ok(target)
    }
    fn sources(&self, io: &Io, out: &[usize]) -> Result<Vec<Option<usize>>> {
        let s = &io.input(0)?.shape;
        let st = strides(s);
        let off = out.len() - s.len();
        let mut v = vec![None; out.iter().product()];
        each_index(out, |k, idx| v[k] = Some(s.iter().enumerate().map(|(d, &n)| if n == 1 { 0 } else { idx[off + d] * st[d] }).sum()));
        Ok(v)
    }
}

struct Gather {
    axis: i32,
    batch_dims: i32,
}

impl Gather {
    fn dims(&self, io: &Io) -> Result<(usize, usize)> {
        let (i, p) = (io.input(0)?, io.input(1)?);
        let axis = axis_of(self.axis as i64, i.shape.len())?;
        let bd = if self.batch_dims < 0 { self.batch_dims + p.shape.len() as i32 } else { self.batch_dims };
        if bd < 0 || bd as usize > axis || bd as usize > p.shape.len() || (0..bd as usize).any(|k| i.shape[k] != p.shape[k]) {
            return err("batch_dims out of range");
        }
        Ok((axis, bd as usize))
    }
}

impl Movement for Gather {
    fn shape(&self, io: &Io) -> Result<Vec<usize>> {
        let (i, p) = (io.input(0)?, io.input(1)?);
        let (axis, bd) = self.dims(io)?;
        Ok([&i.shape[..axis], &p.shape[bd..], &i.shape[axis + 1..]].concat())
    }
    fn sources(&self, io: &Io, _: &[usize]) -> Result<Vec<Option<usize>>> {
        let (i, p) = (io.input(0)?, io.input(1)?);
        let (axis, bd) = self.dims(io)?;
        let s = &i.shape;
        let batch: usize = s[..bd].iter().product();
        let outer: usize = s[bd..axis].iter().product();
        let inner: usize = s[axis + 1..].iter().product();
        let axis_size = s[axis];
        let coords = index_values(p)?;
        let coord_size = coords.len() / batch.max(1);
        let mut v = Vec::with_capacity(batch * outer * coord_size * inner);
        for b in 0..batch {
            for o in 0..outer {
                for c in 0..coord_size {
                    let x = coords[b * coord_size + c];
                    if x < 0 || x as usize >= axis_size {
                        return err(format!("index {x} out of range [0, {axis_size})"));
                    }
                    let from = ((b * outer + o) * axis_size + x as usize) * inner;
                    v.extend((from..from + inner).map(Some));
                }
            }
        }
        Ok(v)
    }
}

struct ReverseV2;

impl Movement for ReverseV2 {
    fn shape(&self, io: &Io) -> Result<Vec<usize>> {
        Ok(io.input(0)?.shape.clone())
    }
    fn sources(&self, io: &Io, out: &[usize]) -> Result<Vec<Option<usize>>> {
        let st = strides(out);
        let axes: Vec<usize> = vals(io, 1)?.iter().map(|&a| axis_of(a, out.len())).collect::<Result<_>>()?;
        let mut v = vec![None; out.iter().product()];
        each_index(out, |k, idx| v[k] = Some(idx.iter().enumerate().map(|(d, &j)| if axes.contains(&d) { out[d] - 1 - j } else { j } * st[d]).sum()));
        Ok(v)
    }
}

struct ReverseSequence {
    seq_dim: i32,
    batch_dim: i32,
}

impl Movement for ReverseSequence {
    fn shape(&self, io: &Io) -> Result<Vec<usize>> {
        Ok(io.input(0)?.shape.clone())
    }
    fn sources(&self, io: &Io, out: &[usize]) -> Result<Vec<Option<usize>>> {
        let st = strides(out);
        let (sd, bd) = (axis_of(self.seq_dim as i64, out.len())?, axis_of(self.batch_dim as i64, out.len())?);
        let lens = vals(io, 1)?;
        if lens.len() != out[bd] || lens.iter().any(|&l| l < 0 || l as usize > out[sd]) {
            return err("seq_lengths must have one length per batch, each at most the sequence dimension");
        }
        let mut v = vec![None; out.iter().product()];
        each_index(out, |k, idx| {
            let len = lens[idx[bd]] as usize;
            v[k] = Some(idx.iter().enumerate().map(|(d, &j)| if d == sd && j < len { len - 1 - j } else { j } * st[d]).sum());
        });
        Ok(v)
    }
}

struct SpaceToDepth {
    block: i32,
}

impl Movement for SpaceToDepth {
    fn shape(&self, io: &Io) -> Result<Vec<usize>> {
        let s = &io.input(0)?.shape;
        let b = self.block as usize;
        if s.len() != 4 || b == 0 || s[1] % b != 0 || s[2] % b != 0 {
            return err("needs a 4-D input whose height and width the block size divides");
        }
        Ok(vec![s[0], s[1] / b, s[2] / b, s[3] * b * b])
    }
    fn sources(&self, io: &Io, out: &[usize]) -> Result<Vec<Option<usize>>> {
        let s = &io.input(0)?.shape;
        let (st, b, c) = (strides(s), self.block as usize, s[3]);
        let mut v = vec![None; out.iter().product()];
        each_index(out, |k, i| {
            let off = i[3] / c;
            v[k] = Some(i[0] * st[0] + (i[1] * b + off / b) * st[1] + (i[2] * b + off % b) * st[2] + i[3] % c);
        });
        Ok(v)
    }
}

struct DepthToSpace {
    block: i32,
}

impl Movement for DepthToSpace {
    fn shape(&self, io: &Io) -> Result<Vec<usize>> {
        let s = &io.input(0)?.shape;
        let b = self.block as usize;
        if s.len() != 4 || b == 0 || s[3] % (b * b) != 0 {
            return err("needs a 4-D input whose depth the block size squared divides");
        }
        Ok(vec![s[0], s[1] * b, s[2] * b, s[3] / (b * b)])
    }
    fn sources(&self, io: &Io, out: &[usize]) -> Result<Vec<Option<usize>>> {
        let s = &io.input(0)?.shape;
        let (st, b, oc) = (strides(s), self.block as usize, out[3]);
        let mut v = vec![None; out.iter().product()];
        each_index(out, |k, i| {
            let ic = ((i[1] % b) * b + i[2] % b) * oc + i[3];
            v[k] = Some(i[0] * st[0] + (i[1] / b) * st[1] + (i[2] / b) * st[2] + ic);
        });
        Ok(v)
    }
}

/// a 3-D input [b, h, c] is handled as [b, h, 1, c] with a block of [bh, 1]
fn as_4d(s: &[usize], block: &[i64]) -> Result<([usize; 4], [usize; 2])> {
    match (s.len(), block.len()) {
        (4, 2) => Ok(([s[0], s[1], s[2], s[3]], [block[0] as usize, block[1] as usize])),
        (3, 1) => Ok(([s[0], s[1], 1, s[2]], [block[0] as usize, 1])),
        _ => err("needs a 3-D or 4-D input with a block shape for its spatial dimensions"),
    }
}

struct SpaceToBatchNd;

impl SpaceToBatchNd {
    fn geometry(io: &Io) -> Result<([usize; 4], [usize; 2], [i64; 4], [usize; 4])> {
        let s = &io.input(0)?.shape;
        let block = vals(io, 1)?;
        let p = vals(io, 2)?;
        let (s4, b) = as_4d(s, &block)?;
        let pads: [i64; 4] = if s.len() == 4 { [p[0], p[1], p[2], p[3]] } else { [p[0], p[1], 0, 0] };
        let (h, w) = (s4[1] as i64 + pads[0] + pads[1], s4[2] as i64 + pads[2] + pads[3]);
        if b[0] == 0 || b[1] == 0 || h % b[0] as i64 != 0 || w % b[1] as i64 != 0 || pads.iter().any(|&x| x < 0) {
            return err("the padded spatial dimensions must be multiples of the block shape");
        }
        Ok((s4, b, pads, [s4[0] * b[0] * b[1], h as usize / b[0], w as usize / b[1], s4[3]]))
    }
}

impl Movement for SpaceToBatchNd {
    fn shape_inputs(&self) -> &'static [usize] {
        &[1, 2]
    }
    fn shape(&self, io: &Io) -> Result<Vec<usize>> {
        let (_, _, _, o) = Self::geometry(io)?;
        Ok(if io.input(0)?.shape.len() == 4 { o.to_vec() } else { vec![o[0], o[1], o[3]] })
    }
    fn sources(&self, io: &Io, _: &[usize]) -> Result<Vec<Option<usize>>> {
        let (s, b, pads, o) = Self::geometry(io)?;
        let st = strides(&s);
        let mut v = vec![None; o.iter().product()];
        each_index(&o, |k, i| {
            let (ib, off) = (i[0] % s[0], i[0] / s[0]);
            let ih = (i[1] * b[0] + off / b[1]) as i64 - pads[0];
            let iw = (i[2] * b[1] + off % b[1]) as i64 - pads[2];
            if ih >= 0 && (ih as usize) < s[1] && iw >= 0 && (iw as usize) < s[2] {
                v[k] = Some(ib * st[0] + ih as usize * st[1] + iw as usize * st[2] + i[3]);
            }
        });
        Ok(v)
    }
    fn fill(&self, io: &Io) -> Result<Option<Vec<u8>>> {
        let i = io.input(0)?;
        Ok((i.quantized() && matches!(i.dtype, TType::U8 | TType::I8)).then(|| vec![i.zero_point() as u8]))
    }
}

struct BatchToSpaceNd;

impl BatchToSpaceNd {
    fn geometry(io: &Io) -> Result<([usize; 4], [usize; 2], [i64; 4], [usize; 4])> {
        let s = &io.input(0)?.shape;
        let block = vals(io, 1)?;
        let c = vals(io, 2)?;
        let (s4, b) = as_4d(s, &block)?;
        let crops: [i64; 4] = if s.len() == 4 { [c[0], c[1], c[2], c[3]] } else { [c[0], c[1], 0, 0] };
        let bb = b[0] * b[1];
        if bb == 0 || s4[0] % bb != 0 || crops.iter().any(|&x| x < 0) {
            return err("the batch must be a multiple of the block shape's product, crops non-negative");
        }
        let h = (s4[1] * b[0]) as i64 - crops[0] - crops[1];
        let w = (s4[2] * b[1]) as i64 - crops[2] - crops[3];
        if h < 0 || w < 0 {
            return err("crops larger than the output");
        }
        Ok((s4, b, crops, [s4[0] / bb, h as usize, w as usize, s4[3]]))
    }
}

impl Movement for BatchToSpaceNd {
    fn shape_inputs(&self) -> &'static [usize] {
        &[1, 2]
    }
    fn shape(&self, io: &Io) -> Result<Vec<usize>> {
        let (_, _, _, o) = Self::geometry(io)?;
        Ok(if io.input(0)?.shape.len() == 4 { o.to_vec() } else { vec![o[0], o[1], o[3]] })
    }
    fn sources(&self, io: &Io, _: &[usize]) -> Result<Vec<Option<usize>>> {
        let (s, b, crops, o) = Self::geometry(io)?;
        let st = strides(&s);
        let mut v = vec![None; o.iter().product()];
        each_index(&o, |k, i| {
            let (h, w) = (i[1] as i64 + crops[0], i[2] as i64 + crops[2]);
            let off = (h as usize % b[0]) * b[1] + w as usize % b[1];
            let ib = off * o[0] + i[0];
            v[k] = Some(ib * st[0] + (h as usize / b[0]) * st[1] + (w as usize / b[1]) * st[2] + i[3]);
        });
        Ok(v)
    }
}

struct MatrixDiag;

impl Movement for MatrixDiag {
    fn shape(&self, io: &Io) -> Result<Vec<usize>> {
        let s = &io.input(0)?.shape;
        let n = *s.last().ok_or_else(|| Error("needs at least one dimension".into()))?;
        Ok([&s[..], &[n]].concat())
    }
    fn sources(&self, _: &Io, out: &[usize]) -> Result<Vec<Option<usize>>> {
        let r = out.len();
        let mut v = vec![None; out.iter().product()];
        each_index(out, |k, i| {
            if i[r - 1] == i[r - 2] {
                v[k] = Some(k / out[r - 1]);
            }
        });
        Ok(v)
    }
}

struct Fill;

impl Movement for Fill {
    fn shape_inputs(&self) -> &'static [usize] {
        &[0]
    }
    fn shape(&self, io: &Io) -> Result<Vec<usize>> {
        let d = vals(io, 0)?;
        if d.iter().any(|&x| x < 0) {
            return err("negative dimension");
        }
        Ok(d.iter().map(|&x| x as usize).collect())
    }
    fn sources(&self, _: &Io, out: &[usize]) -> Result<Vec<Option<usize>>> {
        Ok(vec![Some(0); out.iter().product()])
    }
    fn data_input(&self) -> usize {
        1
    }
}
