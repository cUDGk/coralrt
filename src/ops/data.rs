//! Data ops with several inputs or outputs, or values to compute: SHAPE, RANK, SPLIT, SPLIT_V, SCATTER_ND, RANGE,
//! ONE_HOT, MATRIX_SET_DIAG, SPARSE_TO_DENSE, WHERE, UNIQUE, SELECT, SELECT_V2, ADD_N.

use super::movement::{each_index, strides, take};
use super::util::*;
use super::{Io, Kernel};
use crate::interp::Env;
use crate::tensor::{broadcast_shape, Alloc, TType, Tensor};
use crate::tflite::OpSpec;
use crate::{Error, Result};

pub fn create(s: &OpSpec) -> Result<Option<Box<dyn Kernel>>> {
    Ok(Some(match s.code {
        77 => Box::new(ShapeOp),
        110 => Box::new(Rank),
        49 => Box::new(Split { v: false }),
        102 => Box::new(Split { v: true }),
        122 => Box::new(ScatterNd),
        96 => Box::new(Range),
        85 => Box::new(OneHot { axis: opt(s, 0, -1) }),
        115 => Box::new(MatrixSetDiag),
        68 => Box::new(SparseToDense { validate: opt(s, 0, 0u8) != 0 }),
        109 => Box::new(Where),
        103 => Box::new(Unique),
        64 => Box::new(Select { v2: false }),
        123 => Box::new(Select { v2: true }),
        106 => Box::new(AddN),
        _ => return Ok(None),
    }))
}

fn constant(t: Option<&Tensor>) -> bool {
    t.is_none_or(|t| t.alloc == Alloc::Constant)
}

struct ShapeOp;

impl Kernel for ShapeOp {
    fn prepare(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        let r = io.input(0)?.shape.len();
        io.outs[0].resize(vec![r]);
        Ok(())
    }
    fn eval(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        let s: Vec<i64> = io.input(0)?.shape.iter().map(|&d| d as i64).collect();
        io.outs[0].resize(vec![s.len()]);
        put_ints(&mut io.outs[0], &s)
    }
}

struct Rank;

impl Kernel for Rank {
    fn prepare(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        io.outs[0].resize(vec![]);
        Ok(())
    }
    fn eval(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        let r = io.input(0)?.shape.len() as i64;
        put_ints(&mut io.outs[0], &[r])
    }
}

/// SPLIT (axis, input) into equal parts; SPLIT_V (input, size_splits, axis) into the given sizes (one may be -1)
struct Split {
    v: bool,
}

impl Split {
    fn layout(&self, io: &Io) -> Result<(usize, Vec<usize>)> {
        let (data, axis_t) = if self.v { (io.input(0)?, io.input(2)?) } else { (io.input(1)?, io.input(0)?) };
        let axis = axis_of(*index_values(axis_t)?.first().ok_or_else(|| Error("no axis".into()))?, data.shape.len())?;
        let dim = data.shape[axis];
        let n = io.outs.len();
        let sizes = if self.v {
            let mut s = index_values(io.input(1)?)?;
            if s.len() != n {
                return err(format!("{} sizes for {n} outputs", s.len()));
            }
            let known: i64 = s.iter().filter(|&&x| x != -1).sum();
            match s.iter().filter(|&&x| x == -1).count() {
                0 => {}
                1 => *s.iter_mut().find(|x| **x == -1).unwrap() = dim as i64 - known,
                _ => return err("more than one -1 in size_splits"),
            }
            if s.iter().any(|&x| x < 0) || s.iter().sum::<i64>() != dim as i64 {
                return err(format!("size_splits {s:?} do not add up to {dim}"));
            }
            s.into_iter().map(|x| x as usize).collect()
        } else {
            if n == 0 || dim % n != 0 {
                return err(format!("cannot split a dimension of {dim} into {n}"));
            }
            vec![dim / n; n]
        };
        Ok((axis, sizes))
    }

    fn resize(&self, io: &mut Io) -> Result<()> {
        let (axis, sizes) = self.layout(io)?;
        let shape = io.input(if self.v { 0 } else { 1 })?.shape.clone();
        for (o, &n) in io.outs.iter_mut().zip(&sizes) {
            let mut s = shape.clone();
            s[axis] = n;
            o.resize(s);
        }
        Ok(())
    }
}

impl Kernel for Split {
    fn prepare(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        let shape_inputs = if self.v { [io.opt(1), io.opt(2)] } else { [io.opt(0), None] };
        if shape_inputs.iter().all(|t| constant(*t)) {
            return self.resize(io);
        }
        for o in io.outs.iter_mut() {
            o.alloc = Alloc::Dynamic;
        }
        Ok(())
    }
    fn eval(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        self.resize(io)?;
        let (axis, sizes) = self.layout(io)?;
        let i = io.input(if self.v { 0 } else { 1 })?;
        let es = i.dtype.size();
        let outer: usize = i.shape[..axis].iter().product();
        let inner: usize = i.shape[axis + 1..].iter().product::<usize>() * es;
        let dim = i.shape[axis];
        let d = i.data();
        let mut at = 0;
        for (o, &n) in io.outs.iter_mut().zip(&sizes) {
            let od = o.data_mut();
            for k in 0..outer {
                let src = (k * dim + at) * inner;
                od[k * n * inner..(k + 1) * n * inner].copy_from_slice(&d[src..src + n * inner]);
            }
            at += n;
        }
        Ok(())
    }
}

/// scatter_nd.cc: zeros, then every update added at its index
struct ScatterNd;

impl ScatterNd {
    fn shape(io: &Io) -> Result<Vec<usize>> {
        let s = index_values(io.input(2)?)?;
        if s.iter().any(|&x| x < 0) {
            return err("negative dimension");
        }
        Ok(s.iter().map(|&x| x as usize).collect())
    }
}

impl Kernel for ScatterNd {
    fn prepare(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        if constant(io.opt(2)) {
            let s = Self::shape(io)?;
            io.outs[0].resize(s);
        } else {
            io.outs[0].alloc = Alloc::Dynamic;
        }
        Ok(())
    }
    fn eval(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        let shape = Self::shape(io)?;
        let (idx, upd) = (io.input(0)?, io.input(1)?);
        let o = &mut io.outs[0];
        o.resize(shape.clone());
        o.data_mut().fill(0);
        let nd = *idx.shape.last().ok_or_else(|| Error("indices without dimensions".into()))?;
        if nd > shape.len() {
            return err("indices longer than the output's rank");
        }
        let n: usize = idx.shape[..idx.shape.len() - 1].iter().product();
        let slice: usize = shape[nd..].iter().product();
        let st = strides(&shape);
        let ix = index_values(idx)?;
        for r in 0..n {
            let mut at = 0;
            for j in 0..nd {
                let v = ix[r * nd + j];
                if v < 0 || v as usize >= shape[j] {
                    return err(format!("index {v} out of range [0, {})", shape[j]));
                }
                at += v as usize * st[j];
            }
            let (src, dst) = (r * slice, at);
            match o.dtype {
                TType::F32 => {
                    let u = upd.as_slice::<f32>();
                    let od = o.as_mut_slice::<f32>();
                    for k in 0..slice {
                        od[dst + k] += u[src + k];
                    }
                }
                TType::I32 => {
                    let u = upd.as_slice::<i32>();
                    let od = o.as_mut_slice::<i32>();
                    for k in 0..slice {
                        od[dst + k] = od[dst + k].wrapping_add(u[src + k]);
                    }
                }
                TType::I64 => {
                    let u = upd.as_slice::<i64>();
                    let od = o.as_mut_slice::<i64>();
                    for k in 0..slice {
                        od[dst + k] = od[dst + k].wrapping_add(u[src + k]);
                    }
                }
                TType::U8 | TType::I8 => {
                    let u = upd.data();
                    let od = o.data_mut();
                    for k in 0..slice {
                        od[dst + k] = od[dst + k].wrapping_add(u[src + k]);
                    }
                }
                t => return err(format!("{t:?} is not supported")),
            }
        }
        Ok(())
    }
}

/// range.cc: the length from GetSize, the values by adding delta again and again
struct Range;

impl Range {
    fn len(io: &Io) -> Result<usize> {
        let (s, l, d) = (io.input(0)?, io.input(1)?, io.input(2)?);
        match s.dtype {
            TType::F32 => {
                let (s, l, d) = (s.as_slice::<f32>()[0], l.as_slice::<f32>()[0], d.as_slice::<f32>()[0]);
                if d == 0.0 || !((s >= l && d < 0.0) || (s <= l && d > 0.0)) {
                    return err("delta must be non-zero and point from start to limit");
                }
                Ok(((l - s) / d).abs().ceil() as usize)
            }
            _ => {
                let (s, l, d) = (index_values(s)?[0], index_values(l)?[0], index_values(d)?[0]);
                if d == 0 || !((s >= l && d < 0) || (s <= l && d > 0)) {
                    return err("delta must be non-zero and point from start to limit");
                }
                Ok((((l - s).abs() + d.abs() - 1) / d.abs()) as usize)
            }
        }
    }
}

impl Kernel for Range {
    fn prepare(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        if (0..3).all(|k| constant(io.opt(k))) {
            let n = Self::len(io)?;
            io.outs[0].resize(vec![n]);
        } else {
            io.outs[0].alloc = Alloc::Dynamic;
        }
        Ok(())
    }
    fn eval(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        let n = Self::len(io)?;
        let (s, d) = (io.input(0)?, io.input(2)?);
        let o = &mut io.outs[0];
        o.resize(vec![n]);
        match o.dtype {
            TType::F32 => {
                let (mut v, d) = (s.as_slice::<f32>()[0], d.as_slice::<f32>()[0]);
                for x in o.as_mut_slice::<f32>() {
                    *x = v;
                    v += d;
                }
            }
            _ => {
                let (mut v, d) = (index_values(s)?[0], index_values(d)?[0]);
                let mut out = Vec::with_capacity(n);
                for _ in 0..n {
                    out.push(v);
                    v = v.wrapping_add(d);
                }
                put_ints(o, &out)?;
            }
        }
        Ok(())
    }
}

struct OneHot {
    axis: i32,
}

impl OneHot {
    fn shape(&self, io: &Io) -> Result<(Vec<usize>, usize, usize)> {
        let i = io.input(0)?;
        let depth = index_values(io.input(1)?)?[0];
        if depth < 0 {
            return err("negative depth");
        }
        let r = i.shape.len();
        let axis = if self.axis == -1 { r } else { self.axis as usize };
        if axis > r {
            return err("axis out of range");
        }
        let mut s = i.shape.clone();
        s.insert(axis, depth as usize);
        Ok((s, axis, depth as usize))
    }
}

impl Kernel for OneHot {
    fn prepare(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        if constant(io.opt(1)) {
            let (s, _, _) = self.shape(io)?;
            io.outs[0].resize(s);
        } else {
            io.outs[0].alloc = Alloc::Dynamic;
        }
        Ok(())
    }
    fn eval(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        let (s, axis, depth) = self.shape(io)?;
        let idx = index_values(io.input(0)?)?;
        let (on, off) = (io.input(2)?, io.input(3)?);
        let es = on.dtype.size();
        let (on, off) = (on.data()[..es].to_vec(), off.data()[..es].to_vec());
        let o = &mut io.outs[0];
        o.resize(s.clone());
        let prefix: usize = s[..axis].iter().product();
        let suffix: usize = s[axis + 1..].iter().product();
        let od = o.data_mut();
        for p in 0..prefix {
            for d in 0..depth {
                for q in 0..suffix {
                    let k = (p * depth + d) * suffix + q;
                    let hot = idx[p * suffix + q] == d as i64;
                    od[k * es..(k + 1) * es].copy_from_slice(if hot { &on } else { &off });
                }
            }
        }
        Ok(())
    }
}

struct MatrixSetDiag;

impl Kernel for MatrixSetDiag {
    fn prepare(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        let s = io.input(0)?.shape.clone();
        if s.len() < 2 {
            return err("needs at least 2 dimensions");
        }
        io.outs[0].resize(s);
        Ok(())
    }
    fn eval(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        let (i, d) = (io.input(0)?, io.input(1)?);
        let s = &i.shape;
        let r = s.len();
        let (m, n) = (s[r - 2], s[r - 1]);
        let es = i.dtype.size();
        let o = &mut io.outs[0];
        o.data_mut().copy_from_slice(i.data());
        let batch: usize = s[..r - 2].iter().product();
        let k = m.min(n);
        if d.elements() != batch * k {
            return err("the diagonal does not match the input");
        }
        let (dd, od) = (d.data(), o.data_mut());
        for b in 0..batch {
            for j in 0..k {
                let at = (b * m + j) * n + j;
                od[at * es..(at + 1) * es].copy_from_slice(&dd[(b * k + j) * es..(b * k + j + 1) * es]);
            }
        }
        Ok(())
    }
}

/// sparse_to_dense.cc: the default value everywhere, then each value at its index
struct SparseToDense {
    validate: bool,
}

impl SparseToDense {
    fn shape(io: &Io) -> Result<Vec<usize>> {
        let s = index_values(io.input(1)?)?;
        if s.iter().any(|&x| x < 0) {
            return err("negative dimension");
        }
        Ok(s.iter().map(|&x| x as usize).collect())
    }
}

impl Kernel for SparseToDense {
    fn prepare(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        if constant(io.opt(1)) {
            let s = Self::shape(io)?;
            io.outs[0].resize(s);
        } else {
            io.outs[0].alloc = Alloc::Dynamic;
        }
        Ok(())
    }
    fn eval(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        let shape = Self::shape(io)?;
        let (ix, values, default) = (io.input(0)?, io.input(2)?, io.input(3)?);
        let rank = shape.len();
        let idx = index_values(ix)?;
        // indices: a scalar or [N] (rank-1 outputs), or [N, rank]
        let (n, width) = match ix.shape.len() {
            0 => (1, 1),
            1 => (ix.shape[0], 1),
            _ => (ix.shape[0], ix.shape[1]),
        };
        if width != rank.max(1) {
            return err("indices do not match the output's rank");
        }
        let st = strides(&shape);
        let mut flat = Vec::with_capacity(n);
        for r in 0..n {
            let row = &idx[r * width..(r + 1) * width];
            for (d, &v) in row.iter().enumerate() {
                if v < 0 || v as usize >= shape.get(d).copied().unwrap_or(1) {
                    return err(format!("index {v} out of bounds"));
                }
            }
            if self.validate && r > 0 && row <= &idx[(r - 1) * width..r * width] {
                return err("indices are not strictly increasing");
            }
            flat.push(row.iter().zip(&st).map(|(&v, &s)| v as usize * s).sum::<usize>());
        }
        let es = values.dtype.size();
        let o = &mut io.outs[0];
        o.resize(shape);
        let dv = default.data()[..es].to_vec();
        for c in o.data_mut().chunks_exact_mut(es) {
            c.copy_from_slice(&dv);
        }
        let scalar = values.elements() == 1;
        let (vd, od) = (values.data(), o.data_mut());
        for (r, &at) in flat.iter().enumerate() {
            let src = if scalar { 0 } else { r };
            od[at * es..(at + 1) * es].copy_from_slice(&vd[src * es..(src + 1) * es]);
        }
        Ok(())
    }
}

/// where.cc: the coordinates of the true elements, as int64 [count, rank]
struct Where;

impl Kernel for Where {
    fn prepare(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        io.outs[0].alloc = Alloc::Dynamic;
        Ok(())
    }
    fn eval(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        let c = io.input(0)?;
        let truth: Vec<bool> = match c.dtype {
            TType::Bool => c.data().iter().map(|&b| b != 0).collect(),
            _ => f64s(c)?.iter().map(|&x| x != 0.0).collect(),
        };
        let mut coords = Vec::new();
        each_index(&c.shape, |k, idx| {
            if truth[k] {
                coords.extend(idx.iter().map(|&i| i as i64));
            }
        });
        let r = c.shape.len();
        let o = &mut io.outs[0];
        o.resize(vec![coords.len() / r.max(1), r]);
        put_ints(o, &coords)
    }
}

/// unique.cc: the distinct values in order of first appearance, and each element's index among them
struct Unique;

impl Kernel for Unique {
    fn prepare(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        let n = io.input(0)?.elements();
        io.outs[0].alloc = Alloc::Dynamic;
        io.outs[1].resize(vec![n]);
        Ok(())
    }
    fn eval(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        let i = io.input(0)?;
        if i.shape.len() != 1 {
            return err("the input must be 1-D");
        }
        let es = i.dtype.size();
        let d = i.data();
        let mut firsts: Vec<usize> = Vec::new();
        let mut map: std::collections::HashMap<&[u8], i64> = std::collections::HashMap::new();
        let mut idx = Vec::with_capacity(i.elements());
        for k in 0..i.elements() {
            let key = &d[k * es..(k + 1) * es];
            // float keys compare as values: -0.0 equals 0.0 (std::map with operator<)
            let key = if i.dtype == TType::F32 && f32::from_le_bytes(key.try_into().unwrap()) == 0.0 { &[0u8; 4][..] } else { key };
            let next = firsts.len() as i64;
            let at = *map.entry(key).or_insert_with(|| {
                firsts.push(k);
                next
            });
            idx.push(at);
        }
        io.outs[0].resize(vec![firsts.len()]);
        let src: Vec<Option<usize>> = firsts.iter().map(|&k| Some(k)).collect();
        take(i, &mut io.outs[0], &src, None)?;
        io.outs[1].resize(vec![idx.len()]);
        put_ints(&mut io.outs[1], &idx)
    }
}

/// select.cc: SELECT takes equal shapes, or a rank-1 condition choosing along the first dimension; SELECT_V2
/// broadcasts all three
struct Select {
    v2: bool,
}

impl Select {
    fn shape(&self, io: &Io) -> Result<Vec<usize>> {
        let (c, x, y) = (io.input(0)?, io.input(1)?, io.input(2)?);
        if self.v2 {
            return broadcast_shape(&c.shape, &broadcast_shape(&x.shape, &y.shape)?);
        }
        let rank_one = c.shape.len() == 1 && x.shape.len() > 1 && c.shape[0] == x.shape[0];
        if x.shape != y.shape || !(c.shape == x.shape || rank_one) {
            return err("SELECT needs equal shapes (or a rank-1 condition over the first dimension)");
        }
        Ok(x.shape.clone())
    }
}

impl Kernel for Select {
    fn prepare(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        let s = self.shape(io)?;
        io.outs[0].resize(s);
        Ok(())
    }
    fn eval(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        let out = self.shape(io)?;
        let (c, x, y) = (io.input(0)?, io.input(1)?, io.input(2)?);
        let cd = c.data();
        let n: usize = out.iter().product();
        let es = x.dtype.size();
        let pick = |s: &[usize], idx: &[usize]| -> usize {
            let st = strides(s);
            let off = idx.len() - s.len();
            s.iter().enumerate().map(|(d, &dim)| if dim == 1 { 0 } else { idx[off + d] * st[d] }).sum()
        };
        let o = &mut io.outs[0];
        let (xd, yd) = (x.data(), y.data());
        let od = o.data_mut();
        let rank_one = !self.v2 && c.shape.len() == 1 && x.shape.len() > 1;
        let inner = if rank_one { n / c.shape[0].max(1) } else { 1 };
        each_index(&out, |k, idx| {
            let ck = if rank_one { k / inner } else if self.v2 { pick(&c.shape, idx) } else { k };
            let (src, at) = if cd[ck] != 0 { (xd, if self.v2 { pick(&x.shape, idx) } else { k }) } else { (yd, if self.v2 { pick(&y.shape, idx) } else { k }) };
            od[k * es..(k + 1) * es].copy_from_slice(&src[at * es..(at + 1) * es]);
        });
        Ok(())
    }
}

/// add_n.cc: every input added in order, starting from 0
struct AddN;

impl Kernel for AddN {
    fn prepare(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        let s = io.input(0)?.shape.clone();
        for k in 1..io.n_in() {
            if io.input(k)?.shape != s {
                return err("all inputs must have the same shape");
            }
        }
        io.outs[0].resize(s);
        Ok(())
    }
    fn eval(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        let ins: Vec<&Tensor> = (0..io.n_in()).map(|k| io.input(k)).collect::<Result<_>>()?;
        let o = &mut io.outs[0];
        match o.dtype {
            TType::F32 => {
                let v = o.as_mut_slice::<f32>();
                for (k, x) in v.iter_mut().enumerate() {
                    *x = ins.iter().fold(0.0f32, |s, t| s + t.as_slice::<f32>()[k]);
                }
            }
            TType::I32 => {
                let v = o.as_mut_slice::<i32>();
                for (k, x) in v.iter_mut().enumerate() {
                    *x = ins.iter().fold(0i32, |s, t| s.wrapping_add(t.as_slice::<i32>()[k]));
                }
            }
            t => return err(format!("{t:?} is not supported")),
        }
        Ok(())
    }
}
