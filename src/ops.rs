//! CPU kernels for the ops around the Edge TPU op, written to give the same bytes as TensorFlow Lite 2.5
//! (the tflite_runtime under pycoral 2.0): its fixed-point arithmetic (kernels/internal/common.h,
//! quantization_util.cc), its float/double mix, and its sort order.

use crate::tflite::{Act, Kind, Node, TType, TensorInfo};
use crate::{Error, Result};

pub struct In<'a> {
    pub info: &'a TensorInfo,
    pub data: &'a [u8],
}

pub struct Out<'a> {
    pub info: &'a TensorInfo,
    pub data: &'a mut [u8],
}

fn err<T>(msg: String) -> Result<T> {
    Err(Error(msg))
}

// *** element access (little endian, any alignment) ***

fn rd_i64(t: TType, d: &[u8], i: usize) -> i64 {
    match t {
        TType::U8 => d[i] as i64,
        TType::I8 => d[i] as i8 as i64,
        TType::Bool => (d[i] != 0) as i64,
        TType::I16 => i16::from_le_bytes([d[2 * i], d[2 * i + 1]]) as i64,
        TType::I32 => i32::from_le_bytes(d[4 * i..4 * i + 4].try_into().unwrap()) as i64,
        TType::I64 => i64::from_le_bytes(d[8 * i..8 * i + 8].try_into().unwrap()),
        _ => rd_f64(t, d, i) as i64,
    }
}

fn rd_f64(t: TType, d: &[u8], i: usize) -> f64 {
    match t {
        TType::F32 => rd_f32(d, i) as f64,
        TType::F64 => f64::from_le_bytes(d[8 * i..8 * i + 8].try_into().unwrap()),
        _ => rd_i64(t, d, i) as f64,
    }
}

fn rd_f32(d: &[u8], i: usize) -> f32 {
    f32::from_le_bytes(d[4 * i..4 * i + 4].try_into().unwrap())
}

fn wr_i64(t: TType, d: &mut [u8], i: usize, v: i64) {
    match t {
        TType::U8 | TType::I8 => d[i] = v as u8,
        TType::Bool => d[i] = (v != 0) as u8,
        TType::I16 => d[2 * i..2 * i + 2].copy_from_slice(&(v as i16).to_le_bytes()),
        TType::I32 => d[4 * i..4 * i + 4].copy_from_slice(&(v as i32).to_le_bytes()),
        TType::I64 => d[8 * i..8 * i + 8].copy_from_slice(&v.to_le_bytes()),
        TType::F32 => wr_f32(d, i, v as f32),
        TType::F64 => d[8 * i..8 * i + 8].copy_from_slice(&(v as f64).to_le_bytes()),
        _ => {}
    }
}

fn wr_f32(d: &mut [u8], i: usize, v: f32) {
    d[4 * i..4 * i + 4].copy_from_slice(&v.to_le_bytes());
}

fn type_range(t: TType) -> (i32, i32) {
    match t {
        TType::U8 => (0, 255),
        TType::I8 => (-128, 127),
        TType::I16 => (-32768, 32767),
        _ => (i32::MIN, i32::MAX),
    }
}

// *** fixed point (gemmlowp, kernels/internal/common.h, quantization_util.cc) ***

fn srdhm(a: i32, b: i32) -> i32 {
    if a == b && a == i32::MIN {
        return i32::MAX;
    }
    let ab = a as i64 * b as i64;
    let nudge = if ab >= 0 { 1i64 << 30 } else { 1 - (1i64 << 30) };
    ((ab + nudge) / (1i64 << 31)) as i32
}

fn rounding_divide_by_pot(x: i32, e: i32) -> i32 {
    let mask = ((1i64 << e) - 1) as i32;
    let rem = x & mask;
    let threshold = (mask >> 1) + (x < 0) as i32;
    (x >> e) + (rem > threshold) as i32
}

fn mul_smaller_than_one_exp(x: i32, m: i32, left_shift: i32) -> i32 {
    rounding_divide_by_pot(srdhm(x, m), -left_shift)
}

fn mul_by_quantized_multiplier(x: i32, m: i32, shift: i32) -> i32 {
    let (l, r) = (shift.max(0), (-shift).max(0));
    rounding_divide_by_pot(srdhm(x.wrapping_mul(1 << l), m), r)
}

fn frexp(x: f64) -> (f64, i32) {
    if x == 0.0 || !x.is_finite() {
        return (x, 0);
    }
    let bits = x.to_bits();
    let exp = ((bits >> 52) & 0x7ff) as i32;
    if exp == 0 {
        let (m, e) = frexp(x * 2f64.powi(54));
        return (m, e - 54);
    }
    (f64::from_bits((bits & !(0x7ffu64 << 52)) | (1022u64 << 52)), exp - 1022)
}

/// QuantizeMultiplier: double -> (Q31 multiplier, shift)
fn quantize_multiplier(d: f64) -> (i32, i32) {
    if d == 0.0 {
        return (0, 0);
    }
    let (q, mut shift) = frexp(d);
    let mut q_fixed = (q * (1i64 << 31) as f64).round() as i64;
    if q_fixed == 1i64 << 31 {
        q_fixed /= 2;
        shift += 1;
    }
    if shift < -31 {
        shift = 0;
        q_fixed = 0;
    }
    (q_fixed as i32, shift)
}

fn quantize_multiplier_smaller_than_one(d: f64) -> Result<(i32, i32)> {
    if !(d > 0.0 && d < 1.0) {
        return err(format!("quantized multiplier {d} is outside (0, 1)"));
    }
    Ok(quantize_multiplier(d))
}

/// CalculateActivationRangeQuantized
fn act_range_q(act: Act, o: &TensorInfo) -> (i32, i32) {
    let (qmin, qmax) = type_range(o.dtype);
    let (s, zp) = (o.scale(), o.zero_point());
    let q = |f: f32| zp + (f / s).round() as i32;
    match act {
        Act::None => (qmin, qmax),
        Act::Relu => (qmin.max(q(0.0)), qmax),
        Act::Relu6 => (qmin.max(q(0.0)), qmax.min(q(6.0))),
        Act::ReluN1To1 => (qmin.max(q(-1.0)), qmax.min(q(1.0))),
    }
}

/// CalculateActivationRange for float (int32 uses the same bounds, saturating)
fn act_range_f(act: Act) -> (f32, f32) {
    match act {
        Act::None => (f32::MIN, f32::MAX),
        Act::Relu => (0.0, f32::MAX),
        Act::Relu6 => (0.0, 6.0),
        Act::ReluN1To1 => (-1.0, 1.0),
    }
}

/// std::min(std::max(x, lo), hi), NaN passes through like in C++
fn clamp_f(x: f32, lo: f32, hi: f32) -> f32 {
    let v = if x < lo { lo } else { x };
    if hi < v { hi } else { v }
}

// *** broadcasting ***

/// (index into a, index into b) for every output element, numpy-style broadcasting in row-major order
fn broadcast(out: &[usize], a: &[usize], b: &[usize]) -> Result<Vec<(usize, usize)>> {
    let r = out.len().max(a.len()).max(b.len());
    let pad = |s: &[usize]| [vec![1; r - s.len()], s.to_vec()].concat();
    let (o, a, b) = (pad(out), pad(a), pad(b));
    let strides = |s: &[usize]| -> Result<Vec<usize>> {
        let mut st = vec![0; r];
        let mut acc = 1;
        for i in (0..r).rev() {
            st[i] = if s[i] == o[i] { acc } else if s[i] == 1 { 0 } else { return err(format!("cannot broadcast {s:?} to {o:?}")) };
            acc *= s[i];
        }
        Ok(st)
    };
    let (sa, sb) = (strides(&a)?, strides(&b)?);
    let n: usize = o.iter().product();
    let mut idx = vec![0; r];
    let (mut ia, mut ib) = (0, 0);
    let mut res = Vec::with_capacity(n);
    for _ in 0..n {
        res.push((ia, ib));
        for d in (0..r).rev() {
            idx[d] += 1;
            ia += sa[d];
            ib += sb[d];
            if idx[d] < o[d] {
                break;
            }
            ia -= sa[d] * o[d];
            ib -= sb[d] * o[d];
            idx[d] = 0;
        }
    }
    Ok(res)
}

// *** ops ***

pub fn eval(node: &Node, ins: &[In], outs: &mut [Out]) -> Result<()> {
    match &node.kind {
        Kind::Add | Kind::Sub | Kind::Mul | Kind::Div => binary(node, &ins[0], &ins[1], &mut outs[0]),
        Kind::Quantize => quantize(&ins[0], &mut outs[0]),
        Kind::Dequantize => dequantize(&ins[0], &mut outs[0]),
        Kind::Cast => cast(&ins[0], &mut outs[0]),
        Kind::Reshape => copy_all(&ins[0], &mut outs[0]),
        Kind::ArgMax => arg_max(&ins[0], &ins[1], &mut outs[0]),
        Kind::FloorDiv => floor_div(&ins[0], &ins[1], &mut outs[0]),
        Kind::Sqrt => unary_f32(&ins[0], &mut outs[0], f32::sqrt),
        Kind::Pack { axis } => pack(ins, &mut outs[0], *axis),
        Kind::Unpack { axis } => unpack(&ins[0], outs, *axis),
        Kind::Concat { axis } => concat(node, ins, &mut outs[0], *axis),
        Kind::GatherNd => gather_nd(&ins[0], &ins[1], &mut outs[0]),
        Kind::DetectionPostProcess(o) => dpp::eval(o, ins, outs),
        Kind::EdgeTpu(_) => err("the Edge TPU op runs on the device".into()),
    }
}

fn binary(node: &Node, a: &In, b: &In, o: &mut Out) -> Result<()> {
    let pairs = broadcast(&o.info.shape, &a.info.shape, &b.info.shape)?;
    let (ta, tb, to) = (a.info.dtype, b.info.dtype, o.info.dtype);
    if ta != tb || ta != to {
        return err(format!("{}: mixed types {ta:?} {tb:?} -> {to:?}", node.name));
    }
    match (to, &node.kind) {
        (TType::U8 | TType::I8, Kind::Add | Kind::Sub) => {
            // add.cc / sub.cc Prepare: both inputs rescaled to twice the larger scale, shifted left by 20
            let (sa, sb, so) = (a.info.scale(), b.info.scale(), o.info.scale());
            let left_shift = 20;
            let twice_max = (2.0 * if sa < sb { sb } else { sa }) as f64;
            let (m1, s1) = quantize_multiplier_smaller_than_one(sa as f64 / twice_max)?;
            let (mut m2, s2) = quantize_multiplier_smaller_than_one(sb as f64 / twice_max)?;
            let (mo, s_o) = quantize_multiplier_smaller_than_one(twice_max / (((1 << left_shift) as f32 * so) as f64))?;
            if matches!(node.kind, Kind::Sub) {
                m2 = -m2;
            }
            let (off1, off2, offo) = (-a.info.zero_point(), -b.info.zero_point(), o.info.zero_point());
            let (lo, hi) = act_range_q(node.act, o.info);
            for (i, &(ia, ib)) in pairs.iter().enumerate() {
                let x = (off1 + rd_i64(ta, a.data, ia) as i32) * (1 << left_shift);
                let y = (off2 + rd_i64(tb, b.data, ib) as i32) * (1 << left_shift);
                let sum = mul_smaller_than_one_exp(x, m1, s1) + mul_smaller_than_one_exp(y, m2, s2);
                let v = mul_smaller_than_one_exp(sum, mo, s_o) + offo;
                wr_i64(to, o.data, i, v.clamp(lo, hi) as i64);
            }
        }
        (TType::U8 | TType::I8, Kind::Mul) => {
            // mul.cc Prepare: the product of the float scales, in float, then widened
            let (m, shift) = quantize_multiplier((a.info.scale() * b.info.scale() / o.info.scale()) as f64);
            let (off1, off2, offo) = (-a.info.zero_point(), -b.info.zero_point(), o.info.zero_point());
            let (lo, hi) = act_range_q(node.act, o.info);
            for (i, &(ia, ib)) in pairs.iter().enumerate() {
                let x = off1 + rd_i64(ta, a.data, ia) as i32;
                let y = off2 + rd_i64(tb, b.data, ib) as i32;
                let v = offo + mul_by_quantized_multiplier(x.wrapping_mul(y), m, shift);
                wr_i64(to, o.data, i, v.clamp(lo, hi) as i64);
            }
        }
        (TType::F32, k) => {
            let (lo, hi) = act_range_f(node.act);
            for (i, &(ia, ib)) in pairs.iter().enumerate() {
                let (x, y) = (rd_f32(a.data, ia), rd_f32(b.data, ib));
                let v = match k {
                    Kind::Add => x + y,
                    Kind::Sub => x - y,
                    Kind::Mul => x * y,
                    _ => x / y,
                };
                wr_f32(o.data, i, clamp_f(v, lo, hi));
            }
        }
        (TType::I32 | TType::I64, Kind::Add | Kind::Sub | Kind::Mul) => {
            let (lo, hi) = act_range_f(node.act);
            let (lo, hi) = (if node.act == Act::None { i64::MIN } else { lo as i64 }, if hi == f32::MAX { i64::MAX } else { hi as i64 });
            for (i, &(ia, ib)) in pairs.iter().enumerate() {
                let (x, y) = (rd_i64(ta, a.data, ia), rd_i64(tb, b.data, ib));
                let v = match node.kind {
                    Kind::Add => x.wrapping_add(y),
                    Kind::Sub => x.wrapping_sub(y),
                    _ => x.wrapping_mul(y),
                };
                // int32 arithmetic wraps at 32 bits before the activation clamp
                let v = if to == TType::I32 { v as i32 as i64 } else { v };
                wr_i64(to, o.data, i, v.clamp(lo, hi));
            }
        }
        (t, _) => return err(format!("{} on {t:?} is not supported", node.name)),
    }
    Ok(())
}

fn quantize(i: &In, o: &mut Out) -> Result<()> {
    let (lo, hi) = type_range(o.info.dtype);
    let n = o.info.elements();
    match (i.info.dtype, o.info.dtype) {
        (TType::F32, TType::U8 | TType::I8) => {
            // optimized_ops::AffineQuantize: divides by the scale as a double
            let (scale, zp) = (o.info.scale() as f64, o.info.zero_point());
            for k in 0..n {
                let v = ((rd_f32(i.data, k) as f64) / scale).round() as i32 + zp;
                wr_i64(o.info.dtype, o.data, k, v.clamp(lo, hi) as i64);
            }
        }
        (TType::U8 | TType::I8, TType::U8 | TType::I8) => {
            // Requantize with QuantizeMultiplier(input scale / output scale)
            let (m, shift) = quantize_multiplier(i.info.scale() as f64 / o.info.scale() as f64);
            let (izp, ozp) = (i.info.zero_point(), o.info.zero_point());
            for k in 0..n {
                let v = mul_by_quantized_multiplier(rd_i64(i.info.dtype, i.data, k) as i32 - izp, m, shift) + ozp;
                wr_i64(o.info.dtype, o.data, k, v.clamp(lo, hi) as i64);
            }
        }
        (a, b) => return err(format!("QUANTIZE {a:?} -> {b:?} is not supported")),
    }
    Ok(())
}

fn dequantize(i: &In, o: &mut Out) -> Result<()> {
    if !matches!(i.info.dtype, TType::U8 | TType::I8 | TType::I16) || o.info.dtype != TType::F32 {
        return err(format!("DEQUANTIZE {:?} -> {:?} is not supported", i.info.dtype, o.info.dtype));
    }
    let (scale, zp) = (i.info.scale() as f64, i.info.zero_point() as i64);
    for k in 0..o.info.elements() {
        wr_f32(o.data, k, (scale * (rd_i64(i.info.dtype, i.data, k) - zp) as f64) as f32);
    }
    Ok(())
}

fn cast(i: &In, o: &mut Out) -> Result<()> {
    for k in 0..o.info.elements() {
        match o.info.dtype {
            TType::F32 => wr_f32(o.data, k, if i.info.dtype == TType::F64 { rd_f64(TType::F64, i.data, k) as f32 } else if i.info.dtype == TType::F32 { rd_f32(i.data, k) } else { rd_i64(i.info.dtype, i.data, k) as f32 }),
            TType::Bool => wr_i64(TType::Bool, o.data, k, (rd_f64(i.info.dtype, i.data, k) != 0.0) as i64),
            t => wr_i64(t, o.data, k, rd_i64(i.info.dtype, i.data, k)),
        }
    }
    Ok(())
}

fn copy_all(i: &In, o: &mut Out) -> Result<()> {
    let n = o.data.len();
    o.data.copy_from_slice(i.data.get(..n).ok_or_else(|| Error("RESHAPE to more elements than its input".into()))?);
    Ok(())
}

fn unary_f32(i: &In, o: &mut Out, f: fn(f32) -> f32) -> Result<()> {
    if i.info.dtype != TType::F32 {
        return err(format!("{:?} input is not supported here (float32 only)", i.info.dtype));
    }
    for k in 0..o.info.elements() {
        wr_f32(o.data, k, f(rd_f32(i.data, k)));
    }
    Ok(())
}

fn axis_of(axis: i64, rank: usize) -> Result<usize> {
    let a = if axis < 0 { axis + rank as i64 } else { axis };
    if a < 0 || a as usize >= rank.max(1) {
        return err(format!("axis {axis} out of range for rank {rank}"));
    }
    Ok(a as usize)
}

fn arg_max(i: &In, axis: &In, o: &mut Out) -> Result<()> {
    let shape = &i.info.shape;
    let ax = axis_of(rd_i64(axis.info.dtype, axis.data, 0), shape.len())?;
    let (outer, len, inner) = (shape[..ax].iter().product::<usize>(), shape[ax], shape[ax + 1..].iter().product::<usize>());
    let t = i.info.dtype;
    let float = matches!(t, TType::F32 | TType::F64);
    for out in 0..outer {
        for inn in 0..inner {
            // the first index of the largest value (std::greater: ties keep the earlier one)
            let at = |k: usize| (out * len + k) * inner + inn;
            let mut best = 0;
            for k in 1..len {
                let better = if float { rd_f64(t, i.data, at(k)) > rd_f64(t, i.data, at(best)) } else { rd_i64(t, i.data, at(k)) > rd_i64(t, i.data, at(best)) };
                if better {
                    best = k;
                }
            }
            wr_i64(o.info.dtype, o.data, out * inner + inn, best as i64);
        }
    }
    Ok(())
}

fn floor_div(a: &In, b: &In, o: &mut Out) -> Result<()> {
    let pairs = broadcast(&o.info.shape, &a.info.shape, &b.info.shape)?;
    for (k, &(ia, ib)) in pairs.iter().enumerate() {
        match o.info.dtype {
            TType::F32 => wr_f32(o.data, k, (rd_f32(a.data, ia) / rd_f32(b.data, ib)).floor()),
            TType::I32 => {
                let (x, y) = (rd_i64(a.info.dtype, a.data, ia), rd_i64(b.info.dtype, b.data, ib));
                if y == 0 {
                    return err("FLOOR_DIV: division by 0".into());
                }
                // reference_ops::FloorDiv: std::floor of the double quotient
                wr_i64(TType::I32, o.data, k, (x as f64 / y as f64).floor() as i64);
            }
            t => return err(format!("FLOOR_DIV on {t:?} is not supported")),
        }
    }
    Ok(())
}

/// split `shape` at `axis` into (outer elements, elements from axis on)
fn outer_inner(shape: &[usize], axis: usize) -> (usize, usize) {
    (shape[..axis].iter().product(), shape[axis..].iter().product())
}

fn pack(ins: &[In], o: &mut Out, axis: i32) -> Result<()> {
    let es = o.info.dtype.size();
    let ax = axis_of(axis as i64, o.info.shape.len())?;
    let (outer, _) = outer_inner(&ins[0].info.shape, ax);
    let copy = ins[0].info.elements() / outer.max(1) * es;
    for k in 0..outer {
        for (n, i) in ins.iter().enumerate() {
            let d = (k * ins.len() + n) * copy;
            o.data[d..d + copy].copy_from_slice(&i.data[k * copy..(k + 1) * copy]);
        }
    }
    Ok(())
}

fn unpack(i: &In, outs: &mut [Out], axis: i32) -> Result<()> {
    let es = i.info.dtype.size();
    let ax = axis_of(axis as i64, i.info.shape.len())?;
    let outer: usize = i.info.shape[..ax].iter().product();
    let copy = outs[0].info.elements() / outer.max(1) * es;
    let n = outs.len();
    for k in 0..outer {
        for (m, o) in outs.iter_mut().enumerate() {
            let s = (k * n + m) * copy;
            o.data[k * copy..(k + 1) * copy].copy_from_slice(&i.data[s..s + copy]);
        }
    }
    Ok(())
}

fn concat(node: &Node, ins: &[In], o: &mut Out, axis: i32) -> Result<()> {
    if node.act != Act::None {
        return err("CONCATENATION with a fused activation is not supported".into());
    }
    let es = o.info.dtype.size();
    let ax = axis_of(axis as i64, o.info.shape.len())?;
    let outer: usize = o.info.shape[..ax].iter().product();
    let base: usize = o.info.shape[ax + 1..].iter().product();
    let mut d = 0;
    for k in 0..outer {
        for i in ins {
            let n = i.info.shape[ax] * base;
            let src = &i.data[k * n * es..(k + 1) * n * es];
            let rescale = o.info.dtype == TType::U8 && (i.info.zero_point() != o.info.zero_point() || i.info.scale() != o.info.scale());
            if rescale {
                // ConcatenationWithScaling (uint8 only; int8 inputs share the output's parameters)
                let scale = i.info.scale() * (1.0 / o.info.scale());
                let bias = -(i.info.zero_point() as f32) * scale;
                for (j, &q) in src.iter().enumerate() {
                    let v = (q as f32 * scale + bias).round() as i32 + o.info.zero_point();
                    o.data[d + j] = v.clamp(0, 255) as u8;
                }
            } else {
                o.data[d..d + n * es].copy_from_slice(src);
            }
            d += n * es;
        }
    }
    Ok(())
}

fn gather_nd(p: &In, idx: &In, o: &mut Out) -> Result<()> {
    let es = p.info.dtype.size();
    let ishape = &idx.info.shape;
    let nd = *ishape.last().ok_or_else(|| Error("GATHER_ND indices without dimensions".into()))?;
    let pshape = &p.info.shape;
    if nd > pshape.len() {
        return err(format!("GATHER_ND: {nd} index dimensions for params of rank {}", pshape.len()));
    }
    let slices: usize = ishape[..ishape.len() - 1].iter().product();
    let slice: usize = pshape[nd..].iter().product::<usize>() * es;
    let mut dims_to_count = vec![0; nd];
    let mut remain: usize = pshape.iter().product();
    for (j, c) in dims_to_count.iter_mut().enumerate() {
        *c = remain / pshape[j].max(1);
        remain = *c;
    }
    for s in 0..slices {
        let mut from = 0;
        for j in 0..nd {
            let v = rd_i64(idx.info.dtype, idx.data, s * nd + j);
            if v < 0 || v as usize >= pshape[j] {
                return err(format!("GATHER_ND index {v} out of range [0, {})", pshape[j]));
            }
            from += v as usize * dims_to_count[j];
        }
        o.data[s * slice..(s + 1) * slice].copy_from_slice(&p.data[from * es..from * es + slice]);
    }
    Ok(())
}

/// TFLite_Detection_PostProcess (kernels/detection_postprocess.cc): box decoding and non-max suppression
pub mod dpp {
    use super::{err, rd_f32, wr_f32, In, Out};
    use crate::flex::Ref;
    use crate::tflite::TType;
    use crate::Result;

    pub struct Options {
        max_detections: usize,
        max_classes_per_detection: usize,
        detections_per_class: usize,
        score_threshold: f32,
        iou_threshold: f32,
        num_classes: usize,
        use_regular_nms: bool,
        scale: [f32; 4],
    }

    impl Options {
        pub fn parse(m: Ref) -> Self {
            let f = |k: &str| m.get(k).map_or(0.0, |v| v.as_f64() as f32);
            let i = |k: &str| m.get(k).map_or(0, |v| v.as_i64()) as usize;
            Options {
                max_detections: i("max_detections"),
                max_classes_per_detection: i("max_classes_per_detection"),
                detections_per_class: m.get("detections_per_class").map_or(100, |v| v.as_i64() as usize),
                score_threshold: f("nms_score_threshold"),
                iou_threshold: f("nms_iou_threshold"),
                num_classes: i("num_classes"),
                use_regular_nms: m.get("use_regular_nms").is_some_and(|v| v.as_bool()),
                scale: [f("y_scale"), f("x_scale"), f("h_scale"), f("w_scale")],
            }
        }
    }

    /// std::partial_sort(indices, k) by decreasing value: heap select, then sort_heap. The result for equal values
    /// depends on these exact heap steps (identical in libstdc++ and the MSVC STL), and quantized scores tie a lot.
    fn decreasing_partial_arg_sort(values: &[f32], k: usize) -> Vec<usize> {
        let less = |a: usize, b: usize| values[a] > values[b];
        let mut ix: Vec<usize> = (0..values.len()).collect();
        // __adjust_heap: move the hole at `hole` down to a leaf, then push `v` back up (max-heap under `less`)
        fn adjust(ix: &mut [usize], mut hole: usize, len: usize, v: usize, less: &dyn Fn(usize, usize) -> bool) {
            let top = hole;
            let mut child = hole;
            while child < (len.saturating_sub(1)) / 2 {
                child = 2 * (child + 1);
                if less(ix[child], ix[child - 1]) {
                    child -= 1;
                }
                ix[hole] = ix[child];
                hole = child;
            }
            if len & 1 == 0 && len >= 2 && child == (len - 2) / 2 {
                child = 2 * (child + 1);
                ix[hole] = ix[child - 1];
                hole = child - 1;
            }
            let mut parent = hole.wrapping_sub(1) / 2;
            while hole > top && less(ix[parent], v) {
                ix[hole] = ix[parent];
                hole = parent;
                parent = hole.wrapping_sub(1) / 2;
            }
            ix[hole] = v;
        }
        let mid = k.min(ix.len());
        if mid >= 2 {
            let mut parent = (mid - 2) / 2;
            loop {
                let v = ix[parent];
                adjust(&mut ix, parent, mid, v, &less);
                if parent == 0 {
                    break;
                }
                parent -= 1;
            }
        }
        for i in mid..ix.len() {
            if mid > 0 && less(ix[i], ix[0]) {
                // __pop_heap(first, middle, i)
                let v = ix[i];
                ix[i] = ix[0];
                adjust(&mut ix, 0, mid, v, &less);
            }
        }
        let mut last = mid;
        while last > 1 {
            last -= 1;
            let v = ix[last];
            ix[last] = ix[0];
            adjust(&mut ix, 0, last, v, &less);
        }
        ix.truncate(mid);
        ix
    }

    #[cfg(test)]
    mod tests {
        /// the cases of tests/partial_sort.cpp (same LCG), one line each: n k and the first k indices
        fn cases() -> String {
            let mut state: u64 = 12345;
            let mut next = || {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (state >> 33) as u32
            };
            let mut out = String::new();
            for _ in 0..2000 {
                let n = 1 + next() % 300;
                let k = 1 + next() % n;
                let distinct = 1 + next() % 12;
                let v: Vec<f32> = (0..n).map(|_| (next() % distinct) as f32 * 0.25).collect();
                let ix = super::decreasing_partial_arg_sort(&v, k as usize);
                out += &format!("{n} {k}");
                for i in ix {
                    out += &format!(" {i}");
                }
                out += "\n";
            }
            out
        }

        #[test]
        fn partial_sort_matches_the_cpp_standard_library() {
            if let Ok(path) = std::env::var("PARTIAL_SORT_REFERENCE") {
                let want = std::fs::read_to_string(path).unwrap().replace("\r\n", "\n");
                for (k, (a, b)) in cases().lines().zip(want.lines()).enumerate() {
                    assert_eq!(a, b, "case {k}");
                }
            }
            // FNV-1a of the output of tests/partial_sort.cpp built against the MSVC STL (clang 23, 2026-10-30)
            let h = cases().bytes().fold(0xcbf29ce484222325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100000001b3));
            assert_eq!(h, EXPECTED_FNV);
        }

        const EXPECTED_FNV: u64 = 4091578481920238488;
    }

    type Box4 = [f32; 4]; // ymin, xmin, ymax, xmax

    fn iou(a: &Box4, b: &Box4) -> f32 {
        let area_a = (a[2] - a[0]) * (a[3] - a[1]);
        let area_b = (b[2] - b[0]) * (b[3] - b[1]);
        if area_a <= 0.0 || area_b <= 0.0 {
            return 0.0;
        }
        let fmax = |x: f32, y: f32| if x < y { y } else { x };
        let fmin = |x: f32, y: f32| if y < x { y } else { x };
        let inter = fmax(fmin(a[2], b[2]) - fmax(a[0], b[0]), 0.0) * fmax(fmin(a[3], b[3]) - fmax(a[1], b[1]), 0.0);
        inter / (area_a + area_b - inter)
    }

    /// NonMaxSuppressionSingleClassHelper: indices of the kept boxes, best first
    fn nms_single(o: &Options, boxes: &[Box4], scores: &[f32], max_detections: usize) -> Result<Vec<usize>> {
        if !(o.iou_threshold > 0.0 && o.iou_threshold <= 1.0) {
            return err(format!("detection postprocess: IoU threshold {} outside (0, 1]", o.iou_threshold));
        }
        if boxes.iter().any(|b| b[0] > b[2] || b[1] > b[3]) {
            return err("detection postprocess: a decoded box has min > max".into());
        }
        let (mut keep_scores, mut keep) = (Vec::new(), Vec::new());
        for (i, &s) in scores.iter().enumerate() {
            if s >= o.score_threshold {
                keep_scores.push(s);
                keep.push(i);
            }
        }
        let sorted = decreasing_partial_arg_sort(&keep_scores, keep_scores.len());
        let size = keep.len().min(max_detections);
        let mut active = vec![true; keep.len()];
        let mut num_active = keep.len();
        let mut selected = Vec::new();
        for i in 0..keep.len() {
            if num_active == 0 || selected.len() >= size {
                break;
            }
            if !active[i] {
                continue;
            }
            selected.push(keep[sorted[i]]);
            active[i] = false;
            num_active -= 1;
            for j in i + 1..keep.len() {
                if active[j] && iou(&boxes[keep[sorted[i]]], &boxes[keep[sorted[j]]]) > o.iou_threshold {
                    active[j] = false;
                    num_active -= 1;
                }
            }
        }
        Ok(selected)
    }

    pub fn eval(o: &Options, ins: &[In], outs: &mut [Out]) -> Result<()> {
        let (enc, cls, anc) = (&ins[0], &ins[1], &ins[2]);
        if enc.info.shape.len() != 3 || cls.info.shape.len() != 3 || anc.info.shape.len() != 2 || outs.len() != 4 {
            return err("detection postprocess: unexpected tensor ranks".into());
        }
        let (num_boxes, enc_len, with_bg) = (enc.info.shape[1], enc.info.shape[2], cls.info.shape[2]);
        if enc_len < 4 || cls.info.shape[1] != num_boxes || o.num_classes == 0 || with_bg < o.num_classes || with_bg - o.num_classes > 1 {
            return err("detection postprocess: inconsistent box / class tensors".into());
        }
        let label_offset = with_bg - o.num_classes;
        // Dequantizer: (float(q) - int zero point) * float scale
        let deq = |t: &In, i: usize| -> f32 {
            let zp = t.info.zero_point() as f32 as i32;
            (t.data[i] as f32 - zp as f32) * t.info.scale()
        };
        let mut boxes = Vec::with_capacity(num_boxes);
        for idx in 0..num_boxes {
            let (b, a): ([f32; 4], [f32; 4]) = match enc.info.dtype {
                TType::U8 => (std::array::from_fn(|k| deq(enc, idx * enc_len + k)), std::array::from_fn(|k| deq(anc, idx * 4 + k))),
                TType::F32 => (std::array::from_fn(|k| rd_f32(enc.data, idx * enc_len + k)), std::array::from_fn(|k| rd_f32(anc.data, idx * 4 + k))),
                t => return err(format!("detection postprocess on {t:?} boxes is not supported")),
            };
            // DecodeCenterSizeBoxes, in double like the original
            let s = o.scale;
            let yc = (b[0] as f64 / s[0] as f64 * a[2] as f64 + a[0] as f64) as f32;
            let xc = (b[1] as f64 / s[1] as f64 * a[3] as f64 + a[1] as f64) as f32;
            let hh = (0.5 * (b[2] as f64 / s[2] as f64).exp() * a[2] as f64) as f32;
            let hw = (0.5 * (b[3] as f64 / s[3] as f64).exp() * a[3] as f64) as f32;
            boxes.push([yc - hh, xc - hw, yc + hh, xc + hw]);
        }
        let scores: Vec<f32> = match cls.info.dtype {
            TType::U8 => (0..num_boxes * with_bg).map(|i| deq(cls, i)).collect(),
            TType::F32 => (0..num_boxes * with_bg).map(|i| rd_f32(cls.data, i)).collect(),
            t => return err(format!("detection postprocess on {t:?} scores is not supported")),
        };
        let (rows, n) = if o.use_regular_nms { regular(o, &boxes, &scores, with_bg, label_offset)? } else { fast(o, &boxes, &scores, with_bg, label_offset)? };
        let [ob, oc, os, on] = outs else { unreachable!() };
        let slots = (ob.data.len() / 16).min(oc.data.len() / 4).min(os.data.len() / 4);
        if rows.len() > slots {
            return err(format!("detection postprocess: {} detections for {slots} output slots", rows.len()));
        }
        // the slots past the detections: zeros (the fast path in TFLite leaves them as they were)
        for k in 0..slots {
            let (bx, c, s) = rows.get(k).copied().unwrap_or(([0.0; 4], 0.0, 0.0));
            for (j, v) in bx.iter().enumerate() {
                wr_f32(ob.data, 4 * k + j, *v);
            }
            wr_f32(oc.data, k, c);
            wr_f32(os.data, k, s);
        }
        wr_f32(on.data, 0, n as f32);
        Ok(())
    }

    type Row = (Box4, f32, f32);

    /// NonMaxSuppressionMultiClassFastHelper: NMS on each box's best class, then its top classes
    fn fast(o: &Options, boxes: &[Box4], scores: &[f32], with_bg: usize, off: usize) -> Result<(Vec<Row>, usize)> {
        if o.max_classes_per_detection == 0 {
            return err("detection postprocess: max_classes_per_detection must be > 0".into());
        }
        let per_anchor = o.max_classes_per_detection.min(o.num_classes);
        let mut sorted_classes = Vec::with_capacity(boxes.len());
        let mut max_scores = Vec::with_capacity(boxes.len());
        for row in 0..boxes.len() {
            let s = &scores[row * with_bg + off..row * with_bg + off + o.num_classes];
            let ci = decreasing_partial_arg_sort(s, per_anchor);
            max_scores.push(s[ci[0]]);
            sorted_classes.push(ci);
        }
        let selected = nms_single(o, boxes, &max_scores, o.max_detections)?;
        let mut rows = Vec::new();
        for (out_i, &sel) in selected.iter().enumerate() {
            for col in 0..per_anchor {
                let c = sorted_classes[sel][col];
                // detection outputs are laid out max_classes_per_detection per box
                let slot = o.max_classes_per_detection * out_i + col;
                if rows.len() <= slot {
                    rows.resize(slot + 1, ([0.0; 4], 0.0, 0.0));
                }
                rows[slot] = (boxes[sel], c as f32, scores[sel * with_bg + off + c]);
            }
        }
        Ok((rows, selected.len()))
    }

    /// NonMaxSuppressionMultiClassRegularHelper: NMS per class, merged by score
    fn regular(o: &Options, boxes: &[Box4], scores: &[f32], with_bg: usize, off: usize) -> Result<(Vec<Row>, usize)> {
        if o.detections_per_class == 0 {
            return err("detection postprocess: detections_per_class must be > 0".into());
        }
        let n = boxes.len();
        let mut idx = vec![0usize; n + o.max_detections];
        let mut sc = vec![0f32; n + o.max_detections];
        let mut size = 0;
        for col in 0..o.num_classes {
            let class_scores: Vec<f32> = (0..n).map(|r| scores[r * with_bg + col + off]).collect();
            let selected = nms_single(o, boxes, &class_scores, o.detections_per_class)?;
            let mut out_i = size;
            for &s in &selected {
                idx[out_i] = s * with_bg + col + off;
                sc[out_i] = class_scores[s];
                out_i += 1;
            }
            let k = out_i.min(o.max_detections);
            let order = decreasing_partial_arg_sort(&sc[..out_i], k);
            let (new_idx, new_sc): (Vec<usize>, Vec<f32>) = order.iter().map(|&t| (idx[t], sc[t])).unzip();
            idx[..k].copy_from_slice(&new_idx);
            sc[..k].copy_from_slice(&new_sc);
            size = k;
        }
        let rows = (0..size)
            .map(|i| {
                let anchor = idx[i] / with_bg;
                (boxes[anchor], (idx[i] - anchor * with_bg - off) as f32, sc[i])
            })
            .collect();
        Ok((rows, size))
    }
}
