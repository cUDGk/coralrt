//! What the kernels share: TFLite's fixed-point arithmetic, activation ranges, broadcasting, and whole-tensor
//! conversions between element types.

use crate::tensor::{TType, Tensor};
use crate::tflite::{Act, OpSpec};
use crate::{Error, Result};

pub fn err<T>(msg: impl Into<String>) -> Result<T> {
    Err(Error(msg.into()))
}

// *** options ***

/// field `i` of the builtin options table, or `d` when absent
pub fn opt<T: crate::fb::Scalar>(s: &OpSpec, i: usize, d: T) -> T {
    s.opts.map_or(d, |t| t.get(i, d))
}

pub fn opt_act(s: &OpSpec, i: usize) -> Result<Act> {
    Act::from_code(opt::<i8>(s, i, 0))
}

pub fn opt_vec(s: &OpSpec, i: usize) -> Vec<i32> {
    s.opts.map_or(Vec::new(), |t| t.scalars(i))
}

/// Padding (schema): SAME or VALID
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Padding {
    Same,
    Valid,
}

pub fn opt_padding(s: &OpSpec, i: usize) -> Result<Padding> {
    match opt::<i8>(s, i, 0) {
        0 => Ok(Padding::Same),
        1 => Ok(Padding::Valid),
        p => err(format!("padding {p} is not SAME or VALID")),
    }
}

/// ComputeOutSize and ComputePaddingWithOffset (padding.h) along one axis: (output size, padding before, 1 when the
/// total padding is odd and the extra one goes after)
pub fn out_size_and_pad(p: Padding, in_size: i32, filter: i32, stride: i32, dilation: i32) -> (i32, i32, i32) {
    let effective = (filter - 1) * dilation + 1;
    let out = if stride == 0 {
        0
    } else if p == Padding::Same {
        (in_size + stride - 1) / stride
    } else {
        (in_size + stride - effective) / stride
    };
    let total = ((out - 1) * stride + effective - in_size).max(0);
    (out, total / 2, total % 2)
}

// *** fixed point (gemmlowp, kernels/internal/common.h, quantization_util.cc) ***

pub fn type_range(t: TType) -> (i32, i32) {
    match t {
        TType::U8 => (0, 255),
        TType::I8 => (-128, 127),
        TType::I16 => (-32768, 32767),
        _ => (i32::MIN, i32::MAX),
    }
}

/// SaturatingRoundingDoublingHighMul
pub fn srdhm(a: i32, b: i32) -> i32 {
    if a == b && a == i32::MIN {
        return i32::MAX;
    }
    let ab = a as i64 * b as i64;
    let nudge = if ab >= 0 { 1i64 << 30 } else { 1 - (1i64 << 30) };
    ((ab + nudge) / (1i64 << 31)) as i32
}

pub use super::fixedpoint::rounding_divide_by_pot;

pub fn mul_smaller_than_one_exp(x: i32, m: i32, left_shift: i32) -> i32 {
    rounding_divide_by_pot(srdhm(x, m), -left_shift)
}

/// MultiplyByQuantizedMultiplier
pub fn mul_by_quantized_multiplier(x: i32, m: i32, shift: i32) -> i32 {
    let (l, r) = (shift.max(0), (-shift).max(0));
    rounding_divide_by_pot(srdhm(x.wrapping_mul(1 << l), m), r)
}

/// MultiplyByQuantizedMultiplier for an int64 accumulator (int16 kernels): the multiplier cut to 16 bits, one
/// rounding shift
pub fn mul_by_quantized_multiplier_i64(x: i64, m: i32, shift: i32) -> i32 {
    let reduced = if m < 0x7FFF0000 { ((m + (1 << 15)) >> 16) as i64 } else { 0x7FFF };
    let total_shift = 15 - shift;
    (x.wrapping_mul(reduced).wrapping_add(1i64 << (total_shift - 1)) >> total_shift) as i32
}

/// static_cast<int32_t>(std::round(x)) as x86 runs it (half away from zero; the minimum when out of range or NaN)
/// without the libm call that round() is on baseline x86-64: x - trunc(x) is exact for |x| < 2^31
pub fn round_i32(x: f64) -> i32 {
    if x.is_nan() || x.abs() >= 2147483648.0 {
        return cvtt(x.round(), false) as i32;
    }
    let t = x as i64;
    let f = x - t as f64;
    // rounding up to 2^31 is out of range too
    i32::try_from(t + (f >= 0.5) as i64 - (f <= -0.5) as i64).unwrap_or(i32::MIN)
}

pub fn frexp(x: f64) -> (f64, i32) {
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
pub fn quantize_multiplier(d: f64) -> (i32, i32) {
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

pub fn quantize_multiplier_smaller_than_one(d: f64) -> Result<(i32, i32)> {
    if !(d > 0.0 && d < 1.0) {
        return err(format!("quantized multiplier {d} is outside (0, 1)"));
    }
    Ok(quantize_multiplier(d))
}

/// CalculateActivationRangeQuantized
pub fn act_range_q(act: Act, o: &Tensor) -> (i32, i32) {
    let (qmin, qmax) = type_range(o.dtype);
    let (s, zp) = (o.scale(), o.zero_point());
    let q = |f: f32| zp + (f / s).round() as i32;
    match act {
        Act::Relu => (qmin.max(q(0.0)), qmax),
        Act::Relu6 => (qmin.max(q(0.0)), qmax.min(q(6.0))),
        Act::ReluN1To1 => (qmin.max(q(-1.0)), qmax.min(q(1.0))),
        _ => (qmin, qmax),
    }
}

/// CalculateActivationRange for float (int32 uses the same bounds, saturating)
pub fn act_range_f(act: Act) -> (f32, f32) {
    match act {
        Act::Relu => (0.0, f32::MAX),
        Act::Relu6 => (0.0, 6.0),
        Act::ReluN1To1 => (-1.0, 1.0),
        _ => (f32::MIN, f32::MAX),
    }
}

/// float -> int as x86 converts (cvttss2si / cvttsd2si): truncated, the type's minimum when out of range or NaN.
/// static_cast leaves that undefined; narrower integers go through the 32-bit conversion.
pub fn cvtt(x: f64, bits64: bool) -> i64 {
    let lim = if bits64 { 9223372036854775808.0 } else { 2147483648.0 };
    if x.is_nan() || x >= lim || x < -lim {
        if bits64 { i64::MIN } else { i32::MIN as i64 }
    } else {
        x as i64
    }
}

/// std::min(std::max(x, lo), hi), NaN passes through like in C++
pub fn clamp_f(x: f32, lo: f32, hi: f32) -> f32 {
    let v = if x < lo { lo } else { x };
    if hi < v { hi } else { v }
}

// *** broadcasting ***

/// Call f(output index, index into a, index into b) for every output element, numpy-style broadcasting in row-major
/// order. Neighbouring dimensions both inputs step through alike are merged, so the inner loop runs as long as it can
/// (the whole tensor when the shapes match or one side is a scalar).
pub fn broadcast(out: &[usize], a: &[usize], b: &[usize], mut f: impl FnMut(usize, usize, usize)) -> Result<()> {
    let r = out.len().max(a.len()).max(b.len());
    let pad = |s: &[usize]| [vec![1; r - s.len()], s.to_vec()].concat();
    let (o, a, b) = (pad(out), pad(a), pad(b));
    // (size, stride in a, stride in b) per dimension, innermost first
    let mut dims: Vec<(usize, usize, usize)> = Vec::new();
    let (mut acc_a, mut acc_b) = (1, 1);
    for d in (0..r).rev() {
        let stride = |s: &[usize], acc: usize| match s[d] {
            n if n == o[d] => Ok(acc),
            1 => Ok(0),
            _ => err(format!("cannot broadcast {s:?} to {o:?}")),
        };
        let (sa, sb) = (stride(&a, acc_a)?, stride(&b, acc_b)?);
        (acc_a, acc_b) = (acc_a * a[d], acc_b * b[d]);
        match dims.last_mut() {
            _ if o[d] == 1 => {}
            Some(inner) if inner.1 * inner.0 == sa && inner.2 * inner.0 == sb => inner.0 *= o[d],
            _ => dims.push((o[d], sa, sb)),
        }
    }
    let (n, sa, sb) = dims.first().copied().unwrap_or((1, 0, 0));
    let outer = dims.get(1..).unwrap_or_default();
    let total: usize = o.iter().product();
    let mut idx = vec![0; outer.len()];
    let (mut i, mut ia, mut ib) = (0, 0, 0);
    while i < total {
        for j in 0..n {
            f(i + j, ia + j * sa, ib + j * sb);
        }
        i += n;
        for (k, &(len, sa, sb)) in outer.iter().enumerate() {
            idx[k] += 1;
            ia += sa;
            ib += sb;
            if idx[k] < len {
                break;
            }
            ia -= sa * len;
            ib -= sb * len;
            idx[k] = 0;
        }
    }
    Ok(())
}

// *** whole tensors as one numeric type (the match on the type stays out of the per-element loops) ***

/// the elements of an integer or bool tensor
pub fn ints(t: &Tensor) -> Result<Vec<i64>> {
    Ok(match t.dtype {
        TType::U8 => t.as_slice::<u8>().iter().map(|&x| x as i64).collect(),
        TType::Bool => t.as_slice::<u8>().iter().map(|&x| (x != 0) as i64).collect(),
        TType::I8 => t.as_slice::<i8>().iter().map(|&x| x as i64).collect(),
        TType::I16 => t.as_slice::<i16>().iter().map(|&x| x as i64).collect(),
        TType::I32 => t.as_slice::<i32>().iter().map(|&x| x as i64).collect(),
        TType::U32 => t.as_slice::<u32>().iter().map(|&x| x as i64).collect(),
        TType::I64 => t.as_slice::<i64>().to_vec(),
        TType::U64 => t.as_slice::<u64>().iter().map(|&x| x as i64).collect(),
        d => return err(format!("{d:?} tensors are not supported here")),
    })
}

/// the elements of any numeric tensor as double (exact for every type but 64-bit integers)
pub fn f64s(t: &Tensor) -> Result<Vec<f64>> {
    Ok(match t.dtype {
        TType::F32 => t.as_slice::<f32>().iter().map(|&x| x as f64).collect(),
        TType::F64 => t.as_slice::<f64>().to_vec(),
        TType::F16 => t.as_slice::<u16>().iter().map(|&x| f16_to_f32(x) as f64).collect(),
        _ => ints(t)?.into_iter().map(|x| x as f64).collect(),
    })
}

/// store integers into a tensor of type `t`: integers wrap to its width, floats take the nearest value
pub fn put_ints(o: &mut Tensor, v: &[i64]) -> Result<()> {
    match o.dtype {
        TType::U8 => o.as_mut_slice::<u8>().iter_mut().zip(v).for_each(|(d, &x)| *d = x as u8),
        TType::Bool => o.as_mut_slice::<u8>().iter_mut().zip(v).for_each(|(d, &x)| *d = (x != 0) as u8),
        TType::I8 => o.as_mut_slice::<i8>().iter_mut().zip(v).for_each(|(d, &x)| *d = x as i8),
        TType::I16 => o.as_mut_slice::<i16>().iter_mut().zip(v).for_each(|(d, &x)| *d = x as i16),
        TType::I32 => o.as_mut_slice::<i32>().iter_mut().zip(v).for_each(|(d, &x)| *d = x as i32),
        TType::U32 => o.as_mut_slice::<u32>().iter_mut().zip(v).for_each(|(d, &x)| *d = x as u32),
        TType::I64 => o.as_mut_slice::<i64>().iter_mut().zip(v).for_each(|(d, &x)| *d = x),
        TType::U64 => o.as_mut_slice::<u64>().iter_mut().zip(v).for_each(|(d, &x)| *d = x as u64),
        TType::F32 => o.as_mut_slice::<f32>().iter_mut().zip(v).for_each(|(d, &x)| *d = x as f32),
        TType::F64 => o.as_mut_slice::<f64>().iter_mut().zip(v).for_each(|(d, &x)| *d = x as f64),
        TType::F16 => o.as_mut_slice::<u16>().iter_mut().zip(v).for_each(|(d, &x)| *d = f32_to_f16(x as f32)),
        d => return err(format!("{d:?} tensors are not supported here")),
    }
    Ok(())
}

/// store doubles into a tensor like static_cast: rounded to float, truncated to integers
pub fn put_f64s(o: &mut Tensor, v: &[f64]) -> Result<()> {
    match o.dtype {
        TType::F32 => o.as_mut_slice::<f32>().iter_mut().zip(v).for_each(|(d, &x)| *d = x as f32),
        TType::F64 => o.as_mut_slice::<f64>().iter_mut().zip(v).for_each(|(d, &x)| *d = x),
        TType::F16 => o.as_mut_slice::<u16>().iter_mut().zip(v).for_each(|(d, &x)| *d = f32_to_f16(x as f32)),
        TType::Bool => o.as_mut_slice::<u8>().iter_mut().zip(v).for_each(|(d, &x)| *d = (x != 0.0) as u8),
        _ => put_ints(o, &v.iter().map(|&x| x as i64).collect::<Vec<_>>())?,
    }
    Ok(())
}

/// f of every value a 1-byte integer type holds, indexed by its byte
pub fn byte_table(t: TType, f: impl Fn(i32) -> i32) -> [i32; 256] {
    std::array::from_fn(|b| f(if t == TType::I8 { b as u8 as i8 as i32 } else { b as i32 }))
}

/// An elementwise op on a 1-byte type has only 256 results: work them out once with the kernel itself and look each
/// element up (the same bytes, a fraction of the work on big tensors).
pub fn by_table(i: &Tensor, o: &mut Tensor, kernel: fn(&Tensor, &mut Tensor) -> Result<()>) -> Result<()> {
    let n = i.elements();
    if i.dtype.size() != 1 || n <= 256 || o.dtype == TType::Str {
        return kernel(i, o);
    }
    let mut ti = Tensor { shape: vec![256], ..i.clone_meta() };
    ti.buf = crate::tensor::Buf::new(256);
    ti.buf.bytes_mut().iter_mut().enumerate().for_each(|(k, b)| *b = k as u8);
    let mut to = Tensor { shape: vec![256], ..o.clone_meta() };
    to.resize(vec![256]);
    kernel(&ti, &mut to)?;
    // the element size as a constant: a copy of N bytes is a move, of a runtime length a memcpy call
    fn look_up<const N: usize>(dst: &mut [u8], bytes: &[u8], tab: &[u8]) {
        let tab: Vec<[u8; N]> = tab.chunks_exact(N).map(|c| c.try_into().unwrap()).collect();
        dst.chunks_exact_mut(N).zip(bytes).for_each(|(d, &b)| d.copy_from_slice(&tab[b as usize]));
    }
    let (src, tab) = (&i.data()[..n], to.data());
    match o.dtype.size() {
        1 => look_up::<1>(o.data_mut(), src, tab),
        2 => look_up::<2>(o.data_mut(), src, tab),
        4 => look_up::<4>(o.data_mut(), src, tab),
        8 => look_up::<8>(o.data_mut(), src, tab),
        _ => look_up::<16>(o.data_mut(), src, tab),
    }
    Ok(())
}

// *** half precision (IEEE 754 binary16, round to nearest even like Eigen::half) ***

pub fn f16_to_f32(h: u16) -> f32 {
    let (sign, exp, man) = ((h as u32 & 0x8000) << 16, (h >> 10) & 0x1f, h as u32 & 0x3ff);
    let bits = match exp {
        0 if man == 0 => sign,
        0 => {
            // subnormal: normalize
            let shift = man.leading_zeros() - 21;
            sign | ((113 - shift) << 23) | ((man << shift) & 0x3ff) << 13
        }
        31 => sign | 0x7f80_0000 | (man << 13),
        e => sign | ((e as u32 + 112) << 23) | (man << 13),
    };
    f32::from_bits(bits)
}

pub fn f32_to_f16(f: f32) -> u16 {
    let x = f.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xff) as i32;
    let man = x & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x200 | (man >> 13) as u16 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 31 {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let rest = m & ((1 << shift) - 1);
        let mut v = m >> shift;
        if rest > half || (rest == half && v & 1 == 1) {
            v += 1;
        }
        return sign | v as u16;
    }
    let mut v = ((e as u32) << 10) | (man >> 13);
    let rest = man & 0x1fff;
    if rest > 0x1000 || (rest == 0x1000 && v & 1 == 1) {
        v += 1;
    }
    sign | v as u16
}

/// the axis of a possibly negative `axis` for a tensor of rank `rank`
pub fn axis_of(axis: i64, rank: usize) -> Result<usize> {
    let a = if axis < 0 { axis + rank as i64 } else { axis };
    if a < 0 || a as usize >= rank.max(1) {
        return err(format!("axis {axis} out of range for rank {rank}"));
    }
    Ok(a as usize)
}

/// the int32 / int64 values of a (shape-like) tensor
pub fn index_values(t: &Tensor) -> Result<Vec<i64>> {
    match t.dtype {
        TType::I32 | TType::I64 | TType::I16 | TType::I8 | TType::U8 => ints(t),
        d => err(format!("{}: {d:?} indices are not supported", t.name)),
    }
}

/// output = input's shape, with the input's buffer size
pub fn same_shape(i: &Tensor, o: &mut Tensor) {
    o.resize(i.shape.clone());
}
