//! The parts of tensor_utils that the hybrid and recurrent kernels share, as the x86 build runs them: the portable
//! versions (reference/portable_tensor_utils.cc), since it has neither SSSE3 nor NEON.

use super::eigen;
use super::util::cvtt;
use crate::tflite::Act;

/// SymmetricQuantizeFloats / AsymmetricQuantizeFloats (portable_tensor_utils.cc): (values, scale, zero point)
pub fn quantize_row(v: &[f32], asymmetric: bool) -> (Vec<i32>, f32, i32) {
    if v.is_empty() {
        return (Vec::new(), 1.0, 0);
    }
    // std::minmax_element: the first smallest, the last largest
    let min = v.iter().copied().fold(v[0], |m, x| if x < m { x } else { m });
    let max = v.iter().copied().fold(v[0], |m, x| if x < m { m } else { x });
    if !asymmetric {
        let (a, b) = (min.abs(), max.abs());
        let range = if a < b { b } else { a };
        if range == 0.0 {
            return (vec![0; v.len()], 1.0, 0);
        }
        let inv = 127.0 / range;
        let q = v.iter().map(|&x| (cvtt((x * inv).round() as f64, false) as i32).clamp(-127, 127)).collect();
        return (q, range / 127.0, 0);
    }
    let (rmin, rmax) = ((min as f64).min(0.0), (max as f64).max(0.0));
    if rmin == rmax {
        return (vec![0; v.len()], 1.0, 0);
    }
    let scale = (rmax - rmin) / 255.0;
    let from_min = -128.0 - rmin / scale;
    let from_max = 127.0 - rmax / scale;
    let zp = if 128.0 + (rmin / scale).abs() < 127.0 + (rmax / scale).abs() { from_min } else { from_max };
    let zp = if zp <= -128.0 {
        -128
    } else if zp >= 127.0 {
        127
    } else {
        cvtt(zp.round(), false) as i8 as i32
    };
    let sf = scale as f32;
    let inv = (1.0 / sf as f64) as f32;
    let q = v.iter().map(|&x| (cvtt((zp as f32 + x * inv).round() as f64, false) as i32).clamp(-128, 127)).collect();
    (q, sf, zp)
}

/// BatchQuantizeFloats: each row of n values quantized on its own; (values, scales, zero points)
pub fn batch_quantize(v: &[f32], n: usize, asymmetric: bool) -> (Vec<i32>, Vec<f32>, Vec<i32>) {
    let (mut q, mut scales, mut zps) = (Vec::with_capacity(v.len()), Vec::new(), Vec::new());
    for row in v.chunks(n.max(1)) {
        let (r, s, z) = quantize_row(row, asymmetric);
        q.extend(r);
        scales.push(s);
        zps.push(z);
    }
    (q, scales, zps)
}

/// IsZeroVector (-0.0 counts as zero)
pub fn is_zero(v: &[f32]) -> bool {
    v.iter().all(|&x| x == 0.0)
}

/// MatrixBatchVectorMultiplyAccumulate, float: each result += the row's dot product, summed in order from 0
pub fn mbvma_f32(m: &[f32], rows: usize, cols: usize, v: &[f32], result: &mut [f32]) {
    for r in 0..rows {
        let dot = m[r * cols..(r + 1) * cols].iter().zip(&v[..cols]).fold(0.0f32, |s, (&a, &b)| s + a * b);
        result[r] += dot;
    }
}

/// MatrixBatchVectorMultiplyAccumulate, int8 matrix and vector: result += (dot - row_sum * zero point) * scale
/// (the zero point part only for asymmetric quantization)
pub fn mbvma_i8(m: &[i32], rows: usize, cols: usize, q: &[i32], scale: f32, offset: Option<(&[i32], i32)>, result: &mut [f32]) {
    for r in 0..rows {
        let mut dot = m[r * cols..(r + 1) * cols].iter().zip(&q[..cols]).fold(0i32, |s, (&a, &b)| s.wrapping_add(a * b));
        if let Some((row_sums, zp)) = offset {
            dot = dot.wrapping_sub(row_sums[r].wrapping_mul(zp));
        }
        result[r] += dot as f32 * scale;
    }
}

/// ReductionSumVector over the rows of an int8 matrix
pub fn row_sums(m: &[i32], cols: usize) -> Vec<i32> {
    m.chunks(cols.max(1)).map(|r| r.iter().fold(0i32, |s, &x| s.wrapping_add(x))).collect()
}

/// ApplyActivationToVector
pub fn apply_activation(v: &mut [f32], act: Act) {
    // std::max(a, std::min(x, b)) on NaN: min gives NaN, max gives a
    let clamp = |x: f32, a: f32, b: f32| {
        let m = if b < x { b } else { x };
        if a < m { m } else { a }
    };
    match act {
        Act::None => {}
        Act::Relu => v.iter_mut().for_each(|x| *x = if 0.0 < *x { *x } else { 0.0 }),
        Act::ReluN1To1 => v.iter_mut().for_each(|x| *x = clamp(*x, -1.0, 1.0)),
        Act::Relu6 => v.iter_mut().for_each(|x| *x = clamp(*x, 0.0, 6.0)),
        Act::Tanh => v.iter_mut().for_each(|x| *x = eigen::tanh(*x)),
        Act::SignBit => v.iter_mut().for_each(|x| *x = x.is_sign_negative() as u8 as f32),
    }
}
