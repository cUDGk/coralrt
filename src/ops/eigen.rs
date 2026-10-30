//! Eigen's float approximations that TFLite's optimized kernels call (Eigen f612df27, the one TensorFlow 2.5 pins):
//! the same constants and the same order of operations, so the same bits. pmadd is a multiply then an add (the
//! build without FMA).

fn pmadd(a: f32, b: f32, c: f32) -> f32 {
    a * b + c
}

/// pmin / pmax on SSE: _mm_min_ps(b, a), arguments reversed to propagate NaN like std::min (the first operand)
fn pmin(a: f32, b: f32) -> f32 {
    if b < a { b } else { a }
}
fn pmax(a: f32, b: f32) -> f32 {
    if b > a { b } else { a }
}

/// generic_fast_tanh_float (MathFunctionsImpl.h)
pub fn tanh(a_x: f32) -> f32 {
    let x = pmax(pmin(a_x, 7.90531110763549805), -7.90531110763549805);
    if a_x.abs() < 0.0004 {
        return x;
    }
    let x2 = x * x;
    let mut p = pmadd(x2, -2.76076847742355e-16, 2.00018790482477e-13);
    p = pmadd(x2, p, -8.60467152213735e-11);
    p = pmadd(x2, p, 5.12229709037114e-08);
    p = pmadd(x2, p, 1.48572235717979e-05);
    p = pmadd(x2, p, 6.37261928875436e-04);
    p = pmadd(x2, p, 4.89352455891786e-03);
    p *= x;
    let mut q = pmadd(x2, 1.19825839466702e-06, 1.18534705686654e-04);
    q = pmadd(x2, q, 2.26843463243900e-03);
    q = pmadd(x2, q, 4.89352518554385e-03);
    p / q
}

/// scalar_logistic_op<float> (UnaryFunctors.h): a rational approximation, exp below -9
pub fn logistic(x0: f32) -> f32 {
    if x0 < -9.0 {
        return exp(x0);
    }
    let x = pmin(x0, 15.6437711715698242);
    let x2 = x * x;
    let mut p = pmadd(x2, 4.37031012579801e-11, 1.15627324459942e-07);
    p = pmadd(x2, p, 6.08574864600143e-05);
    p = pmadd(x2, p, 8.51377133304701e-03);
    p = pmadd(x2, p, 2.48287947061529e-01);
    p *= x;
    let mut q = pmadd(x2, 6.10247389755681e-13, 5.76102136993427e-09);
    q = pmadd(x2, q, 6.29106785017040e-06);
    q = pmadd(x2, q, 1.70198817374094e-03);
    q = pmadd(x2, q, 1.16817656904453e-01);
    q = pmadd(x2, q, 9.93151921023180e-01);
    p / q + 0.5
}

/// pexp_float (GenericPacketMathFunctions.h): Cephes' polynomial, then 2^m by pldexp_generic
pub fn exp(x0: f32) -> f32 {
    let x = pmax(pmin(x0, 88.723), -88.723);
    let m = pmadd(x, 1.44269504088896341, 0.5).floor();
    let mut r = pmadd(m, -0.693359375, x);
    r = pmadd(m, 2.12194440e-4, r);
    let r2 = r * r;
    let r3 = r2 * r;
    let mut y = pmadd(1.9875691500E-4, r, 1.3981999507E-3);
    let mut y1 = pmadd(4.1665795894E-2, r, 1.6666665459E-1);
    let y2 = r + 1.0;
    y = pmadd(y, r, 8.3334519073E-3);
    y1 = pmadd(y1, r, 5.0000001201E-1);
    y = pmadd(y, r3, y1);
    y = pmadd(y, r2, y2);
    pmax(ldexp(y, m), x0)
}

/// psqrt<Packet4f> with EIGEN_FAST_MATH (Eigen's default): rsqrtps, one Newton step, then x times that; 0 for zero
/// and denormals. rsqrtps differs between CPU makers, so this runs the instruction itself, as TFLite does on the
/// same machine.
pub fn psqrt(x: f32) -> f32 {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: SSE is part of the x86_64 baseline
    let y = unsafe {
        use std::arch::x86_64::*;
        _mm_cvtss_f32(_mm_rsqrt_ps(_mm_set1_ps(x)))
    };
    #[cfg(not(target_arch = "x86_64"))]
    let y = 1.0 / x.sqrt();
    let y = y * pmadd(x * -0.5, y * y, 1.5);
    if x < f32::MIN_POSITIVE && !(x < 0.0) { 0.0 } else { x * y }
}

/// print<Packet4f> on SSE2 (no SSE4.1): round half to even by adding and subtracting 2^23, then the sign
fn print(a: f32) -> f32 {
    let limit = (1u32 << 23) as f32;
    let abs = a.abs();
    let r = std::hint::black_box(abs + limit) - limit;
    if abs < limit {
        if a < 0.0 { -r } else { r }
    } else {
        a
    }
}

/// pceil / pfloor on SSE2, through print: ceil(-0.7) is +0, floor(-0) is +0
pub fn pceil(a: f32) -> f32 {
    let tmp = print(a);
    tmp + if tmp < a { 1.0 } else { 0.0 }
}

pub fn pfloor(a: f32) -> f32 {
    let tmp = print(a);
    tmp - if tmp > a { 1.0 } else { 0.0 }
}

/// A coefficient-wise op over a whole buffer as Eigen's vectorized assignment runs it (TFLite's buffers start
/// 16-byte aligned): `packet` on the packets of 4, `scalar` (the std:: function) on the elements after the last one
pub fn map_array(v: &mut [f32], packet: impl Fn(f32) -> f32, scalar: impl Fn(f32) -> f32) {
    let full = v.len() / 4 * 4;
    let (head, tail) = v.split_at_mut(full);
    head.iter_mut().for_each(|x| *x = packet(*x));
    tail.iter_mut().for_each(|x| *x = scalar(*x));
}

/// Eigen's redux sum (Redux.h, LinearVectorizedTraversal): packets of 4 from the first 16-byte aligned element, two
/// accumulators, predux as (a0 + a2) + (a1 + a3), then the elements before and after. `offset`: the slice's start
/// in floats from a 16-byte aligned base; 0 as well for an expression, which has no address to align to.
pub fn sum(v: &[f32], offset: usize) -> f32 {
    let size = v.len();
    let aligned_start = ((4 - offset % 4) % 4).min(size);
    let aligned_size = (size - aligned_start) / 4 * 4;
    let aligned_size2 = (size - aligned_start) / 8 * 8;
    let (end, end2) = (aligned_start + aligned_size, aligned_start + aligned_size2);
    if aligned_size == 0 {
        return v[1..].iter().fold(v[0], |s, &x| s + x);
    }
    let load = |k: usize| [v[k], v[k + 1], v[k + 2], v[k + 3]];
    let add = |a: [f32; 4], b: [f32; 4]| [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]];
    let mut p0 = load(aligned_start);
    if aligned_size > 4 {
        let mut p1 = load(aligned_start + 4);
        let mut k = aligned_start + 8;
        while k < end2 {
            p0 = add(p0, load(k));
            p1 = add(p1, load(k + 4));
            k += 8;
        }
        p0 = add(p0, p1);
        if end > end2 {
            p0 = add(p0, load(end2));
        }
    }
    let mut res = (p0[0] + p0[2]) + (p0[1] + p0[3]);
    for &x in &v[..aligned_start] {
        res += x;
    }
    for &x in &v[end..] {
        res += x;
    }
    res
}

/// pldexp_generic: a * 2^e in four factors so no step over- or underflows
fn ldexp(a: f32, exponent: f32) -> f32 {
    let e = exponent.clamp(-278.0, 278.0) as i32;
    let b = e >> 2;
    let pow2 = |k: i32| f32::from_bits(((k + 127) as u32) << 23);
    let c = pow2(b);
    a * c * c * c * pow2(e - 3 * b)
}
