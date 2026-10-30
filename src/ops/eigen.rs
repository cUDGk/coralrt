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

/// Eigen::ceil / Eigen::floor of a whole array as the SSE2 build runs it: packets of 4 through print (ceil(-0.7) is
/// +0, floor(-0) is +0), the elements after the last full packet through std::ceil / std::floor
pub fn ceil_array(v: &[f32], out: &mut [f32]) {
    let full = v.len() / 4 * 4;
    for (k, (d, &a)) in out.iter_mut().zip(v).enumerate() {
        *d = if k < full {
            let tmp = print(a);
            tmp + if tmp < a { 1.0 } else { 0.0 }
        } else {
            a.ceil()
        };
    }
}

pub fn floor_array(v: &[f32], out: &mut [f32]) {
    let full = v.len() / 4 * 4;
    for (k, (d, &a)) in out.iter_mut().zip(v).enumerate() {
        *d = if k < full {
            let tmp = print(a);
            tmp - if tmp > a { 1.0 } else { 0.0 }
        } else {
            a.floor()
        };
    }
}

/// pldexp_generic: a * 2^e in four factors so no step over- or underflows
fn ldexp(a: f32, exponent: f32) -> f32 {
    let e = exponent.clamp(-278.0, 278.0) as i32;
    let b = e >> 2;
    let pow2 = |k: i32| f32::from_bits(((k + 127) as u32) << 23);
    let c = pow2(b);
    a * c * c * c * pow2(e - 3 * b)
}
