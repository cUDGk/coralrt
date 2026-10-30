//! gemmlowp's fixed-point arithmetic (fixedpoint/fixedpoint.h) on int32 raw values, as TFLite's quantized kernels
//! use it. A FixedPoint<int32, IB> is its raw value; the integer bits IB travel as arguments. Where the C++ shifts
//! by an out-of-range amount (undefined behaviour), this does what x86 does: the count is masked.

pub use super::util::srdhm as mul;

/// raw value of 1.0 (the largest value when there are no integer bits)
pub fn one(ib: i32) -> i32 {
    if ib == 0 { i32::MAX } else { 1 << (31 - ib) }
}

/// raw value of 2^exp
pub fn constant_pot(ib: i32, exp: i32) -> i32 {
    1 << (31 - ib + exp)
}

/// RoundingDivideByPOT with x86's masked shift counts
pub fn rounding_divide_by_pot(x: i32, e: i32) -> i32 {
    let mask = (1i64.wrapping_shl(e as u32) - 1) as i32;
    let rem = x & mask;
    let threshold = (mask >> 1) + (x < 0) as i32;
    x.wrapping_shr(e as u32).wrapping_add((rem > threshold) as i32)
}

/// SaturatingRoundingMultiplyByPOT<exp>: a saturating left shift, or a rounding right shift
pub fn mul_pot(x: i32, exp: i32) -> i32 {
    match exp {
        0 => x,
        e if e > 0 => {
            let threshold = (1i32 << (31 - e)) - 1;
            if x > threshold {
                i32::MAX
            } else if x < -threshold {
                i32::MIN
            } else {
                x << e
            }
        }
        e => rounding_divide_by_pot(x, -e),
    }
}

/// Rescale<dst>(FixedPoint<src>)
pub fn rescale(x: i32, src_ib: i32, dst_ib: i32) -> i32 {
    mul_pot(x, src_ib - dst_ib)
}

pub fn rounding_half_sum(a: i32, b: i32) -> i32 {
    let sum = a as i64 + b as i64;
    ((sum + if sum >= 0 { 1 } else { -1 }) / 2) as i32
}

/// exp(a) for a in [-1/4, 0), IB 0
pub fn exp_on_interval_between_negative_one_quarter_and_0_excl(a: i32) -> i32 {
    let constant_term: i32 = 1895147668; // exp(-1/8)
    let constant_1_over_3 = 715827883;
    let x = a.wrapping_add(constant_pot(0, -3));
    let x2 = mul(x, x);
    let x3 = mul(x2, x);
    let x4 = mul(x2, x2);
    let x4_over_4 = mul_pot(x4, -2);
    let rest = mul_pot(mul(x4_over_4.wrapping_add(x3), constant_1_over_3).wrapping_add(x2), -1);
    constant_term.wrapping_add(mul(constant_term, x.wrapping_add(rest)))
}

/// exp(a) for a < 0, a with `ib` integer bits; the result has 0
pub fn exp_on_negative_values(a: i32, ib: i32) -> i32 {
    let fb = 31 - ib;
    let one_quarter = constant_pot(ib, -2);
    let mask = one_quarter - 1;
    let a_mod_quarter_minus_one_quarter = (a & mask).wrapping_sub(one_quarter);
    let mut result = exp_on_interval_between_negative_one_quarter_and_0_excl(rescale(a_mod_quarter_minus_one_quarter, ib, 0));
    let remainder = a_mod_quarter_minus_one_quarter.wrapping_sub(a);
    for (exponent, multiplier) in [(-2, 1672461947), (-1, 1302514674), (0, 790015084), (1, 290630308), (2, 39332535), (3, 720401), (4, 242)] {
        if ib > exponent {
            let shift = fb + exponent;
            if remainder & 1i32.wrapping_shl(shift as u32) != 0 {
                result = mul(result, multiplier);
            }
        }
    }
    if ib > 5 && a < -(1i32 << (36 - ib)) {
        result = 0;
    }
    if a == 0 { one(0) } else { result }
}

/// Newton-Raphson for 1 / (1 + a), a in (0, 1) with IB 0; returns x (IB 2)
fn reciprocal_iterations(a: i32) -> i32 {
    let half_denominator = rounding_half_sum(a, one(0));
    let constant_48_over_17: i32 = 1515870810;
    let constant_neg_32_over_17 = -1010580540;
    let mut x = constant_48_over_17.wrapping_add(mul(half_denominator, constant_neg_32_over_17));
    for _ in 0..3 {
        let half_denominator_times_x = mul(half_denominator, x);
        let one_minus_half_denominator_times_x = one(2).wrapping_sub(half_denominator_times_x);
        x = x.wrapping_add(rescale(mul(x, one_minus_half_denominator_times_x), 4, 2));
    }
    x
}

/// 1 / (1 + a) for a in (0, 1), IB 0
pub fn one_over_one_plus_x_for_x_in_0_1(a: i32) -> i32 {
    // Rescale<0>(ExactMulByPot<-1>(x)): x read with 1 integer bit
    rescale(reciprocal_iterations(a), 1, 0)
}

/// (1 - a) / (1 + a) for a in (0, 1), IB 0
pub fn one_minus_x_over_one_plus_x_for_x_in_0_1(a: i32) -> i32 {
    rescale(reciprocal_iterations(a).wrapping_sub(one(2)), 2, 0)
}

/// tanh(a), a with `ib` integer bits; the result has 0
pub fn tanh(a: i32, ib: i32) -> i32 {
    let n = if a < 0 { a } else { a.wrapping_neg() };
    // neg_tanh_on_negative_values: ExactMulByPot<1> adds an integer bit
    let t = one_minus_x_over_one_plus_x_for_x_in_0_1(exp_on_negative_values(n, ib + 1));
    if a == 0 {
        0
    } else if a < 0 {
        t.wrapping_neg()
    } else {
        t
    }
}

/// logistic(a) = 1 / (1 + exp(-a)), a with `ib` integer bits; the result has 0
pub fn logistic(a: i32, ib: i32) -> i32 {
    let abs = if a > 0 { a } else { a.wrapping_neg() };
    let if_positive = one_over_one_plus_x_for_x_in_0_1(exp_on_negative_values(abs.wrapping_neg(), ib));
    if a == 0 {
        1 << 30
    } else if a > 0 {
        if_positive
    } else {
        one(0).wrapping_sub(if_positive)
    }
}

/// CountLeadingZeros of the bits of x (32 for 0)
pub fn clz(x: u32) -> i32 {
    x.leading_zeros() as i32
}

/// CountLeadingSignBits
pub fn clsb(x: i32) -> i32 {
    if x >= 0 {
        clz(x as u32) - 1
    } else if x != i32::MIN {
        clz((2 * (x.wrapping_neg() as u32)).wrapping_sub(1))
    } else {
        0
    }
}

/// GetReciprocal: (1/x as a Q0 raw value, the bits x has above the unit)
pub fn reciprocal(x: i32, x_integer_digits: i32) -> (i32, i32) {
    let headroom_plus_one = clz(x as u32);
    let num_bits_over_unit = x_integer_digits - headroom_plus_one;
    let shifted_sum_minus_one = (x as u32).wrapping_shl(headroom_plus_one as u32).wrapping_sub(1 << 31) as i32;
    (one_over_one_plus_x_for_x_in_0_1(shifted_sum_minus_one), num_bits_over_unit)
}
