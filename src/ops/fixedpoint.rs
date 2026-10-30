//! gemmlowp's fixed-point arithmetic (fixedpoint/fixedpoint.h) as TFLite's quantized kernels use it, for 32- and
//! 16-bit raw values. A FixedPoint<raw, IB> is its raw value (held in an i64 here) and `Fx` its width; the integer
//! bits IB travel as arguments. Where the C++ shifts by an out-of-range amount (undefined behaviour), this does what
//! x86 does: the count is masked.

/// the raw type's width
#[derive(Clone, Copy)]
pub struct Fx {
    pub bits: i32,
}

pub const I32: Fx = Fx { bits: 32 };
pub const I16: Fx = Fx { bits: 16 };

impl Fx {
    pub fn min(self) -> i64 {
        -(1i64 << (self.bits - 1))
    }
    pub fn max(self) -> i64 {
        (1i64 << (self.bits - 1)) - 1
    }
    /// two's complement wrap to the width (C++ integer overflow as it happens)
    pub fn wrap(self, x: i64) -> i64 {
        let s = 64 - self.bits;
        (x << s) >> s
    }
    /// a 32-bit constant at this width (RescaleConstantInitializer)
    pub fn constant(self, raw32: i32) -> i64 {
        I32.rdbp(raw32 as i64, 32 - self.bits)
    }
    /// raw value of 1.0 (the largest value when there are no integer bits)
    pub fn one(self, ib: i32) -> i64 {
        if ib == 0 { self.max() } else { 1i64 << (self.bits - 1 - ib) }
    }
    /// raw value of 2^exp
    pub fn constant_pot(self, ib: i32, exp: i32) -> i64 {
        1i64 << (self.bits - 1 - ib + exp)
    }

    /// SaturatingRoundingDoublingHighMul
    pub fn mul(self, a: i64, b: i64) -> i64 {
        if a == b && a == self.min() {
            return self.max();
        }
        let ab = a * b;
        let half = 1i64 << (self.bits - 2);
        let nudge = if ab >= 0 { half } else { 1 - half };
        let r = ab + nudge;
        let d = 1i64 << (self.bits - 1);
        // C++ division truncates toward zero
        self.wrap(if r >= 0 { r / d } else { -((-r) / d) })
    }

    /// RoundingDivideByPOT, with x86's masked shift counts
    pub fn rdbp(self, x: i64, e: i32) -> i64 {
        let e = (e as u32) & (self.bits.max(32) as u32 * 2 - 1);
        let mask = self.wrap(1i64.wrapping_shl(e).wrapping_sub(1));
        let rem = x & mask;
        let threshold = (mask >> 1) + (x < 0) as i64;
        let shifted = if self.bits == 32 { (x as i32).wrapping_shr(e) as i64 } else { x >> e.min(63) };
        self.wrap(shifted + (rem > threshold) as i64)
    }

    /// SaturatingRoundingMultiplyByPOT<exp>: a saturating left shift, or a rounding right shift
    pub fn mul_pot(self, x: i64, exp: i32) -> i64 {
        match exp {
            0 => x,
            e if e > 0 => {
                let threshold = (1i64 << (self.bits - 1 - e)) - 1;
                if x > threshold {
                    self.max()
                } else if x < -threshold {
                    self.min()
                } else {
                    self.wrap(x << e)
                }
            }
            e => self.rdbp(x, -e),
        }
    }

    /// Rescale<dst>(FixedPoint<src>)
    pub fn rescale(self, x: i64, src_ib: i32, dst_ib: i32) -> i64 {
        self.mul_pot(x, src_ib - dst_ib)
    }

    pub fn add(self, a: i64, b: i64) -> i64 {
        self.wrap(a + b)
    }
    pub fn sub(self, a: i64, b: i64) -> i64 {
        self.wrap(a - b)
    }
    pub fn neg(self, a: i64) -> i64 {
        self.wrap(-a)
    }

    /// AddSaturatingIf16Bit
    pub fn add_sat_if16(self, a: i64, b: i64) -> i64 {
        if self.bits == 16 { (a + b).clamp(self.min(), self.max()) } else { self.add(a, b) }
    }

    pub fn rounding_half_sum(self, a: i64, b: i64) -> i64 {
        let sum = a + b;
        let s = sum + if sum >= 0 { 1 } else { -1 };
        self.wrap(if s >= 0 { s / 2 } else { -((-s) / 2) })
    }

    /// exp(a) for a in [-1/4, 0), IB 0
    pub fn exp_on_interval_between_negative_one_quarter_and_0_excl(self, a: i64) -> i64 {
        let constant_term = self.constant(1895147668); // exp(-1/8)
        let constant_1_over_3 = self.constant(715827883);
        let x = self.add(a, self.constant_pot(0, -3));
        let x2 = self.mul(x, x);
        let x3 = self.mul(x2, x);
        let x4 = self.mul(x2, x2);
        let x4_over_4 = self.mul_pot(x4, -2);
        let rest = self.mul_pot(self.add(self.mul(self.add(x4_over_4, x3), constant_1_over_3), x2), -1);
        self.add_sat_if16(constant_term, self.mul(constant_term, self.add(x, rest)))
    }

    /// exp(a) for a < 0, a with `ib` integer bits; the result has 0
    pub fn exp_on_negative_values(self, a: i64, ib: i32) -> i64 {
        let fb = self.bits - 1 - ib;
        let one_quarter = self.constant_pot(ib, -2);
        let mask = one_quarter - 1;
        let a_mod_quarter_minus_one_quarter = self.sub(a & mask, one_quarter);
        let mut result = self.exp_on_interval_between_negative_one_quarter_and_0_excl(self.rescale(a_mod_quarter_minus_one_quarter, ib, 0));
        let remainder = self.sub(a_mod_quarter_minus_one_quarter, a);
        for (exponent, multiplier) in [(-2, 1672461947), (-1, 1302514674), (0, 790015084), (1, 290630308), (2, 39332535), (3, 720401), (4, 242)] {
            if ib > exponent {
                let shift = fb + exponent;
                if remainder & self.wrap(1i64.wrapping_shl(shift as u32)) != 0 {
                    result = self.mul(result, self.constant(multiplier));
                }
            }
        }
        if ib > 5 {
            let clamp = self.constant(-(1i32.wrapping_shl((36 - ib) as u32)));
            if a < clamp {
                result = 0;
            }
        }
        if a == 0 { self.one(0) } else { result }
    }

    /// Newton-Raphson for 1 / (1 + a), a in (0, 1) with IB 0; returns x (IB 2)
    fn reciprocal_iterations(self, a: i64) -> i64 {
        let half_denominator = self.rounding_half_sum(a, self.one(0));
        let constant_48_over_17 = self.constant(1515870810);
        let constant_neg_32_over_17 = self.constant(-1010580540);
        let mut x = self.add(constant_48_over_17, self.mul(half_denominator, constant_neg_32_over_17));
        for _ in 0..3 {
            let half_denominator_times_x = self.mul(half_denominator, x);
            let one_minus_half_denominator_times_x = self.sub(self.one(2), half_denominator_times_x);
            x = self.add(x, self.rescale(self.mul(x, one_minus_half_denominator_times_x), 4, 2));
        }
        x
    }

    /// 1 / (1 + a) for a in (0, 1), IB 0
    pub fn one_over_one_plus_x_for_x_in_0_1(self, a: i64) -> i64 {
        // Rescale<0>(ExactMulByPot<-1>(x)): x read with 1 integer bit
        self.rescale(self.reciprocal_iterations(a), 1, 0)
    }

    /// (1 - a) / (1 + a) for a in (0, 1), IB 0
    pub fn one_minus_x_over_one_plus_x_for_x_in_0_1(self, a: i64) -> i64 {
        self.rescale(self.sub(self.reciprocal_iterations(a), self.one(2)), 2, 0)
    }

    /// tanh(a), a with `ib` integer bits; the result has 0
    pub fn tanh(self, a: i64, ib: i32) -> i64 {
        let n = if a < 0 { a } else { self.neg(a) };
        // neg_tanh_on_negative_values: ExactMulByPot<1> adds an integer bit
        let t = self.one_minus_x_over_one_plus_x_for_x_in_0_1(self.exp_on_negative_values(n, ib + 1));
        if a == 0 {
            0
        } else if a < 0 {
            self.neg(t)
        } else {
            t
        }
    }

    /// logistic(a) = 1 / (1 + exp(-a)), a with `ib` integer bits; the result has 0
    pub fn logistic(self, a: i64, ib: i32) -> i64 {
        let abs = if a > 0 { a } else { self.neg(a) };
        let if_positive = self.one_over_one_plus_x_for_x_in_0_1(self.exp_on_negative_values(self.neg(abs), ib));
        if a == 0 {
            self.constant(1 << 30)
        } else if a > 0 {
            if_positive
        } else {
            self.sub(self.one(0), if_positive)
        }
    }
}

// *** int32 conveniences ***

/// RoundingDivideByPOT on int32
pub fn rounding_divide_by_pot(x: i32, e: i32) -> i32 {
    I32.rdbp(x as i64, e) as i32
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
    (I32.one_over_one_plus_x_for_x_in_0_1(shifted_sum_minus_one as i64) as i32, num_bits_over_unit)
}

/// GetInvSqrtQuantizedMultiplierExp: 1/sqrt(input) as (multiplier, shift times reverse_shift)
pub fn inv_sqrt_multiplier_exp(mut input: i32, reverse_shift: i32) -> (i32, i32) {
    if input <= 1 {
        return (i32::MAX, 0);
    }
    let mut shift = 11;
    while input >= 1 << 29 {
        input /= 4;
        shift += 1;
    }
    let max_left_shift_bits = clz(input as u32) as u32 - 1;
    let left_shift_bit_pairs = max_left_shift_bits / 2 - 1;
    shift -= left_shift_bit_pairs as i32;
    input <<= 2 * left_shift_bit_pairs;
    // F3 Newton-Raphson from x = 1, 5 iterations
    let f = I32;
    let fixedpoint_input = (input >> 1) as i64;
    let half_input = f.mul_pot(fixedpoint_input, -1);
    let half_three = ((1 << 28) + (1 << 27)) as i64;
    let mut x = f.one(3);
    for _ in 0..5 {
        let x3 = f.rescale(f.mul(f.mul(x, x), x), 9, 3);
        x = f.rescale(f.sub(f.mul(half_three, x), f.mul(half_input, x3)), 6, 3);
    }
    let half_sqrt_2 = 1518500250;
    let mut out = f.mul(x, half_sqrt_2) as i32;
    if shift < 0 {
        out <<= -shift;
        shift = 0;
    }
    (out, shift * reverse_shift)
}
