//! Elementwise ops on one input (elementwise.cc, activations.cc, exp.cc, neg.cc, floor.cc, ceil.cc, round.cc,
//! zeros_like.cc). Multipliers are worked out in float first where TFLite's Prepare does, then widened.

use super::eigen;
use super::fixedpoint::{self as fp, I16};
use super::util::*;
use super::{Io, Kernel};
use crate::interp::Env;
use crate::tensor::{TType, Tensor};
use crate::tflite::OpSpec;
use crate::Result;

#[derive(Clone, Copy, PartialEq)]
enum Op {
    Abs,
    Neg,
    Exp,
    Log,
    Sin,
    Cos,
    Sqrt,
    Rsqrt,
    Square,
    Ceil,
    Floor,
    Round,
    Not,
    ZerosLike,
    Relu,
    Relu1,
    Relu6,
    HardSwish,
    LeakyRelu(f32),
    Elu,
    Tanh,
    Logistic,
}

pub fn create(s: &OpSpec) -> Result<Option<Box<dyn Kernel>>> {
    let op = match s.code {
        101 => Op::Abs,
        59 => Op::Neg,
        47 => Op::Exp,
        73 => Op::Log,
        66 => Op::Sin,
        108 => Op::Cos,
        75 => Op::Sqrt,
        76 => Op::Rsqrt,
        92 => Op::Square,
        104 => Op::Ceil,
        8 => Op::Floor,
        116 => Op::Round,
        87 => Op::Not,
        93 => Op::ZerosLike,
        19 => Op::Relu,
        20 => Op::Relu1,
        21 => Op::Relu6,
        117 => Op::HardSwish,
        98 => Op::LeakyRelu(opt(s, 0, 0.0f32)),
        111 => Op::Elu,
        28 => Op::Tanh,
        14 => Op::Logistic,
        _ => return Ok(None),
    };
    Ok(Some(Box::new(Unary { op, table: None })))
}

struct Unary {
    op: Op,
    /// 1-byte types: the output for each input byte (the lookup tables of activations.cc, or this op worked out
    /// for all 256 values)
    table: Option<[u8; 256]>,
}

impl Kernel for Unary {
    fn prepare(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        let i = io.input(0)?;
        let o = &mut io.outs[0];
        if self.op != Op::ZerosLike && i.dtype != o.dtype {
            return err(format!("input {:?}, output {:?}", i.dtype, o.dtype));
        }
        o.resize(i.shape.clone());
        self.table = None;
        if matches!(i.dtype, TType::U8 | TType::I8) {
            // every input byte through the op once (a 256-element tensor of all values)
            let mut ti = Tensor { shape: vec![256], ..i.clone_meta() };
            ti.buf = crate::tensor::Buf::new(256);
            ti.buf.bytes_mut().iter_mut().enumerate().for_each(|(k, b)| *b = k as u8);
            let mut to = Tensor { shape: vec![256], alloc: crate::tensor::Alloc::Arena, ..o.clone_meta() };
            to.resize(vec![256]);
            match eval(self.op, &ti, &mut to) {
                Ok(()) => self.table = Some(to.data().try_into().unwrap()),
                // an input that is refused (RSQRT of a negative value) must fail when it shows up, not now
                Err(_) if self.op == Op::Rsqrt => {}
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }

    fn eval(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        let i = io.input(0)?;
        match &self.table {
            Some(t) => {
                for (d, &b) in io.outs[0].data_mut().iter_mut().zip(i.data()) {
                    *d = t[b as usize];
                }
                Ok(())
            }
            None => eval(self.op, i, &mut io.outs[0]),
        }
    }
}

fn map_f32(i: &Tensor, o: &mut Tensor, f: impl Fn(f32) -> f32) {
    for (d, &x) in o.as_mut_slice::<f32>().iter_mut().zip(i.as_slice::<f32>()) {
        *d = f(x);
    }
}

/// apply f to each quantized value of a 1- or 2-byte type, clamped to the type's range
fn map_q(i: &Tensor, o: &mut Tensor, f: impl Fn(i32) -> i32) -> Result<()> {
    let (lo, hi) = type_range(o.dtype);
    match i.dtype {
        TType::U8 => o.as_mut_slice::<u8>().iter_mut().zip(i.as_slice::<u8>()).for_each(|(d, &x)| *d = f(x as i32).clamp(lo, hi) as u8),
        TType::I8 => o.as_mut_slice::<i8>().iter_mut().zip(i.as_slice::<i8>()).for_each(|(d, &x)| *d = f(x as i32).clamp(lo, hi) as i8),
        TType::I16 => o.as_mut_slice::<i16>().iter_mut().zip(i.as_slice::<i16>()).for_each(|(d, &x)| *d = f(x as i32).clamp(lo, hi) as i16),
        t => return err(format!("{t:?} is not supported")),
    }
    Ok(())
}

/// PopulateLookupTable (activations.cc): dequantize, transform in float, requantize
fn table_q(i: &Tensor, o: &mut Tensor, transform: impl Fn(f32) -> f32) -> Result<()> {
    let inverse_scale = 1.0 / o.scale();
    let (izp, ozp, is) = (i.zero_point(), o.zero_point(), i.scale());
    map_q(i, o, |v| {
        let dequantized = is * (v - izp) as f32;
        let rescaled = (transform(dequantized) * inverse_scale).round();
        (rescaled + ozp as f32) as i32
    })
}

/// ReluX (optimized_ops): requantize, then clamp to the activation's range in the output's quantization
fn relu_x(i: &Tensor, o: &mut Tensor, act_min: f32, act_max: f32) -> Result<()> {
    let (tmin, tmax) = type_range(o.dtype);
    let (s, zp) = (o.scale(), o.zero_point());
    let lo = tmin.max(zp + (act_min / s).round() as i32);
    let hi = if act_max == f32::INFINITY { tmax } else { tmax.min(zp + (act_max / s).round() as i32) };
    let (m, shift) = quantize_multiplier((i.scale() / o.scale()) as f64);
    let izp = i.zero_point();
    map_q(i, o, |v| (zp + mul_by_quantized_multiplier(v - izp, m, shift)).clamp(lo, hi))
}

/// RoundToNearest (reference/round.h): halves to even, by way of floor
fn round_to_nearest(v: f32) -> f32 {
    let f = v.floor();
    let diff = v - f;
    // static_cast<int>(floor) as x86 converts: the minimum when out of range
    let fi = cvtt(f as f64, false);
    if diff < 0.5 || (diff == 0.5 && fi % 2 == 0) { f } else { f + 1.0 }
}

/// CheckedLog2 (quantization_util.cc): the rounded log2 of x, and whether x is (close to) a power of two
fn checked_log2(x: f32) -> (i32, bool) {
    let x_log2 = x.ln() * (1.0f32 / 2.0f32.ln());
    let rounded = x_log2.round();
    (rounded as i32, (x_log2 - rounded).abs() < 1e-3)
}

/// TanhPrepare / SigmoidPrepare for int16: (input multiplier, 0 for a power-of-two scale; input left shift)
fn int16_params(input_scale: f32, tanh: bool) -> (i32, i32) {
    let (log2, mut pot) = checked_log2(input_scale);
    let shift = (15 - 3) + log2;
    pot &= shift == 0 || (tanh && shift == 1);
    if pot {
        return (0, shift);
    }
    let mut multiplier = input_scale as f64 * 4096.0 * 3.0;
    let mut shift = 0;
    while multiplier <= 32767.0 / 2.0 && shift <= 30 {
        shift += 1;
        multiplier *= 2.0;
    }
    (multiplier as i32, shift)
}

/// reference_integer_ops::Tanh / Logistic for int16: the sigmoid table, interpolated
fn int16_lut(x: i16, multiplier: i32, left_shift: i32, tanh: bool) -> i16 {
    let (multiplier, left_shift) = if multiplier == 0 { (3 << left_shift, 0) } else { (multiplier, left_shift) };
    let round = if left_shift > 0 { 1 << (left_shift - 1) } else { 0 };
    let input = (x as i32 * multiplier + round) >> left_shift;
    let abs = input.unsigned_abs();
    if tanh {
        let uh = abs >> 8;
        let mut result: i32 = if uh >= 255 {
            0xFFFF << 8
        } else {
            let (ua, ub) = (SIGMOID_TABLE[uh as usize] as u32, SIGMOID_TABLE[uh as usize + 1] as u32);
            let ut = abs & 0xFF;
            ((ua << 8).wrapping_add(ut.wrapping_mul(ub.wrapping_sub(ua)))) as i32
        };
        result = if input >= 0 { result - (1 << (14 + 9)) + (1 << (9 - 2)) } else { -result + (1 << (14 + 9)) + (1 << (9 - 2)) - 1 };
        (result >> (9 - 1)) as i16
    } else {
        let uh = abs >> 9;
        let result: u32 = if uh >= 255 {
            0x7FFF << 10
        } else {
            let (ua, ub) = (SIGMOID_TABLE[uh as usize] as u32, SIGMOID_TABLE[uh as usize + 1] as u32);
            (ua << 9).wrapping_add((abs & 0x1ff).wrapping_mul(ub.wrapping_sub(ua)))
        };
        let result = if input >= 0 { result.wrapping_add(1 << 9) } else { (1u32 << (16 + 9)).wrapping_sub(result).wrapping_add(1 << 9).wrapping_sub(1) };
        (result >> 10) as i16
    }
}

fn eval(op: Op, i: &Tensor, o: &mut Tensor) -> Result<()> {
    let t = i.dtype;
    match (op, t) {
        (Op::Abs, TType::F32) => map_f32(i, o, f32::abs),
        (Op::Abs, TType::I8 | TType::I16) => {
            let (izp, ozp) = (i.zero_point(), o.zero_point());
            if i.scale() == o.scale() {
                map_q(i, o, |v| (v - izp).abs() + ozp)?;
            } else {
                let (m, shift) = quantize_multiplier((i.scale() / o.scale()) as f64);
                map_q(i, o, |v| mul_by_quantized_multiplier((v - izp).abs(), m, shift) + ozp)?;
            }
        }
        (Op::Neg, TType::F32) => map_f32(i, o, |x| -x),
        (Op::Neg, TType::I32) => o.as_mut_slice::<i32>().iter_mut().zip(i.as_slice::<i32>()).for_each(|(d, &x)| *d = x.wrapping_neg()),
        (Op::Neg, TType::I64) => o.as_mut_slice::<i64>().iter_mut().zip(i.as_slice::<i64>()).for_each(|(d, &x)| *d = x.wrapping_neg()),
        (Op::Exp, TType::F32) => map_f32(i, o, f32::exp),
        (Op::Log, TType::F32) => map_f32(i, o, f32::ln),
        (Op::Sin, TType::F32) => map_f32(i, o, f32::sin),
        (Op::Cos, TType::F32) => map_f32(i, o, f32::cos),
        (Op::Sqrt, TType::F32) => map_f32(i, o, f32::sqrt),
        (Op::Rsqrt, TType::F32) => map_f32(i, o, |x| 1.0 / x.sqrt()),
        (Op::Rsqrt, TType::I8) => {
            // elementwise.cc RsqrtEvalQuantized
            let (izp, ozp) = (i.zero_point(), o.zero_point());
            if i.as_slice::<i8>().iter().any(|&v| (v as i32) < izp) {
                return err("Rsqrt is only defined for positive values");
            }
            let (m, shift) = quantize_multiplier(1.0 / ((i.scale() as f64).sqrt() * o.scale() as f64));
            map_q(i, o, |v| {
                let value = v - izp;
                if value == 0 {
                    return 127;
                }
                let (inv, inv_shift) = fp::inv_sqrt_multiplier_exp(value, -1);
                let data = mul_by_quantized_multiplier(1, inv, inv_shift + 20);
                mul_by_quantized_multiplier(data, m, shift - 20) + ozp
            })?;
        }
        (Op::Square, TType::F32) => map_f32(i, o, |x| x * x),
        (Op::Ceil | Op::Floor, TType::F32) => {
            let v = o.as_mut_slice::<f32>();
            v.copy_from_slice(i.as_slice());
            if op == Op::Ceil {
                eigen::map_array(v, eigen::pceil, f32::ceil)
            } else {
                eigen::map_array(v, eigen::pfloor, f32::floor)
            }
        }
        (Op::Round, TType::F32) => map_f32(i, o, round_to_nearest),
        (Op::Not, TType::Bool) => o.data_mut().iter_mut().zip(i.data()).for_each(|(d, &x)| *d = (x == 0) as u8),
        (Op::ZerosLike, TType::F32 | TType::I32 | TType::I64) => o.data_mut().fill(0),
        // std::max(0, x) through Eigen's cwiseMax: NaN stays NaN
        (Op::Relu, TType::F32) => map_f32(i, o, |x| if x < 0.0 { 0.0 } else { x }),
        (Op::Relu, TType::U8 | TType::I8 | TType::I16) => relu_x(i, o, 0.0, f32::INFINITY)?,
        (Op::Relu1, TType::F32) => map_f32(i, o, |x| if x > 1.0 { 1.0 } else if x < -1.0 { -1.0 } else { x }),
        (Op::Relu1, TType::U8 | TType::I8) => relu_x(i, o, -1.0, 1.0)?,
        // std::min(std::max(0.f, x), 6.f): NaN becomes 0
        (Op::Relu6, TType::F32) => map_f32(i, o, |x| {
            let v = if 0.0 < x { x } else { 0.0 };
            if v < 6.0 { v } else { 6.0 }
        }),
        (Op::Relu6, TType::U8 | TType::I8 | TType::I16) => relu_x(i, o, 0.0, 6.0)?,
        (Op::HardSwish, TType::F32) => map_f32(i, o, |x| {
            let v = if 0.0 < x + 3.0 { x + 3.0 } else { 0.0 };
            x * (if v < 6.0 { v } else { 6.0 }) * (1.0f32 / 6.0)
        }),
        (Op::HardSwish, TType::U8 | TType::I8) => hard_swish_q(i, o)?,
        (Op::LeakyRelu(alpha), TType::F32) => map_f32(i, o, |x| if x > 0.0 { x } else { x * alpha }),
        (Op::LeakyRelu(alpha), TType::U8 | TType::I8 | TType::I16) => {
            let (ma, sa) = quantize_multiplier((i.scale() * alpha / o.scale()) as f64);
            let (mi, si) = quantize_multiplier((i.scale() / o.scale()) as f64);
            let (izp, ozp) = (i.zero_point(), o.zero_point());
            map_q(i, o, |v| {
                let x = v - izp;
                ozp + if x >= 0 { mul_by_quantized_multiplier(x, mi, si) } else { mul_by_quantized_multiplier(x, ma, sa) }
            })?;
        }
        (Op::Elu, TType::F32) => map_f32(i, o, |x| if x < 0.0 { x.exp_m1() } else { x }),
        (Op::Elu, TType::I8) => table_q(i, o, |x| if (x as f64) < 0.0 { x.exp() - 1.0 } else { x })?,
        (Op::Tanh, TType::F32) => map_f32(i, o, eigen::tanh),
        (Op::Tanh, TType::U8 | TType::I8) => table_q(i, o, f32::tanh)?,
        (Op::Logistic, TType::F32) => map_f32(i, o, eigen::logistic),
        (Op::Logistic, TType::U8 | TType::I8) => {
            if o.scale() != 1.0 / 256.0 {
                return err("the output scale must be 1/256");
            }
            table_q(i, o, |x| 1.0 / (1.0 + (-x).exp()))?
        }
        (Op::Tanh | Op::Logistic, TType::I16) => {
            let tanh = op == Op::Tanh;
            if i.zero_point() != 0 || o.zero_point() != 0 || checked_log2(o.scale()) != (-15, true) {
                return err("int16 needs zero points 0 and an output scale of 2^-15");
            }
            let (multiplier, shift) = int16_params(i.scale(), tanh);
            let v: Vec<i16> = i
                .as_slice::<i16>()
                .iter()
                .map(|&x| {
                    if multiplier > 0 {
                        int16_lut(x, multiplier, shift, tanh)
                    } else if tanh {
                        // optimized_ops::Tanh: gemmlowp on F3 (doubled first when the shift is 1)
                        let a = if shift == 1 { I16.mul_pot(x as i64, 1) } else { x as i64 };
                        I16.tanh(a, 3) as i16
                    } else {
                        I16.logistic(x as i64, 3) as i16
                    }
                })
                .collect();
            o.as_mut_slice::<i16>().copy_from_slice(&v);
        }
        (op, t) => return err(format!("{t:?} is not supported by {}", name(op))),
    }
    Ok(())
}

fn name(op: Op) -> &'static str {
    match op {
        Op::Abs => "ABS",
        Op::Neg => "NEG",
        Op::Exp => "EXP",
        Op::Log => "LOG",
        Op::Sin => "SIN",
        Op::Cos => "COS",
        Op::Sqrt => "SQRT",
        Op::Rsqrt => "RSQRT",
        Op::Square => "SQUARE",
        Op::Ceil => "CEIL",
        Op::Floor => "FLOOR",
        Op::Round => "ROUND",
        Op::Not => "LOGICAL_NOT",
        Op::ZerosLike => "ZEROS_LIKE",
        Op::Relu => "RELU",
        Op::Relu1 => "RELU_N1_TO_1",
        Op::Relu6 => "RELU6",
        Op::HardSwish => "HARD_SWISH",
        Op::LeakyRelu(_) => "LEAKY_RELU",
        Op::Elu => "ELU",
        Op::Tanh => "TANH",
        Op::Logistic => "LOGISTIC",
    }
}

/// reference_ops::HardSwish (hard_swish.h) with HardSwishPrepare's parameters
fn hard_swish_q(i: &Tensor, o: &mut Tensor) -> Result<()> {
    let hires_input_scale = (1.0f32 / 128.0) * i.scale();
    let reluish_scale = 3.0f32 / 32768.0;
    let (om, oexp) = quantize_multiplier((hires_input_scale / o.scale()) as f64);
    let (rm, rexp) = quantize_multiplier((hires_input_scale / reluish_scale) as f64);
    if oexp > 0 {
        return err("the output multiplier exponent must be <= 0");
    }
    // DownScaleInt32ToInt16Multiplier
    let down = |m: i32| -> i64 { if m >= i32::MAX - (1 << 15) { 32767 } else { ((m + (1 << 15)) >> 16) as i64 } };
    let (om16, rm16) = (down(om), down(rm));
    let sat_shl = |v: i64, amount: i32| (v * (1 << amount)).clamp(-32768, 32767);
    // SaturatingDoublingHighMul without rounding
    let doubling_high_mul = |a: i64, b: i64| if a == b && a == -32768 { 32767 } else { I16.wrap((a * b) / (1 << 15)) };
    let (izp, ozp) = (i.zero_point(), o.zero_point() as i64);
    let (tmin, tmax) = type_range(o.dtype);
    map_q(i, o, |v| {
        let input_value = I16.wrap((v - izp) as i64);
        let hires = I16.wrap(input_value * (1 << 7));
        let preshift = I16.mul(hires, om16);
        let mut reluish = hires;
        if rexp > 0 {
            reluish = sat_shl(reluish, rexp - 1);
        }
        reluish = I16.mul(reluish, rm16);
        if rexp > 0 {
            reluish = sat_shl(reluish, 1);
        }
        if rexp < 0 {
            reluish = I16.rdbp(reluish, -rexp);
        }
        reluish = I16.wrap((reluish + (1 << 15)) >> 1);
        let preshift_output = doubling_high_mul(reluish, preshift);
        let out = I16.wrap(I16.rdbp(preshift_output, -oexp) + ozp);
        out.clamp(tmin as i64, tmax as i64) as i32
    })
}

/// sigmoid_table_uint16 (internal/common.h)
const SIGMOID_TABLE: [u16; 256] = [
    32768, 33451, 34133, 34813, 35493, 36169, 36843, 37513, 38180, 38841, 39498, 40149, 40794, 41432, 42064, 42688, 43304,
    43912, 44511, 45102, 45683, 46255, 46817, 47369, 47911, 48443, 48964, 49475, 49975, 50464, 50942, 51409, 51865, 52311,
    52745, 53169, 53581, 53983, 54374, 54755, 55125, 55485, 55834, 56174, 56503, 56823, 57133, 57433, 57724, 58007, 58280,
    58544, 58800, 59048, 59288, 59519, 59743, 59959, 60168, 60370, 60565, 60753, 60935, 61110, 61279, 61441, 61599, 61750,
    61896, 62036, 62172, 62302, 62428, 62549, 62666, 62778, 62886, 62990, 63090, 63186, 63279, 63368, 63454, 63536, 63615,
    63691, 63765, 63835, 63903, 63968, 64030, 64090, 64148, 64204, 64257, 64308, 64357, 64405, 64450, 64494, 64536, 64576,
    64614, 64652, 64687, 64721, 64754, 64786, 64816, 64845, 64873, 64900, 64926, 64950, 64974, 64997, 65019, 65039, 65060,
    65079, 65097, 65115, 65132, 65149, 65164, 65179, 65194, 65208, 65221, 65234, 65246, 65258, 65269, 65280, 65291, 65301,
    65310, 65319, 65328, 65337, 65345, 65352, 65360, 65367, 65374, 65381, 65387, 65393, 65399, 65404, 65410, 65415, 65420,
    65425, 65429, 65433, 65438, 65442, 65445, 65449, 65453, 65456, 65459, 65462, 65465, 65468, 65471, 65474, 65476, 65479,
    65481, 65483, 65485, 65488, 65489, 65491, 65493, 65495, 65497, 65498, 65500, 65501, 65503, 65504, 65505, 65507, 65508,
    65509, 65510, 65511, 65512, 65513, 65514, 65515, 65516, 65517, 65517, 65518, 65519, 65520, 65520, 65521, 65522, 65522,
    65523, 65523, 65524, 65524, 65525, 65525, 65526, 65526, 65526, 65527, 65527, 65528, 65528, 65528, 65529, 65529, 65529,
    65529, 65530, 65530, 65530, 65530, 65531, 65531, 65531, 65531, 65531, 65532, 65532, 65532, 65532, 65532, 65532, 65533,
    65533, 65533, 65533, 65533, 65533, 65533, 65533, 65534, 65534, 65534, 65534, 65534, 65534, 65534, 65534, 65534, 65534,
    65535,
];
