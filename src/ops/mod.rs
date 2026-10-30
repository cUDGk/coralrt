//! CPU kernels: every op of TensorFlow Lite 2.5's BuiltinOpResolver (the tflite_runtime under pycoral 2.0), written
//! to give the same results: its fixed-point arithmetic (kernels/internal/common.h, quantization_util.cc), its
//! float/double mix and its orderings. Integer and quantized ops match it byte for byte; float ops whose result
//! depends on the summation order of an optimized kernel (convolutions, matrix products) match it within rounding.

pub mod fixedpoint;
pub mod util;

mod binary;
mod data;
mod dpp;
mod eigen;
mod movement;
mod quant;
mod reduce;
mod shape;
mod unary;

use crate::interp::Env;
use crate::tensor::Tensor;
use crate::tflite::OpSpec;
use crate::{Error, Result};

/// An op's tensors while it runs: the inputs (None for an absent optional input), the outputs, and the inputs it
/// writes in place (taken out of `ins` into `vars`).
pub struct Io<'a> {
    pub ins: Vec<Option<&'a Tensor>>,
    pub outs: Vec<Tensor>,
    pub vars: Vec<(usize, Tensor)>,
}

impl<'a> Io<'a> {
    pub fn input(&self, k: usize) -> Result<&'a Tensor> {
        self.ins.get(k).copied().flatten().ok_or_else(|| Error(format!("input {k} is missing")))
    }
    pub fn opt(&self, k: usize) -> Option<&'a Tensor> {
        self.ins.get(k).copied().flatten()
    }
    pub fn n_in(&self) -> usize {
        self.ins.len()
    }
    pub fn var(&mut self, k: usize) -> Result<&mut Tensor> {
        self.vars.iter_mut().find(|(p, _)| *p == k).map(|(_, t)| t).ok_or_else(|| Error(format!("state input {k} is missing")))
    }
}

pub trait Kernel: Send {
    /// work out the outputs' shapes, and whatever depends only on shapes and constant inputs
    fn prepare(&mut self, io: &mut Io, env: &mut Env) -> Result<()>;
    fn eval(&mut self, io: &mut Io, env: &mut Env) -> Result<()>;
    /// positions of the inputs this op updates in place (variable tensors: the state of RNN, LSTM, SVDF)
    fn mutated_inputs(&self) -> &[usize] {
        &[]
    }
    /// the Edge TPU op behind this kernel, if it is one
    fn edgetpu(&self) -> Option<&crate::edgetpu::EdgeTpuOp> {
        None
    }
}

pub fn create(s: &OpSpec) -> Result<Box<dyn Kernel>> {
    for module in [binary::create, data::create, movement::create, quant::create, reduce::create, shape::create, unary::create] {
        if let Some(k) = module(s)? {
            return Ok(k);
        }
    }
    match s.name {
        "edgetpu-custom-op" => Ok(Box::new(crate::edgetpu::EdgeTpuOp::new(s.custom_opts)?)),
        "TFLite_Detection_PostProcess" => Ok(Box::new(dpp::Dpp::new(s.custom_opts))),
        n => Err(Error(format!("op {n} (code {}, version {}) is not supported", s.code, s.version))),
    }
}
