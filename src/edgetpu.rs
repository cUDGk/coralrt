//! The `edgetpu-custom-op`: what libedgetpu does around running an executable (tflite/custom_op.cc,
//! driver/single_tpu_request.cc, api/layer_information.cc): input padding and sign flip, parameter
//! caching, output relayout from the tiles' layout, and the classifier output conversions.

use crate::chip::EdgeTpu;
use crate::exe::{self, DType, ExeType, Executable, Layer, TensorLayout};
use crate::interp::Env;
use crate::ops::{Io, Kernel};
use crate::tensor::{TType, Tensor};
use crate::{Error, Result};
use std::collections::HashMap;

pub struct EdgeTpuOp {
    pub exes: Vec<Executable>,
    infer: usize,
    caching: Vec<usize>,
    /// batches per invoke (from prepare)
    batches: usize,
    /// tells this op's parameters in the chip from another op's (with the same model or caching token)
    id: u64,
}

static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

impl Kernel for EdgeTpuOp {
    fn prepare(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        let ins = (0..io.n_in()).map(|k| io.input(k)).collect::<Result<Vec<_>>>()?;
        let outs: Vec<&Tensor> = io.outs.iter().collect();
        self.batches = self.validate(&ins, &outs)?;
        Ok(())
    }

    fn eval(&mut self, io: &mut Io, env: &mut Env) -> Result<()> {
        let dev = env.device.ok_or_else(|| Error("no Edge TPU".into()))?;
        let ins = (0..io.n_in()).map(|k| io.input(k).map(|t| t.data())).collect::<Result<Vec<_>>>()?;
        let metas: Vec<Tensor> = io.outs.iter().map(|o| o.clone_meta()).collect();
        let infos: Vec<&Tensor> = metas.iter().collect();
        let mut out_data: Vec<&mut [u8]> = io.outs.iter_mut().map(|o| o.data_mut()).collect();
        self.invoke(&mut dev.lock(), self.batches, &ins, &mut out_data, &infos)
    }

    fn edgetpu(&self) -> Option<&EdgeTpuOp> {
        Some(self)
    }
}

/// TFLite type an Edge TPU layer type is exchanged as (custom_op.cc ValidateDataType)
fn tflite_type(d: DType) -> Option<TType> {
    Some(match d {
        DType::U8 => TType::U8,
        DType::I8 => TType::I8,
        DType::U16 | DType::I16 => TType::I16,
        DType::I32 => TType::I32,
        DType::F32 => TType::F32,
        DType::F16 => TType::F16,
        DType::BF16 => return None,
    })
}

/// an output the old compilers produced as float32 / uint16 for a uint8 classifier tensor (custom_op.cc ReFormatOutputs)
fn classifier_conversion(l: &Layer, t: TType) -> bool {
    t == TType::U8 && l.y == 1 && l.x == 1 && matches!(l.dtype, DType::F32 | DType::U16)
}

impl EdgeTpuOp {
    pub fn new(options: &[u8]) -> Result<Self> {
        let mut exes = exe::parse_custom_op(options)?;
        // several chips' executables in one op: keep the USB accelerator's
        if exes.iter().any(|e| e.chip == "beagle") {
            exes.retain(|e| e.chip == "beagle");
        }
        let infer: Vec<usize> = (0..exes.len()).filter(|&i| exes[i].ty != ExeType::ParameterCaching).collect();
        let [infer] = infer[..] else {
            return Err(Error(format!("expected one inference executable in the Edge TPU op, found {}", infer.len())));
        };
        let caching = (0..exes.len()).filter(|&i| exes[i].ty == ExeType::ParameterCaching).collect();
        if exes[infer].outputs.iter().any(|l| l.name.ends_with("_variable_output")) {
            return Err(Error("Edge TPU models with variable (stateful) outputs are not supported".into()));
        }
        Ok(EdgeTpuOp { exes, infer, caching, batches: 1, id: NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed) })
    }

    fn exe(&self) -> &Executable {
        &self.exes[self.infer]
    }

    /// custom_op.cc ValidateInputsAndOutputs: types and sizes, and the batch count (the first dimension may batch)
    pub fn validate(&self, ins: &[&Tensor], outs: &[&Tensor]) -> Result<usize> {
        let e = self.exe();
        if ins.len() != e.inputs.len() || outs.len() < e.outputs.len() {
            return Err(Error(format!(
                "the Edge TPU op has {} inputs / {} outputs, its executable {} / {}",
                ins.len(),
                outs.len(),
                e.inputs.len(),
                e.outputs.len()
            )));
        }
        let batches_of = |t: &Tensor, single: usize| {
            if t.shape.first().is_some_and(|&b| b * single == t.bytes()) { t.shape[0] } else { 1 }
        };
        let mut batches = 0;
        for (t, l) in ins.iter().zip(&e.inputs) {
            if tflite_type(l.dtype) != Some(t.dtype) {
                return Err(Error(format!("input {:?}: tensor type {:?}, Edge TPU layer type {:?}", t.name, t.dtype, l.dtype)));
            }
            let single = l.actual_size_bytes();
            let b = batches_of(t, single);
            if batches != 0 && b != batches {
                return Err(Error("the Edge TPU op's inputs have different batch sizes".into()));
            }
            batches = b;
            if b * single != t.bytes() {
                return Err(Error(format!("input {:?} has {} bytes, the Edge TPU expects {} per batch", t.name, t.bytes(), single)));
            }
        }
        let batches = batches.max(1);
        for (t, l) in outs.iter().zip(&e.outputs) {
            let conv = classifier_conversion(l, t.dtype);
            if !conv && tflite_type(l.dtype) != Some(t.dtype) {
                return Err(Error(format!("output {:?}: tensor type {:?}, Edge TPU layer type {:?}", t.name, t.dtype, l.dtype)));
            }
            let single = l.actual_size_bytes() / l.dtype.size() * t.dtype.size();
            if batches * single != t.bytes() {
                return Err(Error(format!("output {:?} has {} bytes, the Edge TPU produces {} per batch", t.name, t.bytes(), single)));
            }
        }
        Ok(batches)
    }

    pub fn invoke(&self, tpu: &mut EdgeTpu, batches: usize, ins: &[&[u8]], outs: &mut [&mut [u8]], out_info: &[&Tensor]) -> Result<()> {
        let e = self.exe();
        let caching: Vec<&Executable> = self.caching.iter().map(|&i| &self.exes[i]).collect();
        tpu.ensure_cached(self.id, e.caching_token, &caching)?;
        for b in 0..batches {
            let mut inputs = HashMap::new();
            for (l, data) in e.inputs.iter().zip(ins) {
                let n = l.actual_size_bytes();
                inputs.insert(l.name.as_str(), std::sync::Arc::new(host_input(l, &data[b * n..(b + 1) * n])));
            }
            let mut outputs = HashMap::new();
            tpu.run(e, &inputs, &mut outputs)?;
            for ((l, out), info) in e.outputs.iter().zip(outs.iter_mut()).zip(out_info) {
                let host = outputs.get(&l.name).ok_or_else(|| Error(format!("the Edge TPU sent nothing for output {:?}", l.name)))?;
                let mut user = vec![0u8; l.actual_size_bytes()];
                relayout(l, host, &mut user)?;
                if l.dtype.signed() {
                    flip_sign(l, &mut user);
                }
                let single = user.len() / l.dtype.size() * info.dtype.size();
                let dst = &mut out[b * single..(b + 1) * single];
                if classifier_conversion(l, info.dtype) {
                    reformat_classifier(l, &user, dst, info);
                } else {
                    dst.copy_from_slice(&user);
                }
            }
        }
        Ok(())
    }
}

/// SingleTpuRequest::AddInput: scatter iterations into their padded slots, flip the sign bit of signed types
fn host_input(l: &Layer, user: &[u8]) -> Vec<u8> {
    let padded = l.padded_size_bytes();
    let mut host = vec![0u8; padded.max(user.len())];
    if l.execution_count > 1 && user.len() != padded {
        let (p, a) = (padded / l.execution_count, l.actual_size_bytes() / l.execution_count);
        for i in 0..l.execution_count {
            host[p * i..p * i + a].copy_from_slice(&user[a * i..a * (i + 1)]);
        }
    } else {
        host[..user.len()].copy_from_slice(user);
    }
    if l.dtype.signed() {
        flip_sign(l, &mut host);
    }
    host
}

/// LayerInformation::TransformSignedDataType: XOR the most significant byte of the first y*x*z values with 0x80
fn flip_sign(l: &Layer, buf: &mut [u8]) {
    let s = l.dtype.size();
    for i in 0..l.y * l.x * l.z {
        buf[i * s + s - 1] ^= 0x80;
    }
}

fn reformat_classifier(l: &Layer, user: &[u8], dst: &mut [u8], info: &Tensor) {
    let (scale, zp) = (info.scale(), info.zero_point() as f32);
    for (j, d) in dst.iter_mut().enumerate() {
        let v = if l.dtype == DType::F32 {
            let f = f32::from_le_bytes(user[4 * j..4 * j + 4].try_into().unwrap());
            (f / scale + zp) as i32
        } else {
            i16::from_le_bytes(user[2 * j..2 * j + 2].try_into().unwrap()) as i32
        };
        *d = v.clamp(0, 255) as u8;
    }
}

fn dim_len(r: (i32, i32)) -> usize {
    (r.1 - r.0 + 1) as usize
}

fn no_padding(t: &TensorLayout) -> bool {
    (0..t.shape.len().saturating_sub(1)).all(|i| t.stride[i] == t.stride[i + 1] * dim_len(t.shape[i + 1]) as i32)
}

fn needs_relayout(l: &Layer) -> bool {
    match &l.shape_info {
        None => true,
        Some(s) => s.len() > 1 || !no_padding(&s[0].0),
    }
}

/// OutputLayerInformation::Relayout: from the tiles' layout (`src`, padded) to the packed y/x/z tensor (`dest`)
pub fn relayout(l: &Layer, src: &[u8], dest: &mut [u8]) -> Result<()> {
    let dts = l.dtype.size();
    let z_bytes = l.z * dts;
    let exec = l.execution_count;
    let copy = |dest: &mut [u8], d: usize, s: usize, n: usize| -> Result<()> {
        let from = src.get(s..s + n).ok_or_else(|| Error(format!("output {:?}: the Edge TPU sent {} bytes, {} needed", l.name, src.len(), s + n)))?;
        let to = dest.get_mut(d..d + n).ok_or_else(|| Error(format!("output {:?}: relayout past the end of the tensor", l.name)))?;
        to.copy_from_slice(from);
        Ok(())
    };
    if exec == 1 {
        if !needs_relayout(l) {
            let batch = l.shape.as_ref().map_or(1, |s| dim_len(s[0]));
            return copy(dest, 0, 0, batch * l.y * l.x * z_bytes);
        }
        if let Some(si) = &l.shape_info {
            return relayout_with_shape_info(l, si, src, dest);
        }
    } else if l.padded_size_bytes() == l.actual_size_bytes() && !needs_relayout(l) {
        return copy(dest, 0, 0, l.actual_size_bytes());
    }
    if l.y == 1 && l.x == 1 {
        let (padded, actual) = (l.padded_size_bytes(), l.actual_size_bytes());
        if exec == 1 || padded == actual {
            return copy(dest, 0, 0, z_bytes * exec);
        }
        // drop the padding at the end of each execution
        let pad = (padded - actual) / exec;
        for i in 0..exec {
            copy(dest, i * z_bytes, i * (z_bytes + pad), z_bytes)?;
        }
        return Ok(());
    }
    let lay = l.layout.as_ref().ok_or_else(|| Error(format!("output {:?} has no tile layout", l.name)))?;
    // element index of (y, x, z=0) in the tiles' buffer (GetBufferIndex)
    let index = |y: usize, x: usize| -> usize {
        let tile = (lay.y_tile[y] + lay.x_tile[x]) as usize;
        (lay.tile_byte_offset[tile] + lay.y_local_y_offset[y] * lay.x_local_row_size[x] + lay.x_local_byte_offset[x]) as usize
    };
    let z_padded = if l.x > 1 { index(0, 1) - index(0, 0) } else { index(1, 0) - index(0, 0) } * dts;
    // runs of x that live on the same tile
    let mut runs = Vec::new();
    let (mut last_x, mut last_tile) = (0, lay.x_tile[0]);
    for x in 1..l.x {
        if lay.x_tile[x] != last_tile {
            runs.push(x - last_x);
            (last_x, last_tile) = (x, lay.x_tile[x]);
        }
    }
    runs.push(l.x - last_x);
    let mut d = 0;
    if z_bytes != z_padded {
        for y in 0..l.y {
            let mut x0 = 0;
            for &run in &runs {
                let mut s = index(y, x0) * dts;
                for _ in 0..run {
                    copy(dest, d, s, z_bytes)?;
                    d += z_bytes;
                    s += z_padded;
                }
                x0 += run;
            }
        }
    } else if runs.len() > 1 || lay.y_tile[0] != lay.y_tile[l.y - 1] {
        if exec != 1 {
            return Err(Error(format!("output {:?}: relayout of a multi-execution output across tiles is not supported", l.name)));
        }
        for y in 0..l.y {
            let mut x0 = 0;
            for &run in &runs {
                copy(dest, d, index(y, x0) * dts, z_bytes * run)?;
                d += z_bytes * run;
                x0 += run;
            }
        }
    } else {
        copy(dest, 0, 0, l.x * l.y * z_bytes * exec)?;
    }
    Ok(())
}

/// RelayoutWithShapeInformation: copy each streamed slice into the packed layout of the layer's shape
fn relayout_with_shape_info(l: &Layer, slices: &[(TensorLayout, usize)], src: &[u8], dest: &mut [u8]) -> Result<()> {
    let shape = l.shape.clone().ok_or_else(|| Error(format!("output {:?} has slices but no shape", l.name)))?;
    let mut stride = vec![0i32; shape.len()];
    let mut s = 1;
    for i in (0..shape.len()).rev() {
        stride[i] = s;
        s *= dim_len(shape[i]) as i32;
    }
    let packed = TensorLayout { shape, stride };
    for (layout, offset) in slices {
        let src = src.get(*offset..).ok_or_else(|| Error(format!("output {:?}: slice past the end", l.name)))?;
        copy_shape(&layout.shape, layout, src, &packed, dest, l.dtype.size(), 0)?;
    }
    Ok(())
}

fn memory_index(t: &TensorLayout, pos: impl Iterator<Item = i32>) -> usize {
    pos.enumerate().map(|(i, p)| (t.stride[i] * (p - t.shape[i].0)) as usize).sum()
}

fn copy_shape(shape: &exe::Shape, src_l: &TensorLayout, src: &[u8], dst_l: &TensorLayout, dst: &mut [u8], bpe: usize, dim: usize) -> Result<()> {
    let n: usize = shape.iter().map(|&r| dim_len(r)).product();
    let first = |t: &TensorLayout| memory_index(t, shape.iter().map(|r| r.0));
    let last = |t: &TensorLayout| memory_index(t, shape.iter().map(|r| r.1));
    if last(src_l) - first(src_l) + 1 == n && last(dst_l) - first(dst_l) + 1 == n {
        let (s, d) = (first(src_l) * bpe, first(dst_l) * bpe);
        let from = src.get(s..s + n * bpe).ok_or_else(|| Error("output slice past the end of the Edge TPU's data".into()))?;
        dst.get_mut(d..d + n * bpe).ok_or_else(|| Error("output slice outside the tensor".into()))?.copy_from_slice(from);
        return Ok(());
    }
    for i in shape[dim].0..=shape[dim].1 {
        let mut slice = shape.clone();
        slice[dim] = (i, i);
        copy_shape(&slice, src_l, src, dst_l, dst, bpe, dim + 1)?;
    }
    Ok(())
}
