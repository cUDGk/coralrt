//! Running a model: each subgraph's ops in order, prepare (shapes) before the first invoke and after a resize.

use crate::ops::Io;
use crate::tensor::{Alloc, Buf, Tensor};
use crate::tflite::{OpDef, Subgraph};
use crate::{panic_text, Device, Error, Result};
use std::any::Any;
use std::collections::HashMap;

/// How TFLite_Detection_PostProcess orders boxes of equal score in its non-max suppression.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum DetectionSort {
    /// std::partial_sort, as TensorFlow Lite up to 2.5 (pycoral 2.0, the official runtime)
    #[default]
    Tflite25,
    /// std::stable_sort, as TensorFlow Lite 2.6 and later (LiteRT)
    Stable,
}

#[derive(Clone, Debug, Default)]
pub struct Settings {
    pub detection_sort: DetectionSort,
}

/// a subgraph at run time
pub struct Graph {
    pub name: String,
    pub tensors: Vec<Tensor>,
    pub ops: Vec<OpDef>,
    pub inputs: Vec<usize>,
    pub outputs: Vec<usize>,
    /// the shapes are worked out (prepare ran since the last resize)
    pub prepared: bool,
}

impl From<Subgraph> for Graph {
    fn from(s: Subgraph) -> Self {
        Graph { name: s.name, tensors: s.tensors, ops: s.ops, inputs: s.inputs, outputs: s.outputs, prepared: false }
    }
}

/// What an op sees besides its tensors.
pub struct Env<'a> {
    /// the model's subgraphs; the running ones are taken out (None)
    pub graphs: &'a mut [Option<Graph>],
    /// resources by id (hash tables), shared by all subgraphs
    pub resources: &'a mut HashMap<i32, Box<dyn Any + Send>>,
    pub device: Option<&'a Device>,
    pub settings: &'a Settings,
}

impl Env<'_> {
    /// Run subgraph `idx` on these inputs (copied into its input tensors, resized to their shapes) and return its
    /// outputs. Control flow ops (IF, WHILE, CALL_ONCE) use this.
    pub fn run(&mut self, idx: usize, inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
        let mut g = self.take(idx)?;
        let r = (|| {
            set_inputs(&mut g, inputs)?;
            invoke(&mut g, self)?;
            Ok(g.outputs.iter().map(|&o| g.tensors[o].clone()).collect())
        })();
        self.graphs[idx] = Some(g);
        r
    }

    /// prepare subgraph `idx` for inputs of these shapes / types, returning its output tensors' metadata
    pub fn prepare(&mut self, idx: usize, inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
        let mut g = self.take(idx)?;
        let r = (|| {
            set_inputs(&mut g, inputs)?;
            if !g.prepared {
                prepare(&mut g, self)?;
            }
            Ok(g.outputs.iter().map(|&o| g.tensors[o].clone_meta()).collect())
        })();
        self.graphs[idx] = Some(g);
        r
    }

    fn take(&mut self, idx: usize) -> Result<Graph> {
        self.graphs
            .get_mut(idx)
            .ok_or_else(|| Error(format!("no subgraph {idx}")))?
            .take()
            .ok_or_else(|| Error(format!("subgraph {idx} calls itself")))
    }
}

/// copy `inputs` into g's inputs; a shape or type change makes g prepare again
fn set_inputs(g: &mut Graph, inputs: &[&Tensor]) -> Result<()> {
    if inputs.len() != g.inputs.len() {
        return Err(Error(format!("subgraph {:?} takes {} inputs, got {}", g.name, g.inputs.len(), inputs.len())));
    }
    for (&t, src) in g.inputs.iter().zip(inputs) {
        let dst = &mut g.tensors[t];
        if dst.shape != src.shape || dst.dtype != src.dtype {
            g.prepared = false;
        }
        dst.dtype = src.dtype;
        dst.shape = src.shape.clone();
        dst.buf = src.buf.clone();
    }
    Ok(())
}

/// work out every tensor's shape, op by op; variable tensors start from their initial values
pub fn prepare(g: &mut Graph, env: &mut Env) -> Result<()> {
    for t in g.tensors.iter_mut().filter(|t| t.alloc == Alloc::Dynamic) {
        // sized again by its op: back to an ordinary tensor until the op says otherwise
        t.alloc = Alloc::Arena;
    }
    for k in 0..g.ops.len() {
        step(g, k, env, true)?;
    }
    g.prepared = true;
    Ok(())
}

pub fn invoke(g: &mut Graph, env: &mut Env) -> Result<()> {
    if !g.prepared {
        prepare(g, env)?;
    }
    for k in 0..g.ops.len() {
        step(g, k, env, false)?;
    }
    Ok(())
}

/// variable tensors back to their initial values (the model's buffer, else zeros)
pub fn reset_variables(g: &mut Graph) {
    for t in g.tensors.iter_mut().filter(|t| t.alloc == Alloc::Variable) {
        let n = t.elements() * t.dtype.size();
        t.buf = Buf::new(n);
        if let Some(init) = &t.initial {
            let m = n.min(init.len());
            t.buf.bytes_mut()[..m].copy_from_slice(&init[..m]);
        }
    }
}

/// Prepare or run op k. Its outputs, and the inputs it updates in place, leave `g.tensors` while it works and always
/// come back, also after a panic (callers may hold pointers into them).
fn step(g: &mut Graph, k: usize, env: &mut Env, preparing: bool) -> Result<()> {
    let op = &mut g.ops[k];
    let mutated: Vec<usize> = op.kernel.mutated_inputs().to_vec();
    let outs: Vec<Tensor> = op.outputs.iter().map(|&o| std::mem::take(&mut g.tensors[o as usize])).collect();
    let vars: Vec<(usize, Tensor)> = mutated
        .iter()
        .filter_map(|&p| op.inputs.get(p).filter(|&&i| i >= 0).map(|&i| (p, std::mem::take(&mut g.tensors[i as usize]))))
        .collect();
    let tensors = &g.tensors;
    let mut io = Io {
        ins: op.inputs.iter().enumerate().map(|(p, &i)| (i >= 0 && !mutated.contains(&p)).then(|| &tensors[i as usize])).collect(),
        outs,
        vars,
    };
    let kernel = &mut op.kernel;
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| if preparing { kernel.prepare(&mut io, env) } else { kernel.eval(&mut io, env) }))
        .unwrap_or_else(|p| Err(Error(panic_text(&p))));
    let Io { outs, vars, .. } = io;
    for (&o, t) in op.outputs.iter().zip(outs) {
        g.tensors[o as usize] = t;
    }
    for (p, t) in vars {
        g.tensors[op.inputs[p] as usize] = t;
    }
    result.map_err(|e| Error(format!("{} (op {k}{}): {e}", op.name, if preparing { ", prepare" } else { "" })))
}
