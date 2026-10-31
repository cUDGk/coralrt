// From TensorFlow Lite 2.5 (Copyright The TensorFlow Authors, Apache License 2.0), translated and changed; see NOTICE.
// The heap steps are from the Microsoft STL (Copyright (c) Microsoft Corporation, Apache-2.0 WITH LLVM-exception).
//! NON_MAX_SUPPRESSION_V4 / V5 (non_max_suppression.cc, reference_ops::NonMaxSuppression): greedy selection by
//! score with IoU suppression, V5 with soft (Gaussian) score decay. The candidates live in a std::priority_queue,
//! whose order among equal scores follows the heap algorithm, so that is written out as the STL runs it.

use super::util::*;
use super::{Io, Kernel};
use crate::interp::Env;
use crate::tensor::{Alloc, TType};
use crate::tflite::OpSpec;
use crate::Result;

pub fn create(s: &OpSpec) -> Result<Option<Box<dyn Kernel>>> {
    Ok(Some(match s.code {
        120 | 121 => Box::new(Nms),
        _ => return Ok(None),
    }))
}

struct Nms;

impl Nms {
    fn max_output(io: &Io) -> Result<usize> {
        let m = ints(io.input(2)?)?.first().copied().unwrap_or(-1);
        usize::try_from(m).or_else(|_| err("max_output_size must not be negative"))
    }

    fn resize(io: &mut Io, soft: bool) -> Result<()> {
        let m = Self::max_output(io)?;
        io.outs[0].resize(vec![m]);
        if soft {
            io.outs[1].resize(vec![m]);
        }
        Ok(())
    }
}

impl Kernel for Nms {
    fn prepare(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        let soft = io.n_in() == 6;
        if io.n_in() != 5 && !soft {
            return err("NMS takes 5 inputs (6 for soft NMS)");
        }
        let (b, s, m) = (io.input(0)?, io.input(1)?, io.input(2)?);
        let scalar_f = |k: usize| io.input(k).map(|t| t.dtype == TType::F32 && t.shape.is_empty());
        let ok = b.dtype == TType::F32
            && b.shape.len() == 2
            && b.shape[1] == 4
            && s.dtype == TType::F32
            && s.shape == [b.shape[0]]
            && m.dtype == TType::I32
            && m.shape.is_empty()
            && io.input(3)?.dtype == TType::F32
            && io.input(3)?.shape.is_empty()
            && io.input(4)?.shape.is_empty()
            && (!soft || scalar_f(5)?)
            && io.outs.len() == if soft { 3 } else { 2 };
        if !ok {
            return err("NMS takes boxes [n, 4], scores [n] and scalar max size, IoU and score thresholds (and sigma)");
        }
        // 2.5 sets the outputs' types itself
        io.outs[0].dtype = TType::I32;
        if soft {
            io.outs[1].dtype = TType::F32;
        }
        let last = io.outs.len() - 1;
        io.outs[last].dtype = TType::I32;
        io.outs[last].resize(vec![]);
        if m.alloc == Alloc::Constant {
            Self::resize(io, soft)
        } else {
            io.outs[0].alloc = Alloc::Dynamic;
            if soft {
                io.outs[1].alloc = Alloc::Dynamic;
            }
            Ok(())
        }
    }

    fn eval(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        let soft = io.n_in() == 6;
        let max_output = Self::max_output(io)?;
        if io.outs[0].alloc == Alloc::Dynamic {
            Self::resize(io, soft)?;
        }
        let (boxes, scores) = (io.input(0)?.as_slice::<f32>(), io.input(1)?.as_slice::<f32>());
        let iou_threshold = io.input(3)?.as_slice::<f32>()[0];
        let score_threshold = io.input(4)?.as_slice::<f32>()[0];
        let sigma = if soft { io.input(5)?.as_slice::<f32>()[0] } else { 0.0 };
        if sigma < 0.0 {
            return err(format!("Invalid sigma value for soft NMS: {sigma}"));
        }
        let (indices, picked) = suppress(boxes, scores, max_output, iou_threshold, score_threshold, sigma);
        let n = indices.len();
        let mut idx = vec![0i64; max_output];
        idx[..n].iter_mut().zip(&indices).for_each(|(d, &v)| *d = v as i64);
        put_ints(&mut io.outs[0], &idx)?;
        if soft {
            let sc = io.outs[1].as_mut_slice::<f32>();
            sc.fill(0.0);
            sc[..n].copy_from_slice(&picked);
        }
        let last = io.outs.len() - 1;
        put_ints(&mut io.outs[last], &[n as i64])
    }
}

#[derive(Clone, Copy)]
struct Candidate {
    index: usize,
    score: f32,
    suppress_begin: usize,
}

/// std::priority_queue<Candidate, deque, score <> as MSVC's STL keeps it: push_heap sifts up, pop_heap moves the
/// hole down to the larger child (the right one on ties), then sifts the last element up from there
struct Heap(Vec<Candidate>);

impl Heap {
    fn less(a: &Candidate, b: &Candidate) -> bool {
        a.score < b.score
    }

    fn sift_up(&mut self, mut hole: usize, top: usize, val: Candidate) {
        while top < hole {
            let parent = (hole - 1) >> 1;
            if !Self::less(&self.0[parent], &val) {
                break;
            }
            self.0[hole] = self.0[parent];
            hole = parent;
        }
        self.0[hole] = val;
    }

    fn push(&mut self, c: Candidate) {
        self.0.push(c);
        let n = self.0.len();
        self.sift_up(n - 1, 0, c);
    }

    fn pop(&mut self) -> Candidate {
        let top = self.0[0];
        let n = self.0.len();
        if n >= 2 {
            let val = self.0[n - 1];
            let bottom = n - 1;
            let (mut hole, mut idx) = (0usize, 0usize);
            let max_non_leaf = (bottom - 1) >> 1;
            while idx < max_non_leaf {
                idx = 2 * idx + 2;
                if Self::less(&self.0[idx], &self.0[idx - 1]) {
                    idx -= 1;
                }
                self.0[hole] = self.0[idx];
                hole = idx;
            }
            if idx == max_non_leaf && bottom % 2 == 0 {
                self.0[hole] = self.0[bottom - 1];
                hole = bottom - 1;
            }
            self.sift_up(hole, 0, val);
        }
        self.0.pop();
        top
    }
}

/// ComputeIntersectionOverUnion: corners in either order, 0 for an empty box
fn iou(b: &[f32], i: usize, j: usize) -> f32 {
    let mn = |a: f32, c: f32| if c < a { c } else { a };
    let mx = |a: f32, c: f32| if a < c { c } else { a };
    let corners = |k: usize| {
        let r = &b[4 * k..4 * k + 4];
        (mn(r[0], r[2]), mx(r[0], r[2]), mn(r[1], r[3]), mx(r[1], r[3]))
    };
    let (iy0, iy1, ix0, ix1) = corners(i);
    let (jy0, jy1, jx0, jx1) = corners(j);
    let (area_i, area_j) = ((iy1 - iy0) * (ix1 - ix0), (jy1 - jy0) * (jx1 - jx0));
    if area_i <= 0.0 || area_j <= 0.0 {
        return 0.0;
    }
    let inter = mx(mn(iy1, jy1) - mx(iy0, jy0), 0.0) * mx(mn(ix1, jx1) - mx(ix0, jx0), 0.0);
    inter / (area_i + area_j - inter)
}

/// reference_ops::NonMaxSuppression: the selected indices and their (decayed) scores
fn suppress(boxes: &[f32], scores: &[f32], max_output: usize, iou_threshold: f32, score_threshold: f32, sigma: f32) -> (Vec<usize>, Vec<f32>) {
    let mut heap = Heap(Vec::new());
    for (i, &s) in scores.iter().enumerate() {
        if s > score_threshold {
            heap.push(Candidate { index: i, score: s, suppress_begin: 0 });
        }
    }
    let outputs = heap.0.len().min(max_output);
    let (mut selected, mut picked) = (Vec::new(), Vec::new());
    if outputs == 0 {
        return (selected, picked);
    }
    let scale = if sigma > 0.0 { (-0.5 / sigma as f64) as f32 } else { 0.0 };
    while selected.len() < outputs && !heap.0.is_empty() {
        let mut next = heap.pop();
        let original = next.score;
        let mut hard = false;
        for j in (next.suppress_begin..selected.len()).rev() {
            let v = iou(boxes, next.index, selected[j]);
            if v >= iou_threshold {
                hard = true;
                break;
            }
            if sigma > 0.0 {
                next.score *= (scale * v * v).exp();
            }
            if next.score <= score_threshold {
                break;
            }
        }
        next.suppress_begin = selected.len();
        if !hard {
            if next.score == original {
                selected.push(next.index);
                picked.push(next.score);
            }
            if next.score > score_threshold {
                heap.push(next);
            }
        }
    }
    (selected, picked)
}
