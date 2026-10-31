// From TensorFlow Lite 2.5 (Copyright The TensorFlow Authors, Apache License 2.0), translated and changed; see NOTICE.
// The heap steps are from the Microsoft STL (Copyright (c) Microsoft Corporation, Apache-2.0 WITH LLVM-exception).
//! TFLite_Detection_PostProcess (kernels/detection_postprocess.cc): box decoding and non-max suppression

use super::util::err;
use super::{Io, Kernel};
use crate::flex::{self, Ref};
use crate::interp::{DetectionSort, Env};
use crate::tensor::{TType, Tensor};
use crate::Result;

pub struct Dpp {
    o: Options,
}

impl Dpp {
    pub fn new(custom_opts: &[u8]) -> Self {
        Dpp { o: Options::parse(flex::root(custom_opts)) }
    }
}

impl Kernel for Dpp {
    fn prepare(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        // boxes [1, n, 4], classes [1, n], scores [1, n], count [1], n = max detections x classes per detection
        let n = self.o.max_detections * self.o.max_classes_per_detection;
        if io.outs.len() != 4 {
            return err("4 outputs expected");
        }
        for (o, shape) in io.outs.iter_mut().zip([vec![1, n, 4], vec![1, n], vec![1, n], vec![1]]) {
            o.resize(shape);
        }
        Ok(())
    }

    fn eval(&mut self, io: &mut Io, env: &mut Env) -> Result<()> {
        let ins = [io.input(0)?, io.input(1)?, io.input(2)?];
        eval(&self.o, env.settings.detection_sort, ins, &mut io.outs)
    }
}

#[derive(Clone, Copy)]
pub struct Options {
    sort: DetectionSort,
    max_detections: usize,
    max_classes_per_detection: usize,
    detections_per_class: usize,
    score_threshold: f32,
    iou_threshold: f32,
    num_classes: usize,
    use_regular_nms: bool,
    scale: [f32; 4],
}

impl Options {
    pub fn parse(m: Ref) -> Self {
        let f = |k: &str| m.get(k).map_or(0.0, |v| v.as_f64() as f32);
        let i = |k: &str| m.get(k).map_or(0, |v| v.as_i64()) as usize;
        Options {
            sort: DetectionSort::Tflite25,
            max_detections: i("max_detections"),
            max_classes_per_detection: i("max_classes_per_detection"),
            detections_per_class: m.get("detections_per_class").map_or(100, |v| v.as_i64() as usize),
            score_threshold: f("nms_score_threshold"),
            iou_threshold: f("nms_iou_threshold"),
            num_classes: i("num_classes"),
            use_regular_nms: m.get("use_regular_nms").is_some_and(|v| v.as_bool()),
            scale: [f("y_scale"), f("x_scale"), f("h_scale"), f("w_scale")],
        }
    }
}

/// std::partial_sort(indices, k) by decreasing value: heap select, then sort_heap. The result for equal values
/// depends on these exact heap steps (identical in libstdc++ and the MSVC STL), and quantized scores tie a lot.
fn decreasing_partial_arg_sort(values: &[f32], k: usize) -> Vec<usize> {
    if k == 1 && !values.is_empty() {
        // a heap of one: an element replaces the top only when strictly greater, so the first largest stays
        let mut best = 0;
        for (i, &v) in values.iter().enumerate().skip(1) {
            if v > values[best] {
                best = i;
            }
        }
        return vec![best];
    }
    let less = |a: usize, b: usize| values[a] > values[b];
    let mut ix: Vec<usize> = (0..values.len()).collect();
    // __adjust_heap: move the hole at `hole` down to a leaf, then push `v` back up (max-heap under `less`)
    fn adjust(ix: &mut [usize], mut hole: usize, len: usize, v: usize, less: &impl Fn(usize, usize) -> bool) {
        let top = hole;
        let mut child = hole;
        while child < (len.saturating_sub(1)) / 2 {
            child = 2 * (child + 1);
            if less(ix[child], ix[child - 1]) {
                child -= 1;
            }
            ix[hole] = ix[child];
            hole = child;
        }
        if len & 1 == 0 && len >= 2 && child == (len - 2) / 2 {
            child = 2 * (child + 1);
            ix[hole] = ix[child - 1];
            hole = child - 1;
        }
        let mut parent = hole.wrapping_sub(1) / 2;
        while hole > top && less(ix[parent], v) {
            ix[hole] = ix[parent];
            hole = parent;
            parent = hole.wrapping_sub(1) / 2;
        }
        ix[hole] = v;
    }
    let mid = k.min(ix.len());
    if mid >= 2 {
        let mut parent = (mid - 2) / 2;
        loop {
            let v = ix[parent];
            adjust(&mut ix, parent, mid, v, &less);
            if parent == 0 {
                break;
            }
            parent -= 1;
        }
    }
    for i in mid..ix.len() {
        if mid > 0 && less(ix[i], ix[0]) {
            // __pop_heap(first, middle, i)
            let v = ix[i];
            ix[i] = ix[0];
            adjust(&mut ix, 0, mid, v, &less);
        }
    }
    let mut last = mid;
    while last > 1 {
        last -= 1;
        let v = ix[last];
        ix[last] = ix[0];
        adjust(&mut ix, 0, last, v, &less);
    }
    ix.truncate(mid);
    ix
}

#[cfg(test)]
mod tests {
    /// the cases of tests/partial_sort.cpp (same LCG), one line each: n k and the first k indices
    fn cases() -> String {
        let mut state: u64 = 12345;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };
        let mut out = String::new();
        for _ in 0..2000 {
            let n = 1 + next() % 300;
            let k = 1 + next() % n;
            let distinct = 1 + next() % 12;
            let v: Vec<f32> = (0..n).map(|_| (next() % distinct) as f32 * 0.25).collect();
            let ix = super::decreasing_partial_arg_sort(&v, k as usize);
            out += &format!("{n} {k}");
            for i in ix {
                out += &format!(" {i}");
            }
            out += "\n";
        }
        out
    }

    #[test]
    fn partial_sort_matches_the_cpp_standard_library() {
        if let Ok(path) = std::env::var("PARTIAL_SORT_REFERENCE") {
            let want = std::fs::read_to_string(path).unwrap().replace("\r\n", "\n");
            for (k, (a, b)) in cases().lines().zip(want.lines()).enumerate() {
                assert_eq!(a, b, "case {k}");
            }
        }
        // FNV-1a of the output of tests/partial_sort.cpp built against the MSVC STL (clang 23, 2026-10-30)
        let h = cases().bytes().fold(0xcbf29ce484222325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100000001b3));
        assert_eq!(h, EXPECTED_FNV);
    }

    const EXPECTED_FNV: u64 = 4091578481920238488;
}

type Box4 = [f32; 4]; // ymin, xmin, ymax, xmax

fn iou(a: &Box4, b: &Box4) -> f32 {
    let area_a = (a[2] - a[0]) * (a[3] - a[1]);
    let area_b = (b[2] - b[0]) * (b[3] - b[1]);
    if area_a <= 0.0 || area_b <= 0.0 {
        return 0.0;
    }
    let fmax = |x: f32, y: f32| if x < y { y } else { x };
    let fmin = |x: f32, y: f32| if y < x { y } else { x };
    let inter = fmax(fmin(a[2], b[2]) - fmax(a[0], b[0]), 0.0) * fmax(fmin(a[3], b[3]) - fmax(a[1], b[1]), 0.0);
    inter / (area_a + area_b - inter)
}

/// NonMaxSuppressionSingleClassHelper: indices of the kept boxes, best first
fn nms_single(o: &Options, boxes: &[Box4], scores: &[f32], max_detections: usize) -> Result<Vec<usize>> {
    if !(o.iou_threshold > 0.0 && o.iou_threshold <= 1.0) {
        return err(format!("IoU threshold {} outside (0, 1]", o.iou_threshold));
    }
    if boxes.iter().any(|b| b[0] > b[2] || b[1] > b[3]) {
        return err("a decoded box has min > max");
    }
    let (mut keep_scores, mut keep) = (Vec::new(), Vec::new());
    for (i, &s) in scores.iter().enumerate() {
        if s >= o.score_threshold {
            keep_scores.push(s);
            keep.push(i);
        }
    }
    // TFLite 2.5 (pycoral's) sorts with partial_sort; 2.6 and later use stable_sort, which orders equal scores
    // differently (with that one change this op matches TFLite 2.21 byte for byte, tests/oracle/cpu_slices.py)
    let sorted = match o.sort {
        DetectionSort::Tflite25 => decreasing_partial_arg_sort(&keep_scores, keep_scores.len()),
        DetectionSort::Stable => {
            let mut ix: Vec<usize> = (0..keep_scores.len()).collect();
            ix.sort_by(|&a, &b| keep_scores[b].partial_cmp(&keep_scores[a]).unwrap_or(std::cmp::Ordering::Equal));
            ix
        }
    };
    let size = keep.len().min(max_detections);
    let mut active = vec![true; keep.len()];
    let mut num_active = keep.len();
    let mut selected = Vec::new();
    for i in 0..keep.len() {
        if num_active == 0 || selected.len() >= size {
            break;
        }
        if !active[i] {
            continue;
        }
        selected.push(keep[sorted[i]]);
        active[i] = false;
        num_active -= 1;
        for j in i + 1..keep.len() {
            if active[j] && iou(&boxes[keep[sorted[i]]], &boxes[keep[sorted[j]]]) > o.iou_threshold {
                active[j] = false;
                num_active -= 1;
            }
        }
    }
    Ok(selected)
}

fn eval(o: &Options, sort: DetectionSort, ins: [&Tensor; 3], outs: &mut [Tensor]) -> Result<()> {
    let o = &Options { sort, ..*o };
    let [enc, cls, anc] = ins;
    if enc.shape.len() != 3 || cls.shape.len() != 3 || anc.shape.len() != 2 || outs.len() != 4 {
        return err("unexpected tensor ranks");
    }
    let (num_boxes, enc_len, with_bg) = (enc.shape[1], enc.shape[2], cls.shape[2]);
    if enc_len < 4 || cls.shape[1] != num_boxes || o.num_classes == 0 || with_bg < o.num_classes || with_bg - o.num_classes > 1 {
        return err("inconsistent box / class tensors");
    }
    let label_offset = with_bg - o.num_classes;
    // Dequantizer: (float(q) - int zero point) * float scale
    let deq = |t: &Tensor, i: usize| -> f32 {
        let zp = t.zero_point() as f32 as i32;
        (t.data()[i] as f32 - zp as f32) * t.scale()
    };
    let mut boxes = Vec::with_capacity(num_boxes);
    for idx in 0..num_boxes {
        let (b, a): ([f32; 4], [f32; 4]) = match enc.dtype {
            TType::U8 => (std::array::from_fn(|k| deq(enc, idx * enc_len + k)), std::array::from_fn(|k| deq(anc, idx * 4 + k))),
            TType::F32 => {
                let (e, a) = (enc.as_slice::<f32>(), anc.as_slice::<f32>());
                (std::array::from_fn(|k| e[idx * enc_len + k]), std::array::from_fn(|k| a[idx * 4 + k]))
            }
            t => return err(format!("{t:?} boxes are not supported")),
        };
        // DecodeCenterSizeBoxes, in double like the original
        let s = o.scale;
        let yc = (b[0] as f64 / s[0] as f64 * a[2] as f64 + a[0] as f64) as f32;
        let xc = (b[1] as f64 / s[1] as f64 * a[3] as f64 + a[1] as f64) as f32;
        let hh = (0.5 * (b[2] as f64 / s[2] as f64).exp() * a[2] as f64) as f32;
        let hw = (0.5 * (b[3] as f64 / s[3] as f64).exp() * a[3] as f64) as f32;
        boxes.push([yc - hh, xc - hw, yc + hh, xc + hw]);
    }
    let scores: Vec<f32> = match cls.dtype {
        TType::U8 => (0..num_boxes * with_bg).map(|i| deq(cls, i)).collect(),
        TType::F32 => cls.as_slice::<f32>()[..num_boxes * with_bg].to_vec(),
        t => return err(format!("{t:?} scores are not supported")),
    };
    let (rows, n) = if o.use_regular_nms { regular(o, &boxes, &scores, with_bg, label_offset)? } else { fast(o, &boxes, &scores, with_bg, label_offset)? };
    let [ob, oc, os, on] = outs else { unreachable!() };
    let slots = (ob.elements() / 4).min(oc.elements()).min(os.elements());
    if rows.len() > slots {
        return err(format!("{} detections for {slots} output slots", rows.len()));
    }
    // the slots past the detections: zeros (the fast path in TFLite leaves them as they were)
    let (ob, oc, os) = (ob.as_mut_slice::<f32>(), oc.as_mut_slice::<f32>(), os.as_mut_slice::<f32>());
    for k in 0..slots {
        let (bx, c, s) = rows.get(k).copied().unwrap_or(([0.0; 4], 0.0, 0.0));
        ob[4 * k..4 * k + 4].copy_from_slice(&bx);
        oc[k] = c;
        os[k] = s;
    }
    on.as_mut_slice::<f32>()[0] = n as f32;
    Ok(())
}

type Row = (Box4, f32, f32);

/// NonMaxSuppressionMultiClassFastHelper: NMS on each box's best class, then its top classes
fn fast(o: &Options, boxes: &[Box4], scores: &[f32], with_bg: usize, off: usize) -> Result<(Vec<Row>, usize)> {
    if o.max_classes_per_detection == 0 {
        return err("max_classes_per_detection must be > 0");
    }
    let per_anchor = o.max_classes_per_detection.min(o.num_classes);
    let mut sorted_classes = Vec::with_capacity(boxes.len());
    let mut max_scores = Vec::with_capacity(boxes.len());
    for row in 0..boxes.len() {
        let s = &scores[row * with_bg + off..row * with_bg + off + o.num_classes];
        let ci = decreasing_partial_arg_sort(s, per_anchor);
        max_scores.push(s[ci[0]]);
        sorted_classes.push(ci);
    }
    let selected = nms_single(o, boxes, &max_scores, o.max_detections)?;
    let mut rows = Vec::new();
    for (out_i, &sel) in selected.iter().enumerate() {
        for col in 0..per_anchor {
            let c = sorted_classes[sel][col];
            // detection outputs are laid out max_classes_per_detection per box
            let slot = o.max_classes_per_detection * out_i + col;
            if rows.len() <= slot {
                rows.resize(slot + 1, ([0.0; 4], 0.0, 0.0));
            }
            rows[slot] = (boxes[sel], c as f32, scores[sel * with_bg + off + c]);
        }
    }
    Ok((rows, selected.len()))
}

/// NonMaxSuppressionMultiClassRegularHelper: NMS per class, merged by score
fn regular(o: &Options, boxes: &[Box4], scores: &[f32], with_bg: usize, off: usize) -> Result<(Vec<Row>, usize)> {
    if o.detections_per_class == 0 {
        return err("detections_per_class must be > 0");
    }
    let n = boxes.len();
    let mut idx = vec![0usize; n + o.max_detections];
    let mut sc = vec![0f32; n + o.max_detections];
    let mut size = 0;
    for col in 0..o.num_classes {
        let class_scores: Vec<f32> = (0..n).map(|r| scores[r * with_bg + col + off]).collect();
        let selected = nms_single(o, boxes, &class_scores, o.detections_per_class)?;
        let mut out_i = size;
        for &s in &selected {
            idx[out_i] = s * with_bg + col + off;
            sc[out_i] = class_scores[s];
            out_i += 1;
        }
        let k = out_i.min(o.max_detections);
        let order = decreasing_partial_arg_sort(&sc[..out_i], k);
        let (new_idx, new_sc): (Vec<usize>, Vec<f32>) = order.iter().map(|&t| (idx[t], sc[t])).unzip();
        idx[..k].copy_from_slice(&new_idx);
        sc[..k].copy_from_slice(&new_sc);
        size = k;
    }
    let rows = (0..size)
        .map(|i| {
            let anchor = idx[i] / with_bg;
            (boxes[anchor], (idx[i] - anchor * with_bg - off) as f32, sc[i])
        })
        .collect();
    Ok((rows, size))
}
