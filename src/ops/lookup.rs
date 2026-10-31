// From TensorFlow Lite 2.5 (Copyright The TensorFlow Authors, Apache License 2.0), translated and changed; see NOTICE.
// farmhash_fingerprint64 is from farmhash (Copyright (c) 2014 Google, Inc., MIT License, text in NOTICE).
//! Lookups and hashing: EMBEDDING_LOOKUP, EMBEDDING_LOOKUP_SPARSE, HASHTABLE_LOOKUP, LSH_PROJECTION, SKIP_GRAM.

use super::util::*;
use super::{Io, Kernel};
use crate::interp::Env;
use crate::tensor::{Alloc, TType};
use crate::tflite::OpSpec;
use crate::Result;

pub fn create(s: &OpSpec) -> Result<Option<Box<dyn Kernel>>> {
    Ok(Some(match s.code {
        7 => Box::new(EmbeddingLookup),
        33 => Box::new(EmbeddingLookupSparse { combiner: opt(s, 0, 0i8) }),
        10 => Box::new(HashtableLookup),
        15 => Box::new(LshProjection { dense: match opt(s, 0, 0i8) {
            1 => false,
            2 => true,
            t => return err(format!("LSH projection type {t} is not supported")),
        } }),
        30 => Box::new(SkipGram { ngram: opt(s, 0, 0), max_skip: opt(s, 1, 0), all: opt(s, 2, 0u8) != 0 }),
        _ => return Ok(None),
    }))
}

/// embedding_lookup.cc: rows of the value picked by int32 ids; 8-bit values into a float output are dequantized
/// with the value's scale only (the bytes read as int8)
struct EmbeddingLookup;

impl Kernel for EmbeddingLookup {
    fn prepare(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        let (ids, v) = (io.input(0)?, io.input(1)?);
        if ids.shape.len() != 1 || ids.dtype != TType::I32 || v.shape.len() < 2 {
            return err("EMBEDDING_LOOKUP takes 1-D int32 ids and a value of rank 2 or more");
        }
        let mut s = v.shape.clone();
        s[0] = ids.shape[0];
        io.outs[0].resize(s);
        Ok(())
    }

    fn eval(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        let (ids, v) = (ints(io.input(0)?)?, io.input(1)?);
        let o = &mut io.outs[0];
        let rows = v.shape[0];
        if rows == 0 {
            return Ok(());
        }
        let hybrid = matches!(v.dtype, TType::U8 | TType::I8) && o.dtype == TType::F32;
        if !matches!(v.dtype, TType::F32 | TType::U8 | TType::I8) {
            return err(format!("{:?} is not supported", v.dtype));
        }
        let row = v.data().len() / rows;
        for (k, &id) in ids.iter().enumerate() {
            if id < 0 || id as usize >= rows {
                return err(format!("Embedding Lookup: index out of bounds. Got {id}, and bounds are [0, {}]", rows - 1));
            }
            let src = &v.data()[id as usize * row..(id as usize + 1) * row];
            if hybrid {
                let scale = v.scale() as f64;
                for (j, &b) in src.iter().enumerate() {
                    o.as_mut_slice::<f32>()[k * row + j] = (b as i8 as f64 * scale) as f32;
                }
            } else {
                o.data_mut()[k * row..(k + 1) * row].copy_from_slice(src);
            }
        }
        Ok(())
    }
}

/// embedding_lookup_sparse.cc (ids, indices, dense shape, weights, value): weighted sums of value rows into the
/// buckets of a sparse tensor, combined as sum, mean (over the weights) or sqrtn
struct EmbeddingLookupSparse {
    combiner: i8,
}

impl Kernel for EmbeddingLookupSparse {
    fn prepare(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        let (ids, idx, shape, w, v) = (io.input(0)?, io.input(1)?, io.input(2)?, io.input(3)?, io.input(4)?);
        let ok = ids.shape.len() == 1
            && ids.dtype == TType::I32
            && idx.shape.len() == 2
            && idx.dtype == TType::I32
            && shape.shape.len() == 1
            && shape.dtype == TType::I32
            && w.shape.len() == 1
            && w.dtype == TType::F32
            && idx.shape[0] == ids.shape[0]
            && idx.shape[0] == w.shape[0]
            && v.shape.len() >= 2
            && io.outs[0].dtype == TType::F32;
        if !ok {
            return err("EMBEDDING_LOOKUP_SPARSE takes int32 ids, indices and shape, float weights and a value of rank 2 or more");
        }
        io.outs[0].alloc = Alloc::Dynamic;
        Ok(())
    }

    fn eval(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        let (ids, idx_t, shape_t, w, v) = (io.input(0)?, io.input(1)?, io.input(2)?, io.input(3)?, io.input(4)?);
        let (ids, idx, dense) = (ints(ids)?, ints(idx_t)?, ints(shape_t)?);
        let (w, vals) = (w.as_slice::<f32>(), v.as_slice::<f32>());
        let rank = idx_t.shape[1];
        if dense.len() != rank {
            return err("the dense shape must have one size per index column");
        }
        let mut out_shape: Vec<usize> = dense[..rank.saturating_sub(1)].iter().map(|&d| d as usize).collect();
        out_shape.extend_from_slice(&v.shape[1..]);
        let emb: usize = v.shape[1..].iter().product();
        let o = &mut io.outs[0];
        o.resize(out_shape);
        let out = o.as_mut_slice::<f32>();
        out.fill(0.0);
        let finalize = |out: &mut [f32], n: usize, total: f32, squares: f32| {
            if self.combiner != 0 && n > 0 {
                let m = match self.combiner {
                    1 => total,
                    2 => squares.sqrt(),
                    _ => 1.0,
                };
                out.iter_mut().for_each(|x| *x /= m);
            }
        };
        let (mut at, mut n, mut total, mut squares) = (0usize, 0usize, 0.0f32, 0.0f32);
        for (i, &id) in ids.iter().enumerate() {
            if id < 0 || id as usize >= v.shape[0] {
                return err(format!("Embedding Lookup Sparse: index out of bounds. Got {id}, and bounds are [0, {}]", v.shape[0] as i64 - 1));
            }
            let (mut bucket, mut stride) = (0i64, 1i64);
            for k in (0..rank.saturating_sub(1)).rev() {
                bucket += idx[i * rank + k] * stride;
                stride *= dense[k];
            }
            let offset = bucket as usize * emb;
            if offset != at {
                finalize(&mut out[at..at + emb], n, total, squares);
                (n, total, squares, at) = (0, 0.0, 0.0, offset);
            }
            n += 1;
            squares += w[i] * w[i];
            total += w[i];
            for k in 0..emb {
                out[at + k] += vals[id as usize * emb + k] * w[i];
            }
        }
        if at + emb <= out.len() {
            finalize(&mut out[at..at + emb], n, total, squares);
        }
        Ok(())
    }
}

/// hashtable_lookup.cc (lookups, sorted int32 keys, values): the value row of each key found by bsearch, zeros (or an
/// empty string) when missing, and a uint8 hit flag
struct HashtableLookup;

impl Kernel for HashtableLookup {
    fn prepare(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        let (l, k, v) = (io.input(0)?, io.input(1)?, io.input(2)?);
        if l.shape.len() != 1 || l.dtype != TType::I32 || k.shape.len() != 1 || k.dtype != TType::I32 || v.shape.is_empty() || k.shape[0] != v.shape[0] {
            return err("HASHTABLE_LOOKUP takes 1-D int32 lookups and keys, and one value row per key");
        }
        if (v.dtype == TType::Str && v.shape.len() != 1) || io.outs[1].dtype != TType::U8 || io.outs[0].dtype != v.dtype {
            return err("string values must be 1-D, the hits uint8, the output of the values' type");
        }
        if v.dtype != TType::Str {
            let mut s = v.shape.clone();
            s[0] = l.shape[0];
            io.outs[0].resize(s);
        }
        io.outs[1].resize(vec![l.shape[0]]);
        Ok(())
    }

    fn eval(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        let (l, k, v) = (ints(io.input(0)?)?, ints(io.input(1)?)?, io.input(2)?);
        let rows = v.shape[0];
        if rows == 0 {
            return err("the table is empty");
        }
        let found: Vec<Option<usize>> = l.iter().map(|&x| bsearch(x as i32, &k)).collect();
        if v.dtype == TType::Str {
            let vals = v.strings()?;
            let out: Vec<&[u8]> = found.iter().map(|f| f.map_or(&[][..], |i| vals[i])).collect();
            io.outs[0].set_strings(&out, vec![out.len()]);
        } else {
            let row = v.data().len() / rows;
            let o = io.outs[0].data_mut();
            for (n, f) in found.iter().enumerate() {
                let d = &mut o[n * row..(n + 1) * row];
                match f {
                    Some(i) => d.copy_from_slice(&v.data()[i * row..(i + 1) * row]),
                    None => d.fill(0),
                }
            }
        }
        let hits: Vec<i64> = found.iter().map(|f| f.is_some() as i64).collect();
        put_ints(&mut io.outs[1], &hits)
    }
}

/// bsearch as the Windows CRT runs it (ucrt bsearch.cpp), with 2.5's comparison `a - b` (wrapping): which of equal
/// keys is found, and what unsorted keys give, follow from its probing order
fn bsearch(key: i32, keys: &[i64]) -> Option<usize> {
    let cmp = |i: usize| key.wrapping_sub(keys[i] as i32);
    let (mut lo, mut hi, mut num) = (0i64, keys.len() as i64 - 1, keys.len());
    while lo <= hi {
        let half = num / 2;
        if half != 0 {
            let mid = lo + if num & 1 == 1 { half } else { half - 1 } as i64;
            let r = cmp(mid as usize);
            if r == 0 {
                return Some(mid as usize);
            } else if r < 0 {
                hi = mid - 1;
                num = if num & 1 == 1 { half } else { half - 1 };
            } else {
                lo = mid + 1;
                num = half;
            }
        } else if num != 0 {
            return if cmp(lo as usize) == 0 { Some(lo as usize) } else { None };
        } else {
            break;
        }
    }
    None
}

/// lsh_projection.cc (hash seeds [hashes, bits], input, optional weights): each bit the sign of the (weighted) sum of
/// farmhash fingerprints of seed + input row; sparse packs a hash's bits into one int32 per hash
struct LshProjection {
    dense: bool,
}

impl Kernel for LshProjection {
    fn prepare(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        let (h, i) = (io.input(0)?, io.input(1)?);
        if h.shape.len() != 2 || h.shape[1] > 32 || i.shape.is_empty() {
            return err("LSH_PROJECTION takes [hashes, bits <= 32] seeds and an input of rank 1 or more");
        }
        if let Some(w) = io.opt(2) {
            if w.shape.len() != 1 || w.shape[0] != i.shape[0] {
                return err("the weights need one value per input row");
            }
        }
        let n = if self.dense { h.shape[0] * h.shape[1] } else { h.shape[0] };
        io.outs[0].resize(vec![n]);
        Ok(())
    }

    fn eval(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        let (h, i, w) = (io.input(0)?, io.input(1)?, io.opt(2));
        let (hashes, bits) = (h.shape[0], h.shape[1]);
        let seeds = h.as_slice::<f32>();
        let rows = i.shape[0];
        let row = if rows == 0 { 0 } else { i.data().len() / rows };
        let weights = w.map(|t| t.as_slice::<f32>());
        let bit = |seed: f32| -> i64 {
            let mut score = 0.0f64;
            for r in 0..rows {
                let mut key = seed.to_le_bytes().to_vec();
                key.extend_from_slice(&i.data()[r * row..(r + 1) * row]);
                let value = farmhash_fingerprint64(&key) as i64 as f64;
                score += weights.map_or(value, |w| w[r] as f64 * value);
            }
            (score > 0.0) as i64
        };
        let mut out = Vec::new();
        for k in 0..hashes {
            let mut signature = 0i32;
            for j in 0..bits {
                let b = bit(seeds[k * bits + j]);
                if self.dense {
                    out.push(b);
                } else {
                    signature = (signature << 1) | b as i32;
                }
            }
            if !self.dense {
                out.push(signature.wrapping_add((k as i32).wrapping_mul(1i32.wrapping_shl(bits as u32))) as i64);
            }
        }
        put_ints(&mut io.outs[0], &out)
    }
}

/// farmhash's Fingerprint64 (farmhashna::Hash64, google/farmhash 816a4ae, the one TensorFlow 2.5 pins)
fn farmhash_fingerprint64(s: &[u8]) -> u64 {
    const K0: u64 = 0xc3a5c85c97cb3127;
    const K1: u64 = 0xb492b66fbe98f273;
    const K2: u64 = 0x9ae16a3b2f90404f;
    let f64_ = |p: usize| u64::from_le_bytes(s[p..p + 8].try_into().unwrap());
    let f32_ = |p: usize| u32::from_le_bytes(s[p..p + 4].try_into().unwrap()) as u64;
    let rot = |v: u64, k: u32| v.rotate_right(k);
    let shift_mix = |v: u64| v ^ (v >> 47);
    let hash16 = |u: u64, v: u64, mul: u64| {
        let mut a = (u ^ v).wrapping_mul(mul);
        a ^= a >> 47;
        let mut b = (v ^ a).wrapping_mul(mul);
        b ^= b >> 47;
        b.wrapping_mul(mul)
    };
    let weak = |w: u64, x: u64, y: u64, z: u64, mut a: u64, mut b: u64| {
        a = a.wrapping_add(w);
        b = rot(b.wrapping_add(a).wrapping_add(z), 21);
        let c = a;
        a = a.wrapping_add(x).wrapping_add(y);
        b = b.wrapping_add(rot(a, 44));
        (a.wrapping_add(z), b.wrapping_add(c))
    };
    let weak_at = |p: usize, a: u64, b: u64| weak(f64_(p), f64_(p + 8), f64_(p + 16), f64_(p + 24), a, b);
    let len = s.len();
    let n = len as u64;
    if len <= 16 {
        if len >= 8 {
            let mul = K2.wrapping_add(n * 2);
            let a = f64_(0).wrapping_add(K2);
            let b = f64_(len - 8);
            let c = rot(b, 37).wrapping_mul(mul).wrapping_add(a);
            let d = rot(a, 25).wrapping_add(b).wrapping_mul(mul);
            return hash16(c, d, mul);
        }
        if len >= 4 {
            let mul = K2.wrapping_add(n * 2);
            return hash16(n.wrapping_add(f32_(0) << 3), f32_(len - 4), mul);
        }
        if len > 0 {
            let (a, b, c) = (s[0] as u32, s[len >> 1] as u32, s[len - 1] as u32);
            let y = a.wrapping_add(b << 8);
            let z = (len as u32).wrapping_add(c << 2);
            return shift_mix((y as u64).wrapping_mul(K2) ^ (z as u64).wrapping_mul(K0)).wrapping_mul(K2);
        }
        return K2;
    }
    if len <= 32 {
        let mul = K2.wrapping_add(n * 2);
        let a = f64_(0).wrapping_mul(K1);
        let b = f64_(8);
        let c = f64_(len - 8).wrapping_mul(mul);
        let d = f64_(len - 16).wrapping_mul(K2);
        return hash16(
            rot(a.wrapping_add(b), 43).wrapping_add(rot(c, 30)).wrapping_add(d),
            a.wrapping_add(rot(b.wrapping_add(K2), 18)).wrapping_add(c),
            mul,
        );
    }
    if len <= 64 {
        let mul = K2.wrapping_add(n * 2);
        let a = f64_(0).wrapping_mul(K2);
        let b = f64_(8);
        let c = f64_(len - 8).wrapping_mul(mul);
        let d = f64_(len - 16).wrapping_mul(K2);
        let y = rot(a.wrapping_add(b), 43).wrapping_add(rot(c, 30)).wrapping_add(d);
        let z = hash16(y, a.wrapping_add(rot(b.wrapping_add(K2), 18)).wrapping_add(c), mul);
        let e = f64_(16).wrapping_mul(mul);
        let f = f64_(24);
        let g = y.wrapping_add(f64_(len - 32)).wrapping_mul(mul);
        let h = z.wrapping_add(f64_(len - 24)).wrapping_mul(mul);
        return hash16(rot(e.wrapping_add(f), 43).wrapping_add(rot(g, 30)).wrapping_add(h), e.wrapping_add(rot(f.wrapping_add(a), 18)).wrapping_add(g), mul);
    }
    let seed = 81u64;
    let mut x = seed;
    let mut y = seed.wrapping_mul(K1).wrapping_add(113);
    let mut z = shift_mix(y.wrapping_mul(K2).wrapping_add(113)).wrapping_mul(K2);
    let (mut v, mut w) = ((0u64, 0u64), (0u64, 0u64));
    x = x.wrapping_mul(K2).wrapping_add(f64_(0));
    let end = ((len - 1) / 64) * 64;
    let last64 = len - 64;
    let mut p = 0;
    loop {
        x = rot(x.wrapping_add(y).wrapping_add(v.0).wrapping_add(f64_(p + 8)), 37).wrapping_mul(K1);
        y = rot(y.wrapping_add(v.1).wrapping_add(f64_(p + 48)), 42).wrapping_mul(K1);
        x ^= w.1;
        y = y.wrapping_add(v.0).wrapping_add(f64_(p + 40));
        z = rot(z.wrapping_add(w.0), 33).wrapping_mul(K1);
        v = weak_at(p, v.1.wrapping_mul(K1), x.wrapping_add(w.0));
        w = weak_at(p + 32, z.wrapping_add(w.1), y.wrapping_add(f64_(p + 16)));
        std::mem::swap(&mut z, &mut x);
        p += 64;
        if p == end {
            break;
        }
    }
    let mul = K1.wrapping_add((z & 0xff) << 1);
    p = last64;
    w.0 = w.0.wrapping_add(((len - 1) & 63) as u64);
    v.0 = v.0.wrapping_add(w.0);
    w.0 = w.0.wrapping_add(v.0);
    x = rot(x.wrapping_add(y).wrapping_add(v.0).wrapping_add(f64_(p + 8)), 37).wrapping_mul(mul);
    y = rot(y.wrapping_add(v.1).wrapping_add(f64_(p + 48)), 42).wrapping_mul(mul);
    x ^= w.1.wrapping_mul(9);
    y = y.wrapping_add(v.0.wrapping_mul(9)).wrapping_add(f64_(p + 40));
    z = rot(z.wrapping_add(w.0), 33).wrapping_mul(mul);
    v = weak_at(p, v.1.wrapping_mul(mul), x.wrapping_add(w.0));
    w = weak_at(p + 32, z.wrapping_add(w.1), y.wrapping_add(f64_(p + 16)));
    std::mem::swap(&mut z, &mut x);
    hash16(
        hash16(v.0, w.0, mul).wrapping_add(shift_mix(y).wrapping_mul(K0)).wrapping_add(z),
        hash16(v.1, w.1, mul).wrapping_add(x),
        mul,
    )
}

/// skip_gram.cc: the words of the first string (split at C whitespace from the second character on), then every
/// n-gram (or every gram up to n) with at most max_skip words skipped between neighbors, joined by spaces
struct SkipGram {
    ngram: i32,
    max_skip: i32,
    all: bool,
}

impl Kernel for SkipGram {
    fn prepare(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        if io.input(0)?.dtype != TType::Str || io.outs[0].dtype != TType::Str {
            return err("SKIP_GRAM takes and gives strings");
        }
        Ok(())
    }

    fn eval(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        let strs = io.input(0)?.strings()?;
        let s: &[u8] = strs.first().copied().unwrap_or(&[]);
        let space = |c: u8| matches!(c, b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r');
        let mut words: Vec<&[u8]> = Vec::new();
        let mut prev = 0usize;
        for i in 1..s.len() {
            if space(s[i]) {
                if i > prev && !space(s[prev]) {
                    words.push(&s[prev..i]);
                }
                prev = i + 1;
            }
        }
        if s.len() > prev {
            words.push(&s[prev..]);
        }
        let n = self.ngram;
        let mut out: Vec<Vec<u8>> = Vec::new();
        if words.len() as i64 >= n as i64 && n > 0 {
            let mut stack = vec![0i32; n as usize];
            let mut idx = 1i32;
            let num = words.len() as i32;
            while idx >= 0 {
                let step = idx < n && stack[idx as usize] + 1 < num && (idx == 0 || stack[idx as usize] - stack[idx as usize - 1] <= self.max_skip);
                if step {
                    stack[idx as usize] += 1;
                    idx += 1;
                    if idx < n {
                        stack[idx as usize] = stack[idx as usize - 1];
                    }
                } else {
                    let include = idx > 0 && if self.all { idx <= n } else { idx == n };
                    if include {
                        out.push(stack[..idx as usize].iter().map(|&k| words[k as usize]).collect::<Vec<_>>().join(&b' '));
                    }
                    idx -= 1;
                }
            }
        }
        io.outs[0].set_strings(&out, vec![out.len()]);
        Ok(())
    }
}
