//! Tensors at run time: type, shape, quantization and memory.

use crate::{Error, Result};

/// TensorType of schema.fbs (TensorFlow Lite 2.5)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TType {
    F32,
    F16,
    I32,
    U8,
    I64,
    Str,
    Bool,
    I16,
    C64,
    I8,
    F64,
    C128,
    U64,
    Resource,
    Variant,
    U32,
    Other(i8),
}

impl TType {
    pub fn from_code(c: i8) -> Self {
        use TType::*;
        [F32, F16, I32, U8, I64, Str, Bool, I16, C64, I8, F64, C128, U64, Resource, Variant, U32].get(c as usize).copied().unwrap_or(Other(c))
    }

    /// the schema's TensorType code (also what the C API reports)
    pub fn code(self) -> i32 {
        use TType::*;
        match self {
            Other(c) => c as i32,
            t => [F32, F16, I32, U8, I64, Str, Bool, I16, C64, I8, F64, C128, U64, Resource, Variant, U32].iter().position(|&x| x == t).unwrap() as i32,
        }
    }

    /// bytes per element (0 for strings, whose buffer has its own layout)
    pub fn size(self) -> usize {
        use TType::*;
        match self {
            U8 | I8 | Bool => 1,
            F16 | I16 => 2,
            F32 | I32 | U32 => 4,
            I64 | F64 | C64 | U64 => 8,
            C128 => 16,
            // a resource tensor holds the int32 id of the resource
            Resource => 4,
            Str | Variant | Other(_) => 0,
        }
    }
}

/// tensor memory: 8 byte aligned (typed views of any element type are aligned), and it only moves when the tensor
/// is resized to another byte size
#[derive(Default, Clone)]
pub struct Buf {
    words: Vec<u64>,
    len: usize,
}

impl Buf {
    pub fn new(len: usize) -> Self {
        Buf { words: vec![0; len.div_ceil(8)], len }
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn bytes(&self) -> &[u8] {
        // SAFETY: the u64 storage holds at least len initialized bytes
        unsafe { std::slice::from_raw_parts(self.words.as_ptr() as *const u8, self.len) }
    }
    pub fn bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: as above, uniquely borrowed
        unsafe { std::slice::from_raw_parts_mut(self.words.as_mut_ptr() as *mut u8, self.len) }
    }
}

/// an element type a tensor's memory can be viewed as
pub trait Elem: Copy + Default + Send + 'static {
    const TYPES: &'static [TType];
}
macro_rules! elem {
    ($($t:ty => $($tt:ident)|+),*) => {$(impl Elem for $t { const TYPES: &'static [TType] = &[$(TType::$tt),+]; })*};
}
elem!(f32 => F32, f64 => F64, i8 => I8, u8 => U8 | Bool, i16 => I16, u16 => F16, i32 => I32 | Resource, u32 => U32, i64 => I64, u64 => U64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Alloc {
    /// sized when the model is prepared
    #[default]
    Arena,
    /// data from the model file
    Constant,
    /// sized by its op while it runs (the output's shape depends on input values)
    Dynamic,
    /// state kept across invokes (reset by `reset_variables`)
    Variable,
}

#[derive(Clone, Debug, Default)]
pub struct Sparsity {
    pub traversal_order: Vec<i32>,
    pub block_map: Vec<i32>,
    /// per dimension: Dense(size) or Sparse(segments, indices)
    pub dims: Vec<SparseDim>,
}

#[derive(Clone, Debug)]
pub enum SparseDim {
    Dense(usize),
    Sparse(Vec<i32>, Vec<i32>),
}

#[derive(Clone)]
pub struct Tensor {
    pub name: String,
    pub dtype: TType,
    pub shape: Vec<usize>,
    /// the shape with -1 where the model allows any size (else empty)
    pub shape_signature: Vec<i32>,
    pub scales: Vec<f32>,
    pub zero_points: Vec<i64>,
    pub quantized_dimension: i32,
    pub alloc: Alloc,
    pub sparsity: Option<Sparsity>,
    /// what a variable tensor is reset to (its buffer in the model, else zeros)
    pub initial: Option<Vec<u8>>,
    pub buf: Buf,
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{} {:?}{:?}", self.name, self.dtype, self.shape)
    }
}

impl Default for Tensor {
    fn default() -> Self {
        Tensor {
            name: String::new(),
            dtype: TType::F32,
            shape: Vec::new(),
            shape_signature: Vec::new(),
            scales: Vec::new(),
            zero_points: Vec::new(),
            quantized_dimension: 0,
            alloc: Alloc::Arena,
            sparsity: None,
            initial: None,
            buf: Buf::default(),
        }
    }
}

impl Tensor {
    /// everything but the data
    pub fn clone_meta(&self) -> Tensor {
        Tensor {
            name: self.name.clone(),
            dtype: self.dtype,
            shape: self.shape.clone(),
            shape_signature: self.shape_signature.clone(),
            scales: self.scales.clone(),
            zero_points: self.zero_points.clone(),
            quantized_dimension: self.quantized_dimension,
            alloc: self.alloc,
            sparsity: self.sparsity.clone(),
            initial: None,
            buf: Buf::default(),
        }
    }

    pub fn elements(&self) -> usize {
        self.shape.iter().product()
    }
    /// the bytes the shape needs (strings: what the buffer holds)
    pub fn bytes(&self) -> usize {
        if self.dtype == TType::Str { self.buf.len() } else { self.elements() * self.dtype.size() }
    }
    /// per-tensor quantization (TfLiteTensor::params): the first scale / zero point
    pub fn scale(&self) -> f32 {
        self.scales.first().copied().unwrap_or(0.0)
    }
    pub fn zero_point(&self) -> i32 {
        self.zero_points.first().copied().unwrap_or(0) as i32
    }
    pub fn quantized(&self) -> bool {
        !self.scales.is_empty()
    }
    pub fn data(&self) -> &[u8] {
        self.buf.bytes()
    }
    pub fn data_mut(&mut self) -> &mut [u8] {
        self.buf.bytes_mut()
    }

    /// Set the shape; the memory is reallocated (and zeroed) only when its byte size changes, so views of a tensor
    /// whose size stays put stay valid. Strings keep their buffer until written.
    pub fn resize(&mut self, shape: Vec<usize>) {
        self.shape = shape;
        if self.dtype != TType::Str && self.alloc != Alloc::Constant {
            let n = self.elements() * self.dtype.size();
            if n != self.buf.len() {
                self.buf = Buf::new(n);
            }
        }
    }

    /// the elements as T (the tensor's type must be one T stands for)
    pub fn as_slice<T: Elem>(&self) -> &[T] {
        assert!(T::TYPES.contains(&self.dtype), "{} is {:?}, read as {:?}", self.name, self.dtype, T::TYPES);
        let n = self.buf.len() / std::mem::size_of::<T>();
        // SAFETY: the buffer is 8 byte aligned, holds n * size_of::<T>() bytes, and every bit pattern is a valid T
        unsafe { std::slice::from_raw_parts(self.buf.bytes().as_ptr() as *const T, n) }
    }
    pub fn as_mut_slice<T: Elem>(&mut self) -> &mut [T] {
        assert!(T::TYPES.contains(&self.dtype), "{} is {:?}, written as {:?}", self.name, self.dtype, T::TYPES);
        let n = self.buf.len() / std::mem::size_of::<T>();
        // SAFETY: as above, uniquely borrowed
        unsafe { std::slice::from_raw_parts_mut(self.buf.bytes_mut().as_mut_ptr() as *mut T, n) }
    }

    /// The strings of a string tensor. TFLite's layout: int32 count, int32 offsets[count + 1] (from the buffer start),
    /// then the bytes.
    pub fn strings(&self) -> Result<Vec<&[u8]>> {
        let d = self.data();
        if d.len() < 4 {
            return Ok(vec![&[] as &[u8]; self.elements()]);
        }
        let rd = |i: usize| -> Result<usize> {
            d.get(4 * i..4 * i + 4).map(|b| i32::from_le_bytes(b.try_into().unwrap()) as usize).ok_or_else(|| Error(format!("{}: truncated string tensor", self.name)))
        };
        let n = rd(0)?;
        (0..n)
            .map(|k| {
                let (a, b) = (rd(1 + k)?, rd(2 + k)?);
                d.get(a..b).ok_or_else(|| Error(format!("{}: string {k} outside the buffer", self.name)))
            })
            .collect()
    }

    /// write the strings in TFLite's layout and set the shape
    pub fn set_strings(&mut self, v: &[impl AsRef<[u8]>], shape: Vec<usize>) {
        let head = 4 * (v.len() + 2);
        let total = head + v.iter().map(|s| s.as_ref().len()).sum::<usize>();
        let mut buf = Buf::new(total);
        let d = buf.bytes_mut();
        d[..4].copy_from_slice(&(v.len() as i32).to_le_bytes());
        let mut at = head;
        for (k, s) in v.iter().enumerate() {
            d[4 * (k + 1)..4 * (k + 2)].copy_from_slice(&(at as i32).to_le_bytes());
            d[at..at + s.as_ref().len()].copy_from_slice(s.as_ref());
            at += s.as_ref().len();
        }
        d[4 * (v.len() + 1)..4 * (v.len() + 2)].copy_from_slice(&(at as i32).to_le_bytes());
        self.buf = buf;
        self.shape = shape;
    }
}

/// Numpy-style broadcast shape of two shapes (TFLite's CalculateShapeForBroadcast)
pub fn broadcast_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
    let r = a.len().max(b.len());
    (0..r)
        .map(|i| {
            let x = if i + a.len() >= r { a[i + a.len() - r] } else { 1 };
            let y = if i + b.len() >= r { b[i + b.len() - r] } else { 1 };
            match (x, y) {
                _ if x == y => Ok(x),
                (1, _) => Ok(y),
                (_, 1) => Ok(x),
                _ => Err(Error(format!("shapes {a:?} and {b:?} do not broadcast"))),
            }
        })
        .collect()
}
