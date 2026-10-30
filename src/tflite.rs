//! TensorFlow Lite model files (schema.fbs v3 as of TensorFlow Lite 2.5): every subgraph with its tensors and
//! operators. Each operator gets its kernel when the model is read (ops::create).

use crate::fb::Table;
use crate::ops::{self, Kernel};
use crate::tensor::{Alloc, Buf, SparseDim, Sparsity, TType, Tensor};
use crate::{Error, Result};

/// fused activation (ActivationFunctionType)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Act {
    None,
    Relu,
    ReluN1To1,
    Relu6,
    Tanh,
    SignBit,
}

impl Act {
    pub fn from_code(code: i8) -> Result<Act> {
        Ok(match code {
            0 => Act::None,
            1 => Act::Relu,
            2 => Act::ReluN1To1,
            3 => Act::Relu6,
            4 => Act::Tanh,
            5 => Act::SignBit,
            c => return Err(Error(format!("unknown fused activation {c}"))),
        })
    }
}

pub struct OpDef {
    /// "ADD", or the custom op's name
    pub name: String,
    pub code: i32,
    pub version: i32,
    pub inputs: Vec<i32>,
    pub outputs: Vec<i32>,
    pub intermediates: Vec<i32>,
    pub kernel: Box<dyn Kernel>,
}

pub struct Subgraph {
    pub name: String,
    pub tensors: Vec<Tensor>,
    pub ops: Vec<OpDef>,
    pub inputs: Vec<usize>,
    pub outputs: Vec<usize>,
}

/// a SignatureDef: tensor name -> tensor index, inputs and outputs sorted by name
pub struct Signature {
    pub key: String,
    pub inputs: Vec<(String, usize)>,
    pub outputs: Vec<(String, usize)>,
}

pub struct Model {
    pub subgraphs: Vec<Subgraph>,
    pub signatures: Vec<Signature>,
}

/// BuiltinOperator names, by code
pub const BUILTIN: [&str; 140] = [
    "ADD", "AVERAGE_POOL_2D", "CONCATENATION", "CONV_2D", "DEPTHWISE_CONV_2D", "DEPTH_TO_SPACE", "DEQUANTIZE",
    "EMBEDDING_LOOKUP", "FLOOR", "FULLY_CONNECTED", "HASHTABLE_LOOKUP", "L2_NORMALIZATION", "L2_POOL_2D",
    "LOCAL_RESPONSE_NORMALIZATION", "LOGISTIC", "LSH_PROJECTION", "LSTM", "MAX_POOL_2D", "MUL", "RELU",
    "RELU_N1_TO_1", "RELU6", "RESHAPE", "RESIZE_BILINEAR", "RNN", "SOFTMAX", "SPACE_TO_DEPTH", "SVDF", "TANH",
    "CONCAT_EMBEDDINGS", "SKIP_GRAM", "CALL", "CUSTOM", "EMBEDDING_LOOKUP_SPARSE", "PAD",
    "UNIDIRECTIONAL_SEQUENCE_RNN", "GATHER", "BATCH_TO_SPACE_ND", "SPACE_TO_BATCH_ND", "TRANSPOSE", "MEAN",
    "SUB", "DIV", "SQUEEZE", "UNIDIRECTIONAL_SEQUENCE_LSTM", "STRIDED_SLICE", "BIDIRECTIONAL_SEQUENCE_RNN",
    "EXP", "TOPK_V2", "SPLIT", "LOG_SOFTMAX", "DELEGATE", "BIDIRECTIONAL_SEQUENCE_LSTM", "CAST", "PRELU",
    "MAXIMUM", "ARG_MAX", "MINIMUM", "LESS", "NEG", "PADV2", "GREATER", "GREATER_EQUAL", "LESS_EQUAL", "SELECT",
    "SLICE", "SIN", "TRANSPOSE_CONV", "SPARSE_TO_DENSE", "TILE", "EXPAND_DIMS", "EQUAL", "NOT_EQUAL", "LOG",
    "SUM", "SQRT", "RSQRT", "SHAPE", "POW", "ARG_MIN", "FAKE_QUANT", "REDUCE_PROD", "REDUCE_MAX", "PACK",
    "LOGICAL_OR", "ONE_HOT", "LOGICAL_AND", "LOGICAL_NOT", "UNPACK", "REDUCE_MIN", "FLOOR_DIV", "REDUCE_ANY",
    "SQUARE", "ZEROS_LIKE", "FILL", "FLOOR_MOD", "RANGE", "RESIZE_NEAREST_NEIGHBOR", "LEAKY_RELU",
    "SQUARED_DIFFERENCE", "MIRROR_PAD", "ABS", "SPLIT_V", "UNIQUE", "CEIL", "REVERSE_V2", "ADD_N", "GATHER_ND",
    "COS", "WHERE", "RANK", "ELU", "REVERSE_SEQUENCE", "MATRIX_DIAG", "QUANTIZE", "MATRIX_SET_DIAG", "ROUND",
    "HARD_SWISH", "IF", "WHILE", "NON_MAX_SUPPRESSION_V4", "NON_MAX_SUPPRESSION_V5", "SCATTER_ND", "SELECT_V2",
    "DENSIFY", "SEGMENT_SUM", "BATCH_MATMUL", "PLACEHOLDER_FOR_GREATER_OP_CODES", "CUMSUM", "CALL_ONCE",
    "BROADCAST_TO", "RFFT2D", "CONV_3D", "IMAG", "REAL", "COMPLEX_ABS", "HASHTABLE", "HASHTABLE_FIND",
    "HASHTABLE_IMPORT", "HASHTABLE_SIZE",
];

/// what ops::create gets to build a kernel
pub struct OpSpec<'a> {
    pub name: &'a str,
    pub code: i32,
    pub version: i32,
    /// the builtin options table (its type follows from the op)
    pub opts: Option<Table<'a>>,
    pub custom_opts: &'a [u8],
    pub n_subgraphs: usize,
}

fn index_vector(t: Option<(u8, Table)>) -> Vec<i32> {
    // SparseIndexVector: Int32Vector = 1, Uint16Vector = 2, Uint8Vector = 3, each { values }
    match t {
        Some((1, v)) => v.scalars::<i32>(0),
        Some((2, v)) => v.scalars::<u16>(0).into_iter().map(|x| x as i32).collect(),
        Some((3, v)) => v.bytes(0).iter().map(|&x| x as i32).collect(),
        _ => Vec::new(),
    }
}

fn tensor(t: Table, buffers: &[Table]) -> Result<Tensor> {
    let q = t.table(4);
    let data = buffers.get(t.get::<u32>(2, 0) as usize).map(|b| b.bytes(0)).unwrap_or(&[]);
    let sparsity = t.table(6).map(|s| Sparsity {
        traversal_order: s.scalars(0),
        block_map: s.scalars(1),
        dims: s
            .tables(2)
            .into_iter()
            .map(|d| match d.get::<i8>(0, 0) {
                0 => SparseDim::Dense(d.get::<i32>(1, 0) as usize),
                _ => SparseDim::Sparse(
                    index_vector(d.table(3).map(|v| (d.get::<u8>(2, 0), v))),
                    index_vector(d.table(5).map(|v| (d.get::<u8>(4, 0), v))),
                ),
            })
            .collect(),
    });
    let is_variable = t.get::<u8>(5, 0) != 0;
    let mut x = Tensor {
        name: t.str(3).unwrap_or("").to_string(),
        dtype: TType::from_code(t.get::<i8>(1, 0)),
        shape: t.scalars::<i32>(0).into_iter().map(|d| d.max(0) as usize).collect(),
        shape_signature: t.scalars(7),
        scales: q.map(|q| q.scalars(2)).unwrap_or_default(),
        zero_points: q.map(|q| q.scalars(3)).unwrap_or_default(),
        quantized_dimension: q.map_or(0, |q| q.get(6, 0)),
        alloc: Alloc::Arena,
        sparsity,
        initial: None,
        buf: Buf::default(),
    };
    if is_variable {
        x.alloc = Alloc::Variable;
        x.initial = (!data.is_empty()).then(|| data.to_vec());
        x.resize(x.shape.clone());
    } else if !data.is_empty() {
        x.alloc = Alloc::Constant;
        x.buf = Buf::new(data.len());
        x.buf.bytes_mut().copy_from_slice(data);
    } else {
        x.resize(x.shape.clone());
    }
    Ok(x)
}

pub fn parse(buf: &[u8]) -> Result<Model> {
    if buf.len() < 8 || &buf[4..8] != b"TFL3" {
        return Err(Error("not a TensorFlow Lite model (no TFL3 identifier)".into()));
    }
    let m = Table::root(buf);
    // (builtin code, custom name, version): deprecated_builtin_code is an int8, builtin_code (from 2.4) the full code
    let codes: Vec<(i32, Option<&str>, i32)> =
        m.tables(1).into_iter().map(|c| ((c.get::<i8>(0, 0) as i32).max(c.get::<i32>(3, 0)), c.str(1), c.get::<i32>(2, 1))).collect();
    let buffers = m.tables(4);
    let sgs = m.tables(2);
    if sgs.is_empty() {
        return Err(Error("the model has no subgraph".into()));
    }
    let mut subgraphs = Vec::new();
    for sg in &sgs {
        let tensors: Vec<Tensor> = sg.tables(0).into_iter().map(|t| tensor(t, &buffers)).collect::<Result<_>>()?;
        let index = |v: Vec<i32>, optional: bool| -> Result<Vec<i32>> {
            for &i in &v {
                if !(optional && i == -1) && !(i >= 0 && (i as usize) < tensors.len()) {
                    return Err(Error(format!("tensor index {i} out of range")));
                }
            }
            Ok(v)
        };
        let mut ops = Vec::new();
        for op in sg.tables(3) {
            let &(code, custom, version) = codes.get(op.get::<u32>(0, 0) as usize).ok_or_else(|| Error("operator code out of range".into()))?;
            let name = match custom {
                Some(c) if code == 32 => c.to_string(),
                _ => BUILTIN.get(code as usize).map_or(format!("builtin op {code}"), |s| s.to_string()),
            };
            let spec = OpSpec { name: &name, code, version, opts: op.table(4), custom_opts: op.bytes(5), n_subgraphs: sgs.len() };
            let kernel = ops::create(&spec).map_err(|e| Error(format!("{name}: {e}")))?;
            ops.push(OpDef {
                inputs: index(op.scalars(1), true)?,
                outputs: index(op.scalars(2), false)?,
                intermediates: index(op.scalars(8), false)?,
                name,
                code,
                version,
                kernel,
            });
        }
        let inputs = index(sg.scalars(1), false)?.into_iter().map(|i| i as usize).collect();
        let outputs = index(sg.scalars(2), false)?.into_iter().map(|i| i as usize).collect();
        subgraphs.push(Subgraph { name: sg.str(4).unwrap_or("").to_string(), tensors, ops, inputs, outputs });
    }
    // SignatureDefs as TFLite 2.5 reads them: names in a std::map (sorted, a repeated name keeps its last index).
    // Field 2 is method_name in 2.5 and signature_key from 2.6; signatures of other subgraphs (field 4, from 2.7) are
    // left out, as 2.5 runs only subgraph 0 from outside.
    let n0 = subgraphs[0].tensors.len();
    let mut signatures = Vec::new();
    for s in m.tables(7).into_iter().filter(|s| s.get::<u32>(4, 0) == 0) {
        let names = |i: usize| -> Result<Vec<(String, usize)>> {
            let map: std::collections::BTreeMap<String, u32> = s.tables(i).iter().map(|t| (t.str(0).unwrap_or("").into(), t.get(1, 0))).collect();
            map.into_iter()
                .map(|(n, t)| if (t as usize) < n0 { Ok((n, t as usize)) } else { Err(Error(format!("signature tensor index {t} out of range"))) })
                .collect()
        };
        let key = s.str(2).ok_or_else(|| Error("a SignatureDef has no key".into()))?;
        signatures.push(Signature { key: key.into(), inputs: names(0)?, outputs: names(1)? });
    }
    Ok(Model { subgraphs, signatures })
}
