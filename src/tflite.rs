//! TensorFlow Lite model files (schema.fbs v3): the main subgraph's tensors and operators.
//! Tensor shapes are taken as stored: every op here has static shapes in Edge TPU models.

use crate::edgetpu::EdgeTpuOp;
use crate::fb::Table;
use crate::ops::dpp;
use crate::{flex, Error, Result};

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
    Other(i8),
}

impl TType {
    pub fn from_code(c: i8) -> Self {
        match c {
            0 => TType::F32,
            1 => TType::F16,
            2 => TType::I32,
            3 => TType::U8,
            4 => TType::I64,
            5 => TType::Str,
            6 => TType::Bool,
            7 => TType::I16,
            8 => TType::C64,
            9 => TType::I8,
            10 => TType::F64,
            c => TType::Other(c),
        }
    }

    /// the schema's TensorType code (also what the C API reports)
    pub fn code(self) -> i32 {
        match self {
            TType::F32 => 0,
            TType::F16 => 1,
            TType::I32 => 2,
            TType::U8 => 3,
            TType::I64 => 4,
            TType::Str => 5,
            TType::Bool => 6,
            TType::I16 => 7,
            TType::C64 => 8,
            TType::I8 => 9,
            TType::F64 => 10,
            TType::Other(c) => c as i32,
        }
    }

    pub fn size(self) -> usize {
        match self {
            TType::U8 | TType::I8 | TType::Bool => 1,
            TType::F16 | TType::I16 => 2,
            TType::F32 | TType::I32 => 4,
            TType::I64 | TType::F64 | TType::C64 => 8,
            TType::Str | TType::Other(_) => 1,
        }
    }
}

#[derive(Debug)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: TType,
    pub scales: Vec<f32>,
    pub zero_points: Vec<i64>,
    pub quantized_dimension: i32,
    /// contents of a constant tensor
    pub constant: Option<Vec<u8>>,
}

impl TensorInfo {
    pub fn elements(&self) -> usize {
        self.shape.iter().product()
    }
    pub fn bytes(&self) -> usize {
        self.elements() * self.dtype.size()
    }
    /// per-tensor quantization (TfLiteTensor::params): the first scale / zero point
    pub fn scale(&self) -> f32 {
        self.scales.first().copied().unwrap_or(0.0)
    }
    pub fn zero_point(&self) -> i32 {
        self.zero_points.first().copied().unwrap_or(0) as i32
    }
}

/// fused activation of ADD / SUB / MUL / DIV / CONCATENATION (ActivationFunctionType)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Act {
    None,
    Relu,
    ReluN1To1,
    Relu6,
}

pub enum Kind {
    Add,
    Sub,
    Mul,
    Div,
    Quantize,
    Dequantize,
    Cast,
    Reshape,
    ArgMax,
    FloorDiv,
    Sqrt,
    Pack { axis: i32 },
    Unpack { axis: i32 },
    Concat { axis: i32 },
    GatherNd,
    EdgeTpu(Box<EdgeTpuOp>),
    DetectionPostProcess(dpp::Options),
}

pub struct Node {
    pub name: &'static str,
    pub kind: Kind,
    pub act: Act,
    pub inputs: Vec<i32>,
    pub outputs: Vec<i32>,
}

/// a SignatureDef: tensor name -> tensor index, inputs and outputs sorted by name
pub struct Signature {
    pub key: String,
    pub inputs: Vec<(String, usize)>,
    pub outputs: Vec<(String, usize)>,
}

pub struct Model {
    pub tensors: Vec<TensorInfo>,
    pub nodes: Vec<Node>,
    pub inputs: Vec<usize>,
    pub outputs: Vec<usize>,
    pub signatures: Vec<Signature>,
}

/// BuiltinOperator names, by code
const BUILTIN: [&str; 140] = [
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
    "BROADCAST_TO", "RFFT2D", "CONV_3D", "?", "?", "?", "HASHTABLE", "HASHTABLE_FIND", "HASHTABLE_IMPORT",
    "HASHTABLE_SIZE",
];

fn tensor(t: Table, buffers: &[Table]) -> TensorInfo {
    let q = t.table(4);
    let data = buffers.get(t.get::<u32>(2, 0) as usize).map(|b| b.bytes(0)).unwrap_or(&[]);
    TensorInfo {
        name: t.str(3).unwrap_or("").to_string(),
        shape: t.scalars::<i32>(0).into_iter().map(|d| d.max(0) as usize).collect(),
        dtype: TType::from_code(t.get::<i8>(1, 0)),
        scales: q.map(|q| q.scalars(2)).unwrap_or_default(),
        zero_points: q.map(|q| q.scalars(3)).unwrap_or_default(),
        quantized_dimension: q.map_or(0, |q| q.get(6, 0)),
        constant: (!data.is_empty()).then(|| data.to_vec()),
    }
}

fn act(code: i8) -> Result<Act> {
    Ok(match code {
        0 => Act::None,
        1 => Act::Relu,
        2 => Act::ReluN1To1,
        3 => Act::Relu6,
        c => return Err(Error(format!("fused activation {c} is not supported"))),
    })
}

pub fn parse(buf: &[u8]) -> Result<Model> {
    if buf.len() < 8 || &buf[4..8] != b"TFL3" {
        return Err(Error("not a TensorFlow Lite model (no TFL3 identifier)".into()));
    }
    let m = Table::root(buf);
    let codes: Vec<(i32, Option<&str>)> = m
        .tables(1)
        .into_iter()
        .map(|c| ((c.get::<i8>(0, 0) as i32).max(c.get::<i32>(3, 0)), c.str(1)))
        .collect();
    let buffers = m.tables(4);
    let sg = *m.tables(2).first().ok_or_else(|| Error("the model has no subgraph".into()))?;
    let tensors: Vec<TensorInfo> = sg.tables(0).into_iter().map(|t| tensor(t, &buffers)).collect();
    let mut nodes = Vec::new();
    for op in sg.tables(3) {
        let (code, custom) = *codes.get(op.get::<u32>(0, 0) as usize).ok_or_else(|| Error("operator code out of range".into()))?;
        let options = op.table(4);
        let opt_i32 = |i: usize| options.map_or(0, |o| o.get::<i32>(i, 0));
        let opt_act = |i: usize| act(options.map_or(0, |o| o.get::<i8>(i, 0)));
        let (name, kind, a): (&'static str, Kind, Act) = match code {
            0 => ("ADD", Kind::Add, opt_act(0)?),
            41 => ("SUB", Kind::Sub, opt_act(0)?),
            18 => ("MUL", Kind::Mul, opt_act(0)?),
            42 => ("DIV", Kind::Div, opt_act(0)?),
            114 => ("QUANTIZE", Kind::Quantize, Act::None),
            6 => ("DEQUANTIZE", Kind::Dequantize, Act::None),
            53 => ("CAST", Kind::Cast, Act::None),
            22 => ("RESHAPE", Kind::Reshape, Act::None),
            56 => ("ARG_MAX", Kind::ArgMax, Act::None),
            90 => ("FLOOR_DIV", Kind::FloorDiv, Act::None),
            75 => ("SQRT", Kind::Sqrt, Act::None),
            83 => ("PACK", Kind::Pack { axis: opt_i32(1) }, Act::None),
            88 => ("UNPACK", Kind::Unpack { axis: opt_i32(1) }, Act::None),
            2 => ("CONCATENATION", Kind::Concat { axis: opt_i32(0) }, opt_act(1)?),
            107 => ("GATHER_ND", Kind::GatherNd, Act::None),
            32 => match custom {
                Some("edgetpu-custom-op") => ("edgetpu-custom-op", Kind::EdgeTpu(Box::new(EdgeTpuOp::new(op.bytes(5))?)), Act::None),
                Some("TFLite_Detection_PostProcess") => {
                    ("TFLite_Detection_PostProcess", Kind::DetectionPostProcess(dpp::Options::parse(flex::root(op.bytes(5)))), Act::None)
                }
                other => return Err(Error(format!("custom op {:?} is not supported", other.unwrap_or("")))),
            },
            c => {
                let n = BUILTIN.get(c as usize).copied().unwrap_or("?");
                return Err(Error(format!("op {n} ({c}) is not supported on the CPU side yet")));
            }
        };
        nodes.push(Node { name, kind, act: a, inputs: op.scalars(1), outputs: op.scalars(2) });
    }
    let index = |v: Vec<i32>| -> Result<Vec<usize>> {
        v.into_iter()
            .map(|i| if i >= 0 && (i as usize) < tensors.len() { Ok(i as usize) } else { Err(Error(format!("tensor index {i} out of range"))) })
            .collect()
    };
    let (inputs, outputs) = (index(sg.scalars(1))?, index(sg.scalars(2))?);
    for n in &nodes {
        index(n.outputs.clone())?;
        index(n.inputs.iter().copied().filter(|&i| i >= 0).collect())?;
    }
    // SignatureDefs as TFLite 2.5 reads them: names in a std::map (sorted, a repeated name keeps its last index).
    // Field 2 is method_name in 2.5 and signature_key from 2.6; signatures of other subgraphs (field 4, from 2.7) are
    // left out, as only subgraph 0 runs here.
    let mut signatures = Vec::new();
    for s in m.tables(7).into_iter().filter(|s| s.get::<u32>(4, 0) == 0) {
        let names = |i: usize| -> Result<Vec<(String, usize)>> {
            let map: std::collections::BTreeMap<String, u32> = s.tables(i).iter().map(|t| (t.str(0).unwrap_or("").into(), t.get(1, 0))).collect();
            map.into_iter().map(|(n, t)| Ok((n, index(vec![t as i32])?[0]))).collect()
        };
        let key = s.str(2).ok_or_else(|| Error("a SignatureDef has no key".into()))?;
        signatures.push(Signature { key: key.into(), inputs: names(0)?, outputs: names(1)? });
    }
    Ok(Model { tensors, nodes, inputs, outputs, signatures })
}
