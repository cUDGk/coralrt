//! DarwiNN executables inside an `edgetpu-custom-op` (schema: libedgetpu executable/executable.fbs).

use crate::fb::{at, Table};
use crate::{flex, Error, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    U8,
    U16,
    I32,
    BF16,
    F16,
    F32,
    I8,
    I16,
}

impl DType {
    fn from_code(c: i16) -> Result<Self> {
        Ok(match c {
            0 => DType::U8,
            1 => DType::U16,
            2 => DType::I32,
            3 => DType::BF16,
            4 => DType::F16,
            5 => DType::F32,
            8 => DType::I8,
            9 => DType::I16,
            _ => return Err(Error(format!("unknown Edge TPU layer data type {c}"))),
        })
    }

    pub fn size(self) -> usize {
        match self {
            DType::U8 | DType::I8 => 1,
            DType::U16 | DType::I16 | DType::BF16 | DType::F16 => 2,
            DType::I32 | DType::F32 => 4,
        }
    }

    /// libedgetpu flips the sign bit of these on the way in and out (LayerInformation::SignedDataType)
    pub fn signed(self) -> bool {
        matches!(self, DType::I8 | DType::I16)
    }
}

/// per coordinate, where the tiles put an output (OutputLayout in executable.fbs)
#[derive(Debug, Default)]
pub struct OutputLayout {
    pub y_tile: Vec<i32>,
    pub x_tile: Vec<i32>,
    pub tile_byte_offset: Vec<i32>,
    pub x_local_byte_offset: Vec<i32>,
    pub y_local_y_offset: Vec<i32>,
    pub x_local_row_size: Vec<i32>,
}

/// inclusive (start, end) ranges per dimension
pub type Shape = Vec<(i32, i32)>;

#[derive(Debug)]
pub struct TensorLayout {
    pub shape: Shape,
    pub stride: Vec<i32>,
}

#[derive(Debug)]
pub struct Layer {
    pub name: String,
    /// padded size of one iteration
    pub size_bytes: usize,
    pub y: usize,
    pub x: usize,
    pub z: usize,
    pub zero_point: i32,
    pub scale: f32,
    pub dtype: DType,
    pub execution_count: usize,
    pub shape: Option<Shape>,
    pub layout: Option<OutputLayout>,
    /// output slices streamed from the tiles (OutputShapeInfo): layout and byte offset of each
    pub shape_info: Option<Vec<(TensorLayout, usize)>>,
}

impl Layer {
    pub fn actual_size_bytes(&self) -> usize {
        let n = match &self.shape {
            Some(s) => s.iter().map(|&(a, b)| (b - a + 1) as usize).product(),
            None => self.y * self.x * self.z,
        };
        n * self.dtype.size() * self.execution_count
    }

    pub fn padded_size_bytes(&self) -> usize {
        self.size_bytes * self.execution_count
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Desc {
    Output,
    Input,
    Parameter,
    Scratch,
}

#[derive(Debug)]
pub enum Hint {
    Instruction(usize),
    /// a DMA of `size` bytes at `offset` of the named input / output layer (or of the parameters)
    Dma { outfeed: bool, desc: Desc, name: String, offset: usize, size: usize },
    Interrupt(u8),
    Fence,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExeType {
    StandAlone,
    ParameterCaching,
    ExecutionOnly,
}

#[derive(Debug)]
pub struct Executable {
    pub name: String,
    pub ty: ExeType,
    pub batch_size: usize,
    pub scratch_size: usize,
    pub bitstreams: Vec<Vec<u8>>,
    pub parameters: Vec<u8>,
    pub hints: Vec<Hint>,
    pub fully_deterministic: bool,
    pub inputs: Vec<Layer>,
    pub outputs: Vec<Layer>,
    pub chip: String,
    pub caching_token: u64,
}

fn shape(t: Option<Table>) -> Option<Shape> {
    t.map(|t| t.structs(0, 8).into_iter().map(|r| (at::<i32>(r, 0), at::<i32>(r, 4))).collect())
}

fn layer(t: Table) -> Result<Layer> {
    let num = t.table(5);
    let out = if t.get::<u8>(7, 0) == 1 { t.table(8) } else { None };
    let layout = out.and_then(|o| o.table(0)).map(|l| OutputLayout {
        y_tile: l.scalars(0),
        x_tile: l.scalars(1),
        tile_byte_offset: l.scalars(2),
        x_local_byte_offset: l.scalars(3),
        y_local_y_offset: l.scalars(4),
        x_local_row_size: l.scalars(5),
    });
    let shape_info = out.and_then(|o| o.table(2)).map(|si| {
        let offsets: Vec<i32> = si.scalars(1);
        si.tables(0)
            .into_iter()
            .zip(offsets)
            .map(|(l, off)| (TensorLayout { shape: shape(l.table(0)).unwrap_or_default(), stride: l.scalars(1) }, off as usize))
            .collect()
    });
    Ok(Layer {
        name: t.str(0).unwrap_or("").to_string(),
        size_bytes: t.get::<i32>(1, 0) as usize,
        y: t.get::<i32>(2, 0) as usize,
        x: t.get::<i32>(3, 0) as usize,
        z: t.get::<i32>(4, 0) as usize,
        zero_point: num.map_or(0, |n| n.get(0, 0)),
        scale: num.map_or(0.0, |n| n.get(1, 0.0)),
        dtype: DType::from_code(t.get::<i16>(6, 0))?,
        execution_count: t.get::<i32>(9, 1).max(1) as usize,
        shape: shape(t.table(11)),
        layout,
        shape_info,
    })
}

fn meta(t: Option<Table>) -> Result<(Desc, String)> {
    let Some(t) = t else { return Ok((Desc::Output, String::new())) };
    let desc = match t.get::<i16>(0, 0) {
        0 => Desc::Output,
        1 => Desc::Input,
        2 => Desc::Parameter,
        3 => Desc::Scratch,
        d => return Err(Error(format!("unknown DMA description {d}"))),
    };
    Ok((desc, t.str(2).unwrap_or("").to_string()))
}

pub fn parse_executable(buf: &[u8]) -> Result<Executable> {
    let e = Table::root(buf);
    let mut hints = Vec::new();
    let dh = e.table(7);
    for h in dh.map(|d| d.tables(0)).unwrap_or_default() {
        let ht = h.table(1);
        let outfeed = h.get::<i16>(2, 0) == 1;
        hints.push(match (h.get::<u8>(0, 0), ht) {
            (1, Some(ht)) => {
                let (desc, name) = meta(ht.table(0))?;
                Hint::Dma { outfeed, desc, name, offset: ht.get::<i32>(1, 0) as usize, size: ht.get::<i32>(2, 0) as usize }
            }
            (2, Some(ht)) => Hint::Instruction(ht.get::<i32>(0, 0) as usize),
            (3, Some(ht)) => Hint::Interrupt(ht.get::<i16>(0, 0) as u8),
            (4, _) => Hint::Fence,
            (k, _) => return Err(Error(format!("unknown DMA hint type {k}"))),
        });
    }
    let ty = match e.get::<i16>(13, 0) {
        0 => ExeType::StandAlone,
        1 => ExeType::ParameterCaching,
        2 => ExeType::ExecutionOnly,
        t => return Err(Error(format!("unknown executable type {t}"))),
    };
    Ok(Executable {
        name: e.str(1).unwrap_or("").to_string(),
        ty,
        batch_size: e.get::<i32>(3, 1).max(1) as usize,
        scratch_size: e.get::<i32>(4, 0) as usize,
        bitstreams: e.tables(5).into_iter().map(|b| b.bytes(0).to_vec()).collect(),
        parameters: e.bytes(6).to_vec(),
        hints,
        fully_deterministic: dh.is_some_and(|d| d.get::<u8>(1, 0) != 0),
        inputs: e.tables(8).into_iter().map(layer).collect::<Result<_>>()?,
        outputs: e.tables(9).into_iter().map(layer).collect::<Result<_>>()?,
        chip: e.str(10).unwrap_or("").to_string(),
        caching_token: e.get::<u64>(14, 0),
    })
}

/// the executables of one serialized Package ("DWN1")
pub fn parse_package(buf: &[u8]) -> Result<Vec<Executable>> {
    if buf.len() < 8 || &buf[4..8] != b"DWN1" {
        return Err(Error("the Edge TPU op does not hold a DWN1 package".into()));
    }
    let multi = Table::root(Table::root(buf).bytes(1));
    multi.blobs(0).into_iter().map(parse_executable).collect()
}

/// executables of an `edgetpu-custom-op`, from its FlexBuffer options (libedgetpu tflite/custom_op_data.cc):
/// "4" is the first serialized package, "7" the others, "6" the chip of each.
pub fn parse_custom_op(options: &[u8]) -> Result<Vec<Executable>> {
    let m = flex::root(options);
    let first = m.get("4").ok_or_else(|| Error("the Edge TPU op has no executable".into()))?;
    let mut packages = vec![first.as_bytes()];
    if let Some(rest) = m.get("7") {
        packages.extend((0..rest.len()).map(|i| rest.index(i).as_bytes()));
    }
    let mut exes = Vec::new();
    for p in packages {
        exes.extend(parse_package(p)?);
    }
    Ok(exes)
}
