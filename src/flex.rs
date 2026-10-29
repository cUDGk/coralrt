//! Minimal FlexBuffers reader, for the custom op options (the Edge TPU op and
//! TFLite_Detection_PostProcess store theirs as a FlexBuffer map).

use crate::fb::at;

const INT: u8 = 1;
const UINT: u8 = 2;
const FLOAT: u8 = 3;
const KEY: u8 = 4;
const STRING: u8 = 5;
const INDIRECT_INT: u8 = 6;
const INDIRECT_UINT: u8 = 7;
const INDIRECT_FLOAT: u8 = 8;
const MAP: u8 = 9;
const VECTOR: u8 = 10;
const VECTOR_INT: u8 = 11;
const VECTOR_KEY: u8 = 14;
const BLOB: u8 = 25;
const BOOL: u8 = 26;
const VECTOR_BOOL: u8 = 36;

fn uint(b: &[u8], o: usize, w: usize) -> u64 {
    match w {
        1 => at::<u8>(b, o) as u64,
        2 => at::<u16>(b, o) as u64,
        4 => at::<u32>(b, o) as u64,
        _ => at::<u64>(b, o),
    }
}

fn int(b: &[u8], o: usize, w: usize) -> i64 {
    match w {
        1 => at::<i8>(b, o) as i64,
        2 => at::<i16>(b, o) as i64,
        4 => at::<i32>(b, o) as i64,
        _ => at::<i64>(b, o),
    }
}

fn float(b: &[u8], o: usize, w: usize) -> f64 {
    if w == 4 { at::<f32>(b, o) as f64 } else { at::<f64>(b, o) }
}

/// a value: where it is stored, the width of that slot, the width of what it points to, and its type
#[derive(Clone, Copy)]
pub struct Ref<'a> {
    buf: &'a [u8],
    off: usize,
    parent_width: usize,
    byte_width: usize,
    ty: u8,
}

pub fn root(buf: &[u8]) -> Ref<'_> {
    let n = buf.len();
    let width = buf[n - 1] as usize;
    let packed = buf[n - 2];
    Ref { buf, off: n - 2 - width, parent_width: width, byte_width: 1 << (packed & 3), ty: packed >> 2 }
}

impl<'a> Ref<'a> {
    fn target(&self) -> usize {
        self.off - uint(self.buf, self.off, self.parent_width) as usize
    }

    pub fn as_i64(&self) -> i64 {
        match self.ty {
            INT => int(self.buf, self.off, self.parent_width),
            UINT | BOOL => uint(self.buf, self.off, self.parent_width) as i64,
            FLOAT => float(self.buf, self.off, self.parent_width) as i64,
            INDIRECT_INT => int(self.buf, self.target(), self.byte_width),
            INDIRECT_UINT => uint(self.buf, self.target(), self.byte_width) as i64,
            INDIRECT_FLOAT => float(self.buf, self.target(), self.byte_width) as i64,
            _ => 0,
        }
    }

    pub fn as_f64(&self) -> f64 {
        match self.ty {
            FLOAT => float(self.buf, self.off, self.parent_width),
            INDIRECT_FLOAT => float(self.buf, self.target(), self.byte_width),
            _ => self.as_i64() as f64,
        }
    }

    pub fn as_bool(&self) -> bool {
        self.as_i64() != 0
    }

    /// contents of a string, blob or key
    pub fn as_bytes(&self) -> &'a [u8] {
        match self.ty {
            STRING | BLOB => {
                let t = self.target();
                let n = uint(self.buf, t - self.byte_width, self.byte_width) as usize;
                &self.buf[t..t + n]
            }
            KEY => {
                let t = self.target();
                let n = self.buf[t..].iter().position(|&c| c == 0).unwrap_or(self.buf.len() - t);
                &self.buf[t..t + n]
            }
            _ => &[],
        }
    }

    pub fn len(&self) -> usize {
        match self.ty {
            MAP | VECTOR | VECTOR_INT..=VECTOR_KEY | VECTOR_BOOL => {
                uint(self.buf, self.target() - self.byte_width, self.byte_width) as usize
            }
            _ => 0,
        }
    }

    /// element `i` of a vector, a typed vector, or the values of a map
    pub fn index(&self, i: usize) -> Ref<'a> {
        let (t, w) = (self.target(), self.byte_width);
        let ty = match self.ty {
            // untyped: one packed type byte per element, after the elements
            MAP | VECTOR => {
                let packed = self.buf[t + self.len() * w + i];
                return Ref { buf: self.buf, off: t + i * w, parent_width: w, byte_width: 1 << (packed & 3), ty: packed >> 2 };
            }
            VECTOR_INT..=VECTOR_KEY => self.ty - VECTOR_INT + INT,
            VECTOR_BOOL => BOOL,
            other => panic!("flexbuffer value of type {other} is not a vector"),
        };
        Ref { buf: self.buf, off: t + i * w, parent_width: w, byte_width: w, ty }
    }

    /// value of `key` in a map
    pub fn get(&self, key: &str) -> Option<Ref<'a>> {
        if self.ty != MAP {
            return None;
        }
        let (t, w) = (self.target(), self.byte_width);
        // a map is a vector of values preceded by (keys vector offset, keys byte width)
        let keys_slot = t - 3 * w;
        let keys = keys_slot - uint(self.buf, keys_slot, w) as usize;
        let keys_width = uint(self.buf, t - 2 * w, w) as usize;
        (0..self.len())
            .find(|&i| {
                let k = Ref { buf: self.buf, off: keys + i * keys_width, parent_width: keys_width, byte_width: keys_width, ty: KEY };
                k.as_bytes() == key.as_bytes()
            })
            .map(|i| self.index(i))
    }
}
