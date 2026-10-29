//! Minimal schema-less FlatBuffers reader. A field is addressed by its index in the .fbs table
//! (a union takes two slots: the type, then the value). Reads past the end of the buffer panic;
//! the public entry points turn that into an `Error` for malformed files (see `crate::guard`).

pub trait Scalar: Copy {
    const SIZE: usize;
    fn read(b: &[u8]) -> Self;
}

macro_rules! scalar {
    ($($t:ty),*) => {$(
        impl Scalar for $t {
            const SIZE: usize = std::mem::size_of::<$t>();
            fn read(b: &[u8]) -> Self { <$t>::from_le_bytes(b[..Self::SIZE].try_into().unwrap()) }
        }
    )*};
}
scalar!(u8, i8, u16, i16, u32, i32, u64, i64, f32, f64);

pub fn at<T: Scalar>(buf: &[u8], off: usize) -> T {
    T::read(&buf[off..off + T::SIZE])
}

#[derive(Clone, Copy)]
pub struct Table<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Table<'a> {
    pub fn root(buf: &'a [u8]) -> Self {
        Table { buf, pos: at::<u32>(buf, 0) as usize }
    }

    /// absolute position of field `idx`, None when it is absent (default value)
    fn field(&self, idx: usize) -> Option<usize> {
        let vt = (self.pos as i64 - at::<i32>(self.buf, self.pos) as i64) as usize;
        let vtlen = at::<u16>(self.buf, vt) as usize;
        if 4 + 2 * idx >= vtlen {
            return None;
        }
        let fo = at::<u16>(self.buf, vt + 4 + 2 * idx) as usize;
        (fo != 0).then_some(self.pos + fo)
    }

    fn indirect(&self, o: usize) -> usize {
        o + at::<u32>(self.buf, o) as usize
    }

    pub fn get<T: Scalar>(&self, idx: usize, default: T) -> T {
        self.field(idx).map_or(default, |o| at(self.buf, o))
    }

    pub fn table(&self, idx: usize) -> Option<Table<'a>> {
        self.field(idx).map(|o| Table { buf: self.buf, pos: self.indirect(o) })
    }

    /// (start of the elements, element count) of a vector field
    fn vec(&self, idx: usize) -> Option<(usize, usize)> {
        let v = self.indirect(self.field(idx)?);
        Some((v + 4, at::<u32>(self.buf, v) as usize))
    }

    pub fn bytes(&self, idx: usize) -> &'a [u8] {
        self.vec(idx).map_or(&[], |(s, n)| &self.buf[s..s + n])
    }

    pub fn str(&self, idx: usize) -> Option<&'a str> {
        self.field(idx)?;
        std::str::from_utf8(self.bytes(idx)).ok()
    }

    pub fn scalars<T: Scalar>(&self, idx: usize) -> Vec<T> {
        self.vec(idx).map_or(Vec::new(), |(s, n)| (0..n).map(|i| at(self.buf, s + i * T::SIZE)).collect())
    }

    pub fn tables(&self, idx: usize) -> Vec<Table<'a>> {
        self.vec(idx).map_or(Vec::new(), |(s, n)| {
            (0..n).map(|i| Table { buf: self.buf, pos: self.indirect(s + 4 * i) }).collect()
        })
    }

    /// vector of strings (or of [ubyte] blobs, which have the same layout)
    pub fn blobs(&self, idx: usize) -> Vec<&'a [u8]> {
        self.vec(idx).map_or(Vec::new(), |(s, n)| {
            (0..n)
                .map(|i| {
                    let v = self.indirect(s + 4 * i);
                    &self.buf[v + 4..v + 4 + at::<u32>(self.buf, v) as usize]
                })
                .collect()
        })
    }

    /// vector of inline structs of `size` bytes each
    pub fn structs(&self, idx: usize, size: usize) -> Vec<&'a [u8]> {
        self.vec(idx).map_or(Vec::new(), |(s, n)| (0..n).map(|i| &self.buf[s + i * size..s + (i + 1) * size]).collect())
    }
}
