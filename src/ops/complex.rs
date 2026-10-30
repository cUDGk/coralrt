//! Complex numbers: REAL, IMAG, COMPLEX_ABS (complex_support.cc) and RFFT2D (rfft2d.cc), whose transform is Takuya
//! Ooura's rdft2d in double (third_party/oourafft, compiled by build.rs as in the official runtime).

use std::ffi::c_int;

use super::util::*;
use super::{Io, Kernel};
use crate::interp::Env;
use crate::tensor::{Alloc, TType};
use crate::tflite::OpSpec;
use crate::Result;

unsafe extern "C" {
    fn rdft2d(n1: c_int, n2: c_int, isgn: c_int, a: *mut *mut f64, t: *mut f64, ip: *mut c_int, w: *mut f64);
}

#[derive(Clone, Copy, PartialEq)]
enum Part {
    Real,
    Imag,
    Abs,
}

pub fn create(s: &OpSpec) -> Result<Option<Box<dyn Kernel>>> {
    Ok(Some(match s.code {
        134 => Box::new(Complex(Part::Real)),
        133 => Box::new(Complex(Part::Imag)),
        135 => Box::new(Complex(Part::Abs)),
        131 => Box::new(Rfft2d),
        _ => return Ok(None),
    }))
}

struct Complex(Part);

impl Kernel for Complex {
    fn prepare(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        let i = io.input(0)?;
        let want = match i.dtype {
            TType::C64 => TType::F32,
            TType::C128 => TType::F64,
            t => return err(format!("{t:?} is not complex")),
        };
        if io.outs[0].dtype != want {
            return err(format!("the output must be {want:?}"));
        }
        same_shape(i, &mut io.outs[0]);
        Ok(())
    }

    fn eval(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        let i = io.input(0)?;
        let o = &mut io.outs[0];
        // std::real / std::imag, and std::abs: hypotf / hypot in the VS 2019 standard library
        if i.dtype == TType::C64 {
            for (d, p) in o.as_mut_slice::<f32>().iter_mut().zip(i.as_slice::<f32>().chunks_exact(2)) {
                *d = match self.0 {
                    Part::Real => p[0],
                    Part::Imag => p[1],
                    Part::Abs => p[0].hypot(p[1]),
                };
            }
        } else {
            for (d, p) in o.as_mut_slice::<f64>().iter_mut().zip(i.as_slice::<f64>().chunks_exact(2)) {
                *d = match self.0 {
                    Part::Real => p[0],
                    Part::Imag => p[1],
                    Part::Abs => p[0].hypot(p[1]),
                };
            }
        }
        Ok(())
    }
}

/// RFFT2D (input [.., H, W] float, fft_length [2] powers of two): per 2-D slice, the input cropped or zero padded to
/// the lengths, rdft2d forward, Rfft2dReorder (the full width/2 + 1 columns, imaginary parts negated), complex64 out
struct Rfft2d;

impl Rfft2d {
    fn lengths(io: &Io) -> Result<(usize, usize)> {
        let l = ints(io.input(1)?)?;
        let pow2 = |v: i64| v > 0 && (v & (v - 1)) == 0 && v <= u32::MAX as i64;
        if l.len() != 2 || !pow2(l[0]) || !pow2(l[1]) {
            return err("the FFT lengths must be two powers of two");
        }
        Ok((l[0] as usize, l[1] as usize))
    }

    fn resize(io: &mut Io) -> Result<()> {
        let (h, w) = Self::lengths(io)?;
        let mut s = io.input(0)?.shape.clone();
        let n = s.len();
        s[n - 2] = h;
        s[n - 1] = w / 2 + 1;
        io.outs[0].resize(s);
        Ok(())
    }
}

impl Kernel for Rfft2d {
    fn prepare(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        let (i, l) = (io.input(0)?, io.input(1)?);
        if i.shape.len() < 2 || i.dtype != TType::F32 || l.shape != [2] || l.dtype != TType::I32 {
            return err("RFFT2D takes float32 of rank 2 or more and two int32 lengths");
        }
        // 2.5 makes the output complex64 itself
        io.outs[0].dtype = TType::C64;
        if l.alloc == Alloc::Constant {
            Self::resize(io)
        } else {
            io.outs[0].alloc = Alloc::Dynamic;
            Ok(())
        }
    }

    fn eval(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        if io.outs[0].alloc == Alloc::Dynamic {
            Self::resize(io)?;
        }
        let (h, w) = Self::lengths(io)?;
        let i = io.input(0)?;
        let n = i.shape.len();
        let (ih, iw) = (i.shape[n - 2], i.shape[n - 1]);
        let slices: usize = i.shape[..n - 2].iter().product();
        let x = i.as_slice::<f32>();
        let out = io.outs[0].as_mut_slice::<f32>();
        let mut rows: Vec<Vec<f64>> = vec![vec![0.0; w + 2]; h];
        let working = h.max(w / 2);
        let mut ip = vec![0 as c_int; 2 + (working as f64).sqrt() as usize];
        let mut wa = vec![0.0f64; working / 2 + w / 4];
        let cols = w / 2 + 1;
        for s in 0..slices {
            for (r, row) in rows.iter_mut().enumerate() {
                row.fill(0.0);
                if r < ih {
                    for (d, &v) in row.iter_mut().zip(&x[(s * ih + r) * iw..(s * ih + r) * iw + iw.min(w)]) {
                        *d = v as f64;
                    }
                }
            }
            ip.fill(0);
            wa.fill(0.0);
            let mut ptrs: Vec<*mut f64> = rows.iter_mut().map(|r| r.as_mut_ptr()).collect();
            // SAFETY: rdft2d works on h rows of w + 2 doubles (it uses w), with ip and w sized as rfft2d.cc sizes them,
            // and allocates its own column buffer when t is null
            unsafe { rdft2d(h as c_int, w as c_int, 1, ptrs.as_mut_ptr(), std::ptr::null_mut(), ip.as_mut_ptr(), wa.as_mut_ptr()) };
            reorder(&mut rows, h, w);
            for (r, row) in rows.iter().enumerate() {
                for j in 0..cols {
                    let at = ((s * h + r) * cols + j) * 2;
                    out[at] = row[2 * j] as f32;
                    out[at + 1] = row[2 * j + 1] as f32;
                }
            }
        }
        Ok(())
    }
}

/// Rfft2dReorder: Ooura's packed result spread to the full width / 2 + 1 columns, imaginary parts negated
fn reorder(a: &mut [Vec<f64>], h: usize, w: usize) {
    let half = h >> 1;
    for i in half + 1..h {
        let (real, img) = (a[i][0], a[i][1]);
        a[i][w] = img;
        a[i][w + 1] = real;
        a[h - i][w] = img;
        a[h - i][w + 1] = -real;
        a[i][0] = a[h - i][0];
        a[i][1] = -a[h - i][1];
    }
    let temp = a[0][1];
    a[0][w + 1] = 0.0;
    a[0][1] = 0.0;
    a[half][w] = a[half][1];
    a[half][w + 1] = 0.0;
    a[half][1] = 0.0;
    a[0][w] = temp;
    for row in a.iter_mut().take(h) {
        for j in (1..w + 2).step_by(2) {
            row[j] = -row[j];
        }
    }
}
