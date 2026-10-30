//! The C ABI (include/coralrt.h), for C, C++, Python (ctypes), C#, Go, Node and anything else with an FFI.
//! Errors: functions return NULL / -1 and leave a message for `coral_last_error` on the calling thread.
//! Panics never cross the boundary.

use crate::{Interpreter, Options, Performance};
use std::cell::RefCell;
use std::ffi::{c_char, CStr, CString};

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

fn set_error(msg: String) {
    LAST_ERROR.with(|e| *e.borrow_mut() = Some(CString::new(msg.replace('\0', " ")).unwrap()));
}

/// run `f` at the boundary: an error or panic becomes `fail` plus a message for coral_last_error
fn call<T>(fail: T, f: impl FnOnce() -> crate::Result<T>) -> T {
    match crate::guard("coralrt", f) {
        Ok(v) => v,
        Err(e) => {
            set_error(e.0);
            fail
        }
    }
}

/// SAFETY: `s` is NULL or a NUL terminated string
unsafe fn opt_str<'a>(s: *const c_char) -> crate::Result<Option<&'a str>> {
    if s.is_null() {
        return Ok(None);
    }
    // SAFETY: by the caller's contract
    unsafe { CStr::from_ptr(s) }.to_str().map(Some).map_err(|_| crate::Error("argument is not UTF-8".into()))
}

fn copy_str(s: &str, buf: *mut c_char, len: usize) -> i32 {
    if !buf.is_null() && len > 0 {
        let n = s.len().min(len - 1);
        // SAFETY: the caller gave `len` writable bytes at buf
        unsafe {
            std::ptr::copy_nonoverlapping(s.as_ptr(), buf as *mut u8, n);
            *buf.add(n) = 0;
        }
    }
    s.len() as i32
}

pub struct CoralInterpreter {
    it: Interpreter,
    names: Vec<CString>,
    /// per signature: its key, then its input names, then its output names
    sig_names: Vec<Vec<CString>>,
}

fn c_string(s: &str) -> CString {
    CString::new(s.replace('\0', " ")).unwrap()
}

#[unsafe(no_mangle)]
pub extern "C" fn coral_last_error() -> *const c_char {
    LAST_ERROR.with(|e| e.borrow().as_ref().map_or(std::ptr::null(), |s| s.as_ptr()))
}

#[unsafe(no_mangle)]
pub extern "C" fn coral_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const c_char
}

#[unsafe(no_mangle)]
pub extern "C" fn coral_list_devices() -> i32 {
    call(-1, || Ok(crate::list_devices()?.len() as i32))
}

/// copies the usb path of the i-th stick into buf (NUL terminated, truncated to len); returns its full length
#[unsafe(no_mangle)]
pub extern "C" fn coral_device_path(i: i32, buf: *mut c_char, len: usize) -> i32 {
    call(-1, || {
        let all = crate::list_devices()?;
        let d = all.get(i as usize).ok_or_else(|| crate::Error(format!("no Edge TPU number {i}")))?;
        Ok(copy_str(&d.path, buf, len))
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn coral_device_in_bootloader(i: i32) -> i32 {
    call(-1, || {
        let all = crate::list_devices()?;
        Ok(all.get(i as usize).ok_or_else(|| crate::Error(format!("no Edge TPU number {i}")))?.in_bootloader as i32)
    })
}

fn new_interpreter(model: &[u8], device: *const c_char, performance: *const c_char) -> crate::Result<*mut CoralInterpreter> {
    // SAFETY: the C contract of these arguments
    let (device, perf) = unsafe { (opt_str(device)?, opt_str(performance)?) };
    let performance = perf.map(Performance::parse).transpose()?;
    let it = Interpreter::from_bytes(model, &Options { device, performance })?;
    let names = (0..it.tensor_count()).map(|t| c_string(&it.tensor(t).name)).collect();
    let sig_names = it
        .signatures()
        .iter()
        .map(|s| std::iter::once(&s.key).chain(s.inputs.iter().chain(&s.outputs).map(|(n, _)| n)).map(|n| c_string(n)).collect())
        .collect();
    Ok(Box::into_raw(Box::new(CoralInterpreter { it, names, sig_names })))
}

#[unsafe(no_mangle)]
pub extern "C" fn coral_interpreter_new(model_path: *const c_char, device: *const c_char, performance: *const c_char) -> *mut CoralInterpreter {
    call(std::ptr::null_mut(), || {
        // SAFETY: NUL terminated path
        let path = unsafe { opt_str(model_path)? }.ok_or_else(|| crate::Error("model_path is NULL".into()))?;
        let bytes = std::fs::read(path).map_err(|e| crate::Error(format!("{path}: {e}")))?;
        new_interpreter(&bytes, device, performance)
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn coral_interpreter_new_from_buffer(data: *const u8, len: usize, device: *const c_char, performance: *const c_char) -> *mut CoralInterpreter {
    call(std::ptr::null_mut(), || {
        if data.is_null() {
            return Err(crate::Error("model data is NULL".into()));
        }
        // SAFETY: the caller gives len readable bytes
        new_interpreter(unsafe { std::slice::from_raw_parts(data, len) }, device, performance)
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn coral_interpreter_free(it: *mut CoralInterpreter) {
    if !it.is_null() {
        // SAFETY: made by Box::into_raw in new_interpreter, freed once by the caller's contract
        drop(unsafe { Box::from_raw(it) });
    }
}

/// SAFETY: `it` comes from coral_interpreter_new* and is not freed
unsafe fn get<'a>(it: *const CoralInterpreter) -> crate::Result<&'a CoralInterpreter> {
    // SAFETY: by the caller's contract
    unsafe { it.as_ref() }.ok_or_else(|| crate::Error("interpreter is NULL".into()))
}

unsafe fn get_mut<'a>(it: *mut CoralInterpreter) -> crate::Result<&'a mut CoralInterpreter> {
    // SAFETY: by the caller's contract
    unsafe { it.as_mut() }.ok_or_else(|| crate::Error("interpreter is NULL".into()))
}

fn tensor_of(c: &CoralInterpreter, t: i32) -> crate::Result<&crate::TensorInfo> {
    if t < 0 || t as usize >= c.it.tensor_count() {
        return Err(crate::Error(format!("tensor {t} out of range")));
    }
    Ok(c.it.tensor(t as usize))
}

#[unsafe(no_mangle)]
pub extern "C" fn coral_interpreter_invoke(it: *mut CoralInterpreter) -> i32 {
    call(-1, || unsafe { get_mut(it) }?.it.invoke().map(|_| 0))
}

#[unsafe(no_mangle)]
pub extern "C" fn coral_input_count(it: *const CoralInterpreter) -> i32 {
    call(-1, || Ok(unsafe { get(it) }?.it.inputs().len() as i32))
}

#[unsafe(no_mangle)]
pub extern "C" fn coral_output_count(it: *const CoralInterpreter) -> i32 {
    call(-1, || Ok(unsafe { get(it) }?.it.outputs().len() as i32))
}

/// tensor index of the i-th input
#[unsafe(no_mangle)]
pub extern "C" fn coral_input_tensor(it: *const CoralInterpreter, i: i32) -> i32 {
    call(-1, || unsafe { get(it) }?.it.inputs().get(i as usize).map(|&t| t as i32).ok_or_else(|| crate::Error(format!("no input {i}"))))
}

#[unsafe(no_mangle)]
pub extern "C" fn coral_output_tensor(it: *const CoralInterpreter, i: i32) -> i32 {
    call(-1, || unsafe { get(it) }?.it.outputs().get(i as usize).map(|&t| t as i32).ok_or_else(|| crate::Error(format!("no output {i}"))))
}

#[unsafe(no_mangle)]
pub extern "C" fn coral_tensor_count(it: *const CoralInterpreter) -> i32 {
    call(-1, || Ok(unsafe { get(it) }?.it.tensor_count() as i32))
}

/// valid while the interpreter lives
#[unsafe(no_mangle)]
pub extern "C" fn coral_tensor_name(it: *const CoralInterpreter, t: i32) -> *const c_char {
    call(std::ptr::null(), || {
        let c = unsafe { get(it) }?;
        tensor_of(c, t)?;
        Ok(c.names[t as usize].as_ptr())
    })
}

/// TensorFlow Lite TensorType: 0 float32, 1 float16, 2 int32, 3 uint8, 4 int64, 6 bool, 7 int16, 9 int8, 10 float64
#[unsafe(no_mangle)]
pub extern "C" fn coral_tensor_type(it: *const CoralInterpreter, t: i32) -> i32 {
    call(-1, || Ok(tensor_of(unsafe { get(it) }?, t)?.dtype.code()))
}

#[unsafe(no_mangle)]
pub extern "C" fn coral_tensor_rank(it: *const CoralInterpreter, t: i32) -> i32 {
    call(-1, || Ok(tensor_of(unsafe { get(it) }?, t)?.shape.len() as i32))
}

#[unsafe(no_mangle)]
pub extern "C" fn coral_tensor_dim(it: *const CoralInterpreter, t: i32, d: i32) -> i32 {
    call(-1, || {
        let s = &tensor_of(unsafe { get(it) }?, t)?.shape;
        s.get(d as usize).map(|&v| v as i32).ok_or_else(|| crate::Error(format!("dimension {d} out of range")))
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn coral_tensor_bytes(it: *const CoralInterpreter, t: i32) -> isize {
    call(-1, || Ok(tensor_of(unsafe { get(it) }?, t)?.bytes() as isize))
}

/// number of quantization scales (0: not quantized, 1: per tensor, more: per channel along the quantized dimension)
#[unsafe(no_mangle)]
pub extern "C" fn coral_tensor_quant_count(it: *const CoralInterpreter, t: i32) -> i32 {
    call(-1, || Ok(tensor_of(unsafe { get(it) }?, t)?.scales.len() as i32))
}

#[unsafe(no_mangle)]
pub extern "C" fn coral_tensor_quant_scale(it: *const CoralInterpreter, t: i32, k: i32) -> f32 {
    call(f32::NAN, || tensor_of(unsafe { get(it) }?, t)?.scales.get(k as usize).copied().ok_or_else(|| crate::Error(format!("no scale {k}"))))
}

#[unsafe(no_mangle)]
pub extern "C" fn coral_tensor_quant_zero_point(it: *const CoralInterpreter, t: i32, k: i32) -> i64 {
    call(0, || tensor_of(unsafe { get(it) }?, t)?.zero_points.get(k as usize).copied().ok_or_else(|| crate::Error(format!("no zero point {k}"))))
}

#[unsafe(no_mangle)]
pub extern "C" fn coral_tensor_quantized_dimension(it: *const CoralInterpreter, t: i32) -> i32 {
    call(-1, || Ok(tensor_of(unsafe { get(it) }?, t)?.quantized_dimension))
}

/// The tensor's memory (8 byte aligned), valid while the interpreter lives: write inputs and read outputs in place.
#[unsafe(no_mangle)]
pub extern "C" fn coral_tensor_data(it: *mut CoralInterpreter, t: i32) -> *mut u8 {
    call(std::ptr::null_mut(), || {
        let c = unsafe { get_mut(it) }?;
        tensor_of(c, t)?;
        Ok(c.it.data_mut(t as usize).as_mut_ptr())
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn coral_set_input(it: *mut CoralInterpreter, i: i32, data: *const u8, len: usize) -> i32 {
    call(-1, || {
        if data.is_null() {
            return Err(crate::Error("input data is NULL".into()));
        }
        // SAFETY: the caller gives len readable bytes
        unsafe { get_mut(it) }?.it.set_input(i as usize, unsafe { std::slice::from_raw_parts(data, len) }).map(|_| 0)
    })
}

/// copies the i-th output into data (which must hold the whole tensor); returns the bytes copied
#[unsafe(no_mangle)]
pub extern "C" fn coral_get_output(it: *const CoralInterpreter, i: i32, data: *mut u8, len: usize) -> isize {
    call(-1, || {
        let out = unsafe { get(it) }?.it.output(i as usize)?;
        if data.is_null() || len < out.len() {
            return Err(crate::Error(format!("output {i} needs {} bytes", out.len())));
        }
        // SAFETY: the caller gives len >= out.len() writable bytes
        unsafe { std::ptr::copy_nonoverlapping(out.as_ptr(), data, out.len()) };
        Ok(out.len() as isize)
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn coral_signature_count(it: *const CoralInterpreter) -> i32 {
    call(-1, || Ok(unsafe { get(it) }?.it.signatures().len() as i32))
}

/// valid while the interpreter lives
#[unsafe(no_mangle)]
pub extern "C" fn coral_signature_key(it: *const CoralInterpreter, s: i32) -> *const c_char {
    call(std::ptr::null(), || {
        let c = unsafe { get(it) }?;
        Ok(c.sig_names.get(s as usize).ok_or_else(|| crate::Error(format!("no signature {s}")))?[0].as_ptr())
    })
}

/// the inputs (io 0) or outputs (io 1) of signature s, and where their names start in sig_names[s]
fn signature_io(c: &CoralInterpreter, s: i32, io: i32) -> crate::Result<(&[(String, usize)], usize)> {
    let sig = c.it.signatures().get(s as usize).ok_or_else(|| crate::Error(format!("no signature {s}")))?;
    match io {
        0 => Ok((&sig.inputs, 1)),
        1 => Ok((&sig.outputs, 1 + sig.inputs.len())),
        _ => Err(crate::Error(format!("io must be 0 (inputs) or 1 (outputs), not {io}"))),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn coral_signature_tensor_count(it: *const CoralInterpreter, s: i32, io: i32) -> i32 {
    call(-1, || Ok(signature_io(unsafe { get(it) }?, s, io)?.0.len() as i32))
}

/// valid while the interpreter lives
#[unsafe(no_mangle)]
pub extern "C" fn coral_signature_tensor_name(it: *const CoralInterpreter, s: i32, io: i32, k: i32) -> *const c_char {
    call(std::ptr::null(), || {
        let c = unsafe { get(it) }?;
        let (list, at) = signature_io(c, s, io)?;
        if k < 0 || k as usize >= list.len() {
            return Err(crate::Error(format!("signature {s} has no tensor {k} there")));
        }
        Ok(c.sig_names[s as usize][at + k as usize].as_ptr())
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn coral_signature_tensor_index(it: *const CoralInterpreter, s: i32, io: i32, k: i32) -> i32 {
    call(-1, || {
        let (list, _) = signature_io(unsafe { get(it) }?, s, io)?;
        list.get(k as usize).map(|&(_, t)| t as i32).ok_or_else(|| crate::Error(format!("signature {s} has no tensor {k} there")))
    })
}

/// usb path of the stick the interpreter runs on (empty: the model has no Edge TPU op); returns the full length
#[unsafe(no_mangle)]
pub extern "C" fn coral_interpreter_device_path(it: *const CoralInterpreter, buf: *mut c_char, len: usize) -> i32 {
    call(-1, || Ok(copy_str(&unsafe { get(it) }?.it.device().map(|d| d.path()).unwrap_or_default(), buf, len)))
}
