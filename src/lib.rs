//! coralrt runs Coral Edge TPU models (`*_edgetpu.tflite`) on the Coral USB Accelerator without libedgetpu,
//! tflite_runtime or pycoral: a USB driver for the stick, the `edgetpu-custom-op`, and the CPU ops that compiled
//! models keep around it, computing the same bytes as pycoral 2.0 (tflite_runtime 2.5 + libedgetpu).
//! Rust uses this crate directly; every other language goes through the C ABI in `capi` (include/coralrt.h).
//!
//! Sources ported: libedgetpu (Apache 2.0), TensorFlow Lite 2.5 kernels (Apache 2.0), tinycoral (MIT).

mod capi;
mod chip;
mod edgetpu;
pub mod exe;
mod fb;
mod flex;
mod ops;
pub mod tflite;
mod usb;

pub use chip::Performance;
pub use tflite::{Signature, TType, TensorInfo};

use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard, Weak};

#[derive(Debug, Clone)]
pub struct Error(pub String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

/// Run `f`, turning a panic into an `Error`. The parsers index straight into the file's bytes, so a malformed or
/// truncated model panics on an out-of-range read; at the API boundary that is an error about the input.
fn guard<T>(what: &str, f: impl FnOnce() -> Result<T>) -> Result<T> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).unwrap_or_else(|p| Err(Error(format!("{what}: {}", panic_text(&p)))))
}

fn panic_text(p: &Box<dyn std::any::Any + Send>) -> String {
    p.downcast_ref::<&str>().map(|s| s.to_string()).or_else(|| p.downcast_ref::<String>().cloned()).unwrap_or_else(|| "malformed input".into())
}

#[derive(Clone, Debug)]
pub struct DeviceInfo {
    /// "bus-port.port"
    pub path: String,
    /// still in the bootloader (no firmware yet: the first open loads it)
    pub in_bootloader: bool,
}

pub fn list_devices() -> Result<Vec<DeviceInfo>> {
    Ok(chip::list()?.into_iter().map(|(path, in_bootloader)| DeviceInfo { path, in_bootloader }).collect())
}

/// An open Coral USB Accelerator, shared by the interpreters that use it (one USB handle per stick and process).
#[derive(Clone)]
pub struct Device(Arc<Mutex<chip::EdgeTpu>>);

static OPEN: Mutex<Vec<Weak<Mutex<chip::EdgeTpu>>>> = Mutex::new(Vec::new());

impl Device {
    /// `selector` like pycoral's `device`: None, "" or "usb" for the first stick, ":N" or "usb:N" for the N-th
    /// (as `list_devices` orders them), "usb:<bus-port path>" for a given one. A stick this process already
    /// opened is shared, at the performance it was opened with.
    pub fn open(selector: Option<&str>, performance: Performance) -> Result<Device> {
        let sel = selector.unwrap_or("").trim();
        let rest = sel.strip_prefix("usb").unwrap_or(sel);
        let rest = rest.strip_prefix(':').unwrap_or(rest);
        if !sel.is_empty() && !sel.starts_with("usb") && !sel.starts_with(':') {
            return Err(Error(format!("unsupported device {sel:?} (only USB sticks: \"usb\", \":N\", \"usb:N\", \"usb:<path>\")")));
        }
        let all = list_devices()?;
        let path = if rest.is_empty() {
            None
        } else if let Ok(n) = rest.parse::<usize>() {
            Some(all.get(n).ok_or_else(|| Error(format!("no Edge TPU number {n} ({} found)", all.len())))?.path.clone())
        } else {
            Some(rest.to_string())
        };
        let mut open = OPEN.lock().unwrap_or_else(|e| e.into_inner());
        open.retain(|w| w.strong_count() > 0);
        for w in open.iter() {
            if let Some(d) = w.upgrade() {
                let same = path.as_ref().is_none_or(|p| d.lock().unwrap_or_else(|e| e.into_inner()).path == *p);
                if same {
                    return Ok(Device(d));
                }
            }
        }
        let tpu = guard("opening the Edge TPU", || chip::EdgeTpu::open(path.as_deref(), performance))?;
        let d = Arc::new(Mutex::new(tpu));
        open.push(Arc::downgrade(&d));
        Ok(Device(d))
    }

    fn lock(&self) -> MutexGuard<'_, chip::EdgeTpu> {
        self.0.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn path(&self) -> String {
        self.lock().path.clone()
    }

    pub fn performance(&self) -> Performance {
        self.lock().performance
    }

    /// 3 = USB 2 high speed, 4 = USB 3 super speed
    pub fn usb_speed(&self) -> i32 {
        self.lock().usb_speed
    }

    pub fn efuse_revision(&self) -> Result<u32> {
        self.lock().efuse_revision()
    }
}

/// tensor storage: 8 byte aligned (so typed views of any element type are aligned), never moved once allocated
#[derive(Default)]
struct Buf {
    words: Vec<u64>,
    len: usize,
}

impl Buf {
    fn new(len: usize) -> Self {
        Buf { words: vec![0; len.div_ceil(8)], len }
    }
    fn bytes(&self) -> &[u8] {
        // SAFETY: the u64 storage holds at least len initialized bytes
        unsafe { std::slice::from_raw_parts(self.words.as_ptr() as *const u8, self.len) }
    }
    fn bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: as above, uniquely borrowed
        unsafe { std::slice::from_raw_parts_mut(self.words.as_mut_ptr() as *mut u8, self.len) }
    }
}

pub struct Options<'a> {
    /// which stick (see `Device::open`); used when the model has an Edge TPU op
    pub device: Option<&'a str>,
    /// the clock; default: the CORALRT_PERFORMANCE environment variable, else High (the official "throttled" runtime)
    pub performance: Option<Performance>,
}

impl Default for Options<'_> {
    fn default() -> Self {
        Options { device: None, performance: None }
    }
}

/// A model with its tensors, like tflite's Interpreter: write the inputs, `invoke`, read the outputs.
pub struct Interpreter {
    model: tflite::Model,
    bufs: Vec<Buf>,
    batches: Vec<usize>,
    device: Option<Device>,
}

impl Interpreter {
    pub fn from_file(path: impl AsRef<std::path::Path>, options: &Options) -> Result<Self> {
        let p = path.as_ref();
        let bytes = std::fs::read(p).map_err(|e| Error(format!("{}: {e}", p.display())))?;
        Self::from_bytes(&bytes, options)
    }

    pub fn from_bytes(model: &[u8], options: &Options) -> Result<Self> {
        let model = guard("reading the model", || tflite::parse(model))?;
        let mut bufs: Vec<Buf> = model.tensors.iter().map(|t| Buf::new(t.bytes())).collect();
        for (b, t) in bufs.iter_mut().zip(&model.tensors) {
            if let Some(c) = &t.constant {
                let n = b.len.min(c.len());
                b.bytes_mut()[..n].copy_from_slice(&c[..n]);
            }
        }
        let mut batches = vec![1; model.nodes.len()];
        for (n, node) in model.nodes.iter().enumerate() {
            if let tflite::Kind::EdgeTpu(op) = &node.kind {
                let ins: Vec<&TensorInfo> = node.inputs.iter().map(|&i| &model.tensors[i as usize]).collect();
                let outs: Vec<&TensorInfo> = node.outputs.iter().map(|&i| &model.tensors[i as usize]).collect();
                batches[n] = op.validate(&ins, &outs)?;
            }
        }
        let device = if model.nodes.iter().any(|n| matches!(n.kind, tflite::Kind::EdgeTpu(_))) {
            let perf = match options.performance {
                Some(p) => p,
                None => match std::env::var("CORALRT_PERFORMANCE") {
                    Ok(s) => Performance::parse(&s)?,
                    Err(_) => Performance::High,
                },
            };
            Some(Device::open(options.device, perf)?)
        } else {
            None
        };
        Ok(Interpreter { model, bufs, batches, device })
    }

    /// share an already open stick instead of opening one from `Options`
    pub fn from_bytes_on(model: &[u8], device: Device) -> Result<Self> {
        let path = device.path();
        Self::from_bytes(model, &Options { device: Some(&format!("usb:{path}")), performance: Some(device.performance()) })
    }

    pub fn device(&self) -> Option<&Device> {
        self.device.as_ref()
    }

    pub fn inputs(&self) -> &[usize] {
        &self.model.inputs
    }

    pub fn outputs(&self) -> &[usize] {
        &self.model.outputs
    }

    /// the model's SignatureDefs (most Edge TPU models have none)
    pub fn signatures(&self) -> &[Signature] {
        &self.model.signatures
    }

    pub fn tensor_count(&self) -> usize {
        self.model.tensors.len()
    }

    pub fn tensor(&self, index: usize) -> &TensorInfo {
        &self.model.tensors[index]
    }

    pub fn data(&self, index: usize) -> &[u8] {
        self.bufs[index].bytes()
    }

    pub fn data_mut(&mut self, index: usize) -> &mut [u8] {
        self.bufs[index].bytes_mut()
    }

    /// copy `data` into the k-th input (sizes must match)
    pub fn set_input(&mut self, k: usize, data: &[u8]) -> Result<()> {
        let t = *self.model.inputs.get(k).ok_or_else(|| Error(format!("no input {k}")))?;
        let b = self.bufs[t].bytes_mut();
        if b.len() != data.len() {
            return Err(Error(format!("input {k} takes {} bytes, got {}", b.len(), data.len())));
        }
        b.copy_from_slice(data);
        Ok(())
    }

    pub fn output(&self, k: usize) -> Result<&[u8]> {
        let t = *self.model.outputs.get(k).ok_or_else(|| Error(format!("no output {k}")))?;
        Ok(self.bufs[t].bytes())
    }

    /// names of the ops, in execution order (e.g. to see what runs on the CPU)
    pub fn op_names(&self) -> Vec<&'static str> {
        self.model.nodes.iter().map(|n| n.name).collect()
    }

    /// the Edge TPU executables of each `edgetpu-custom-op`
    pub fn executables(&self) -> Vec<&exe::Executable> {
        self.model.nodes.iter().filter_map(|n| if let tflite::Kind::EdgeTpu(op) = &n.kind { Some(&op.exes) } else { None }).flatten().collect()
    }

    pub fn invoke(&mut self) -> Result<()> {
        for (n, node) in self.model.nodes.iter().enumerate() {
            // the outputs leave `bufs` while the op writes them, and always come back (also after a panic): callers
            // may hold pointers into them
            let mut taken: Vec<(usize, Buf)> = node.outputs.iter().map(|&o| (o as usize, std::mem::take(&mut self.bufs[o as usize]))).collect();
            let result = {
                let tensors = &self.model.tensors;
                let bufs = &self.bufs;
                let ins: Vec<ops::In> =
                    node.inputs.iter().filter(|&&i| i >= 0).map(|&i| ops::In { info: &tensors[i as usize], data: bufs[i as usize].bytes() }).collect();
                let mut outs: Vec<ops::Out> = taken.iter_mut().map(|(o, b)| ops::Out { info: &tensors[*o], data: b.bytes_mut() }).collect();
                let device = &self.device;
                let batches = self.batches[n];
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match &node.kind {
                    tflite::Kind::EdgeTpu(op) => {
                        let dev = device.as_ref().ok_or_else(|| Error("no Edge TPU".into()))?;
                        let in_data: Vec<&[u8]> = ins.iter().map(|i| i.data).collect();
                        let infos: Vec<&TensorInfo> = outs.iter().map(|o| o.info).collect();
                        let mut out_data: Vec<&mut [u8]> = outs.iter_mut().map(|o| &mut *o.data).collect();
                        op.invoke(&mut dev.lock(), batches, &in_data, &mut out_data, &infos)
                    }
                    _ => ops::eval(node, &ins, &mut outs),
                }))
                .unwrap_or_else(|p| Err(Error(panic_text(&p))))
            };
            for (o, b) in taken {
                self.bufs[o] = b;
            }
            result.map_err(|e| Error(format!("{} (op {n}): {e}", node.name)))?;
        }
        Ok(())
    }
}

/// These need a stick and the recorded official outputs (tests/oracle/dump_pycoral.py):
/// `cargo test --release -- --ignored`
#[cfg(test)]
mod device_tests {
    use super::*;
    use std::time::Duration;

    const ROOT: &str = env!("CARGO_MANIFEST_DIR");

    #[test]
    #[ignore]
    fn a_failed_run_is_recovered_by_the_next_one() {
        let model = format!("{ROOT}/../test_data/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite");
        let input = std::fs::read(format!("{ROOT}/target/oracle/classify_photo.in.bin")).unwrap();
        let want = std::fs::read(format!("{ROOT}/target/oracle/classify_photo.out0.bin")).unwrap();
        let mut it = Interpreter::from_file(model, &Options::default()).unwrap();
        it.set_input(0, &input).unwrap();
        let dev = it.device().unwrap().clone();
        for _ in 0..3 {
            // the transfers time out midway: the chip is left with data in flight
            dev.lock().timeout = Duration::from_millis(1); // libusb counts in ms, and 0 means no timeout
            let err = it.invoke().unwrap_err();
            assert!(err.0.contains("timed out"), "{err}");
            dev.lock().timeout = Duration::from_secs(6);
            it.invoke().unwrap();
            assert!(it.output(0).unwrap() == want.as_slice(), "the run after the failure gave other bytes");
        }
    }
}
