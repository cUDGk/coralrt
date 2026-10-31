//! coralrt runs TensorFlow Lite models, and Coral Edge TPU models (`*_edgetpu.tflite`) on the Coral USB Accelerator,
//! without libedgetpu, tflite_runtime or pycoral: a USB driver for the stick, the `edgetpu-custom-op`, and the ops of
//! TensorFlow Lite 2.5, computing the same bytes as pycoral 2.0 (tflite_runtime 2.5 + libedgetpu).
//! Rust uses this crate directly; every other language goes through the C ABI in `capi` (include/coralrt.h).
//!
//! Sources ported: TensorFlow Lite 2.5, libedgetpu, gemmlowp, ruy, the Microsoft STL, Eigen, farmhash and tinycoral;
//! NOTICE says which file comes from which, and under what license.

mod capi;
mod chip;
mod edgetpu;
pub mod exe;
mod fb;
mod flex;
mod interp;
mod ops;
pub mod tensor;
pub mod tflite;
mod usb;

pub use chip::Performance;
pub use interp::DetectionSort;
pub use tensor::{TType, Tensor};
pub use tflite::Signature;

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

pub struct Options<'a> {
    /// which stick (see `Device::open`); used when the model has an Edge TPU op
    pub device: Option<&'a str>,
    /// the clock; default: the CORALRT_PERFORMANCE environment variable, else High (the official "throttled" runtime)
    pub performance: Option<Performance>,
    /// how TFLite_Detection_PostProcess orders boxes of equal score; default: as the official runtime (TFLite 2.5)
    pub detection_sort: DetectionSort,
}

impl Default for Options<'_> {
    fn default() -> Self {
        Options { device: None, performance: None, detection_sort: DetectionSort::Tflite25 }
    }
}

/// A model with its tensors, like tflite's Interpreter: write the inputs, `invoke`, read the outputs. Tensor indices
/// are those of the main subgraph.
pub struct Interpreter {
    graphs: Vec<Option<interp::Graph>>,
    signatures: Vec<Signature>,
    resources: std::collections::HashMap<i32, Box<dyn std::any::Any + Send>>,
    settings: interp::Settings,
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
        let uses_tpu = model.subgraphs.iter().flat_map(|g| &g.ops).any(|o| o.kernel.edgetpu().is_some());
        let device = if uses_tpu {
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
        let mut it = Interpreter {
            graphs: model.subgraphs.into_iter().map(|g| Some(g.into())).collect(),
            signatures: model.signatures,
            resources: Default::default(),
            settings: interp::Settings { detection_sort: options.detection_sort },
            device,
        };
        it.allocate_tensors()?;
        Ok(it)
    }

    /// share an already open stick instead of opening one from `Options`
    pub fn from_bytes_on(model: &[u8], device: Device) -> Result<Self> {
        let path = device.path();
        Self::from_bytes(model, &Options { device: Some(&format!("usb:{path}")), performance: Some(device.performance()), ..Default::default() })
    }

    pub fn device(&self) -> Option<&Device> {
        self.device.as_ref()
    }

    fn main(&self) -> &interp::Graph {
        self.graphs[0].as_ref().expect("the main subgraph is in place between calls")
    }

    fn main_mut(&mut self) -> &mut interp::Graph {
        self.graphs[0].as_mut().expect("the main subgraph is in place between calls")
    }

    /// Run `f` on the main subgraph with the environment its ops see (the subgraph is taken out meanwhile).
    fn with_main<T>(&mut self, f: impl FnOnce(&mut interp::Graph, &mut interp::Env) -> Result<T>) -> Result<T> {
        let mut g = self.graphs[0].take().expect("the main subgraph is in place between calls");
        let mut env = interp::Env { graphs: &mut self.graphs, resources: &mut self.resources, device: self.device.as_ref(), settings: &self.settings };
        let r = f(&mut g, &mut env);
        self.graphs[0] = Some(g);
        r
    }

    /// Work out every tensor's shape from the inputs' (after `resize_input`), and reset the variable tensors.
    pub fn allocate_tensors(&mut self) -> Result<()> {
        self.with_main(|g, env| {
            interp::prepare(g, env)?;
            interp::reset_variables(g);
            Ok(())
        })
    }

    /// Give input k another shape; `allocate_tensors` (or the next `invoke`) works out the rest. Tensors whose size
    /// changes move, so views into them must be taken again.
    pub fn resize_input(&mut self, k: usize, shape: &[usize]) -> Result<()> {
        let g = self.main_mut();
        let t = *g.inputs.get(k).ok_or_else(|| Error(format!("no input {k}")))?;
        if g.tensors[t].shape != shape {
            g.tensors[t].resize(shape.to_vec());
            g.prepared = false;
        }
        Ok(())
    }

    /// variable tensors (the state of RNN / LSTM ops) back to their initial values
    pub fn reset_variables(&mut self) {
        for g in self.graphs.iter_mut().flatten() {
            interp::reset_variables(g);
        }
    }

    pub fn inputs(&self) -> &[usize] {
        &self.main().inputs
    }

    pub fn outputs(&self) -> &[usize] {
        &self.main().outputs
    }

    /// the model's SignatureDefs (most Edge TPU models have none)
    pub fn signatures(&self) -> &[Signature] {
        &self.signatures
    }

    pub fn tensor_count(&self) -> usize {
        self.main().tensors.len()
    }

    pub fn tensor(&self, index: usize) -> &Tensor {
        &self.main().tensors[index]
    }

    pub fn data(&self, index: usize) -> &[u8] {
        self.main().tensors[index].data()
    }

    pub fn data_mut(&mut self, index: usize) -> &mut [u8] {
        self.main_mut().tensors[index].data_mut()
    }

    /// set a string tensor's strings (its shape: one dimension of their count unless given)
    pub fn set_strings(&mut self, index: usize, strings: &[&[u8]], shape: Option<&[usize]>) -> Result<()> {
        let t = &mut self.main_mut().tensors[index];
        if t.dtype != TType::Str {
            return Err(Error(format!("tensor {index} is {:?}, not a string tensor", t.dtype)));
        }
        let shape = shape.map_or(vec![strings.len()], |s| s.to_vec());
        if shape.iter().product::<usize>() != strings.len() {
            return Err(Error(format!("{} strings for shape {shape:?}", strings.len())));
        }
        t.set_strings(strings, shape);
        Ok(())
    }

    /// copy `data` into the k-th input (sizes must match)
    pub fn set_input(&mut self, k: usize, data: &[u8]) -> Result<()> {
        let g = self.main_mut();
        let t = *g.inputs.get(k).ok_or_else(|| Error(format!("no input {k}")))?;
        let b = g.tensors[t].data_mut();
        if b.len() != data.len() {
            return Err(Error(format!("input {k} takes {} bytes, got {}", b.len(), data.len())));
        }
        b.copy_from_slice(data);
        Ok(())
    }

    pub fn output(&self, k: usize) -> Result<&[u8]> {
        let g = self.main();
        let t = *g.outputs.get(k).ok_or_else(|| Error(format!("no output {k}")))?;
        Ok(g.tensors[t].data())
    }

    /// the ops of the main subgraph in execution order: (name, input tensors, output tensors); -1 is an absent input
    pub fn ops(&self) -> Vec<(&str, &[i32], &[i32])> {
        self.main().ops.iter().map(|o| (o.name.as_str(), &o.inputs[..], &o.outputs[..])).collect()
    }

    /// names of the ops, in execution order (e.g. to see what runs on the CPU)
    pub fn op_names(&self) -> Vec<&str> {
        self.main().ops.iter().map(|o| o.name.as_str()).collect()
    }

    /// the Edge TPU executables of each `edgetpu-custom-op`
    pub fn executables(&self) -> Vec<&exe::Executable> {
        self.graphs.iter().flatten().flat_map(|g| &g.ops).filter_map(|o| o.kernel.edgetpu()).flat_map(|e| &e.exes).collect()
    }

    pub fn invoke(&mut self) -> Result<()> {
        self.with_main(interp::invoke)
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
