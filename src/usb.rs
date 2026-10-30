//! libusb, loaded at run time (no link-time dependency): control transfers for the registers and
//! DFU, and batches of async bulk transfers for the data path.
//!
//! On Windows libusb talks to the stick through WinUSB (the official Edge TPU runtime binds it). Only the
//! firmware upload uses a second context on the UsbDk backend, like libedgetpu (libusb_options_windows.cc):
//! leaving the DFU bootloader needs a USB reset that WinUSB can't do without administrator rights. UsbDk
//! is not used for the rest because opening through it resets the port, and a reset makes a stick whose
//! chip is stuck mid-transfer drop off the bus until it is replugged (seen 2026-10-30, and in tinycoral).
//! The official runtime installs both UsbDk and the libusb-1.0.dll this loads.

use crate::{Error, Result};
use std::ffi::{c_char, c_int, c_long, c_uint, c_void, CStr};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

type Ptr = *mut c_void;

#[repr(C)]
#[derive(Default)]
pub struct DeviceDescriptor {
    pub length: u8,
    pub descriptor_type: u8,
    pub bcd_usb: u16,
    pub class: u8,
    pub subclass: u8,
    pub protocol: u8,
    pub max_packet_size0: u8,
    pub vendor: u16,
    pub product: u16,
    pub bcd_device: u16,
    pub manufacturer: u8,
    pub product_index: u8,
    pub serial: u8,
    pub num_configurations: u8,
}

#[repr(C)]
struct Transfer {
    dev_handle: Ptr,
    flags: u8,
    endpoint: u8,
    ty: u8,
    timeout: c_uint,
    status: c_int,
    length: c_int,
    actual_length: c_int,
    callback: extern "system" fn(*mut Transfer),
    user_data: Ptr,
    buffer: *mut u8,
    num_iso_packets: c_int,
}

#[repr(C)]
struct Timeval {
    sec: c_long,
    usec: c_long,
}

const TRANSFER_TYPE_BULK: u8 = 2;
const OPTION_USE_USBDK: c_int = 1;
const STATUS: [&str; 7] = ["completed", "error", "timed out", "cancelled", "stall", "no device", "overflow"];

macro_rules! libusb_fns {
    ($($name:ident: fn($($arg:ty),*) $(-> $ret:ty)?;)*) => {
        struct Lib {
            _lib: libloading::Library,
            set_option: unsafe extern "C" fn(Ptr, c_int, ...) -> c_int,
            $($name: unsafe extern "system" fn($($arg),*) $(-> $ret)?,)*
        }
        impl Lib {
            fn load() -> std::result::Result<Lib, String> {
                let name = std::env::var("CORALRT_LIBUSB").unwrap_or_else(|_| DEFAULT_LIBUSB.to_string());
                // SAFETY: libusb's initializers have no side effects beyond the library itself
                let lib = unsafe { libloading::Library::new(&name) }.map_err(|e| format!("cannot load {name}: {e}"))?;
                // SAFETY: the signatures match libusb.h (LIBUSB_CALL is the system ABI, set_option is variadic cdecl)
                unsafe {
                    Ok(Lib {
                        set_option: *lib.get(b"libusb_set_option\0").map_err(|e| e.to_string())?,
                        $($name: *lib.get(concat!("libusb_", stringify!($name), "\0").as_bytes()).map_err(|e| e.to_string())?,)*
                        _lib: lib,
                    })
                }
            }
        }
    };
}

#[cfg(windows)]
const DEFAULT_LIBUSB: &str = "libusb-1.0.dll";
#[cfg(target_os = "macos")]
const DEFAULT_LIBUSB: &str = "libusb-1.0.0.dylib";
#[cfg(all(unix, not(target_os = "macos")))]
const DEFAULT_LIBUSB: &str = "libusb-1.0.so.0";

libusb_fns! {
    init: fn(*mut Ptr) -> c_int;
    get_device_list: fn(Ptr, *mut *mut Ptr) -> isize;
    free_device_list: fn(*mut Ptr, c_int);
    get_device_descriptor: fn(Ptr, *mut DeviceDescriptor) -> c_int;
    get_bus_number: fn(Ptr) -> u8;
    get_port_numbers: fn(Ptr, *mut u8, c_int) -> c_int;
    open: fn(Ptr, *mut Ptr) -> c_int;
    close: fn(Ptr);
    get_device: fn(Ptr) -> Ptr;
    get_device_speed: fn(Ptr) -> c_int;
    claim_interface: fn(Ptr, c_int) -> c_int;
    release_interface: fn(Ptr, c_int) -> c_int;
    control_transfer: fn(Ptr, u8, u8, u16, u16, *mut u8, u16, c_uint) -> c_int;
    reset_device: fn(Ptr) -> c_int;
    alloc_transfer: fn(c_int) -> *mut Transfer;
    free_transfer: fn(*mut Transfer);
    submit_transfer: fn(*mut Transfer) -> c_int;
    cancel_transfer: fn(*mut Transfer) -> c_int;
    handle_events_timeout_completed: fn(Ptr, *const Timeval, *mut c_int) -> c_int;
    strerror: fn(c_int) -> *const c_char;
}

fn lib() -> Result<&'static Lib> {
    static LIB: OnceLock<std::result::Result<Lib, String>> = OnceLock::new();
    LIB.get_or_init(Lib::load).as_ref().map_err(|e| Error(e.clone()))
}

/// the libusb backend of a context (see the module comment)
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Default,
    UsbDk,
}

struct Ctx {
    lib: &'static Lib,
    ctx: Ptr,
}

// SAFETY: a libusb context may be used from any thread
unsafe impl Send for Ctx {}
unsafe impl Sync for Ctx {}

fn ctx(backend: Backend) -> Result<&'static Ctx> {
    static CTX: [OnceLock<std::result::Result<Ctx, String>>; 2] = [OnceLock::new(), OnceLock::new()];
    CTX[backend as usize]
        .get_or_init(|| {
            let lib = lib().map_err(|e| e.0)?;
            let mut ctx: Ptr = std::ptr::null_mut();
            // SAFETY: init writes the new context to ctx
            let rc = unsafe { (lib.init)(&mut ctx) };
            if rc < 0 {
                return Err(format!("libusb_init failed ({rc})"));
            }
            if backend == Backend::UsbDk && cfg!(windows) {
                // fails when UsbDk is not installed: the context then uses WinUSB as well, the reset after the
                // firmware upload fails and opening the stick afterwards asks for a replug
                // SAFETY: valid context, the option takes no argument
                let _ = unsafe { (lib.set_option)(ctx, OPTION_USE_USBDK) };
            }
            Ok(Ctx { lib, ctx })
        })
        .as_ref()
        .map_err(|e| Error(e.clone()))
}

fn check(rc: c_int, what: &str) -> Result<c_int> {
    if rc >= 0 {
        return Ok(rc);
    }
    let msg = lib().ok().map_or(String::new(), |l| {
        // SAFETY: strerror returns a static string for any code
        unsafe { CStr::from_ptr((l.strerror)(rc)) }.to_string_lossy().into_owned()
    });
    Err(Error(format!("{what}: {msg} ({rc})")))
}

/// a USB device on the bus, by vendor/product id and "bus-port.port" path
#[derive(Clone, Debug)]
pub struct Found {
    pub vendor: u16,
    pub product: u16,
    pub path: String,
}

fn for_each_device<T>(c: &Ctx, mut f: impl FnMut(Ptr, &DeviceDescriptor, String) -> Option<T>) -> Result<Option<T>> {
    let mut list: *mut Ptr = std::ptr::null_mut();
    // SAFETY: the list is freed below; devices are only used while it is alive
    let n = unsafe { (c.lib.get_device_list)(c.ctx, &mut list) };
    check(n as c_int, "libusb_get_device_list")?;
    let mut out = None;
    for i in 0..n as usize {
        // SAFETY: i < n
        let dev = unsafe { *list.add(i) };
        let mut d = DeviceDescriptor::default();
        // SAFETY: d is a valid descriptor struct
        if unsafe { (c.lib.get_device_descriptor)(dev, &mut d) } < 0 {
            continue;
        }
        let mut ports = [0u8; 8];
        // SAFETY: ports has room for 8 entries
        let np = unsafe { (c.lib.get_port_numbers)(dev, ports.as_mut_ptr(), 8) }.max(0) as usize;
        let ports: Vec<String> = ports[..np].iter().map(|p| p.to_string()).collect();
        // SAFETY: valid device
        let path = format!("{}-{}", unsafe { (c.lib.get_bus_number)(dev) }, ports.join("."));
        if let Some(v) = f(dev, &d, path) {
            out = Some(v);
            break;
        }
    }
    // SAFETY: frees the list and unreferences devices we did not open
    unsafe { (c.lib.free_device_list)(list, 1) };
    Ok(out)
}

/// The devices with one of `ids`. The paths are those of the backend: UsbDk numbers a bus per hub and keeps only the
/// port on that hub, so the stick at "2-6.3" under WinUSB is at "3-3" under UsbDk.
pub fn list(backend: Backend, ids: &[(u16, u16)]) -> Result<Vec<Found>> {
    let mut found = Vec::new();
    for_each_device::<()>(ctx(backend)?, |_, d, path| {
        if ids.contains(&(d.vendor, d.product)) {
            found.push(Found { vendor: d.vendor, product: d.product, path });
        }
        None
    })?;
    Ok(found)
}

/// an opened device; interfaces claimed through it are released on drop
pub struct Handle {
    c: &'static Ctx,
    h: Ptr,
    claimed: Vec<c_int>,
}

// SAFETY: a libusb device handle may move between threads (the caller serializes its use)
unsafe impl Send for Handle {}

impl Handle {
    /// open the device at `path` (as returned by `list`)
    pub fn open(backend: Backend, vendor: u16, product: u16, path: &str) -> Result<Handle> {
        let c = ctx(backend)?;
        let h = for_each_device(c, |dev, d, p| {
            if d.vendor != vendor || d.product != product || p != path {
                return None;
            }
            let mut h: Ptr = std::ptr::null_mut();
            // SAFETY: open writes the handle, which holds its own reference to the device
            Some(check(unsafe { (c.lib.open)(dev, &mut h) }, "libusb_open").map(|_| h))
        })?;
        match h {
            Some(h) => Ok(Handle { c, h: h?, claimed: Vec::new() }),
            None => Err(Error(format!("usb device {vendor:04x}:{product:04x} at {path} not found"))),
        }
    }

    pub fn claim(&mut self, iface: c_int) -> Result<()> {
        // SAFETY: valid handle
        check(unsafe { (self.c.lib.claim_interface)(self.h, iface) }, "libusb_claim_interface")?;
        self.claimed.push(iface);
        Ok(())
    }

    /// 1 low, 2 full, 3 high, 4 super, 5 super+ (0 unknown)
    pub fn speed(&self) -> i32 {
        // SAFETY: valid handle
        unsafe { (self.c.lib.get_device_speed)((self.c.lib.get_device)(self.h)) }
    }

    pub fn control_in(&self, request_type: u8, request: u8, value: u16, index: u16, len: usize, timeout_ms: u32) -> Result<Vec<u8>> {
        let mut buf = vec![0u8; len];
        // SAFETY: buf holds len bytes
        let n = unsafe { (self.c.lib.control_transfer)(self.h, request_type, request, value, index, buf.as_mut_ptr(), len as u16, timeout_ms) };
        buf.truncate(check(n, "control transfer in")? as usize);
        Ok(buf)
    }

    pub fn control_out(&self, request_type: u8, request: u8, value: u16, index: u16, data: &[u8], timeout_ms: u32) -> Result<()> {
        let mut buf = data.to_vec();
        // SAFETY: buf holds data.len() bytes; libusb does not write to an OUT buffer
        let n = unsafe { (self.c.lib.control_transfer)(self.h, request_type, request, value, index, buf.as_mut_ptr(), buf.len() as u16, timeout_ms) };
        if check(n, "control transfer out")? as usize != data.len() {
            return Err(Error(format!("control transfer out: sent {n} of {} bytes", data.len())));
        }
        Ok(())
    }

    pub fn reset(&self) -> Result<()> {
        // SAFETY: valid handle
        check(unsafe { (self.c.lib.reset_device)(self.h) }, "libusb_reset_device").map(|_| ())
    }

    /// Runs a set of bulk transfers concurrently: every IN is queued before the OUTs (the device
    /// produces the INs while it consumes the OUTs). Each `InStream` is a byte stream on its endpoint:
    /// a short packet ends every device DMA, so one transfer can end early and the next one continues
    /// the stream; transfers are resubmitted until `want` bytes arrived.
    pub fn bulk_batch(&self, outs: &[(u8, &[u8])], ins: &mut [InStream], timeout: Duration) -> Result<()> {
        let mut b = Batch { c: self.c, h: self.h, slots: Vec::new() };
        let r = b.run(outs, ins, timeout);
        b.free();
        r
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        // SAFETY: valid handle; it is not used after this
        unsafe {
            for &i in &self.claimed {
                (self.c.lib.release_interface)(self.h, i);
            }
            (self.c.lib.close)(self.h);
        }
    }
}

/// bytes wanted from one IN endpoint, in transfers of the given sizes, and what arrived
pub struct InStream {
    pub ep: u8,
    pub want: usize,
    pub chunks: Vec<usize>,
    pub got: Vec<u8>,
}

impl InStream {
    pub fn new(ep: u8, chunks: Vec<usize>) -> Self {
        InStream { ep, want: chunks.iter().sum(), chunks, got: Vec::new() }
    }
}

struct Slot {
    t: *mut Transfer,
    done: Box<c_int>,
    buf: Vec<u8>,
    /// index of the InStream this IN feeds, None for an OUT
    stream: Option<usize>,
    order: usize,
}

struct Batch<'c> {
    c: &'c Ctx,
    h: Ptr,
    slots: Vec<Slot>,
}

extern "system" fn on_done(t: *mut Transfer) {
    // SAFETY: user_data points to the slot's boxed flag, alive until the slot is freed
    unsafe { *((*t).user_data as *mut c_int) = 1 };
}

impl Batch<'_> {
    fn submit(&mut self, ep: u8, buf: Vec<u8>, len: usize, stream: Option<usize>, order: usize, timeout_ms: u32) -> Result<()> {
        // SAFETY: a fresh transfer without iso packets
        let t = unsafe { (self.c.lib.alloc_transfer)(0) };
        if t.is_null() {
            return Err(Error("libusb_alloc_transfer failed".into()));
        }
        let mut slot = Slot { t, done: Box::new(0), buf, stream, order };
        // SAFETY: t is a valid transfer; the buffer and the flag outlive it (both owned by the slot)
        unsafe {
            *t = Transfer {
                dev_handle: self.h,
                flags: 0,
                endpoint: ep,
                ty: TRANSFER_TYPE_BULK,
                timeout: timeout_ms,
                status: 0,
                length: len as c_int,
                actual_length: 0,
                callback: on_done,
                user_data: &mut *slot.done as *mut c_int as Ptr,
                buffer: slot.buf.as_mut_ptr(),
                num_iso_packets: 0,
            };
        }
        // SAFETY: the transfer is fully initialized
        let rc = unsafe { (self.c.lib.submit_transfer)(t) };
        if rc < 0 {
            *slot.done = 1;
        }
        self.slots.push(slot);
        check(rc, "libusb_submit_transfer").map(|_| ())
    }

    fn pending(&self) -> bool {
        self.slots.iter().any(|s| *s.done == 0)
    }

    fn handle_events(&self, max: Duration) {
        let tv = Timeval { sec: max.as_secs() as c_long, usec: max.subsec_micros() as c_long };
        let mut completed: c_int = 0;
        // SAFETY: valid context and timeval; the callbacks only touch the slots' flags
        unsafe { (self.c.lib.handle_events_timeout_completed)(self.c.ctx, &tv, &mut completed) };
    }

    fn run(&mut self, outs: &[(u8, &[u8])], ins: &mut [InStream], timeout: Duration) -> Result<()> {
        let tmo = timeout.as_millis() as u32;
        let mut order = 0;
        for (i, s) in ins.iter().enumerate() {
            for &n in &s.chunks {
                self.submit(s.ep, vec![0; n], n, Some(i), order, tmo)?;
                order += 1;
            }
        }
        for &(ep, data) in outs {
            self.submit(ep, data.to_vec(), data.len(), None, 0, tmo)?;
        }
        let deadline = Instant::now() + timeout;
        // SAFETY (status reads): only of completed transfers
        let failed = |s: &Slot| *s.done != 0 && unsafe { (*s.t).status } != 0;
        loop {
            // one transfer failed: cancel the rest before anything else touches the device, a bulk OUT
            // the chip can't drain otherwise blocks the usb firmware (and the control transfers) for good
            if self.slots.iter().any(failed) {
                self.cancel_all();
                break;
            }
            // a stream whose transfers all ended short gets the rest requested right away: the chip sends the rest
            // only once an IN is pending, and until then the OUTs behind it can't drain (deadlock)
            for (i, s) in ins.iter().enumerate() {
                let mine = || self.slots.iter().filter(move |x| x.stream == Some(i));
                if mine().all(|x| *x.done != 0) {
                    // SAFETY: completed transfers
                    let got: usize = mine().map(|x| unsafe { (*x.t).actual_length } as usize).sum();
                    if got < s.want {
                        order += 1;
                        self.submit(s.ep, vec![0; s.want - got], s.want - got, Some(i), order, tmo)?;
                    }
                }
            }
            if !self.pending() {
                break;
            }
            if Instant::now() > deadline {
                // SAFETY: reads of the transfers' public fields
                let stuck: Vec<String> =
                    self.slots.iter().filter(|s| *s.done == 0).map(|s| unsafe { format!("{:#04x}:{}B", (*s.t).endpoint, (*s.t).length) }).collect();
                self.cancel_all();
                return Err(Error(format!("usb transfers timed out, waiting for {}", stuck.join(" "))));
            }
            self.handle_events(Duration::from_millis(100));
        }
        // report the first real failure, not the transfers cancelled because of it
        if let Some(bad) = self.slots.iter().filter(|s| failed(s)).min_by_key(|s| unsafe { ((*s.t).status == 3) as u8 }) {
            let (st, ep) = unsafe { ((*bad.t).status, (*bad.t).endpoint) };
            return Err(Error(format!("usb transfer on endpoint {ep:#04x} {}", STATUS.get(st as usize).unwrap_or(&"failed"))));
        }
        for (i, s) in ins.iter_mut().enumerate() {
            let mut mine: Vec<&Slot> = self.slots.iter().filter(|x| x.stream == Some(i)).collect();
            mine.sort_by_key(|x| x.order);
            s.got.clear();
            for x in mine {
                // SAFETY: completed transfer
                let n = unsafe { (*x.t).actual_length } as usize;
                s.got.extend_from_slice(&x.buf[..n]);
            }
        }
        for s in &self.slots {
            // SAFETY: completed transfer
            let (len, got) = unsafe { ((*s.t).length, (*s.t).actual_length) };
            if s.stream.is_none() && len != got {
                return Err(Error(format!("usb transfer out: sent {got} of {len} bytes")));
            }
        }
        Ok(())
    }

    /// cancel what is in flight and wait for libusb to hand every transfer back
    fn cancel_all(&mut self) {
        for s in &self.slots {
            if *s.done == 0 {
                // SAFETY: submitted transfer
                unsafe { (self.c.lib.cancel_transfer)(s.t) };
            }
        }
        let until = Instant::now() + Duration::from_secs(5);
        while self.pending() && Instant::now() < until {
            self.handle_events(Duration::from_millis(50));
        }
    }

    fn free(&mut self) {
        if self.pending() {
            self.cancel_all();
        }
        for s in self.slots.drain(..) {
            if *s.done != 0 {
                // SAFETY: the transfer is no longer in flight
                unsafe { (self.c.lib.free_transfer)(s.t) };
            } else {
                // libusb still owns it (the device stopped answering a cancel): leak the transfer and its buffer
                // rather than free memory the driver may still write to
                std::mem::forget(s);
            }
        }
    }
}
