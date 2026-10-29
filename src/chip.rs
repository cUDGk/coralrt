//! The Coral USB Accelerator ("beagle"): firmware upload, chip bring-up and running DarwiNN executables.
//! Ports of libedgetpu driver/usb (usb_driver.cc, usb_dfu_commands.cc) and driver/beagle
//! (beagle_top_level_handler.cc), by way of tinycoral's coral/device.py, dfu.py and runtime.py.

use crate::exe::{Desc, Executable, Hint};
use crate::usb::{self, Backend, Handle, InStream};
use crate::{Error, Result};
use std::collections::HashMap;
use std::time::{Duration, Instant};

pub const DFU: (u16, u16) = (0x1a6e, 0x089a);
pub const APP: (u16, u16) = (0x18d1, 0x9302);
/// libedgetpu driver/usb/apex_latest_single_ep.bin (Apache 2.0, see firmware/LICENSE.libedgetpu)
static FIRMWARE: &[u8] = include_bytes!("../firmware/apex_latest_single_ep.bin");

const TAG_INSTRUCTIONS: u32 = 0;
const TAG_INPUT: u32 = 1;
const TAG_PARAMETERS: u32 = 2;
const TAG_INT0: u8 = 4;
const EP_OUT: u8 = 0x01;
const EP_DATA_IN: u8 = 0x81;
const EP_EVENT_IN: u8 = 0x82;
const TIMEOUT: Duration = Duration::from_secs(6);

/// register addresses (libedgetpu driver/config/beagle/beagle_csr_offsets.h)
mod reg {
    pub const GCBB_CREDIT0: u32 = 0x1907c;
    pub const OMC0_00: u32 = 0x1a000;
    pub const OMC0_D4: u32 = 0x1a0d4;
    pub const OMC0_D8: u32 = 0x1a0d8;
    pub const SCU_CTRL_0: u32 = 0x1a30c;
    pub const SCU_CTRL_2: u32 = 0x1a314;
    pub const SCU_CTRL_3: u32 = 0x1a318;
    pub const SCU_CTR_7: u32 = 0x1a33c;
    pub const SLV_ABM_EN: u32 = 0x1a500;
    pub const SLV_ERR_RESP_ISR_MASK: u32 = 0x1a558;
    pub const MST_ABM_EN: u32 = 0x1a600;
    pub const MST_ERR_RESP_ISR_MASK: u32 = 0x1a658;
    pub const RAMBIST_CTRL_1: u32 = 0x1a704;
    pub const DEEP_SLEEP: u32 = 0x40020;
    pub const SCALAR_CORE_RUN_CONTROL: u32 = 0x44018;
    pub const FATAL_ERR_INT_CONTROL: u32 = 0x486c0;
    pub const TILECONFIG0: u32 = 0x48788;
    pub const IDLE_REGISTER: u32 = 0x4a000;
    pub const OUTFEED_CHUNK_LENGTH: u32 = 0x4c058;
    pub const DESCR_EP: u32 = 0x4c148;
    pub const MULTI_BO_EP: u32 = 0x4c160;
    /// scalarCore, avDataPop, parameterPop, infeed, outfeed
    pub const SC_RUN_CONTROLS: [u32; 5] = [0x44018, 0x44158, 0x44198, 0x441d8, 0x44218];
    /// op, wideToNarrow, narrowToWide, ringBusConsumer0/1, ringBusProducer, meshBus0..3
    pub const TILE_RUN_CONTROLS: [u32; 10] = [0x400c0, 0x40110, 0x40150, 0x40190, 0x401d0, 0x40210, 0x40250, 0x40298, 0x402e0, 0x40328];
    /// top_level_int_0..3_control
    pub const TOP_LEVEL_INT_CONTROLS: [u32; 4] = [0x4c070, 0x4c080, 0x4c090, 0x4c0a0];
}

/// libedgetpu's performance setting: the chip clock (the official "throttled" runtime is High, "direct" is Max)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Performance {
    Low,
    Medium,
    High,
    Max,
}

impl Performance {
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s.to_ascii_lowercase().as_str() {
            "low" => Performance::Low,
            "medium" => Performance::Medium,
            "high" => Performance::High,
            "max" => Performance::Max,
            _ => return Err(Error(format!("unknown performance {s:?} (low, medium, high, max)"))),
        })
    }

    /// scu_ctrl_3 (gcb clock divider, axi at 125 MHz, 8051 at 250 MHz) per BeagleTopLevelHandler::QuitReset over USB:
    /// the gcb clock is 500 MHz >> divider
    fn scu_ctrl_3(self) -> (u32, u32, u32) {
        match self {
            Performance::Low => (3, 1, 1),
            Performance::Medium => (2, 1, 0),
            Performance::High => (1, 1, 0),
            Performance::Max => (0, 0, 0),
        }
    }
}

fn bits(v: u32, lo: u32, n: u32) -> u32 {
    (v >> lo) & ((1 << n) - 1)
}

fn setbits(v: u32, lo: u32, n: u32, x: u32) -> u32 {
    let m = ((1u32 << n) - 1) << lo;
    (v & !m) | ((x << lo) & m)
}

/// the sticks on the bus: (path, is it still in the DFU bootloader)
pub fn list() -> Result<Vec<(String, bool)>> {
    Ok(usb::list(&[DFU, APP])?.into_iter().map(|f| (f.path, (f.vendor, f.product) == DFU)).collect())
}

/// Load the firmware over USB DFU 1.1 (usb_dfu_commands.cc), then reset the stick so it comes back as 18d1:9302.
fn load_firmware(path: &str) -> Result<()> {
    const DNLOAD: u8 = 1;
    const UPLOAD: u8 = 2;
    const GETSTATUS: u8 = 3;
    // through UsbDk on Windows: the reset at the end needs it (see usb.rs)
    let mut dev = Handle::open(Backend::UsbDk, DFU.0, DFU.1, path)?;
    let cfg = dev.control_in(0x80, 6, 0x0200, 0, 512, 1000)?;
    // the DFU interface (class 0xFE, subclass 1, no endpoints) and the transfer size of the DFU functional descriptor
    let (mut iface, mut xfer, mut i) = (None, None, 0);
    while i + 1 < cfg.len() && cfg[i] != 0 {
        let (len, ty) = (cfg[i] as usize, cfg[i + 1]);
        if ty == 4 && i + 6 < cfg.len() && cfg[i + 4] == 0 && cfg[i + 5] == 0xFE && cfg[i + 6] == 1 {
            iface = Some(cfg[i + 2]);
        }
        if ty == 0x21 && i + 6 < cfg.len() {
            xfer = Some(u16::from_le_bytes([cfg[i + 5], cfg[i + 6]]) as usize);
        }
        i += len;
    }
    let (Some(iface), Some(xfer)) = (iface, xfer) else { return Err(Error("the bootloader has no DFU interface".into())) };
    dev.claim(iface as i32)?;
    let blocks = FIRMWARE.len().div_ceil(xfer);
    for blk in 0..=blocks {
        // the last block is empty: it ends the download
        let chunk = &FIRMWARE[(blk * xfer).min(FIRMWARE.len())..((blk + 1) * xfer).min(FIRMWARE.len())];
        dev.control_out(0x21, DNLOAD, blk as u16, iface as u16, chunk, 1000)?;
        let st = dev.control_in(0xA1, GETSTATUS, 0, iface as u16, 6, 1000)?;
        let want = if blk < blocks { 5 } else { 2 }; // dfuDNLOAD-IDLE, then dfuIDLE
        if st.len() < 6 || st[0] != 0 || st[4] != want {
            return Err(Error(format!("firmware block {blk}: DFU status {st:02x?}")));
        }
    }
    let mut back = Vec::new();
    for blk in 0.. {
        let chunk = dev.control_in(0xA1, UPLOAD, blk, iface as u16, xfer, 1000)?;
        back.extend_from_slice(&chunk);
        if chunk.len() < xfer {
            break;
        }
    }
    if back.get(..FIRMWARE.len()) != Some(FIRMWARE) {
        return Err(Error("firmware readback mismatch".into()));
    }
    // the stick drops off the bus while resetting, so this may report an error after it took effect
    let _ = dev.reset();
    Ok(())
}

pub struct EdgeTpu {
    usb: Handle,
    pub path: String,
    pub performance: Performance,
    pub usb_speed: i32,
    /// parameter caching token of the executable whose parameters are in the chip's SRAM
    cached_token: u64,
    /// a transfer failed: drain the pipes and re-initialize the chip before the next run
    needs_recovery: bool,
}

impl EdgeTpu {
    /// Open the stick at `path` (from `list`), or the first one; load the firmware first if it is in the bootloader.
    pub fn open(path: Option<&str>, performance: Performance) -> Result<Self> {
        let all = list()?;
        let (path, in_dfu) = match path {
            Some(p) => all.into_iter().find(|(q, _)| q == p).ok_or_else(|| Error(format!("no Edge TPU at usb path {p}")))?,
            None => all.into_iter().next().ok_or_else(|| Error("no Edge TPU found".into()))?,
        };
        if in_dfu {
            load_firmware(&path)?;
        }
        // after the firmware upload (or a just closed handle of another process) the stick may need a moment
        let start = Instant::now();
        let usb = loop {
            let tried = Handle::open(Backend::Default, APP.0, APP.1, &path).and_then(|mut h| h.claim(0).map(|_| h));
            match tried {
                Ok(h) => break h,
                Err(e) if start.elapsed() > Duration::from_secs(10) => {
                    return Err(Error(format!("{e}{}", if in_dfu { " (after the firmware upload: replug the stick)" } else { "" })));
                }
                Err(_) => std::thread::sleep(Duration::from_millis(100)),
            }
        };
        let usb_speed = usb.speed();
        if usb.control_in(0xC0, 1, (reg::SCU_CTRL_0 & 0xffff) as u16, (reg::SCU_CTRL_0 >> 16) as u16, 4, 500).is_err() {
            return Err(Error("the Edge TPU does not answer: replug it".into()));
        }
        let mut tpu = EdgeTpu { usb, path, performance, usb_speed, cached_token: 0, needs_recovery: false };
        tpu.init()?;
        Ok(tpu)
    }

    // *** registers: vendor control requests, bRequest 0 = 64 bit, 1 = 32 bit, the address split over wValue / wIndex ***
    fn read32(&self, a: u32) -> Result<u32> {
        let b = self.usb.control_in(0xC0, 1, (a & 0xffff) as u16, (a >> 16) as u16, 4, 1000)?;
        Ok(u32::from_le_bytes(b.get(..4).ok_or_else(|| Error(format!("short read of {a:#x}")))?.try_into().unwrap()))
    }
    fn write32(&self, a: u32, v: u32) -> Result<()> {
        self.usb.control_out(0x40, 1, (a & 0xffff) as u16, (a >> 16) as u16, &v.to_le_bytes(), 1000)
    }
    fn read64(&self, a: u32) -> Result<u64> {
        let b = self.usb.control_in(0xC0, 0, (a & 0xffff) as u16, (a >> 16) as u16, 8, 1000)?;
        Ok(u64::from_le_bytes(b.get(..8).ok_or_else(|| Error(format!("short read of {a:#x}")))?.try_into().unwrap()))
    }
    fn write64(&self, a: u32, v: u64) -> Result<()> {
        self.usb.control_out(0x40, 0, (a & 0xffff) as u16, (a >> 16) as u16, &v.to_le_bytes(), 1000)
    }
    fn poll(&self, what: &str, mut read: impl FnMut() -> Result<u64>, want: u64, mask: u64) -> Result<()> {
        let start = Instant::now();
        loop {
            let v = read()?;
            if v & mask == want & mask {
                return Ok(());
            }
            if start.elapsed() > Duration::from_secs(1) {
                return Err(Error(format!("chip init: {what} is {v:#x}, waited for {want:#x}")));
            }
        }
    }

    /// UsbDriver::DoOpen in single bulk-out endpoint mode
    fn init(&mut self) -> Result<()> {
        // BeagleTopLevelHandler::Open: phy modes, and whether the gcb is hardware clock gated
        self.write32(reg::SCU_CTRL_0, setbits(setbits(self.read32(reg::SCU_CTRL_0)?, 8, 3, 0), 11, 3, 0))?;
        let gated = bits(self.read32(reg::SCU_CTRL_2)?, 18, 2) == 1;
        if gated {
            // DisableHardwareClockGate: force the clock on
            self.write32(reg::SCU_CTRL_2, setbits(self.read32(reg::SCU_CTRL_2)?, 18, 2, 2))?;
        }
        // EnableReset: rg_force_sleep = 3, wait for cur_pwr_state == 2, pulse gcbb_credit0
        let s3 = self.read32(reg::SCU_CTRL_3)?;
        if bits(s3, 22, 2) != 3 {
            self.write32(reg::SCU_CTRL_3, setbits(s3, 22, 2, 3))?;
            self.poll("scu_ctrl_3", || self.read32(reg::SCU_CTRL_3).map(u64::from), 2 << 8, 3 << 8)?;
            self.write32(reg::GCBB_CREDIT0, 0xF)?;
            self.write32(reg::GCBB_CREDIT0, 0)?;
        }
        // QuitReset: rg_force_sleep = 2 and the clocks, wait for cur_pwr_state == 0
        let (clkdiv, axi125, u8051_250) = self.performance.scu_ctrl_3();
        let mut s3 = setbits(self.read32(reg::SCU_CTRL_3)?, 22, 2, 2);
        s3 = setbits(setbits(setbits(s3, 28, 2, clkdiv), 30, 1, axi125), 31, 1, u8051_250);
        self.write32(reg::SCU_CTRL_3, s3)?;
        self.poll("scu_ctrl_3", || self.read32(reg::SCU_CTRL_3).map(u64::from), 0, 3 << 8)?;
        self.poll("scalarCoreRunControl", || self.read64(reg::SCALAR_CORE_RUN_CONTROL), 0, !0)?;
        self.write64(reg::IDLE_REGISTER, 1)?;
        self.write64(reg::TILECONFIG0, 0x7f)?; // broadcast to all tiles
        self.poll("tileconfig0", || self.read64(reg::TILECONFIG0), 0x7f, !0)?;
        self.write64(reg::DEEP_SLEEP, (30 << 8) | 2)?; // to_wake_delay = 30, to_sleep_delay = 2
        if !gated {
            // EnableHardwareClockGate (libedgetpu leaves a stick that came up gated forced on)
            self.write32(reg::SCU_CTRL_2, setbits(self.read32(reg::SCU_CTRL_2)?, 18, 2, 1))?;
        }
        self.write64(reg::DESCR_EP, 0xF0)?; // only the scalar core's host interrupts on the event endpoint
        self.write64(reg::MULTI_BO_EP, 0)?; // single bulk-out endpoint
        // USB 2 high speed: 256 byte chunks (libedgetpu's workaround for b/73181174), else 1 KB
        self.write64(reg::OUTFEED_CHUNK_LENGTH, if self.usb_speed == 3 { 0x20 } else { 0x80 })?;
        // RunControl::kMoveToRun
        for r in reg::SC_RUN_CONTROLS {
            self.write64(r, 1)?;
        }
        self.write64(reg::TILECONFIG0, 0x7f)?;
        for r in reg::TILE_RUN_CONTROLS {
            self.write64(r, 1)?;
        }
        // RegisterAndEnableAllInterrupts
        self.write64(reg::FATAL_ERR_INT_CONTROL, 1)?;
        for r in reg::TOP_LEVEL_INT_CONTROLS {
            self.write64(r, 1)?;
        }
        // BeagleTopLevelInterruptManager::DoEnableInterrupts: thermal warning, mbist, pcie error, thermal shutdown
        self.write32(reg::OMC0_D4, self.read32(reg::OMC0_D4)? | 0x8000_0001)?;
        self.write32(reg::RAMBIST_CTRL_1, 0x7f)?;
        self.write32(reg::SCU_CTR_7, 0x3f)?;
        self.write32(reg::SLV_ABM_EN, 1)?;
        self.write32(reg::MST_ABM_EN, 1)?;
        self.write32(reg::SLV_ERR_RESP_ISR_MASK, 3)?;
        self.write32(reg::MST_ERR_RESP_ISR_MASK, 3)?;
        self.write32(reg::OMC0_D8, self.read32(reg::OMC0_D8)? | 0x8000_0000)?;
        self.cached_token = 0;
        Ok(())
    }

    /// efuse revision of the chip (a cheap register read, to check the stick answers)
    pub fn efuse_revision(&self) -> Result<u32> {
        Ok(bits(self.read32(reg::OMC0_00)?, 24, 8))
    }

    /// after a failed transfer: drain what the chip still sends and re-initialize it (the usb firmware keeps running)
    fn recover(&mut self) -> Result<()> {
        for ep in [EP_DATA_IN, EP_EVENT_IN] {
            let mut s = [InStream::new(ep, vec![1 << 16])];
            while self.usb.bulk_batch(&[], &mut s, Duration::from_millis(50)).is_ok() && !s[0].got.is_empty() {}
        }
        self.init()?;
        self.needs_recovery = false;
        Ok(())
    }

    /// Run one executable. `inputs` maps input layer names to their (padded) bytes; returns the bytes of each output layer
    /// (padded, as the chip lays them out). Parameters of a caching executable stay in the chip until another token replaces them.
    pub fn run(&mut self, exe: &Executable, inputs: &HashMap<&str, Vec<u8>>, outputs: &mut HashMap<String, Vec<u8>>) -> Result<()> {
        if self.needs_recovery {
            self.recover()?;
        }
        let r = self.run_hints(exe, inputs, outputs);
        if r.is_err() {
            self.needs_recovery = true;
            self.cached_token = 0;
        }
        r
    }

    /// Parameter caching: run the caching executables when another model's parameters are in the chip.
    pub fn ensure_cached(&mut self, token: u64, caching: &[&Executable]) -> Result<()> {
        if caching.is_empty() || (token != 0 && token == self.cached_token) {
            return Ok(());
        }
        let mut sink = HashMap::new();
        for exe in caching {
            self.run(exe, &HashMap::new(), &mut sink)?;
        }
        self.cached_token = token;
        Ok(())
    }

    /// Follow the compiler's DMA hints in segments: the OUTs (instructions, inputs, parameters), then the INs
    /// (outputs, interrupts) due before the next OUT. Within a segment every IN is queued before the OUTs; a segment's OUTs
    /// only start once all earlier INs completed: in single endpoint mode a bulk OUT can block a pending bulk IN until the
    /// device deadlocks (libedgetpu usb_driver.cc enforces the same order). A fence closes the segment.
    fn run_hints(&mut self, exe: &Executable, inputs: &HashMap<&str, Vec<u8>>, outputs: &mut HashMap<String, Vec<u8>>) -> Result<()> {
        struct Segment<'h> {
            outs: Vec<Vec<u8>>,
            data: Vec<&'h Hint>,
            interrupts: Vec<u8>,
        }
        let new_seg = || Segment { outs: Vec::new(), data: Vec::new(), interrupts: Vec::new() };
        let mut segs = vec![new_seg()];
        let packet = |tag: u32, data: &[u8]| {
            let mut p = Vec::with_capacity(8 + data.len());
            p.extend_from_slice(&(data.len() as u32).to_le_bytes());
            p.extend_from_slice(&tag.to_le_bytes());
            p.extend_from_slice(data);
            p
        };
        for h in &exe.hints {
            let seg = segs.last_mut().unwrap();
            let has_ins = !seg.data.is_empty() || !seg.interrupts.is_empty();
            match h {
                Hint::Instruction(_) | Hint::Dma { outfeed: false, .. } if has_ins => segs.push(new_seg()),
                Hint::Fence if has_ins || !seg.outs.is_empty() => segs.push(new_seg()),
                _ => {}
            }
            let seg = segs.last_mut().unwrap();
            match h {
                Hint::Instruction(i) => {
                    let bs = exe.bitstreams.get(*i).ok_or_else(|| Error(format!("instruction chunk {i} missing")))?;
                    seg.outs.push(packet(TAG_INSTRUCTIONS, bs));
                }
                Hint::Dma { outfeed: false, desc, name, offset, size } => {
                    let (src, tag) = match desc {
                        Desc::Input => (inputs.get(name.as_str()).ok_or_else(|| Error(format!("no input for layer {name:?}")))?, TAG_INPUT),
                        Desc::Parameter => (&exe.parameters, TAG_PARAMETERS),
                        d => return Err(Error(format!("{d:?} DMAs to the device are not supported"))),
                    };
                    let data = src.get(*offset..offset + size).ok_or_else(|| {
                        Error(format!("{desc:?} {name:?} has {} bytes, the program reads {size} at {offset}", src.len()))
                    })?;
                    seg.outs.push(packet(tag, data));
                }
                Hint::Dma { outfeed: true, desc: Desc::Output, .. } => seg.data.push(h),
                Hint::Dma { outfeed: true, desc, .. } => return Err(Error(format!("{desc:?} DMAs from the device are not supported"))),
                Hint::Interrupt(n) => seg.interrupts.push(*n),
                Hint::Fence => {}
            }
        }
        // USB 2 high speed: bulk INs of at most 256 bytes (matches the outfeed chunk set in init)
        let cap = if self.usb_speed == 3 { 256 } else { usize::MAX };
        for seg in segs {
            let mut chunks = Vec::new();
            for h in &seg.data {
                if let Hint::Dma { size, .. } = h {
                    let mut left = *size;
                    while left > 0 {
                        chunks.push(left.min(cap));
                        left -= left.min(cap);
                    }
                }
            }
            let mut ins = vec![InStream::new(EP_DATA_IN, chunks), InStream::new(EP_EVENT_IN, vec![16; seg.interrupts.len()])];
            let outs: Vec<(u8, &[u8])> = seg.outs.iter().map(|p| (EP_OUT, p.as_slice())).collect();
            self.usb.bulk_batch(&outs, &mut ins, TIMEOUT)?;
            let mut at = 0;
            for h in &seg.data {
                if let Hint::Dma { name, offset, size, .. } = h {
                    let layer = exe.outputs.iter().find(|l| &l.name == name).ok_or_else(|| Error(format!("DMA to unknown output {name:?}")))?;
                    let buf = outputs.entry(name.clone()).or_insert_with(|| vec![0; layer.padded_size_bytes()]);
                    if buf.len() < offset + size {
                        buf.resize(offset + size, 0);
                    }
                    buf[*offset..offset + size].copy_from_slice(&ins[0].got[at..at + size]);
                    at += size;
                }
            }
            for (ev, want) in ins[1].got.chunks(16).zip(&seg.interrupts) {
                let tag = ev[12] & 0xF;
                if tag != TAG_INT0 + want {
                    return Err(Error(format!("expected interrupt {want}, the chip sent tag {tag}")));
                }
            }
        }
        Ok(())
    }
}
