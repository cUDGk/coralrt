//! The audio custom ops of TFLite 2.5's BuiltinOpResolver: AudioSpectrogram (audio_spectrogram.cc,
//! internal/spectrogram.cc) through Ooura's rdft, the same C as the official runtime, and Mfcc (mfcc.cc,
//! internal/mfcc*.cc), all in double as there.

use super::util::*;
use super::{Io, Kernel};
use crate::flex;
use crate::interp::Env;
use crate::tensor::TType;
use crate::Result;

unsafe extern "C" {
    fn rdft(n: i32, isgn: i32, a: *mut f64, ip: *mut i32, w: *mut f64);
}

/// flexbuffer map values as 2.5 reads them: a missing key is 0 / false
fn int_opt(custom: &[u8], key: &str) -> i64 {
    if custom.is_empty() {
        return 0;
    }
    flex::root(custom).get(key).map_or(0, |r| r.as_i64())
}

pub struct AudioSpectrogram {
    window_size: i32,
    stride: i32,
    magnitude_squared: bool,
    height: usize,
}

impl AudioSpectrogram {
    pub fn new(custom: &[u8]) -> AudioSpectrogram {
        AudioSpectrogram {
            window_size: int_opt(custom, "window_size") as i32,
            stride: int_opt(custom, "stride") as i32,
            magnitude_squared: int_opt(custom, "magnitude_squared") != 0,
            height: 0,
        }
    }
}

/// internal::Spectrogram after Initialize: a periodic Hann window, the FFT size the next power of two, and the queue
/// of samples, which carries over from one channel to the next as 2.5's single Spectrogram does
struct Spectrogram {
    window: Vec<f64>,
    step: usize,
    fft_length: usize,
    queue: Vec<f64>,
    samples_to_next_step: usize,
    fft: Vec<f64>,
    ip: Vec<i32>,
    w: Vec<f64>,
}

impl Spectrogram {
    fn new(window_length: i32, step: i32) -> Result<Spectrogram> {
        if window_length < 2 || step < 1 {
            return err("the spectrogram needs a window of 2 samples or more and a positive stride");
        }
        let pi = 1f64.atan() * 4.0;
        let n = window_length as usize;
        let window = (0..n).map(|i| 0.5 - 0.5 * ((2.0 * pi * i as f64) / n as f64).cos()).collect();
        let fft_length = n.next_power_of_two();
        let half = fft_length / 2;
        Ok(Spectrogram {
            window,
            step: step as usize,
            fft_length,
            queue: Vec::new(),
            samples_to_next_step: n,
            fft: vec![0.0; fft_length + 2],
            ip: vec![0; 2 + (half as f64).sqrt() as usize],
            w: vec![0.0; half],
        })
    }

    fn channels(&self) -> usize {
        1 + self.fft_length / 2
    }

    /// ComputeSquaredMagnitudeSpectrogram: one row per full window
    fn squared_magnitudes(&mut self, input: &[f32]) -> Vec<Vec<f32>> {
        let (mut rows, mut start) = (Vec::new(), 0);
        let wl = self.window.len();
        loop {
            // GetNextWindowOfSamples
            let remaining = input.len() - start;
            if self.samples_to_next_step > remaining {
                self.queue.extend(input[start..].iter().map(|&x| x as f64));
                self.samples_to_next_step -= remaining;
                return rows;
            }
            self.queue.extend(input[start..start + self.samples_to_next_step].iter().map(|&x| x as f64));
            start += self.samples_to_next_step;
            self.queue.drain(..self.queue.len() - wl);
            self.samples_to_next_step = self.step;
            // ProcessCoreFFT
            let n = self.fft_length;
            for j in 0..wl {
                self.fft[j] = self.queue[j] * self.window[j];
            }
            self.fft[wl..n].fill(0.0);
            // SAFETY: the buffers have the sizes rdft needs for n: a n (+2), ip 2 + sqrt(n/2), w n/2
            unsafe { rdft(n as i32, 1, self.fft.as_mut_ptr(), self.ip.as_mut_ptr(), self.w.as_mut_ptr()) };
            self.fft[n] = self.fft[1];
            self.fft[n + 1] = 0.0;
            self.fft[1] = 0.0;
            rows.push((0..self.channels()).map(|i| (self.fft[2 * i] * self.fft[2 * i] + self.fft[2 * i + 1] * self.fft[2 * i + 1]) as f32).collect());
        }
    }
}

impl Kernel for AudioSpectrogram {
    fn prepare(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        if io.n_in() != 1 || io.outs.len() != 1 {
            return err("AudioSpectrogram takes 1 input and 1 output");
        }
        let i = io.input(0)?;
        if i.shape.len() != 2 || i.dtype != TType::F32 || io.outs[0].dtype != TType::F32 {
            return err("AudioSpectrogram takes float32 [samples, channels]");
        }
        let s = Spectrogram::new(self.window_size, self.stride)?;
        let rest = i.shape[0] as i64 - self.window_size as i64;
        self.height = if rest < 0 { 0 } else { 1 + (rest / self.stride as i64) as usize };
        io.outs[0].resize(vec![i.shape[1], self.height, s.channels()]);
        Ok(())
    }

    fn eval(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        let i = io.input(0)?;
        let mut s = Spectrogram::new(self.window_size, self.stride)?;
        let (samples, channels, width) = (i.shape[0], i.shape[1], s.channels());
        let x = i.as_slice::<f32>();
        let out = io.outs[0].as_mut_slice::<f32>();
        for c in 0..channels {
            let column: Vec<f32> = (0..samples).map(|k| x[k * channels + c]).collect();
            let rows = s.squared_magnitudes(&column);
            if rows.len() != self.height {
                // the queue left by the previous channel shifts this channel's windows
                return err(format!("channel {c} gives {} windows, not {} (2.5 fails the same way)", rows.len(), self.height));
            }
            for (r, row) in rows.iter().enumerate() {
                let o = &mut out[(c * self.height + r) * width..][..width];
                for (d, &v) in o.iter_mut().zip(row) {
                    *d = if self.magnitude_squared { v } else { v.sqrt() };
                }
            }
        }
        Ok(())
    }
}

pub struct Mfcc {
    upper: f32,
    lower: f32,
    filterbank_channels: i32,
    dct_coefficients: i32,
}

impl Mfcc {
    pub fn new(custom: &[u8]) -> Mfcc {
        Mfcc {
            upper: int_opt(custom, "upper_frequency_limit") as f32,
            lower: int_opt(custom, "lower_frequency_limit") as f32,
            filterbank_channels: int_opt(custom, "filterbank_channel_count") as i32,
            dct_coefficients: int_opt(custom, "dct_coefficient_count") as i32,
        }
    }
}

fn freq_to_mel(f: f64) -> f64 {
    1127.0 * (f / 700.0).ln_1p()
}

/// MfccMelFilterbank and MfccDct after Initialize
struct MfccState {
    start: i32,
    end: i32,
    band_mapper: Vec<i32>,
    weights: Vec<f64>,
    channels: usize,
    cosines: Vec<Vec<f64>>,
}

impl Mfcc {
    /// Mfcc::Initialize; None when 2.5's Initialize fails (and Compute then gives nothing)
    fn init(&self, input_length: usize, sample_rate: f64) -> Option<MfccState> {
        let (lower, upper, channels) = (self.lower as f64, self.upper as f64, self.filterbank_channels);
        if channels < 1 || sample_rate <= 0.0 || input_length < 2 || lower < 0.0 || upper <= lower {
            return None;
        }
        let nc = channels as usize;
        let mel_low = freq_to_mel(lower);
        let mel_spacing = (freq_to_mel(upper) - mel_low) / (nc + 1) as f64;
        let centers: Vec<f64> = (0..=nc).map(|i| mel_low + mel_spacing * (i + 1) as f64).collect();
        let hz_per_sbin = 0.5 * sample_rate / (input_length - 1) as f64;
        let start = cvtt(1.5 + lower / hz_per_sbin, false) as i32;
        let end = cvtt(upper / hz_per_sbin, false) as i32;
        let mut band_mapper = vec![0i32; input_length];
        let mut channel = 0usize;
        for (i, b) in band_mapper.iter_mut().enumerate() {
            let melf = freq_to_mel(i as f64 * hz_per_sbin);
            if (i as i32) < start || (i as i32) > end {
                *b = -2;
            } else {
                while channel < nc && centers[channel] < melf {
                    channel += 1;
                }
                *b = channel as i32 - 1;
            }
        }
        let weights = (0..input_length)
            .map(|i| {
                let ch = band_mapper[i];
                if (i as i32) < start || (i as i32) > end {
                    0.0
                } else if ch >= 0 {
                    let ch = ch as usize;
                    (centers[ch + 1] - freq_to_mel(i as f64 * hz_per_sbin)) / (centers[ch + 1] - centers[ch])
                } else {
                    (centers[0] - freq_to_mel(i as f64 * hz_per_sbin)) / (centers[0] - mel_low)
                }
            })
            .collect();
        // MfccDct::Initialize
        let count = self.dct_coefficients;
        if count < 1 || count > channels {
            return None;
        }
        let fnorm = (2.0 / nc as f64).sqrt();
        let arg = 1f64.atan() * 4.0 / nc as f64;
        let cosines = (0..count as usize).map(|i| (0..nc).map(|j| fnorm * (i as f64 * arg * (j as f64 + 0.5)).cos()).collect()).collect();
        Some(MfccState { start, end, band_mapper, weights, channels: nc, cosines })
    }
}

impl MfccState {
    /// Mfcc::Compute: the mel filterbank over the square roots, the log (floored at 1e-12), the DCT
    fn compute(&self, frame: &[f64]) -> Vec<f64> {
        let mut working = Vec::new();
        // the filterbank gives nothing when the input is too short, and the DCT then sums nothing
        if frame.len() > self.end as usize {
            working = vec![0.0; self.channels];
            for i in self.start.max(0)..=self.end {
                let i = i as usize;
                let spec = frame[i].sqrt();
                let weighted = spec * self.weights[i];
                let channel = self.band_mapper[i];
                if channel >= 0 {
                    working[channel as usize] += weighted;
                }
                if channel + 1 < self.channels as i32 {
                    working[(channel + 1) as usize] += spec - weighted;
                }
            }
            for v in working.iter_mut() {
                // a NaN stays NaN (f64::max would floor it)
                *v = if *v < 1e-12 { 1e-12f64.ln() } else { v.ln() };
            }
        }
        self.cosines.iter().map(|c| c.iter().zip(&working).fold(0.0f64, |s, (&c, &x)| s + c * x)).collect()
    }
}

impl Kernel for Mfcc {
    fn prepare(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        if io.n_in() != 2 || io.outs.len() != 1 {
            return err("Mfcc takes 2 inputs and 1 output");
        }
        let (wav, rate) = (io.input(0)?, io.input(1)?);
        if wav.shape.len() != 3 || rate.elements() != 1 || wav.dtype != TType::F32 || rate.dtype != TType::I32 || io.outs[0].dtype != TType::F32 {
            return err("Mfcc takes a float32 [channels, samples, bins] spectrogram and an int32 sample rate");
        }
        if self.dct_coefficients < 0 {
            return err("a negative DCT coefficient count");
        }
        io.outs[0].resize(vec![wav.shape[0], wav.shape[1], self.dct_coefficients as usize]);
        Ok(())
    }

    fn eval(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        let (wav, rate) = (io.input(0)?, io.input(1)?);
        let (channels, samples, bins) = (wav.shape[0], wav.shape[1], wav.shape[2]);
        let count = self.dct_coefficients as usize;
        let state = self.init(bins, rate.as_slice::<i32>()[0] as f64);
        let x = wav.as_slice::<f32>();
        let out = io.outs[0].as_mut_slice::<f32>();
        for k in 0..channels * samples {
            let frame: Vec<f64> = x[k * bins..(k + 1) * bins].iter().map(|&v| v as f64).collect();
            let Some(coefficients) = state.as_ref().map(|s| s.compute(&frame)).filter(|c| c.len() == count) else {
                return err("the Mfcc settings do not initialize (2.5 gives no coefficients and fails)");
            };
            for (d, &v) in out[k * count..(k + 1) * count].iter_mut().zip(&coefficients) {
                *d = v as f32;
            }
        }
        Ok(())
    }
}
