//! Compiles Takuya Ooura's FFT package (third_party/oourafft, the OouraFFT 1.0 that TensorFlow 2.5 builds as fft2d),
//! which RFFT2D and AudioSpectrogram run: the same C as the official runtime, so the same doubles.

fn main() {
    let dir = "third_party/oourafft";
    println!("cargo:rerun-if-changed={dir}");
    cc::Build::new()
        .file(format!("{dir}/fftsg.c"))
        .file(format!("{dir}/fftsg2d.c"))
        // its rdft / cdft / ... stay inside coralrt's shared library
        .flag_if_supported("-fvisibility=hidden")
        .warnings(false)
        .compile("oourafft");
}
