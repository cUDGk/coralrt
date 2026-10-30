"""Builds the coralrt wheel into dist/: the native library (cargo build --release) copied next to the package for
the build only, then a py3-none-<platform> wheel with coralrt, pycoral and tflite_runtime in it.

  python build_wheel.py
"""
import pathlib
import shutil
import subprocess
import sys

here = pathlib.Path(__file__).resolve().parent
crate = here.parent
name = {"win32": "coralrt.dll", "darwin": "libcoralrt.dylib"}.get(sys.platform, "libcoralrt.so")
subprocess.run(["cargo", "build", "--release"], cwd=crate, check=True)
lib = here / "coralrt" / name
shutil.copy2(crate / "target" / "release" / name, lib)
try:
  subprocess.run([sys.executable, "-m", "pip", "wheel", ".", "--no-deps", "--no-build-isolation", "-w", "dist"], cwd=here, check=True)
finally:
  # the source tree then loads target/release's library again, not a stale copy
  lib.unlink()
  shutil.rmtree(here / "build", ignore_errors=True)
