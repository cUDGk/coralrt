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
# the licenses of everything in the library go into the wheel too (setuptools takes LICENSE* and NOTICE* from here)
notices = [shutil.copy2(crate / n, here / n) for n in ("LICENSE", "NOTICE")]
try:
  # isolated: the build gets the setuptools of pyproject.toml, not whatever the running Python has (an old
  # packaging there breaks setuptools' license metadata)
  subprocess.run([sys.executable, "-m", "pip", "wheel", ".", "--no-deps", "-w", "dist"], cwd=here, check=True)
finally:
  # the source tree then loads target/release's library again, not a stale copy
  lib.unlink()
  for n in notices:
    pathlib.Path(n).unlink()
  shutil.rmtree(here / "build", ignore_errors=True)
