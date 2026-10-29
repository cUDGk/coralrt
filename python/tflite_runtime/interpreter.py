"""tflite_runtime.interpreter on coralrt: Interpreter and load_delegate for Edge TPU models.

  import tflite_runtime.interpreter as tflite
  it = tflite.Interpreter(model_path, experimental_delegates=[tflite.load_delegate("edgetpu.dll")])

Only the Edge TPU delegate exists, and only the ops coralrt implements run on the CPU side (enough for the
models the Edge TPU compiler produces for classification, detection and pose; others raise when loaded).
"""
import os

import coralrt
from coralrt import Interpreter


def load_delegate(library, options=None):
  if "edgetpu" not in os.path.basename(os.fspath(library)).lower():
    raise ValueError(f"Failed to load delegate from {library}: coralrt provides only the Edge TPU delegate")
  return coralrt.Delegate(options)


__all__ = ["Interpreter", "load_delegate"]
