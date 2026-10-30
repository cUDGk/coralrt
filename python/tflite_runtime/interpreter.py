"""tflite_runtime.interpreter on coralrt: the names tflite_runtime 2.5's module has.

  import tflite_runtime.interpreter as tflite
  it = tflite.Interpreter(model_path, experimental_delegates=[tflite.load_delegate("edgetpu.dll")])

Only the Edge TPU delegate exists; every op of TensorFlow Lite 2.5's BuiltinOpResolver runs on the CPU side.
"""
import os

import coralrt
from coralrt import Delegate, Interpreter, OpResolverType, SignatureRunner


def load_delegate(library, options=None):
  if "edgetpu" not in os.path.basename(os.fspath(library)).lower():
    raise ValueError(f"Failed to load delegate from {library}: coralrt provides only the Edge TPU delegate")
  return coralrt.Delegate(options)


class InterpreterWithCustomOps(Interpreter):
  """tflite_runtime's interpreter with native custom op registerers: coralrt cannot load native TFLite kernels."""

  def __init__(self, custom_op_registerers=None, **kwargs):
    if custom_op_registerers:
      raise ValueError("coralrt cannot load native TensorFlow Lite custom op registerers")
    super().__init__(**kwargs)


__all__ = ["Delegate", "Interpreter", "InterpreterWithCustomOps", "OpResolverType", "SignatureRunner", "load_delegate"]
