# From pycoral 2.0 and libcoral (Copyright Google LLC, Apache License 2.0), rewritten; see NOTICE.
"""pycoral.utils.edgetpu on coralrt: the same functions as pycoral 2.0, without libedgetpu or tflite_runtime.

make_interpreter returns a coralrt.Interpreter (tflite_runtime's API), so pycoral.adapters work on it unchanged.
Not available: GStreamer buffers in run_inference, and the PCIe Edge TPU (coralrt drives the USB stick).
"""
import ctypes

import numpy as np

import coralrt
from coralrt import Interpreter


def load_edgetpu_delegate(options=None):
  """A delegate object for make_interpreter / tflite_runtime.Interpreter; only the "device" option is used."""
  return coralrt.Delegate(options)


def make_interpreter(model_path_or_content, device=None, delegate=None):
  """Creates an Interpreter for an Edge TPU model (path as str, contents as bytes).

  device: None / "usb" for any stick, ":N" / "usb:N" for the N-th (see list_edge_tpus), "usb:<path>".
  delegate: from load_edgetpu_delegate; if given, its "device" option is used instead of `device`.
  """
  if delegate:
    device = delegate.options.get("device")
  if isinstance(model_path_or_content, bytes):
    return Interpreter(model_content=model_path_or_content, device=device)
  return Interpreter(model_path=model_path_or_content, device=device)


def list_edge_tpus():
  return coralrt.list_edge_tpus()


def get_runtime_version():
  return f"coralrt {coralrt.version()} (Edge TPU runtime without libedgetpu, executables of RuntimeVersion 14)"


def set_verbosity(verbosity):
  """libedgetpu's log level; coralrt reports through exceptions only, so there is nothing to set."""
  return True


def supports_dmabuf(interpreter):
  return False


def run_inference(interpreter, input_data):
  """Writes a raw input (bytes, a numpy array, or a (ctypes.c_void_p, size) pair) to the first input and invokes."""
  view = interpreter.tensor(interpreter.get_input_details()[0]["index"])().reshape(-1).view(np.uint8)
  if isinstance(input_data, tuple) and len(input_data) == 2 and isinstance(input_data[0], ctypes.c_void_p):
    pointer, size = input_data
    if size < view.size:
      raise ValueError(f"input size={size}, expected={view.size}.")
    data = np.ctypeslib.as_array(ctypes.cast(pointer, ctypes.POINTER(ctypes.c_uint8)), (view.size,))
  elif isinstance(input_data, (bytes, bytearray, memoryview, np.ndarray)):
    data = np.frombuffer(bytes(input_data) if not isinstance(input_data, np.ndarray) else input_data.tobytes(), np.uint8)
    if data.size < view.size:
      raise ValueError(f"input size={data.size}, expected={view.size}.")
  else:
    raise TypeError(f"run_inference: unsupported input type {type(input_data).__name__} (GStreamer buffers need pycoral's native module)")
  view[...] = data[:view.size]
  interpreter.invoke()
