"""coralrt for Python: Coral Edge TPU models (*_edgetpu.tflite) on the Coral USB Accelerator from any Python 3
and numpy (1 or 2), through ctypes over coralrt's C ABI. No tflite_runtime, libedgetpu or pycoral needed.

  from coralrt import Interpreter
  it = Interpreter("model_edgetpu.tflite")        # device="usb:0", performance="max" optional
  it.set_tensor(it.get_input_details()[0]["index"], image)
  it.invoke()
  scores = it.get_tensor(it.get_output_details()[0]["index"])

The Interpreter mirrors tflite_runtime's, so the pycoral and tflite_runtime packages next to this one (thin
compatibility layers) run code written for them unchanged.

The native library is coralrt.dll / libcoralrt.so / libcoralrt.dylib: $CORALRT_LIB, else next to this file,
else the crate's target/release (for an editable install of the source tree).
"""
import ctypes
import os
import sys

import numpy as np

__all__ = ["Interpreter", "Delegate", "list_edge_tpus", "version"]

_NAME = {"win32": "coralrt.dll", "darwin": "libcoralrt.dylib"}.get(sys.platform, "libcoralrt.so")
_HERE = os.path.dirname(os.path.abspath(__file__))


def _load():
  for path in (os.environ.get("CORALRT_LIB"), os.path.join(_HERE, _NAME), os.path.join(_HERE, "..", "..", "target", "release", _NAME)):
    if path and os.path.exists(path):
      return ctypes.CDLL(os.path.abspath(path))
  raise ImportError(f"{_NAME} not found: build it with `cargo build --release` in the coralrt crate, or set CORALRT_LIB")


_lib = _load()
_I = ctypes.c_void_p
_S = ctypes.c_char_p
for _name, _args, _ret in [
    ("coral_last_error", [], _S),
    ("coral_version", [], _S),
    ("coral_list_devices", [], ctypes.c_int32),
    ("coral_device_path", [ctypes.c_int32, ctypes.c_char_p, ctypes.c_size_t], ctypes.c_int32),
    ("coral_device_in_bootloader", [ctypes.c_int32], ctypes.c_int32),
    ("coral_interpreter_new", [_S, _S, _S], _I),
    ("coral_interpreter_new_from_buffer", [ctypes.c_char_p, ctypes.c_size_t, _S, _S], _I),
    ("coral_interpreter_free", [_I], None),
    ("coral_interpreter_invoke", [_I], ctypes.c_int32),
    ("coral_input_count", [_I], ctypes.c_int32),
    ("coral_output_count", [_I], ctypes.c_int32),
    ("coral_input_tensor", [_I, ctypes.c_int32], ctypes.c_int32),
    ("coral_output_tensor", [_I, ctypes.c_int32], ctypes.c_int32),
    ("coral_tensor_count", [_I], ctypes.c_int32),
    ("coral_tensor_name", [_I, ctypes.c_int32], _S),
    ("coral_tensor_type", [_I, ctypes.c_int32], ctypes.c_int32),
    ("coral_tensor_rank", [_I, ctypes.c_int32], ctypes.c_int32),
    ("coral_tensor_dim", [_I, ctypes.c_int32, ctypes.c_int32], ctypes.c_int32),
    ("coral_tensor_bytes", [_I, ctypes.c_int32], ctypes.c_ssize_t),
    ("coral_tensor_quant_count", [_I, ctypes.c_int32], ctypes.c_int32),
    ("coral_tensor_quant_scale", [_I, ctypes.c_int32, ctypes.c_int32], ctypes.c_float),
    ("coral_tensor_quant_zero_point", [_I, ctypes.c_int32, ctypes.c_int32], ctypes.c_int64),
    ("coral_tensor_quantized_dimension", [_I, ctypes.c_int32], ctypes.c_int32),
    ("coral_tensor_data", [_I, ctypes.c_int32], ctypes.c_void_p),
    ("coral_interpreter_device_path", [_I, ctypes.c_char_p, ctypes.c_size_t], ctypes.c_int32),
]:
  _f = getattr(_lib, _name)
  _f.argtypes, _f.restype = _args, _ret

# TensorFlow Lite TensorType -> numpy
_DTYPES = {0: np.float32, 1: np.float16, 2: np.int32, 3: np.uint8, 4: np.int64, 6: np.bool_, 7: np.int16, 9: np.int8, 10: np.float64}


def _error():
  msg = _lib.coral_last_error()
  return msg.decode("utf-8", "replace") if msg else "unknown error"


def _check(rc):
  if rc is None or (isinstance(rc, int) and rc < 0):
    raise RuntimeError(_error())
  return rc


def _c(s):
  return None if s is None else str(s).encode()


def version():
  return _lib.coral_version().decode()


def list_edge_tpus():
  """The sticks on the bus, like pycoral's list_edge_tpus: [{'type': 'usb', 'path': '2-1'}, ...]"""
  out = []
  for i in range(_check(_lib.coral_list_devices())):
    buf = ctypes.create_string_buffer(64)
    _check(_lib.coral_device_path(i, buf, 64))
    out.append({"type": "usb", "path": buf.value.decode()})
  return out


class Delegate:
  """What load_delegate("edgetpu...") returns: carries the delegate options (only "device" is used)."""

  def __init__(self, options=None):
    self.options = dict(options or {})


class Interpreter:
  """A model on the Edge TPU, with tflite_runtime's Interpreter API.

  device: None / "usb" (first stick), ":N" / "usb:N" (N-th, as list_edge_tpus orders them), "usb:<path>".
  performance: None ($CORALRT_PERFORMANCE, else "high" = the official throttled runtime, 250 MHz), "max"
  (500 MHz, the official direct runtime, runs hotter), "medium", "low".
  experimental_delegates: Edge TPU delegates from load_delegate (their "device" option is honored).
  Tensors are allocated at construction and never move: views from tensor() stay valid.
  """

  def __init__(self, model_path=None, model_content=None, experimental_delegates=None, num_threads=None, device=None, performance=None, **_unused):
    for d in experimental_delegates or []:
      if isinstance(d, Delegate) and device is None:
        device = d.options.get("device")
    if model_content is not None:
      data = bytes(model_content)
      self._h = _lib.coral_interpreter_new_from_buffer(data, len(data), _c(device), _c(performance))
    elif model_path is not None:
      self._h = _lib.coral_interpreter_new(os.fsencode(os.fspath(model_path)), _c(device), _c(performance))
    else:
      raise ValueError("Interpreter needs model_path or model_content")
    if not self._h:
      raise ValueError(_error())

  def __del__(self):
    h, self._h = getattr(self, "_h", None), None
    if h:
      _lib.coral_interpreter_free(h)

  # *** tflite_runtime's API ***
  def allocate_tensors(self):
    """Tensors are allocated with the interpreter; kept for API compatibility."""

  def invoke(self):
    _check(_lib.coral_interpreter_invoke(self._h))

  def get_input_details(self):
    return [self._details(_check(_lib.coral_input_tensor(self._h, i))) for i in range(_check(_lib.coral_input_count(self._h)))]

  def get_output_details(self):
    return [self._details(_check(_lib.coral_output_tensor(self._h, i))) for i in range(_check(_lib.coral_output_count(self._h)))]

  def get_tensor_details(self):
    return [self._details(t) for t in range(_check(_lib.coral_tensor_count(self._h)))]

  def get_signature_list(self):
    return {}

  def tensor(self, tensor_index):
    """A function returning a writable numpy view of the tensor's memory."""
    self._details(tensor_index)  # raises for a bad index now, like tflite
    return lambda: self._view(tensor_index)

  def set_tensor(self, tensor_index, value):
    d = self._details(tensor_index)
    value = np.asarray(value)
    if value.dtype != d["dtype"]:
      raise ValueError(f"Cannot set tensor: Got value of type {value.dtype.name} but expected type {np.dtype(d['dtype']).name} for input {tensor_index}, name: {d['name']}")
    if value.shape != tuple(d["shape"]):
      raise ValueError(f"Cannot set tensor: Dimension mismatch. Got {value.shape} but expected {tuple(d['shape'])} for input {tensor_index}.")
    self._view(tensor_index)[...] = value

  def get_tensor(self, tensor_index):
    return self._view(tensor_index).copy()

  def resize_tensor_input(self, input_index, tensor_size, strict=False):
    raise NotImplementedError("coralrt runs Edge TPU models at the shapes they were compiled for")

  # *** coralrt ***
  @property
  def device_path(self):
    """usb path of the stick the model runs on ("" without an Edge TPU op)"""
    buf = ctypes.create_string_buffer(64)
    _check(_lib.coral_interpreter_device_path(self._h, buf, 64))
    return buf.value.decode()

  def _details(self, t):
    if not isinstance(t, (int, np.integer)) or not 0 <= int(t) < _lib.coral_tensor_count(self._h):
      raise ValueError(f"Tensor with no shape found: {t}")
    t = int(t)
    rank = _check(_lib.coral_tensor_rank(self._h, t))
    shape = np.array([_lib.coral_tensor_dim(self._h, t, d) for d in range(rank)], np.int32)
    q = _check(_lib.coral_tensor_quant_count(self._h, t))
    scales = np.array([_lib.coral_tensor_quant_scale(self._h, t, k) for k in range(q)], np.float32)
    zero_points = np.array([_lib.coral_tensor_quant_zero_point(self._h, t, k) for k in range(q)], np.int32)
    code = _check(_lib.coral_tensor_type(self._h, t))
    return {
        "name": _lib.coral_tensor_name(self._h, t).decode("utf-8", "replace"),
        "index": t,
        "shape": shape,
        "shape_signature": shape.copy(),
        "dtype": _DTYPES.get(code, np.uint8),
        "quantization": (float(scales[0]), int(zero_points[0])) if q else (0.0, 0),
        "quantization_parameters": {"scales": scales, "zero_points": zero_points, "quantized_dimension": _lib.coral_tensor_quantized_dimension(self._h, t)},
        "sparsity_parameters": {},
    }

  def _view(self, t):
    d = self._details(t)
    n = _check(_lib.coral_tensor_bytes(self._h, t))
    buf = (ctypes.c_uint8 * n).from_address(_lib.coral_tensor_data(self._h, t))
    buf.interpreter = self  # the view keeps the native memory alive
    return np.frombuffer(buf, dtype=d["dtype"]).reshape(d["shape"])
