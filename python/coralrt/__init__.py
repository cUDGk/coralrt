"""coralrt for Python: TensorFlow Lite models, and Coral Edge TPU models (*_edgetpu.tflite) on the Coral USB
Accelerator, from any Python 3 and numpy (1 or 2), through ctypes over coralrt's C ABI. No tflite_runtime,
libedgetpu or pycoral needed.

  from coralrt import Interpreter
  it = Interpreter("model_edgetpu.tflite")        # device="usb:0", performance="max" optional
  it.set_tensor(it.get_input_details()[0]["index"], image)
  it.invoke()
  scores = it.get_tensor(it.get_output_details()[0]["index"])

The Interpreter mirrors tflite_runtime 2.5's, so the pycoral and tflite_runtime packages next to this one (thin
compatibility layers) run code written for them unchanged.

The native library is coralrt.dll / libcoralrt.so / libcoralrt.dylib: $CORALRT_LIB, else next to this file,
else the crate's target/release (for an editable install of the source tree).
"""
import ctypes
import enum
import os
import sys
import weakref

import numpy as np

__all__ = ["Interpreter", "Delegate", "OpResolverType", "SignatureRunner", "list_edge_tpus", "version"]

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
_i32 = ctypes.c_int32
_P = ctypes.POINTER
for _name, _args, _ret in [
    ("coral_last_error", [], _S),
    ("coral_version", [], _S),
    ("coral_list_devices", [], _i32),
    ("coral_device_path", [_i32, ctypes.c_char_p, ctypes.c_size_t], _i32),
    ("coral_device_in_bootloader", [_i32], _i32),
    ("coral_interpreter_new", [_S, _S], _I),
    ("coral_interpreter_new_from_buffer", [ctypes.c_char_p, ctypes.c_size_t, _S], _I),
    ("coral_interpreter_free", [_I], None),
    ("coral_interpreter_invoke", [_I], _i32),
    ("coral_interpreter_allocate_tensors", [_I], _i32),
    ("coral_interpreter_resize_input", [_I, _i32, _P(_i32), _i32], _i32),
    ("coral_interpreter_reset_variables", [_I], _i32),
    ("coral_input_count", [_I], _i32),
    ("coral_output_count", [_I], _i32),
    ("coral_input_tensor", [_I, _i32], _i32),
    ("coral_output_tensor", [_I, _i32], _i32),
    ("coral_tensor_count", [_I], _i32),
    ("coral_tensor_name", [_I, _i32], _S),
    ("coral_tensor_type", [_I, _i32], _i32),
    ("coral_tensor_rank", [_I, _i32], _i32),
    ("coral_tensor_dim", [_I, _i32, _i32], _i32),
    ("coral_tensor_signature_rank", [_I, _i32], _i32),
    ("coral_tensor_signature_dim", [_I, _i32, _i32], _i32),
    ("coral_tensor_bytes", [_I, _i32], ctypes.c_ssize_t),
    ("coral_tensor_quant_count", [_I, _i32], _i32),
    ("coral_tensor_quant_scale", [_I, _i32, _i32], ctypes.c_float),
    ("coral_tensor_quant_zero_point", [_I, _i32, _i32], ctypes.c_int64),
    ("coral_tensor_quantized_dimension", [_I, _i32], _i32),
    ("coral_tensor_data", [_I, _i32], ctypes.c_void_p),
    ("coral_tensor_string_count", [_I, _i32], _i32),
    ("coral_tensor_string", [_I, _i32, _i32, _P(ctypes.c_size_t)], ctypes.c_void_p),
    ("coral_set_tensor_strings", [_I, _i32, _P(ctypes.c_char_p), _P(ctypes.c_size_t), _i32, _P(_i32), _i32], _i32),
    ("coral_op_count", [_I], _i32),
    ("coral_op_name", [_I, _i32], _S),
    ("coral_op_tensor_count", [_I, _i32, _i32], _i32),
    ("coral_op_tensor", [_I, _i32, _i32, _i32], _i32),
    ("coral_interpreter_device_path", [_I, ctypes.c_char_p, ctypes.c_size_t], _i32),
    ("coral_signature_count", [_I], _i32),
    ("coral_signature_key", [_I, _i32], _S),
    ("coral_signature_tensor_count", [_I, _i32, _i32], _i32),
    ("coral_signature_tensor_name", [_I, _i32, _i32, _i32], _S),
    ("coral_signature_tensor_index", [_I, _i32, _i32, _i32], _i32),
]:
  _f = getattr(_lib, _name)
  _f.argtypes, _f.restype = _args, _ret

# TensorFlow Lite TensorType -> numpy (interpreter_wrapper/numpy.cc), and its names (TfLiteTypeGetName)
_DTYPES = {0: np.float32, 1: np.float16, 2: np.int32, 3: np.uint8, 4: np.int64, 5: np.bytes_, 6: np.bool_, 7: np.int16,
           8: np.complex64, 9: np.int8, 10: np.float64, 11: np.complex128, 12: np.uint64, 13: np.object_, 14: np.object_,
           15: np.uint32}
_TYPE_NAMES = ["FLOAT32", "FLOAT16", "INT32", "UINT8", "INT64", "STRING", "BOOL", "INT16", "COMPLEX64", "INT8", "FLOAT64",
               "COMPLEX128", "UINT64", "RESOURCE", "VARIANT", "UINT32"]
_STRING, _OBJECT_TYPES = 5, (13, 14)


def _type_of_array(a):
  """TfLiteTypeFromPyArray"""
  if a.dtype.kind in "OSU":
    return _STRING
  for code, t in _DTYPES.items():
    if code not in (_STRING,) + _OBJECT_TYPES and a.dtype == np.dtype(t):
      return code
  return -1


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


class OpResolverType(enum.Enum):
  """tflite_runtime's op resolver choice; coralrt has one set of kernels (those of BUILTIN), so all behave alike."""
  AUTO = 0
  BUILTIN = 1
  BUILTIN_REF = 2
  BUILTIN_WITHOUT_DEFAULT_DELEGATES = 3


class SignatureRunner:
  """Runs a SignatureDef by its names, like tflite_runtime 2.5's: runner(x=..., y=...) -> {output name: array}."""

  def __init__(self, interpreter=None, signature_def_name=None):
    if not interpreter:
      raise ValueError("None interpreter provided.")
    if not signature_def_name:
      raise ValueError("None signature_def_name provided.")
    self._interpreter = interpreter
    self._signature_def_name = signature_def_name
    signature_defs = interpreter._get_full_signature_list()
    if signature_def_name not in signature_defs:
      raise ValueError("Invalid signature_def_name provided.")
    self._signature_def = signature_defs[signature_def_name]
    self._outputs = self._signature_def["outputs"].items()
    self._inputs = self._signature_def["inputs"]

  def __call__(self, **kwargs):
    if len(kwargs) != len(self._inputs):
      raise ValueError("Invalid number of inputs provided for running a SignatureDef, "
                       "expected %s vs provided %s" % (len(kwargs), len(self._inputs)))
    for input_name, value in kwargs.items():
      if input_name not in self._inputs:
        raise ValueError("Invalid Input name (%s) for SignatureDef" % input_name)
      self._interpreter.resize_tensor_input(self._inputs[input_name], value.shape)
    self._interpreter.allocate_tensors()
    for input_name, value in kwargs.items():
      self._interpreter._set_input_tensor(input_name, value=value, method_name=self._signature_def_name)
    self._interpreter.invoke()
    return {name: self._interpreter.get_tensor(index) for name, index in self._outputs}


class Interpreter:
  """A TensorFlow Lite model (on the Edge TPU if it has an Edge TPU op), with tflite_runtime 2.5's Interpreter API.

  device: None / "usb" (first stick), ":N" / "usb:N" (N-th, as list_edge_tpus orders them), "usb:<path>".
  performance: None ($CORALRT_PERFORMANCE, else "high" = the official throttled runtime, 250 MHz), "max"
  (500 MHz, the official direct runtime, runs hotter), "medium", "low".
  detection_sort: "tflite2.5" (default, as the official runtime) or "stable" (TFLite 2.6 and later) for the order
  TFLite_Detection_PostProcess gives boxes of equal score.
  experimental_delegates: Edge TPU delegates from load_delegate (their "device" option is honored).
  num_threads, experimental_op_resolver_type, experimental_preserve_all_tensors: accepted; coralrt runs on one thread
  and keeps every tensor.
  Tensors are allocated at construction. As in tflite_runtime, views from tensor() must not be held across
  invoke(), allocate_tensors() or resize_tensor_input(): those raise while one is alive.
  """

  def __init__(self, model_path=None, model_content=None, experimental_delegates=None, num_threads=None,
               experimental_op_resolver_type=OpResolverType.AUTO, experimental_preserve_all_tensors=False, device=None,
               performance=None, detection_sort=None):
    if num_threads is not None:
      if not isinstance(num_threads, int):
        raise ValueError("type of num_threads should be int")
      if num_threads < 1:
        raise ValueError("num_threads should >= 1")
    for d in experimental_delegates or []:
      if isinstance(d, Delegate) and device is None:
        device = d.options.get("device")
    options = ",".join(f"{k}={v}" for k, v in [("device", device), ("performance", performance), ("detection_sort", detection_sort)] if v is not None)
    if model_path and not model_content:
      self._h = _lib.coral_interpreter_new(os.fsencode(os.fspath(model_path)), _c(options))
    elif model_content and not model_path:
      data = bytes(model_content)
      self._h = _lib.coral_interpreter_new_from_buffer(data, len(data), _c(options))
    elif not model_content and not model_path:
      raise ValueError("`model_path` or `model_content` must be specified.")
    else:
      raise ValueError("Can't both provide `model_path` and `model_content`")
    if not self._h:
      raise ValueError(_error())
    self._views = []  # weak references to the buffers behind live tensor() views
    self._signature_defs = self.get_signature_list()

  def __del__(self):
    h, self._h = getattr(self, "_h", None), None
    if h:
      _lib.coral_interpreter_free(h)

  # *** tflite_runtime's API ***
  def allocate_tensors(self):
    self._ensure_safe()
    _check(_lib.coral_interpreter_allocate_tensors(self._h))

  def invoke(self):
    self._ensure_safe()
    if _lib.coral_interpreter_invoke(self._h) < 0:
      raise RuntimeError(_error())

  def reset_all_variables(self):
    _check(_lib.coral_interpreter_reset_variables(self._h))

  def resize_tensor_input(self, input_index, tensor_size, strict=False):
    self._ensure_safe()
    tensor_size = np.array(tensor_size, dtype=np.int32)
    inputs = [_lib.coral_input_tensor(self._h, i) for i in range(_check(_lib.coral_input_count(self._h)))]
    if int(input_index) not in inputs:
      raise ValueError(f"Invalid tensor index {input_index} exceeds max tensor index {len(inputs)}")
    if strict:
      d = self._details(input_index)
      sig = d["shape_signature"]
      if len(sig) != len(tensor_size) or any(s != -1 and s != t for s, t in zip(sig, tensor_size)):
        raise ValueError("ResizeInputTensorStrict only allows mutating unknown dimensions identified by -1.")
    dims = (_i32 * len(tensor_size))(*tensor_size)
    if _lib.coral_interpreter_resize_input(self._h, inputs.index(int(input_index)), dims, len(tensor_size)) < 0:
      raise ValueError(_error())

  def get_input_details(self):
    return [self._details(_check(_lib.coral_input_tensor(self._h, i))) for i in range(_check(_lib.coral_input_count(self._h)))]

  def get_output_details(self):
    return [self._details(_check(_lib.coral_output_tensor(self._h, i))) for i in range(_check(_lib.coral_output_count(self._h)))]

  def get_tensor_details(self):
    return [self._details(t) for t in range(_check(_lib.coral_tensor_count(self._h)))]

  def _get_op_details(self, op_index):
    op_index = int(op_index)
    name = _lib.coral_op_name(self._h, op_index)
    if name is None:
      raise ValueError(_error())
    tensors = lambda io: np.array([_lib.coral_op_tensor(self._h, op_index, io, k) for k in range(_check(_lib.coral_op_tensor_count(self._h, op_index, io)))], np.int32)
    return {"index": op_index, "op_name": name.decode(), "inputs": tensors(0), "outputs": tensors(1)}

  def _get_ops_details(self):
    return [self._get_op_details(i) for i in range(_check(_lib.coral_op_count(self._h)))]

  def get_signature_list(self):
    return {key: {io: list(names) for io, names in sig.items()} for key, sig in self._get_full_signature_list().items()}

  def _get_full_signature_list(self):
    """{key: {"inputs": {name: tensor index}, "outputs": {...}}}, as tflite_runtime 2.5 returns it (pycoral reads it)"""
    h, out = self._h, {}
    for s in range(_check(_lib.coral_signature_count(h))):
      out[_lib.coral_signature_key(h, s).decode("utf-8", "replace")] = {
          io: {_lib.coral_signature_tensor_name(h, s, k, i).decode("utf-8", "replace"): _check(_lib.coral_signature_tensor_index(h, s, k, i))
               for i in range(_check(_lib.coral_signature_tensor_count(h, s, k)))}
          for k, io in enumerate(("inputs", "outputs"))
      }
    return out

  def _set_input_tensor(self, input_name, value, method_name=None):
    if method_name is None:
      if len(self._signature_defs) != 1:
        raise ValueError("SignatureDef method_name is None and model has {0} Signatures. "
                         "None is only allowed when the model has 1 SignatureDef".format(len(self._signature_defs)))
      method_name = next(iter(self._signature_defs))
    inputs = self._get_full_signature_list().get(method_name, {}).get("inputs", {})
    if input_name not in inputs:
      raise ValueError(f"Invalid input tensor name {input_name} for SignatureDef {method_name}")
    self.set_tensor(inputs[input_name], value)

  def get_signature_runner(self, method_name=None):
    if method_name is None:
      if len(self._signature_defs) != 1:
        raise ValueError("SignatureDef method_name is None and model has {0} Signatures. "
                         "None is only allowed when the model has 1 SignatureDef".format(len(self._signature_defs)))
      method_name = next(iter(self._signature_defs))
    return SignatureRunner(interpreter=self, signature_def_name=method_name)

  def tensor(self, tensor_index):
    """A function returning a writable numpy view of the tensor's memory."""
    self._details(tensor_index)  # raises for a bad index now, like tflite
    return lambda: self._view(tensor_index)

  def set_tensor(self, tensor_index, value):
    d = self._details(tensor_index)
    a = np.asarray(value)
    code = _check(_lib.coral_tensor_type(self._h, d["index"]))
    got = _type_of_array(a)
    if got != code:
      name = _TYPE_NAMES[got] if 0 <= got < len(_TYPE_NAMES) else "NOTYPE"
      raise ValueError(f"Cannot set tensor: Got value of type {name} but expected type {_TYPE_NAMES[code]} for input {tensor_index}, name: {d['name']} ")
    if a.ndim != len(d["shape"]):
      raise ValueError(f"Cannot set tensor: Dimension mismatch. Got {a.ndim} but expected {len(d['shape'])} for input {tensor_index}.")
    for j, (x, y) in enumerate(zip(a.shape, d["shape"])):
      if x != y:
        raise ValueError(f"Cannot set tensor: Dimension mismatch. Got {x} but expected {y} for dimension {j} of input {tensor_index}.")
    if code == _STRING:
      items = [a.tobytes()] if a.ndim == 0 else [s if isinstance(s, bytes) else str(s).encode() for s in a.reshape(-1).tolist()]
      data = (ctypes.c_char_p * len(items))(*items)
      lens = (ctypes.c_size_t * len(items))(*[len(s) for s in items])
      dims = (_i32 * len(d["shape"]))(*d["shape"])
      _check(_lib.coral_set_tensor_strings(self._h, d["index"], data, lens, len(items), dims, len(d["shape"])))
      return
    n = _check(_lib.coral_tensor_bytes(self._h, d["index"]))
    if a.nbytes != n:
      raise ValueError(f"numpy array had {a.nbytes} bytes but expected {n} bytes.")
    ctypes.memmove(_lib.coral_tensor_data(self._h, d["index"]), np.ascontiguousarray(a).ctypes.data, n)

  def get_tensor(self, tensor_index):
    d = self._details(tensor_index)
    code = _check(_lib.coral_tensor_type(self._h, d["index"]))
    if code == _STRING or code in _OBJECT_TYPES:
      out = np.empty(tuple(d["shape"]), dtype=object)
      flat = out.reshape(-1)
      if code == _STRING:
        n = ctypes.c_size_t()
        for k in range(_check(_lib.coral_tensor_string_count(self._h, d["index"]))):
          p = _lib.coral_tensor_string(self._h, d["index"], k, ctypes.byref(n))
          flat[k] = ctypes.string_at(p, n.value) if n.value else b""
      return out
    a = self._view(tensor_index, track=False).copy()
    return a[()] if a.ndim == 0 else a

  # *** coralrt ***
  @property
  def device_path(self):
    """usb path of the stick the model runs on ("" without an Edge TPU op)"""
    buf = ctypes.create_string_buffer(64)
    _check(_lib.coral_interpreter_device_path(self._h, buf, 64))
    return buf.value.decode()

  def _live_views(self):
    self._views = [r for r in self._views if r() is not None]
    return self._views

  def _ensure_safe(self):
    if self._live_views():
      raise RuntimeError("There is at least 1 reference to internal data in the interpreter in the form of a numpy "
                         "array or slice. Be sure to only hold the function returned from tensor() if you are using "
                         "raw data access.")

  def _details(self, t):
    if not isinstance(t, (int, np.integer)) or not 0 <= int(t) < _lib.coral_tensor_count(self._h):
      raise ValueError(f"Tensor with no shape found: {t}")
    t = int(t)
    rank = _check(_lib.coral_tensor_rank(self._h, t))
    shape = np.array([_lib.coral_tensor_dim(self._h, t, d) for d in range(rank)], np.int32)
    sig = np.array([_lib.coral_tensor_signature_dim(self._h, t, d) for d in range(_check(_lib.coral_tensor_signature_rank(self._h, t)))], np.int32)
    q = _check(_lib.coral_tensor_quant_count(self._h, t))
    scales = np.array([_lib.coral_tensor_quant_scale(self._h, t, k) for k in range(q)], np.float32)
    zero_points = np.array([_lib.coral_tensor_quant_zero_point(self._h, t, k) for k in range(q)], np.int32)
    code = _check(_lib.coral_tensor_type(self._h, t))
    return {
        "name": _lib.coral_tensor_name(self._h, t).decode("utf-8", "replace"),
        "index": t,
        "shape": shape,
        "shape_signature": sig,
        "dtype": _DTYPES.get(code, np.uint8),
        "quantization": (float(scales[0]), int(zero_points[0])) if q else (0.0, 0),
        "quantization_parameters": {"scales": scales, "zero_points": zero_points, "quantized_dimension": _lib.coral_tensor_quantized_dimension(self._h, t)},
        "sparsity_parameters": {},
    }

  def _view(self, t, track=True):
    d = self._details(t)
    if d["dtype"] in (np.bytes_, np.object_):
      raise ValueError("string and resource tensors have no numpy view; use get_tensor / set_tensor")
    n = _check(_lib.coral_tensor_bytes(self._h, t))
    ptr = _lib.coral_tensor_data(self._h, t)
    if not n or not ptr:
      return np.zeros(d["shape"], d["dtype"])
    buf = (ctypes.c_uint8 * n).from_address(ptr)
    buf.interpreter = self  # the view keeps the native memory alive
    if track:
      self._live_views().append(weakref.ref(buf))
    return np.frombuffer(buf, dtype=d["dtype"]).reshape(d["shape"])
