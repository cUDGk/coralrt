"""The description language of the op cases (cases.py): one model per case, made of one op (or a few) with its
tensors. Plain Python and numpy, so both build.py (TensorFlow's schema) and official.py (tflite_runtime 2.5) read it.
"""
import numpy as np


class T:
  """A tensor of a case.

  shape: the shape stored in the model (inputs get it unless `resize` gives another)
  dtype: numpy-style name: float32, float16, float64, int8, uint8, int16, int32, int64, uint32, uint64, bool, string,
         complex64, resource
  quant: (scale, zero_point), or (scales, zero_points, quantized_dimension) for per-channel
  data: a constant's values (array-like); the tensor then has a buffer in the model
  value: the values an input gets at run time instead of random ones
  gen: how random values are drawn: ("uniform", lo, hi), ("int", lo, hi) (inclusive), ("choice", values)
  var: a variable tensor (state kept across invokes); its `data`, if any, is the initial value
  sig: shape_signature (-1 for any size)
  resize: the shape the input is resized to before allocate_tensors
  sparsity: a dict for SparsityParameters: traversal_order, block_map, dim_metadata [(format, dense_size, segments, indices)]
  """

  def __init__(self, shape, dtype="float32", quant=None, data=None, value=None, gen=None, var=False, sig=None, resize=None,
               name=None, sparsity=None):
    self.shape, self.dtype, self.quant, self.gen, self.var, self.sig, self.resize, self.name = list(shape), dtype, quant, gen, var, sig, resize, name
    self.data = None if data is None else np.asarray(data, dtype=object if dtype == "string" else dtype)
    self.value = None if value is None else np.asarray(value, dtype=object if dtype == "string" else dtype)
    self.sparsity = sparsity


class Op:
  """op: a builtin name ("ADD") or "CUSTOM:<name>"; inputs / outputs: indices into the graph's tensors (-1 = absent);
  options: (options table name, {field: value}) with the schema's ObjectAPI field names; custom: custom_options bytes,
  or a function making them (called only by build.py, so official.py needs no flexbuffers)"""

  def __init__(self, op, inputs, outputs, options=None, custom=None, version=1, intermediates=None):
    self.op, self.inputs, self.outputs, self.options, self.custom, self.version = op, inputs, outputs, options, custom, version
    self.intermediates = intermediates or []


class Graph:
  def __init__(self, tensors, ops, inputs, outputs, name=""):
    self.tensors, self.ops, self.inputs, self.outputs, self.name = tensors, ops, inputs, outputs, name


class Case:
  """tol: 0 = every output byte must match; else floats may differ by tol * max(1, |expected|)
  invokes: how many times to invoke (state carried by variable tensors), each with new random inputs"""

  def __init__(self, name, graphs, tol=0, invokes=1, seed=None):
    self.name, self.graphs, self.tol, self.invokes = name, graphs, tol, invokes
    self.seed = seed if seed is not None else sum(map(ord, name))


def op_case(name, op, inputs, outputs, options=None, custom=None, version=1, tol=0, intermediates=None, invokes=1):
  """A case of one op: inputs and outputs are lists of T; every non-constant input is a model input."""
  tensors = [t for t in inputs if t is not None] + list(outputs) + list(intermediates or [])
  # indices of the inputs among `tensors` (absent ones, None, are -1)
  idx, k = [], 0
  for t in inputs:
    idx.append(-1 if t is None else k)
    k += t is not None
  outs = list(range(k, k + len(outputs)))
  inter = list(range(k + len(outputs), k + len(outputs) + len(intermediates or [])))
  model_inputs = [i for i, t in zip(idx, inputs) if t is not None and t.data is None and not t.var]
  g = Graph(tensors, [Op(op, idx, outs, options, custom, version, inter)], model_inputs, outs)
  return Case(name, [g], tol=tol, invokes=invokes)


def string_buffer(strings):
  """TFLite's string tensor layout: int32 count, int32 offsets[count + 1], the bytes"""
  strings = [s if isinstance(s, bytes) else str(s).encode() for s in strings]
  head = 4 * (len(strings) + 2)
  offsets, at = [], head
  for s in strings:
    offsets.append(at)
    at += len(s)
  offsets.append(at)
  return np.array([len(strings)] + offsets, "<i4").tobytes() + b"".join(strings)


def tensor_bytes(dtype, values):
  if dtype == "string":
    return string_buffer(np.asarray(values, dtype=object).reshape(-1).tolist())
  return np.ascontiguousarray(np.asarray(values, dtype=dtype)).tobytes()


def random_values(t, rng, shape=None):
  """the values an input gets: its `value`, else drawn by `gen`, else a default range per type"""
  if t.value is not None:
    return t.value
  shape = list(shape if shape is not None else (t.resize or t.shape))
  gen = t.gen
  if t.dtype == "string":
    words = gen[1] if gen and gen[0] == "choice" else [b"", b"a", b"bc", b"hello", b"tflite", b"\xe3\x81\x82"]
    return np.array([words[rng.integers(len(words))] for _ in range(int(np.prod(shape)))], dtype=object).reshape(shape)
  if gen and gen[0] == "choice":
    return np.array(gen[1], dtype=t.dtype)[rng.integers(len(gen[1]), size=shape)]
  if t.dtype == "bool":
    return rng.integers(0, 2, size=shape).astype(bool)
  if t.dtype.startswith("float") or t.dtype.startswith("complex"):
    lo, hi = (gen[1], gen[2]) if gen else (-4.0, 4.0)
    v = rng.uniform(lo, hi, size=shape)
    if t.dtype.startswith("complex"):
      v = v + 1j * rng.uniform(lo, hi, size=shape)
    return v.astype(t.dtype)
  info = np.iinfo(t.dtype)
  lo, hi = (gen[1], gen[2]) if gen else ((info.min, info.max) if info.bits <= 16 else (-100, 100) if info.min < 0 else (0, 100))
  return rng.integers(lo, hi, size=shape, endpoint=True).astype(t.dtype)
