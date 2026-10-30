"""Reading and editing .tflite models without TensorFlow, for pycoral.learn.

An edit writes the tables it changes anew in front of the original model; they point into it for everything they
keep (flatbuffers offsets point forward, so new tables before the old bytes can refer to them). The original bytes
stay as they are, 16-byte aligned from the start of the file as before.
"""
import struct

import numpy as np

_TFL3 = b"TFL3"

# field kinds of the schema's tables, by field id: "ref" for an offset (table, vector, string), else a struct format
_MODEL = ["I", "ref", "ref", "ref", "ref", "ref", "ref", "ref"]
_SUBGRAPH = ["ref", "ref", "ref", "ref", "ref", "i"]
_TENSOR = ["ref", "b", "I", "ref", "ref", "?", "ref", "ref", "?", "ref"]
_OPERATOR = ["I", "ref", "ref", "B", "ref", "ref", "b", "ref", "ref", "Q", "Q", "B", "ref", "i"]

# BuiltinOptions
FULLY_CONNECTED_OPTIONS, SOFTMAX_OPTIONS, RESHAPE_OPTIONS = 8, 9, 17


def _u32(buf, pos):
  return struct.unpack_from("<I", buf, pos)[0]


def quantize(values, scale, zero_point, lo, hi):
  """libcoral's Quantize: std::round(zero_point + x / scale) (halves away from zero), clamped by std::min and
  std::max in float, so NaN goes to hi."""
  x = np.float32(zero_point) + np.asarray(values, np.float32) / np.float32(scale)
  t = np.trunc(x)
  r = t + (x - t >= 0.5) - (x - t <= -0.5)
  r = np.where(r < hi, r, np.float32(hi))
  return np.where(lo < r, r, np.float32(lo))


def quantization_params(f_min, f_max):
  """libcoral's QuantizationParams<uint8_t>: (scale, zero point) for [f_min, f_max] (f_min <= 0 <= f_max), in float."""
  f_min, f_max = np.float32(f_min), np.float32(f_max)
  if f_min == f_max:
    return np.float32(0), 0
  scale = np.float32((f_max - f_min) / np.float32(255))
  from_min = np.float32(0) - f_min / scale
  from_max = np.float32(255) - f_max / scale
  zp = from_min if abs(f_min / scale) < np.float32(255) + abs(f_max / scale) else from_max
  if zp < 0:
    return scale, 0
  if zp > 255:
    return scale, 255
  t = np.trunc(zp)
  return scale, int(t + (zp - t >= 0.5))


class Table:
  """A table of a flatbuffer `buf` at `pos`."""

  def __init__(self, buf, pos):
    self.buf, self.pos = buf, pos
    self.vt = pos - struct.unpack_from("<i", buf, pos)[0]
    self.vt_len = struct.unpack_from("<H", buf, self.vt)[0]

  def field(self, fid):
    """The field's position in the buffer, or 0 when it is absent."""
    o = 4 + 2 * fid
    if o >= self.vt_len:
      return 0
    rel = struct.unpack_from("<H", self.buf, self.vt + o)[0]
    return self.pos + rel if rel else 0

  def scalar(self, fid, fmt, default=0):
    p = self.field(fid)
    return struct.unpack_from("<" + fmt, self.buf, p)[0] if p else default

  def target(self, fid):
    p = self.field(fid)
    return p + _u32(self.buf, p) if p else 0

  def table(self, fid):
    t = self.target(fid)
    return Table(self.buf, t) if t else None

  def vector(self, fid):
    """(position of the first element, length), (0, 0) when absent."""
    t = self.target(fid)
    return (t + 4, _u32(self.buf, t)) if t else (0, 0)

  def tables(self, fid):
    start, n = self.vector(fid)
    return [Table(self.buf, start + 4 * i + _u32(self.buf, start + 4 * i)) for i in range(n)]

  def array(self, fid, fmt):
    start, n = self.vector(fid)
    return list(struct.unpack_from("<%d%s" % (n, fmt), self.buf, start)) if n else []

  def bytes(self, fid):
    start, n = self.vector(fid)
    return bytes(self.buf[start:start + n])


def _width(kind):
  return 4 if kind == "ref" else struct.calcsize("<" + kind)


class _Builder:
  """flatbuffers written back to front in front of `old`. A reference is the distance of an object from the end of
  the buffer, so references to old objects hold however much is written before them."""

  def __init__(self, old):
    self.tail = bytes(old) + bytes(-len(old) % 16)
    self.rev = bytearray()  # the new bytes, last one first

  def size(self):
    return len(self.tail) + len(self.rev)

  def old(self, pos):
    """A reference to the old object at `pos`."""
    return len(self.tail) - pos

  def _align(self, size, extra=0):
    self.rev += bytes(-(self.size() + extra) % size)

  def _put(self, data):
    self.rev += data[::-1]

  def scalar(self, fmt, value):
    data = struct.pack("<" + fmt, value)
    self._align(len(data))
    self._put(data)
    return self.size()

  def offset(self, ref):
    self._align(4)
    self._put(struct.pack("<I", self.size() + 4 - ref))
    return self.size()

  def blob(self, data, n, alignment=4):
    """A vector of n elements whose bytes are `data`."""
    self._align(max(alignment, 4), len(data))
    self._put(bytes(data))
    self._put(struct.pack("<I", n))
    return self.size()

  def vector(self, fmt, values, alignment=4):
    return self.blob(struct.pack("<%d%s" % (len(values), fmt), *values), len(values), alignment)

  def string(self, s):
    data = s.encode() if isinstance(s, str) else bytes(s)
    self._align(4, len(data) + 1)
    self._put(b"\0")
    self._put(data)
    self._put(struct.pack("<I", len(data)))
    return self.size()

  def offsets(self, refs):
    self._align(4, 4 * len(refs))
    for r in reversed(refs):
      self._put(struct.pack("<I", self.size() + 4 - r))
    self._put(struct.pack("<I", len(refs)))
    return self.size()

  def table(self, fields):
    """fields: {field id: (kind, value)}, a reference for "ref"."""
    start = self.size()
    where = {}
    # the widest first keeps the padding small
    for fid, (kind, value) in sorted(fields.items(), key=lambda f: -_width(f[1][0])):
      where[fid] = self.offset(value) if kind == "ref" else self.scalar(kind, value)
    self._align(4)
    end = self.size() + 4
    n = max(fields) + 1 if fields else 0
    slots = [0] * n
    for fid, at in where.items():
      slots[fid] = end - at
    vtable = struct.pack("<HH%dH" % n, 4 + 2 * n, end - start, *slots)
    # the vtable right before the table: the table's offset to it is the vtable's length
    self._put(struct.pack("<i", len(vtable)))
    self._put(vtable)
    return end

  def copy(self, t, kinds, changes):
    """Table `t` again with `changes` ({field id: (kind, value)}, None to drop a field); the other fields keep t's
    values, offsets referring to t's objects."""
    for fid in range(len(kinds), (t.vt_len - 4) // 2):
      if t.field(fid):
        raise ValueError("the model has a table field this editor does not know")
    fields = {}
    for fid, kind in enumerate(kinds):
      if fid in changes:
        if changes[fid] is not None:
          fields[fid] = changes[fid]
        continue
      p = t.field(fid)
      if p:
        fields[fid] = ("ref", self.old(p + _u32(t.buf, p))) if kind == "ref" else (kind, struct.unpack_from("<" + kind, t.buf, p)[0])
    return self.table(fields)

  def finish(self, root):
    self._align(16, 8)
    self._put(_TFL3)
    self._put(struct.pack("<I", self.size() + 4 - root))
    return bytes(self.rev[::-1]) + self.tail


class Model:
  """A .tflite model: what subgraph 0 holds, and edits to it written out by serialize()."""

  def __init__(self, data):
    self.data = bytes(data)
    if len(self.data) < 8 or self.data[4:8] != _TFL3:
      raise ValueError("not a TensorFlow Lite model")
    root = self.root = Table(self.data, _u32(self.data, 0))
    self.opcodes = [max(c.scalar(0, "b"), c.scalar(3, "i")) for c in root.tables(1)]
    self.subgraphs = root.tables(2)
    if not self.subgraphs:
      raise ValueError("the model has no subgraph")
    sg = self.subgraphs[0]
    self.tensors, self.ops, self.buffers = sg.tables(0), sg.tables(3), root.tables(4)
    self.outputs = sg.array(2, "i")
    if any(b.scalar(1, "Q") > 1 for b in self.buffers):
      raise ValueError("models with buffers outside the flatbuffer are not supported")
    s = root.target(3)
    self.description = self.data[s + 4:s + 4 + _u32(self.data, s)].decode("utf-8", "replace") if s else ""
    # edits
    self.tensor_changes = {}  # index: {"shape": [...], "quant": {...}}
    self.new_tensors = []  # (shape, type, buffer, name, quant)
    self.buffer_changes = {}  # index: bytes
    self.new_buffers = []
    self.reshape_changes = {}  # op index: new_shape
    self.new_ops = []  # (opcode index, inputs, outputs, options type, options fields)
    self.new_opcodes = []
    self.new_outputs = None
    self.new_description = None

  # *** reading ***

  def op_code(self, i):
    return self.opcodes[self.ops[i].scalar(0, "I")]

  def op_inputs(self, i):
    return self.ops[i].array(1, "i")

  def op_outputs(self, i):
    return self.ops[i].array(2, "i")

  def shape(self, t):
    return self.tensors[t].array(0, "i")

  def quant(self, t):
    """{"min", "max", "scale", "zero_point"} lists (empty without quantization)."""
    q = self.tensors[t].table(4)
    keys = ("min", "max", "scale", "zero_point")
    if q is None:
      return {k: [] for k in keys}
    return dict(zip(keys, (q.array(0, "f"), q.array(1, "f"), q.array(2, "f"), q.array(3, "q"))))

  def buffer_index(self, t):
    return self.tensors[t].scalar(2, "I")

  def buffer_data(self, b):
    return self.buffer_changes.get(b, self.buffers[b].bytes(0))

  # *** editing ***

  def add_buffer(self, data=b""):
    self.new_buffers.append(bytes(data))
    return len(self.buffers) + len(self.new_buffers) - 1

  def add_tensor(self, shape, ttype, buffer, name, quant=None):
    self.new_tensors.append((list(shape), ttype, buffer, name, quant))
    return len(self.tensors) + len(self.new_tensors) - 1

  def opcode_index(self, code):
    """FindOrCreateOpcode for a builtin operator."""
    codes = self.opcodes + self.new_opcodes
    if code in codes:
      return codes.index(code)
    self.new_opcodes.append(code)
    return len(codes)

  def add_op(self, code, inputs, outputs, options_type, options):
    self.new_ops.append((self.opcode_index(code), list(inputs), list(outputs), options_type, options))
    return len(self.ops) + len(self.new_ops) - 1

  def serialize(self):
    b = _Builder(self.data)

    def quant(q):
      return b.table({0: ("ref", b.vector("f", q["min"])), 1: ("ref", b.vector("f", q["max"])),
                      2: ("ref", b.vector("f", q["scale"])), 3: ("ref", b.vector("q", q["zero_point"]))})

    def buffer(data):
      return b.table({0: ("ref", b.blob(data, len(data), 16))} if data else {})

    # tensors
    tensors = [b.old(t.pos) for t in self.tensors]
    for i, ch in sorted(self.tensor_changes.items()):
      changes = {}
      if "shape" in ch:
        changes[0] = ("ref", b.vector("i", ch["shape"]))
      if "quant" in ch:
        changes[4] = ("ref", quant(ch["quant"]))
      tensors[i] = b.copy(self.tensors[i], _TENSOR, changes)
    for shape, ttype, buf, name, q in self.new_tensors:
      fields = {0: ("ref", b.vector("i", shape)), 1: ("b", ttype), 2: ("I", buf), 3: ("ref", b.string(name))}
      if q:
        fields[4] = ("ref", quant(q))
      tensors.append(b.table(fields))
    # operators
    ops = [b.old(o.pos) for o in self.ops]
    for i, new_shape in sorted(self.reshape_changes.items()):
      opts = b.table({0: ("ref", b.vector("i", new_shape))})
      ops[i] = b.copy(self.ops[i], _OPERATOR, {4: ("ref", opts)})
    for opcode, inputs, outputs, options_type, options in self.new_ops:
      fields = {}
      for fid, (kind, value) in options.items():
        fields[fid] = ("ref", b.vector("i", value)) if kind == "ints" else (kind, value)
      opts = b.table(fields)
      ops.append(b.table({0: ("I", opcode), 1: ("ref", b.vector("i", inputs)), 2: ("ref", b.vector("i", outputs)),
                          3: ("B", options_type), 4: ("ref", opts)}))
    # the subgraph
    sg_changes = {}
    if self.tensor_changes or self.new_tensors:
      sg_changes[0] = ("ref", b.offsets(tensors))
    if self.reshape_changes or self.new_ops:
      sg_changes[3] = ("ref", b.offsets(ops))
    if self.new_outputs is not None:
      sg_changes[2] = ("ref", b.vector("i", self.new_outputs))
    subgraphs = [b.old(s.pos) for s in self.subgraphs]
    subgraphs[0] = b.copy(self.subgraphs[0], _SUBGRAPH, sg_changes)
    # the model
    changes = {2: ("ref", b.offsets(subgraphs))}
    if self.buffer_changes or self.new_buffers:
      refs = [b.old(x.pos) for x in self.buffers]
      for i, data in sorted(self.buffer_changes.items()):
        refs[i] = buffer(data)
      refs += [buffer(data) for data in self.new_buffers]
      changes[4] = ("ref", b.offsets(refs))
    if self.new_opcodes:
      refs = [b.old(c.pos) for c in self.root.tables(1)]
      refs += [b.table({0: ("b", min(code, 127)), 3: ("i", code)}) for code in self.new_opcodes]
      changes[1] = ("ref", b.offsets(refs))
    if self.new_description is not None:
      changes[3] = ("ref", b.string(self.new_description))
    return b.finish(b.copy(self.root, _MODEL, changes))
