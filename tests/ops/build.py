"""Write the model of every op case (cases.py) as a .tflite file, with TensorFlow's flatbuffer schema (ObjectAPI).
Needs TensorFlow (the .venv-tf one); the models use only what TensorFlow Lite 2.5's schema has.

  python -I build.py <out dir> [case name prefix ...]
"""
import os, sys

import flatbuffers
import numpy as np
from tensorflow.lite.python import schema_py_generated as schema

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spec import Case, tensor_bytes  # noqa: E402
import cases  # noqa: E402

TYPES = {"float32": 0, "float16": 1, "int32": 2, "uint8": 3, "int64": 4, "string": 5, "bool": 6, "int16": 7, "complex64": 8,
         "int8": 9, "float64": 10, "complex128": 11, "uint64": 12, "resource": 13, "variant": 14, "uint32": 15}


def build(case: Case) -> bytes:
  m = schema.ModelT()
  m.version = 3
  m.description = case.name
  m.buffers = [schema.BufferT()]  # buffer 0: the empty one
  codes, m.operatorCodes, m.subgraphs = {}, [], []
  for g in case.graphs:
    sg = schema.SubGraphT()
    sg.name = g.name
    sg.tensors = []
    for i, t in enumerate(g.tensors):
      tt = schema.TensorT()
      tt.name = t.name or f"t{i}"
      tt.shape = np.array(t.shape, np.int32)
      tt.type = TYPES[t.dtype]
      tt.buffer = 0
      tt.isVariable = bool(t.var)
      if t.sig is not None:
        tt.shapeSignature = np.array(t.sig, np.int32)
      if t.quant is not None:
        q = schema.QuantizationParametersT()
        if len(t.quant) == 3:
          q.scale, q.zeroPoint, q.quantizedDimension = np.array(t.quant[0], np.float32), np.array(t.quant[1], np.int64), t.quant[2]
        else:
          q.scale, q.zeroPoint = np.array([t.quant[0]], np.float32), np.array([t.quant[1]], np.int64)
        tt.quantization = q
      if t.data is not None:
        b = schema.BufferT()
        b.data = np.frombuffer(tensor_bytes(t.dtype, t.data), np.uint8)
        m.buffers.append(b)
        tt.buffer = len(m.buffers) - 1
      if t.sparsity is not None:
        sp = schema.SparsityParametersT()
        sp.traversalOrder = np.array(t.sparsity["traversal_order"], np.int32)
        sp.blockMap = np.array(t.sparsity.get("block_map", []), np.int32)
        sp.dimMetadata = []
        for fmt, dense, segments, indices in t.sparsity["dim_metadata"]:
          d = schema.DimensionMetadataT()
          d.format, d.denseSize = fmt, dense
          if fmt == 1:
            d.arraySegmentsType, d.arrayIndicesType = 1, 1  # Int32Vector
            d.arraySegments, d.arrayIndices = schema.Int32VectorT(), schema.Int32VectorT()
            d.arraySegments.values, d.arrayIndices.values = np.array(segments, np.int32), np.array(indices, np.int32)
          sp.dimMetadata.append(d)
        tt.sparsity = sp
      sg.tensors.append(tt)
    sg.inputs, sg.outputs = np.array(g.inputs, np.int32), np.array(g.outputs, np.int32)
    sg.operators = []
    for op in g.ops:
      if op.op.startswith("CUSTOM:"):
        code, custom = 32, op.op[len("CUSTOM:"):]
      else:
        code, custom = getattr(schema.BuiltinOperator, op.op), None
      key = (code, custom, op.version)
      if key not in codes:
        codes[key] = len(m.operatorCodes)
        oc = schema.OperatorCodeT()
        oc.deprecatedBuiltinCode, oc.builtinCode, oc.version = min(code, 127), code, op.version
        if custom is not None:
          oc.customCode = custom
        m.operatorCodes.append(oc)
      o = schema.OperatorT()
      o.opcodeIndex = codes[key]
      o.inputs, o.outputs = np.array(op.inputs, np.int32), np.array(op.outputs, np.int32)
      if op.intermediates:
        o.intermediates = np.array(op.intermediates, np.int32)
      if op.options is not None:
        name, fields = op.options
        o.builtinOptionsType = getattr(schema.BuiltinOptions, name)
        opts = getattr(schema, name + "T")()
        for k, v in fields.items():
          if not hasattr(opts, k):
            raise AttributeError(f"{case.name}: {name} has no field {k}")
          setattr(opts, k, v)
        o.builtinOptions = opts
      if op.custom is not None:
        o.customOptions = np.frombuffer(op.custom() if callable(op.custom) else op.custom, np.uint8)
      sg.operators.append(o)
    m.subgraphs.append(sg)
  b = flatbuffers.Builder(1024)
  b.Finish(m.Pack(b), file_identifier=b"TFL3")
  return bytes(b.Output())


if __name__ == "__main__":
  out = os.path.join(sys.argv[1], "models")
  os.makedirs(out, exist_ok=True)
  prefixes = sys.argv[2:]
  n = 0
  for case in cases.all_cases():
    if prefixes and not any(case.name.startswith(p) for p in prefixes):
      continue
    with open(os.path.join(out, case.name + ".tflite"), "wb") as f:
      f.write(build(case))
    n += 1
  print(f"{n} models in {out}")
