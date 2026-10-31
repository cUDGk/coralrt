"""Compares the models make_models.py wrote with pycoral 2.0 and with coralrt: every tensor (name, shape, type,
quantization, data), operator, input, output and the description; then runs the CPU models of both on
tflite_runtime 2.5, so run it with the official Python.
  python compare_models.py <official dir> <coralrt dir>"""
import importlib.util
import pathlib
import sys

import numpy as np
from tflite_runtime.interpreter import Interpreter

# coralrt's flatbuffer reader, by path: the official pycoral is the one importable here
spec = importlib.util.spec_from_file_location("ctf", pathlib.Path(__file__).parents[2] / "python/pycoral/learn/_tflite.py")
ctf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ctf)

# the options compared: the new shape of a RESHAPE and the beta of a SOFTMAX (not those of the other ops)
RESHAPE, SOFTMAX = 22, 25


def describe(path):
  m = ctf.Model(path.read_bytes())
  tensors = []
  for t in range(len(m.tensors)):
    tt = m.tensors[t]
    q = {k: [float(x) for x in v] for k, v in m.quant(t).items()}
    tensors.append((tt.bytes(3), m.shape(t), tt.scalar(1, "b"), q, m.buffer_data(m.buffer_index(t))))
  ops = []
  for i in range(len(m.ops)):
    o = m.ops[i]
    opts = o.table(4)
    extra = None
    if opts is not None and m.op_code(i) == RESHAPE:
      extra = opts.array(0, "i")
    if opts is not None and m.op_code(i) == SOFTMAX:
      extra = opts.scalar(0, "f")
    ops.append((m.op_code(i), m.op_inputs(i), m.op_outputs(i), o.scalar(3, "B"), extra, o.bytes(5)))
  return {"tensors": tensors, "ops": ops, "outputs": m.outputs, "inputs": m.subgraphs[0].array(1, "i"),
          "description": m.description, "buffers": len(m.buffers), "subgraphs": len(m.subgraphs)}


official, mine = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
bad = 0
names = sorted(p.name for p in official.glob("*.tflite"))
for n in names:
  a, b = describe(official / n), describe(mine / n)
  diffs = [k for k in a if a[k] != b[k]]
  if "tensors" in diffs:
    for i, (x, y) in enumerate(zip(a["tensors"], b["tensors"])):
      if x != y:
        fields = [f for f, u, v in zip(("name", "shape", "type", "quant", "data"), x, y) if u != v]
        print("  tensor", i, x[0], fields, x[1:4] if "data" not in fields else "", y[1:4] if "data" not in fields else "")
        break
  print("SAME " if not diffs else "DIFF ", n, diffs)
  bad += bool(diffs)

# the CPU models: the same outputs on tflite_runtime 2.5 (or the same refusal)
for n in names:
  if "edgetpu" in n:
    continue
  outs = []
  for d in (official, mine):
    try:
      it = Interpreter(model_path=str(d / n))
      it.allocate_tensors()
      inp = it.get_input_details()[0]
      it.set_tensor(inp["index"], np.random.default_rng(3).integers(0, 256, inp["shape"], dtype=np.uint8))
      it.invoke()
      outs.append(it.get_tensor(it.get_output_details()[0]["index"]).copy())
    except RuntimeError as e:
      outs.append(str(e).split("Node")[0][-60:])
  if all(isinstance(o, np.ndarray) for o in outs):
    same = np.array_equal(outs[0], outs[1])
  else:
    same = outs[0] == outs[1]
  print("RUN  ", "same" if same else "DIFF", n, outs[0].reshape(-1)[:6] if isinstance(outs[0], np.ndarray) else outs[0])
  bad += not same
print("problems:", bad)
sys.exit(1 if bad else 0)
