"""Run every op case's model (build.py) in the official tflite_runtime 2.5 on the CPU, and record what goes in and
comes out for op_check (examples/op_check.rs). Needs the official Python (pycoral 2.0's .venv).

  python -I official.py <out dir> [case name prefix ...]

<out dir>/oracle/manifest.txt, one line per case, tab separated:
  case  status  tol  invokes  inputs  outputs  message
status: ok, load_error (2.5 cannot read the model: not a case), run_error (2.5 refuses it: coralrt must too);
inputs / outputs: "shape:dtype" separated by ";" with shapes like 2x3 (empty for a scalar).
Files: <case>.<invoke>.in<k>.bin and <case>.<invoke>.out<k>.bin (strings in TFLite's string layout).
"""
import os, subprocess, sys, tempfile

import numpy as np
from tflite_runtime.interpreter import Interpreter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spec import random_values, tensor_bytes  # noqa: E402
import cases  # noqa: E402

NAMES = {np.dtype(k): k for k in ["float32", "float16", "float64", "int8", "uint8", "int16", "int32", "int64", "uint32", "uint64",
                                 "bool", "complex64", "complex128"]}


def dtype_name(d):
  d = np.dtype(d)
  return "string" if d.kind in "OSU" else NAMES[d]


def fmt(shape, dtype):
  return "x".join(str(int(s)) for s in shape) + ":" + dtype


def record(case, models, oracle):
  """run one case; its manifest line"""
  path = os.path.join(models, case.name + ".tflite")
  g = case.graphs[0]
  ins = [g.tensors[i] for i in g.inputs]
  status, msg, in_desc, out_desc = "ok", "", [], []
  try:
    it = Interpreter(model_path=path)
  except Exception as e:
    return "	".join([case.name, "load_error", str(case.tol), str(case.invokes), "", "", str(e).splitlines()[0][:200]])
  try:
    details = it.get_input_details()
    for d, t in zip(details, ins):
      if t.resize is not None:
        it.resize_tensor_input(d["index"], t.resize)
    it.allocate_tensors()
    rng = np.random.default_rng(case.seed)
    for inv in range(case.invokes):
      for k, (d, t) in enumerate(zip(details, ins)):
        v = random_values(t, rng)
        it.set_tensor(d["index"], v)
        with open(os.path.join(oracle, f"{case.name}.{inv}.in{k}.bin"), "wb") as f:
          f.write(tensor_bytes(t.dtype, v))
        if inv == 0:
          in_desc.append(fmt(np.shape(v), t.dtype))
      it.invoke()
      out_desc = []
      for k, d in enumerate(it.get_output_details()):
        v = it.get_tensor(d["index"])
        name = "string" if d["dtype"] in (np.bytes_, np.object_) else dtype_name(d["dtype"])
        with open(os.path.join(oracle, f"{case.name}.{inv}.out{k}.bin"), "wb") as f:
          f.write(tensor_bytes(name, v))
        out_desc.append(fmt(np.shape(v), name))
  except Exception as e:
    status, msg = "run_error", (str(e).strip().splitlines() or [type(e).__name__])[-1][:200]
  return "	".join([case.name, status, str(case.tol), str(case.invokes), ";".join(in_desc), ";".join(out_desc), msg])


def worker(out, names):
  """record the named cases, a line each to stdout; "start <case>" first, so the parent knows which one crashed"""
  models, oracle = os.path.join(out, "models"), os.path.join(out, "oracle")
  by_name = {c.name: c for c in cases.all_cases()}
  for name in names:
    print("start " + name, flush=True)
    print("line " + record(by_name[name], models, oracle), flush=True)


def main():
  # tflite_runtime 2.5 aborts the process on some malformed cases: the cases run in a child process, and one that
  # takes the child down is recorded as refused (coralrt must refuse it too, without crashing)
  out = sys.argv[1]
  oracle = os.path.join(out, "oracle")
  os.makedirs(oracle, exist_ok=True)
  prefixes = sys.argv[2:]
  todo = [c.name for c in cases.all_cases() if not prefixes or any(c.name.startswith(p) for p in prefixes)]
  lines = []
  while todo:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
      f.write("
".join(todo))
    run = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "--worker", out, f.name], capture_output=True, text=True)
    os.unlink(f.name)
    started = None
    for l in run.stdout.splitlines():
      if l.startswith("start "):
        started = l[6:]
      elif l.startswith("line "):
        lines.append(l[5:])
        started = None
    done = {l.split("	")[0] for l in lines}
    if started is not None:
      case = next(c for c in cases.all_cases() if c.name == started)
      lines.append("	".join([started, "run_error", str(case.tol), str(case.invokes), "", "", f"tflite 2.5 crashed (exit {run.returncode})"]))
      done.add(started)
    elif run.returncode != 0:
      raise RuntimeError(run.stderr[-2000:])
    todo = [n for n in todo if n not in done]
  with open(os.path.join(oracle, "manifest.txt"), "w", encoding="utf-8", newline="
") as f:
    f.write("
".join(lines) + "
")
  counts = {}
  for l in lines:
    counts[l.split("	")[1]] = counts.get(l.split("	")[1], 0) + 1
  print(f"{len(lines)} cases: {counts}")


if __name__ == "__main__":
  if sys.argv[1] == "--worker":
    worker(sys.argv[2], open(sys.argv[3]).read().split("
"))
  else:
    main()
