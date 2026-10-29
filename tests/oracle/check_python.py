"""Run the recorded pycoral cases (dump_pycoral.py) through the pycoral compatibility layer on this Python,
compare every output byte, and time each model.

  python check_python.py <test_data dir> <oracle dir>
"""
import os, sys, time
import numpy as np
from pycoral.utils.edgetpu import make_interpreter

DATA, ORACLE = sys.argv[1], sys.argv[2]
print(f"python {sys.version.split()[0]}, numpy {np.__version__}")
interps, failed, cases = {}, 0, 0
for line in open(os.path.join(ORACLE, "manifest.txt")):
  case, model, n = line.split()
  if model not in interps:
    interps[model] = make_interpreter(os.path.join(DATA, model))
    interps[model].allocate_tensors()
  it = interps[model]
  inp = it.get_input_details()[0]
  it.set_tensor(inp["index"], np.fromfile(os.path.join(ORACLE, f"{case}.in.bin"), inp["dtype"]).reshape(inp["shape"]))
  it.invoke()
  outs = it.get_output_details()
  # detection: the slots past the count are not defined by TFLite, compare the detections only
  count = int(it.get_tensor(outs[3]["index"])[0]) if len(outs) == 4 and "PostProcess" in outs[0]["name"] else None
  cases += 1
  for k, o in enumerate(outs):
    got = it.get_tensor(o["index"])
    want = np.fromfile(os.path.join(ORACLE, f"{case}.out{k}.bin"), o["dtype"]).reshape(got.shape)
    if count is not None and k < 3:
      got, want = got[:, :count], want[:, :count]
    if got.tobytes() != want.tobytes():
      print(f"MISMATCH {case} output {k}: {np.count_nonzero(got != want)} of {got.size} values differ")
      failed += 1
print(f"{cases} cases, {failed} mismatching outputs")
for line in open(os.path.join(ORACLE, "timing.txt")):
  model, ms = line.split()
  it = interps[model]
  ts = []
  for _ in range(100):
    st = time.perf_counter()
    it.invoke()
    ts.append((time.perf_counter() - st) * 1e3)
  print(f"{model}: coralrt {np.median(ts):.2f} ms, pycoral {ms} ms")
sys.exit(1 if failed else 0)
