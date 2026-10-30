"""Speed of the official stack (Python 3.9, pycoral 2.0) against coralrt's, taking turns so heat and background load
hit both alike: each round runs every model in one process per side, the median of the rounds is reported.
Only invoke() is timed, as pycoral's examples do.

  python -I bench.py <official python> <coralrt python> <test_data dir> [rounds] [invokes]
"""
import json, os, statistics, subprocess, sys, time

if sys.argv[1] == "--worker":
  from pycoral.utils.edgetpu import make_interpreter
  data, n = sys.argv[2], int(sys.argv[3])
  out = {}
  for model in sorted(f for f in os.listdir(data) if f.endswith("_edgetpu.tflite")):
    it = make_interpreter(os.path.join(data, model))
    it.allocate_tensors()
    for _ in range(10):
      it.invoke()
    times = []
    for _ in range(n):
      t = time.perf_counter()
      it.invoke()
      times.append(time.perf_counter() - t)
    out[model] = statistics.median(times) * 1000
  print(json.dumps(out))
  sys.exit()

official, coralrt, data = sys.argv[1:4]
rounds = int(sys.argv[4]) if len(sys.argv) > 4 else 5
n = sys.argv[5] if len(sys.argv) > 5 else "200"
ms = {"official": [], "coralrt": []}
for r in range(rounds):
  for who, py in [("official", official), ("coralrt", coralrt)][:: 1 if r % 2 == 0 else -1]:
    run = subprocess.run([py, "-I", os.path.abspath(__file__), "--worker", data, n], capture_output=True, text=True, timeout=600)
    if run.returncode:
      sys.exit(f"{who}: {run.stderr}")
    ms[who].append(json.loads(run.stdout.strip().splitlines()[-1]))
print(f"{'model':56} {'official':>9} {'coralrt':>9}  ms per invoke, median of {rounds} rounds x {n}")
for model in ms["official"][0]:
  a, b = (statistics.median(x[model] for x in ms[who]) for who in ("official", "coralrt"))
  print(f"{model:56} {a:9.2f} {b:9.2f}  ({(b / a - 1) * 100:+.1f}%)")
