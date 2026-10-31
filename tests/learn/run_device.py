"""Runs with whichever pycoral the running Python imports: the Edge TPU imprinting models from make_models.py, the
backprop models from train_backprop.py (CPU and Edge TPU), and inception_v3 as a 2-segment pipeline and whole. Saves every output to <out dir>/results.npz for compare_device.py.
  python run_device.py <test_data> <models dir> <backprop dir> <out dir>"""
import pathlib
import sys

import numpy as np
from PIL import Image
from pycoral.pipeline.pipelined_model_runner import PipelinedModelRunner
from pycoral.utils.edgetpu import make_interpreter

td, models, bp, out = (pathlib.Path(a) for a in sys.argv[1:5])
out.mkdir(parents=True, exist_ok=True)
results = {}


def run(path, n=3):
  it = make_interpreter(str(path))
  it.allocate_tensors()
  d = it.get_input_details()[0]
  for k in range(n):
    it.set_tensor(d["index"], np.random.default_rng(k).integers(0, 256, d["shape"], dtype=np.uint8))
    it.invoke()
    results[f"{path.stem}_{k}"] = it.get_tensor(it.get_output_details()[0]["index"]).copy()


for p in sorted(models.glob("*l2norm*edgetpu*.tflite")) + sorted(bp.glob("*.tflite")):
  run(p)

images = [np.asarray(Image.open(td / n).convert("RGB").resize((299, 299), Image.BILINEAR), np.uint8)[None]
          for n in ("parrot.jpg", "grace_hopper.bmp")]
images += [np.random.default_rng(100 + k).integers(0, 256, (1, 299, 299, 3), dtype=np.uint8) for k in range(3)]
segments = [make_interpreter(str(td / f"pipeline/inception_v3_299_quant_segment_{i}_of_2_edgetpu.tflite")) for i in range(2)]
for s in segments:
  s.allocate_tensors()
runner = PipelinedModelRunner(segments)
name = segments[0].get_input_details()[0]["name"]
for img in images:
  runner.push({name: img})
runner.push({})
k = 0
while r := runner.pop():
  results[f"pipe_{k}"] = list(r.values())[0]
  k += 1
del runner
full = make_interpreter(str(td / "inception_v3_299_quant_edgetpu.tflite"))
full.allocate_tensors()
for j, img in enumerate(images):
  full.set_tensor(full.get_input_details()[0]["index"], img)
  full.invoke()
  results[f"full_{j}"] = full.get_tensor(full.get_output_details()[0]["index"]).copy()
np.savez(out / "results.npz", **results)
print(len(results), "outputs;", "pipeline results", k)
