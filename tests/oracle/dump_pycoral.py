"""Record what the official stack (pycoral 2.0 + tflite_runtime 2.5 + libedgetpu, Python 3.9) outputs, as the
reference coralrt must match byte for byte.

  ..\\..\\..\\.venv\\Scripts\\python.exe dump_pycoral.py <test_data dir> <out dir>

Writes <case>.in.bin and <case>.out<k>.bin per case, manifest.txt (case, model, number of outputs) and
timing.txt (median invoke milliseconds per model).
"""
import os, sys, time
import numpy as np
from PIL import Image
from pycoral.utils.edgetpu import make_interpreter

DATA, OUT = sys.argv[1], sys.argv[2]
MODELS = {
  "classify": ("mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite", "parrot.jpg"),
  "ssd": ("ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite", "grace_hopper.bmp"),
  "lightning": ("movenet_single_pose_lightning_ptq_edgetpu.tflite", "grace_hopper.bmp"),
  "thunder": ("movenet_single_pose_thunder_ptq_edgetpu.tflite", "grace_hopper.bmp"),
}

os.makedirs(OUT, exist_ok=True)
manifest, timing = [], []
for key, (model, image) in MODELS.items():
  it = make_interpreter(os.path.join(DATA, model))
  it.allocate_tensors()
  inp = it.get_input_details()[0]
  shape, dtype = inp["shape"], inp["dtype"]
  # the photo (resized like the pycoral examples), noise, and the two extremes
  img = Image.open(os.path.join(DATA, image)).convert("RGB").resize((shape[2], shape[1]), Image.LANCZOS)
  rng = np.random.default_rng(1234)
  cases = {"photo": np.asarray(img, dtype=dtype).reshape(shape)}
  for s in range(3):
    cases[f"noise{s}"] = rng.integers(0, 256, size=shape).astype(dtype)
  cases["zeros"] = np.zeros(shape, dtype)
  cases["max"] = np.full(shape, 255, dtype)
  outs = it.get_output_details()
  for name, x in cases.items():
    it.set_tensor(inp["index"], x)
    it.invoke()
    case = f"{key}_{name}"
    x.tofile(os.path.join(OUT, f"{case}.in.bin"))
    for k, o in enumerate(outs):
      it.get_tensor(o["index"]).tofile(os.path.join(OUT, f"{case}.out{k}.bin"))
    manifest.append(f"{case}\t{model}\t{len(outs)}")
  ts = []
  for _ in range(100):
    st = time.perf_counter()
    it.invoke()
    ts.append((time.perf_counter() - st) * 1e3)
  timing.append(f"{model}\t{np.median(ts):.3f}")
  print(key, "ok", f"{np.median(ts):.2f} ms")
  del it

open(os.path.join(OUT, "manifest.txt"), "w").write("\n".join(manifest) + "\n")
open(os.path.join(OUT, "timing.txt"), "w").write("\n".join(timing) + "\n")
