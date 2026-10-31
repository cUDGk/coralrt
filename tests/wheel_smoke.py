"""Smoke test of an installed coralrt wheel (run it outside the source tree): pycoral's classification of the
parrot, imprinting, backprop and the pipeline. Without a stick the CPU model is classified and the pipeline left out;
--cpu does that without looking for a stick at all (no libusb needed, as on CI).
  python -I wheel_smoke.py <test_data> [--cpu]"""
import pathlib
import sys

import numpy as np
from PIL import Image

import coralrt
from pycoral.adapters import classify, common
from pycoral.learn.backprop.softmax_regression import SoftmaxRegression
from pycoral.learn.imprinting.engine import ImprintingEngine
from pycoral.pipeline.pipelined_model_runner import PipelinedModelRunner
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import list_edge_tpus, make_interpreter

td = pathlib.Path(sys.argv[1])
print(sys.version.split()[0], "coralrt", coralrt.version(), "from", pathlib.Path(coralrt.__file__).parent)
devices = [] if "--cpu" in sys.argv[2:] else list_edge_tpus()
print("devices", devices)
tpu = "_edgetpu" if devices else ""
it = make_interpreter(str(td / f"mobilenet_v2_1.0_224_inat_bird_quant{tpu}.tflite"))
it.allocate_tensors()
parrot = Image.open(td / "parrot.jpg").convert("RGB")
common.set_input(it, parrot.resize(common.input_size(it), Image.LANCZOS))
it.invoke()
top = classify.get_classes(it, top_k=1)[0]
label = read_label_file(str(td / "inat_bird_labels.txt")).get(top.id, top.id)
print("parrot:", label, round(top.score, 5))
assert label == "Ara macao (Scarlet Macaw)", label
e = ImprintingEngine(str(td / f"mobilenet_v1_1.0_224_l2norm_quant{tpu}.tflite"))
e.train(np.ones(e.embedding_dim, np.float32), 0)
print("imprinting model", len(e.serialize_model()), "bytes")
sr = SoftmaxRegression(1024, 2)
print("backprop model", len(sr.serialize_model(str(td / f"mobilenet_v1_1.0_224_quant_embedding_extractor{tpu}.tflite"))), "bytes")
if devices:
  segs = [make_interpreter(str(td / f"pipeline/inception_v3_299_quant_segment_{i}_of_2_edgetpu.tflite")) for i in range(2)]
  for s in segs:
    s.allocate_tensors()
  runner = PipelinedModelRunner(segs)
  runner.push({segs[0].get_input_details()[0]["name"]: np.asarray(parrot.resize((299, 299)), np.uint8)[None]})
  runner.push({})
  print("pipeline top class", int(list(runner.pop().values())[0].argmax()), runner.pop())
