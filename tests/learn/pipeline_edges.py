"""The pipeline API's edges with whichever pycoral runs it (needs the stick): segments in the wrong order, a float
input, the end, push and pop after the end. The printout of pycoral 2.0 and of coralrt must be the same.
  python pipeline_edges.py <test_data>"""
import pathlib
import sys

import numpy as np
from pycoral.pipeline.pipelined_model_runner import PipelinedModelRunner
from pycoral.utils.edgetpu import make_interpreter

td = pathlib.Path(sys.argv[1])
segs = [make_interpreter(str(td / f"pipeline/inception_v3_299_quant_segment_{i}_of_2_edgetpu.tflite")) for i in range(2)]
for s in segs:
  s.allocate_tensors()
try:
  PipelinedModelRunner(segs[::-1])
except ValueError as e:
  print("reversed:", type(e).__name__, e)
runner = PipelinedModelRunner(segs)
name = segs[0].get_input_details()[0]["name"]
try:
  runner.push({name: np.zeros((1, 299, 299, 3), np.float32)})
except ValueError as e:
  print("float input:", type(e).__name__, e)
runner.set_input_queue_size(2)
runner.set_output_queue_size(2)
runner.push({name: np.zeros((1, 299, 299, 3), np.uint8)})
r = runner.pop()
print("pop:", sorted((k, v.shape, v.dtype.name) for k, v in r.items()))
runner.push({})
print("pop after end:", runner.pop())
try:
  runner.push({name: np.zeros((1, 299, 299, 3), np.uint8)})
except RuntimeError as e:
  print("push after end:", type(e).__name__, e)
try:
  runner.push({})
except RuntimeError as e:
  print("end twice:", type(e).__name__, e)
print("interpreters:", len(runner.interpreters()))
