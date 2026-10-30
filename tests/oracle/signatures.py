"""Print each model's signature lists, and for models with a signature the outputs it gives for a fixed input, read
through the signature's names. Run it with the official Python (tflite_runtime 2.5) and with coralrt's: the output
must be the same.

  python -I signatures.py <model.tflite> ...
"""
import sys

import numpy as np
from pycoral.utils.edgetpu import make_interpreter

for path in sys.argv[1:]:
  it = make_interpreter(path)
  it.allocate_tensors()
  full = it._get_full_signature_list()
  print(path.replace("\\", "/").rsplit("/", 1)[-1], full, it.get_signature_list())
  for key, sig in full.items():
    for k, (name, t) in enumerate(sig["inputs"].items()):
      d = next(d for d in it.get_input_details() if d["index"] == t)
      it.set_tensor(t, (np.arange(np.prod(d["shape"])) * (k + 2) - 3).astype(d["dtype"]).reshape(d["shape"]))
    it.invoke()
    for name, t in sig["outputs"].items():
      print(f"  {key}/{name}:", it.get_tensor(t).tolist())
