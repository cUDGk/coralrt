"""Makes imprinting and backprop models with whichever pycoral the running Python imports (pycoral 2.0 for the
reference, coralrt's for the check), from the same inputs both times.
  python make_models.py <test_data> <out dir>"""
import pathlib
import sys

import numpy as np
from pycoral.learn.backprop.softmax_regression import SoftmaxRegression
from pycoral.learn.imprinting.engine import ImprintingEngine

td, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
emb = np.random.default_rng(7).random((9, 1024), dtype=np.float32)

for name in ["mobilenet_v1_1.0_224_l2norm_quant", "mobilenet_v1_1.0_224_l2norm_quant_edgetpu"]:
  for keep in [False, True]:
    tag = f"{name}_{int(keep)}"
    e = ImprintingEngine(str(td / f"{name}.tflite"), keep)
    (out / f"{tag}_extractor.tflite").write_bytes(e.serialize_extractor_model())
    base = e.num_classes
    for c in range(3):
      for k in range(3):
        e.train(emb[3 * c + k], base + c)
    (out / f"{tag}_trained.tflite").write_bytes(e.serialize_model())
    e.train(emb[0], base)
    (out / f"{tag}_retrained.tflite").write_bytes(e.serialize_model())
    # training further from a model trained by imprinting
    e2 = ImprintingEngine(str(out / f"{tag}_trained.tflite"), keep_classes=True)
    e2.train(emb[4], base + 1)
    e2.train(emb[5], base + 3)
    (out / f"{tag}_continued.tflite").write_bytes(e2.serialize_model())
    print(tag, "classes", base, e.num_classes, e2.num_classes, "dim", e.embedding_dim)

# backprop: zero starting weights, one sample, one step: deterministic and exact on both
x = np.zeros((4, 1024), np.float32)
x[:, 0] = 0.5
for classes in [2, 3]:
  sr = SoftmaxRegression(1024, classes, weight_scale=0.0)
  sr.train_with_sgd({"data_train": x, "labels_train": [1] * 4, "data_val": x, "labels_val": [1] * 4},
                    num_iter=1, learning_rate=0.1, batch_size=1, print_every=0)
  for name in ["mobilenet_v1_1.0_224_quant_embedding_extractor", "mobilenet_v1_1.0_224_quant_embedding_extractor_edgetpu"]:
    (out / f"{name}_backprop{classes}.tflite").write_bytes(sr.serialize_model(str(td / f"{name}.tflite")))
  print("backprop", classes, "accuracy", sr.get_accuracy(x, [1] * 4))
