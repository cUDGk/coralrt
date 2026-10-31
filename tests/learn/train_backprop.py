"""coralrt only (needs the stick): embeddings of made-up images from the Edge TPU extractor, a softmax regression
trained on them, and the layer appended to both extractors (CPU and Edge TPU).
  python train_backprop.py <test_data> <out dir>"""
import pathlib
import sys

import numpy as np
from pycoral.learn.backprop.softmax_regression import SoftmaxRegression
from pycoral.utils.edgetpu import make_interpreter

td, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
it = make_interpreter(str(td / "mobilenet_v1_1.0_224_quant_embedding_extractor_edgetpu.tflite"))
it.allocate_tensors()
inp, outp = it.get_input_details()[0], it.get_output_details()[0]
rng = np.random.default_rng(11)
embeddings, labels = [], []
for k in range(60):
  label = k % 3
  # three kinds of images: dark, middle, bright
  img = rng.integers(0, 80, inp["shape"], dtype=np.uint8) + np.uint8(80 * label)
  it.set_tensor(inp["index"], img)
  it.invoke()
  scale, zp = outp["quantization"]
  embeddings.append(scale * (it.get_tensor(outp["index"]).reshape(-1).astype(np.float32) - zp))
  labels.append(label)
x = np.array(embeddings, np.float32)
data = {"data_train": x[:45], "labels_train": labels[:45], "data_val": x[45:], "labels_val": labels[45:]}
np.random.seed(5)
sr = SoftmaxRegression(x.shape[1], 3)
sr.train_with_sgd(data, num_iter=300, learning_rate=0.01, batch_size=16, print_every=0)
print("train", sr.get_accuracy(x[:45], labels[:45]), "val", sr.get_accuracy(x[45:], labels[45:]))
for name in ["mobilenet_v1_1.0_224_quant_embedding_extractor", "mobilenet_v1_1.0_224_quant_embedding_extractor_edgetpu"]:
  (out / f"{name}_trained.tflite").write_bytes(sr.serialize_model(str(td / f"{name}.tflite")))
