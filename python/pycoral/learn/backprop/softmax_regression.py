"""pycoral.learn.backprop.softmax_regression on coralrt: a softmax regression layer (fully connected + softmax)
trained on the CPU with SGD on embeddings from an embedding extractor, then appended to that model, as libcoral's
SoftmaxRegressionModel does it (in float32, numpy in place of Eigen; the starting weights are random either way).
"""
import logging

import numpy as np

from pycoral.learn import _tflite

_FULLY_CONNECTED, _RESHAPE, _SOFTMAX = 9, 22, 25
_UINT8, _INT32 = 3, 2


def _softmax(logits):
  exps = np.exp(logits - logits.max(axis=1, keepdims=True))
  return exps / exps.sum(axis=1, keepdims=True)


class SoftmaxRegression:
  """The last layer of a classification model (fully connected, then softmax), trained on device."""

  def __init__(self, feature_dim=None, num_classes=None, weight_scale=0.01, reg=0.0):
    """feature_dim: the embedding's length; num_classes: the number of classes; weight_scale: the factor for the
    starting weights (standard normal draws); reg: the regularization strength (added to the loss)."""
    self._dim, self._classes = int(feature_dim), int(num_classes)
    self._reg = np.float32(reg)
    self._w = (np.float32(weight_scale) * np.random.standard_normal((self._dim, self._classes))).astype(np.float32)
    self._b = np.zeros((1, self._classes), np.float32)
    self._logit_min, self._logit_max = np.float32(np.inf), np.float32(-np.inf)

  def _logits(self, x):
    return (np.asarray(x, np.float32) @ self._w + self._b).astype(np.float32)

  def _step(self, x, labels, learning_rate):
    """GetLoss, GetGrads and SgdUpdate for one batch; returns the loss."""
    logits = self._logits(x)
    self._logit_min = min(self._logit_min, logits.min())
    self._logit_max = max(self._logit_max, logits.max())
    probs = _softmax(logits).astype(np.float32)
    loss = np.float32(np.mean(-(labels * np.log(probs)).sum(axis=1)))
    loss = np.float32(loss + 0.5 * self._reg * (self._w.T @ self._w).sum())
    dprobs = (np.float32(1.0 / len(probs)) * -labels / probs).astype(np.float32)
    # SoftmaxGradient: each row times its local gradient diag(p) (I - p), made first as libcoral does
    local = probs[:, :, None] * (np.eye(self._classes, dtype=np.float32)[None] - probs[:, None, :])
    dlogits = np.einsum("ij,ijk->ik", dprobs, local).astype(np.float32)
    self._w -= np.float32(learning_rate) * (x.T @ dlogits).astype(np.float32)
    self._b -= np.float32(learning_rate) * dlogits.sum(axis=0, keepdims=True)
    return loss

  def serialize_model(self, in_model_path):
    """Appends the learned layer (weights and biases quantized to uint8 and int32) to the embedding extractor at
    in_model_path; returns the new model as bytes."""
    with open(in_model_path, "rb") as f:
      m = _tflite.Model(f.read())
    if len(m.subgraphs) != 1:
      raise RuntimeError("Model must have one and only one subgraph. Actual: %d." % len(m.subgraphs))
    if len(m.outputs) != 1:
      raise RuntimeError("Model must have one and only one output. Actual: %d." % len(m.outputs))
    logging.info("Logit min: %s, max: %s", self._logit_min, self._logit_max)
    embedding = m.outputs[0]
    in_shape, in_quant = m.shape(embedding), m.quant(embedding)
    dim = in_shape[-1]
    # the weights in Eigen's column-major order: one row of the kernel per class
    weights = self._w.T.reshape(-1)
    w_min, w_max = min(np.float32(0), weights.min()), max(np.float32(0), weights.max())
    w_scale, w_zp = _tflite.quantization_params(w_min, w_max)
    kernel = _tflite.quantize(weights, w_scale, w_zp, 0, 255).astype(np.uint8)
    biases = self._b.reshape(-1)
    b_min, b_max = min(np.float32(0), biases.min()), max(np.float32(0), biases.max())
    b_scale = np.float32(np.float32(in_quant["scale"][0]) * w_scale)
    bias = _tflite.quantize(biases, b_scale, 0, -2147483648.0, 2147483647.0)
    # static_cast<int32_t> on x86: out of range is the minimum
    bias = np.where(bias >= 2147483648.0, -2147483648.0, bias).astype(np.int64).astype(np.int32)
    o_min, o_max = min(np.float32(0), self._logit_min), max(np.float32(0), self._logit_max)
    o_scale, o_zp = _tflite.quantization_params(o_min, o_max)
    n = len(weights) // dim
    # AppendLinearLayer: a fully connected layer after the graph's output
    w_t = m.add_tensor([n, dim], _UINT8, m.add_buffer(kernel.tobytes()), "Appended/FC/Weights",
                       {"min": [w_min], "max": [w_max], "scale": [w_scale], "zero_point": [w_zp]})
    b_t = m.add_tensor([n], _INT32, m.add_buffer(bias.astype("<i4").tobytes()), "Appended/FC/Bias",
                       {"min": [b_min], "max": [b_max], "scale": [b_scale], "zero_point": [0]})
    out_quant = {"min": [o_min], "max": [o_max], "scale": [o_scale], "zero_point": [o_zp]}
    out = m.add_tensor(in_shape[:-1] + [n], _UINT8, m.add_buffer(), "Appended/FC/Output", out_quant)
    m.add_op(_FULLY_CONNECTED, [embedding, w_t, b_t], [out], _tflite.FULLY_CONNECTED_OPTIONS, {})
    shape = in_shape[:-1] + [n]
    if len(in_shape) == 4:
      shape = [shape[0], shape[-1]]
      reshaped = m.add_tensor(shape, _UINT8, m.add_buffer(), "Imprinting/Reshape/Output", out_quant)
      m.add_op(_RESHAPE, [out], [reshaped], _tflite.RESHAPE_OPTIONS, {0: ("ints", shape)})
      out = reshaped
    softmax = m.add_tensor(shape, _UINT8, m.add_buffer(), "Imprinting/Softmax/Output",
                           {"min": [0.0], "max": [1.0], "scale": [np.float32(1.0) / np.float32(256)], "zero_point": [0]})
    m.add_op(_SOFTMAX, [out], [softmax], _tflite.SOFTMAX_OPTIONS, {0: ("f", 1.0)})
    m.new_outputs = [softmax]
    return m.serialize()

  def get_accuracy(self, mat_x, labels):
    """The fraction of the rows of mat_x (N x feature_dim embeddings) whose top class is their label."""
    result = self._logits(mat_x).argmax(axis=1)
    labels = list(labels)
    return float(np.float32(sum(int(r == l) for r, l in zip(result, labels))) / np.float32(len(labels)))

  def train_with_sgd(self, data, num_iter, learning_rate, batch_size=100, print_every=100):
    """Trains with stochastic gradient descent on data = {"data_train", "labels_train", "data_val", "labels_val"}
    (embeddings N x feature_dim and their labels): num_iter batches of batch_size rows drawn at random."""
    x = np.asarray(data["data_train"], np.float32)
    if x.ndim != 2:
      raise RuntimeError("Incompatible buffer dimension!")
    labels = np.asarray(data["labels_train"], np.int64)
    for i in range(num_iter):
      idx = np.random.randint(0, len(x), batch_size)
      onehot = np.zeros((batch_size, self._classes), np.float32)
      onehot[np.arange(batch_size), labels[idx]] = 1.0
      loss = self._step(x[idx], onehot, learning_rate)
      if print_every > 0 and i % print_every == 0:
        logging.info("Loss: %s", loss)
        logging.info("Train Acc: %s", self.get_accuracy(x, labels))
        logging.info("Valid Acc: %s", self.get_accuracy(data["data_val"], data["labels_val"]))
