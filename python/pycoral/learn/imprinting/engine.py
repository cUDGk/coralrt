"""pycoral.learn.imprinting.engine on coralrt: weight imprinting (low-shot transfer learning) for classification
models that end with L2Norm, a 1x1 Conv2d, Mul, Reshape and Softmax, as libcoral's ImprintingEngine does it.

Each class's weights are the L2-normalized sum of its training embeddings, quantized with the kernel's scale and
zero point. The sums' norms of the classes trained here are kept in the model's description ("<class> <norm>"
lines), so a model trained here can be trained further with keep_classes=True.
"""
import numpy as np

from pycoral.learn import _tflite

_L2_NORMALIZATION, _CONV_2D, _MUL, _RESHAPE, _SOFTMAX, _QUANTIZE = 11, 3, 18, 22, 25, 114


def _parse_metadata(description):
  """ParseTrainingMetadata: "label norm" pairs; any malformed pair drops them all."""
  tokens = description.replace("\n", " ").split(" ") if description else []
  metadata = {}
  for i in range(0, len(tokens) - 1, 2):
    try:
      metadata[int(tokens[i])] = np.float32(float(tokens[i + 1]))
    except ValueError:
      return {}
  return metadata


def _l2_normalize(v):
  """L2Normalize: the norm summed in order in float; the vector divided by it unless it is ~0."""
  total = np.float32(0)
  for x in v:
    total = np.float32(total + x * x)
  norm = np.float32(np.sqrt(total))
  if abs(norm) > 1e-5:
    v = (v / norm).astype(np.float32)
  return v, norm


class _ImprintingModel:
  """ImprintingModel::Create: the model checked, and what training and serializing need from it."""

  def __init__(self, data):
    m = self.model = _tflite.Model(data)
    codes = [m.op_code(i) for i in range(len(m.ops))]
    if _L2_NORMALIZATION not in codes:
      raise ValueError("Unsupported model architecture. Input model must have an L2Norm layer.")
    l2norm = codes.index(_L2_NORMALIZATION) if codes.count(_L2_NORMALIZATION) == 1 else -1
    if len(codes) < 4 or codes[-4:] != [_CONV_2D, _MUL, _RESHAPE, _SOFTMAX]:
      raise ValueError("The last 4 operators must be Conv2d, Mul, Reshape, Softmax")
    fc = len(codes) - 4
    if fc - l2norm > 2 or (fc - l2norm == 2 and codes[l2norm + 1] != _QUANTIZE):
      raise ValueError("Unsupported model architecture. L2Norm operator must be followed with the last Conv2d operator.")
    embedding = m.op_outputs(l2norm)[0]
    shape = m.shape(embedding)
    if len(shape) != 4 or shape[:3] != [1, 1, 1]:
      raise ValueError("Embedding extractor's output tensor should be [1, 1, 1, x]")
    bias_zp = m.quant(m.op_inputs(fc)[2])["zero_point"][0]
    if not 0 <= bias_zp <= 255:
      raise ValueError("bias_zero_point is out of [0, 255] range!")
    softmax_out = m.op_outputs(len(codes) - 1)[0]
    if len(m.shape(softmax_out)) != 2 or m.shape(softmax_out)[0] != 1:
      raise ValueError("Logit output tensor should be [1, x]")
    if m.outputs != [softmax_out]:
      raise ValueError("Graph must have output from Softmax")
    self.fc = fc
    self.embedding_dim = shape[3]
    kernel = m.op_inputs(fc)[1]
    q = m.quant(kernel)
    self.scale, self.zero_point = np.float32(q["scale"][0]), int(q["zero_point"][0])
    self.kernel_data = m.buffer_data(m.buffer_index(kernel))
    m.new_outputs = [embedding]
    self.extractor = m.serialize()
    m.new_outputs = None

  def existing_classes(self):
    """LoadExistingClasses: the kernel's rows dequantized; those trained here scaled back by their norms."""
    metadata = _parse_metadata(self.model.description)
    data = np.frombuffer(self.kernel_data, np.uint8)
    classes = []
    for i in range(len(data) // self.embedding_dim):
      row = data[i * self.embedding_dim:(i + 1) * self.embedding_dim].astype(np.int32) - self.zero_point
      weights = (self.scale * row.astype(np.float32)).astype(np.float32)
      trainable = i in metadata
      if trainable:
        weights = (weights * metadata[i]).astype(np.float32)
      classes.append([weights, trainable])
    return classes

  def serialize(self, classes):
    """SerializeModel / UpdateImprintingModel: new kernel, bias and shapes for the number of classes."""
    m, n = self.model, len(classes)
    data = bytearray()
    metadata = {}
    for i, (weights, trainable) in enumerate(classes):
      normalized, norm = _l2_normalize(weights)
      data += _tflite.quantize(normalized, self.scale, self.zero_point, 0, 255).astype(np.uint8).tobytes()
      if trainable:
        metadata[i] = norm
    fc_in, fc_out = m.op_inputs(self.fc), m.op_outputs(self.fc)
    kernel, bias = fc_in[1], fc_in[2]
    changes = m.tensor_changes = {}
    m.buffer_changes = {m.buffer_index(kernel): bytes(data)}
    shape = m.shape(kernel)
    changes[kernel] = {"shape": [n] + shape[1:]}
    # every byte of the bias is its zero point
    bias_zp = int(m.quant(bias)["zero_point"][0])
    m.buffer_changes[m.buffer_index(bias)] = bytes([bias_zp]) * (n * 4)
    changes[bias] = {"shape": [n] + m.shape(bias)[1:]}
    out_shape = m.shape(fc_out[0])
    changes[fc_out[0]] = {"shape": out_shape[:3] + [n] + out_shape[4:],
                          "quant": {"min": [-1.0], "max": [1.0], "scale": [np.float32(1.0) / np.float32(128)], "zero_point": [128]}}
    # Mul: its output's range from the scale factor's first quantized value
    mul = self.fc + 1
    factor = m.op_inputs(mul)[1]
    fq = m.quant(factor)
    q0 = np.float32(m.buffer_data(m.buffer_index(factor))[0])
    scale_factor = np.float32(np.float32(fq["scale"][0]) * np.float32(q0 - np.float32(fq["zero_point"][0])))
    mul_quant = {"min": [-scale_factor], "max": [scale_factor], "scale": [np.float32(np.float32(1.0) / np.float32(128)) * scale_factor], "zero_point": [128]}
    mul_out = m.op_outputs(mul)[0]
    s = m.shape(mul_out)
    changes[mul_out] = {"shape": s[:3] + [n] + s[4:], "quant": mul_quant}
    # Reshape: its output, its shape input and its new_shape option
    reshape = self.fc + 2
    reshape_out = m.op_outputs(reshape)[0]
    changes[reshape_out] = {"shape": m.shape(reshape_out)[:-1] + [n], "quant": mul_quant}
    reshape_in = m.op_inputs(reshape)
    if len(reshape_in) == 2:
      b = m.buffer_index(reshape_in[1])
      shape_data = bytearray(m.buffer_data(b))
      shape_data[4:8] = int(n).to_bytes(4, "little", signed=True)
      m.buffer_changes[b] = bytes(shape_data)
    opts = m.ops[reshape].table(4)
    new_shape = opts.array(0, "i") if opts is not None and m.ops[reshape].scalar(3, "B") == _tflite.RESHAPE_OPTIONS else []
    m.reshape_changes = {reshape: new_shape[:-1] + [n]} if new_shape else {}
    softmax_out = m.op_outputs(self.fc + 3)[0]
    s = m.shape(softmax_out)
    changes[softmax_out] = {"shape": [s[0], n] + s[2:]}
    m.new_outputs = [softmax_out]
    m.new_description = "".join("%d %f\n" % (k, v) for k, v in sorted(metadata.items()))
    return m.serialize()


class ImprintingEngine:
  """Performs weight imprinting (transfer learning) with the given model."""

  def __init__(self, model_path, keep_classes=False):
    """model_path: a .tflite model made for imprinting (it ends with L2Norm, Conv2d, Mul, Reshape, Softmax).
    keep_classes: keep the pre-trained model's classes (training adds classes after them), or drop them."""
    try:
      with open(model_path, "rb") as f:
        data = f.read()
    except OSError:
      raise ValueError("Failed to open file: " + str(model_path)) from None
    self._model = _ImprintingModel(data)
    self._classes = self._model.existing_classes() if keep_classes else []

  @property
  def embedding_dim(self):
    """Returns number of embedding dimensions."""
    return self._model.embedding_dim

  @property
  def num_classes(self):
    """Returns number of currently trained classes."""
    return len(self._classes)

  def serialize_extractor_model(self):
    """Returns the embedding extractor model (the model with the L2Norm output as its output) as bytes."""
    return self._model.extractor

  def serialize_model(self):
    """Returns the newly trained model as bytes."""
    if not self._classes:
      raise RuntimeError("Model is not trained.")
    return self._model.serialize(self._classes)

  def train(self, embedding, class_id):
    """Adds an embedding (numpy array of embedding_dim values) to class class_id: the next new class
    (num_classes) or a class trained with imprinting (not one of the pre-trained model's)."""
    embedding = np.asarray(embedding, np.float32)
    if embedding.shape != (self._model.embedding_dim,):
      raise RuntimeError("Invalid weights array shape.")
    n = len(self._classes)
    if class_id < n and not self._classes[class_id][1]:
      raise RuntimeError("Cannot change the base model classes not trained with imprinting method!")
    if class_id > n:
      raise RuntimeError("The class index of a new category is too large!")
    if class_id == n:
      self._classes.append([np.zeros(self._model.embedding_dim, np.float32), True])
    self._classes[class_id][0] = (self._classes[class_id][0] + embedding).astype(np.float32)
