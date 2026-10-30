"""Write a small float model that has a SignatureDef (the test models have none), for signatures.py. Needs
TensorFlow (the .venv-tf one). The output names are chosen so that sorting them changes their order.

  python -I make_signature_model.py <out.tflite>
"""
import sys, tempfile

import tensorflow as tf


class M(tf.Module):
  @tf.function(input_signature=[tf.TensorSpec([1, 4], tf.float32, name="x"), tf.TensorSpec([1, 4], tf.float32, name="y")])
  def __call__(self, x, y):
    return {"sum": x + y, "diff": x - y, "Prod": x * y}


with tempfile.TemporaryDirectory() as d:
  m = M()
  tf.saved_model.save(m, d, signatures=m.__call__)
  data = tf.lite.TFLiteConverter.from_saved_model(d).convert()
with open(sys.argv[1], "wb") as f:
  f.write(data)
