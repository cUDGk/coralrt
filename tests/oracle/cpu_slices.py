"""Check coralrt's CPU ops without a stick: cut each Edge TPU model after its edgetpu-custom-op (the op's outputs
become the inputs), run that CPU-only model in TensorFlow Lite on random inputs, and record the cases in the
oracle format of dump_pycoral.py, for oracle_check / check_python.

  ..\\..\\..\\.venv-tf\\Scripts\\python.exe cpu_slices.py <test_data dir> <out dir>

TensorFlow 2.21 is the reference here (pycoral's TFLite 2.5 needs the stick); kernels that changed between the
two would show up as mismatches to look into.
"""
import os, sys
import numpy as np
import flatbuffers
import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as schema

DATA, OUT = sys.argv[1], sys.argv[2]
MODELS = ["movenet_single_pose_lightning_ptq_edgetpu.tflite", "movenet_single_pose_thunder_ptq_edgetpu.tflite",
          "ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"]
os.makedirs(OUT, exist_ok=True)
manifest = []
for name in MODELS:
  m = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(open(os.path.join(DATA, name), "rb").read(), 0))
  sg = m.subgraphs[0]
  codes = [c.customCode.decode() if c.customCode else None for c in m.operatorCodes]
  cut = next(i for i, op in enumerate(sg.operators) if codes[op.opcodeIndex] == "edgetpu-custom-op")
  sg.inputs = np.array(sg.operators[cut].outputs, np.int32)
  sg.operators = sg.operators[cut + 1:]
  b = flatbuffers.Builder(0)
  b.Finish(m.Pack(b), file_identifier=b"TFL3")
  sliced = name.replace("_edgetpu.tflite", "_cpu.tflite")
  open(os.path.join(OUT, sliced), "wb").write(b.Output())

  # TFLite's own kernels: the default XNNPACK delegate would take over some ops with its own rounding
  it = tf.lite.Interpreter(model_path=os.path.join(OUT, sliced),
                           experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
  it.allocate_tensors()
  ins, outs = it.get_input_details(), it.get_output_details()
  rng = np.random.default_rng(7)
  for c in range(6):
    case = f"{sliced[:-7]}_{c}"
    for k, d in enumerate(ins):
      info = np.iinfo(d["dtype"])
      # case 0: all zero points (quantized zeros), case 1: extremes, others: noise
      x = rng.integers(info.min, info.max + 1, size=d["shape"]).astype(d["dtype"])
      if c == 0:
        x[...] = d["quantization"][1]
      elif c == 1:
        x = np.where(rng.random(d["shape"]) < 0.5, info.min, info.max).astype(d["dtype"])
      it.set_tensor(d["index"], x)
      x.tofile(os.path.join(OUT, f"{case}.in{k}.bin"))
    it.invoke()
    for k, d in enumerate(outs):
      it.get_tensor(d["index"]).tofile(os.path.join(OUT, f"{case}.out{k}.bin"))
    manifest.append(f"{case}\t{sliced}\t{len(outs)}\t{len(ins)}")
  print(sliced, "inputs", [tuple(d["shape"]) for d in ins], "outputs", len(outs))

open(os.path.join(OUT, "manifest.txt"), "w").write("\n".join(manifest) + "\n")
