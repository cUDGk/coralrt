"""The op cases: for every op of TensorFlow Lite 2.5's BuiltinOpResolver, small models over its types, shapes and
options. build.py writes them, official.py records tflite_runtime 2.5 on them, op_check compares coralrt.
"""
from spec import T, op_case

ACT = {"NONE": 0, "RELU": 1, "RELU_N1_TO_1": 2, "RELU6": 3, "TANH": 4, "SIGN_BIT": 5}
TT = {"float32": 0, "float16": 1, "int32": 2, "uint8": 3, "int64": 4, "string": 5, "bool": 6, "int16": 7, "complex64": 8,
      "int8": 9, "float64": 10, "complex128": 11, "uint64": 12, "uint32": 15}

# quantization of 8-bit tensors in the cases: (scale, zero point) per type
Q = {"uint8": (0.05, 128), "int8": (0.05, 0), "int16": (0.001, 0)}
Q2 = {"uint8": (0.03, 100), "int8": (0.03, -10), "int16": (0.0007, 0)}
QO = {"uint8": (0.1, 120), "int8": (0.1, -5), "int16": (0.002, 0)}


def binary_cases():
  out = []
  shapes = [([2, 3, 4], [2, 3, 4]), ([2, 3, 4], [4]), ([1, 3, 1], [2, 1, 5]), ([2, 1, 3, 1, 2], [1, 4, 1, 5, 1]), ([], [3, 2]), ([5], [])]
  for op, opt in [("ADD", "AddOptions"), ("SUB", "SubOptions"), ("MUL", "MulOptions"), ("DIV", "DivOptions")]:
    for act in ["NONE", "RELU", "RELU6", "RELU_N1_TO_1"]:
      for k, (a, b) in enumerate(shapes):
        out.append(op_case(f"{op}_f32_{act}_{k}", op, [T(a), T(b, gen=("uniform", 0.5, 4) if op == "DIV" else None)], [T([])],
                           (opt, {"fusedActivationFunction": ACT[act]})))
    types = {"ADD": ["uint8", "int8", "int32", "int64"], "SUB": ["uint8", "int8", "int32", "int64"], "MUL": ["uint8", "int8", "int32", "int64"],
             "DIV": ["int32", "uint8"]}[op]
    for dt in types:
      for k, (a, b) in enumerate(shapes[:5]):
        q = dt in Q and (op != "DIV" or dt == "uint8")
        gen = ("int", 1, 50) if op == "DIV" and dt == "int32" else ("int", -1000, 1000) if dt == "int32" and op == "MUL" else None
        ins = [T(a, dt, Q[dt] if q else None), T(b, dt, Q2[dt] if q else None, gen=gen)]
        for act in ["NONE", "RELU"] if k == 0 else ["NONE"]:
          out.append(op_case(f"{op}_{dt}_{act}_{k}", op, ins, [T([], dt, QO[dt] if q else None)], (opt, {"fusedActivationFunction": ACT[act]})))
  for dt in ["float32", "int32"]:
    for k, (a, b) in enumerate(shapes[:4]):
      out.append(op_case(f"FLOOR_DIV_{dt}_{k}", "FLOOR_DIV", [T(a, dt), T(b, dt, gen=("int", 1, 7) if dt == "int32" else ("uniform", 0.5, 3))], [T([], dt)]))
  return out


def quant_cases():
  out = []
  for dt in ["uint8", "int8"]:
    out.append(op_case(f"QUANTIZE_f32_{dt}", "QUANTIZE", [T([3, 50], gen=("uniform", -20, 20))], [T([3, 50], dt, QO[dt])]))
    out.append(op_case(f"QUANTIZE_f32_{dt}_big", "QUANTIZE", [T([4, 300], gen=("uniform", -20, 20))], [T([4, 300], dt, QO[dt])]))
    for dt2 in ["uint8", "int8"]:
      out.append(op_case(f"QUANTIZE_{dt}_{dt2}", "QUANTIZE", [T([4, 100], dt, Q[dt])], [T([4, 100], dt2, QO[dt2])]))
    out.append(op_case(f"DEQUANTIZE_{dt}", "DEQUANTIZE", [T([4, 100], dt, Q[dt])], [T([4, 100])]))
  out.append(op_case("DEQUANTIZE_int16", "DEQUANTIZE", [T([4, 100], "int16", Q["int16"])], [T([4, 100])]))
  out.append(op_case("DEQUANTIZE_float16", "DEQUANTIZE", [T([4, 25], "float16", data=[[i * 0.37 - 9 for i in range(25)]] * 4)], [T([4, 25])]))
  kinds = ["float32", "int32", "int64", "uint8", "int8", "int16", "bool", "float64", "uint32", "complex64"]
  for a in kinds:
    for b in kinds:
      if a != b:
        out.append(op_case(f"CAST_{a}_{b}", "CAST", [T([2, 7], a, gen=("uniform", -300, 300) if a.startswith("float") else None)], [T([2, 7], b)],
                           ("CastOptions", {"inDataType": TT[a], "outDataType": TT[b]})))
  return out


def shape_cases():
  out = []
  out.append(op_case("RESHAPE_opts", "RESHAPE", [T([2, 3, 4])], [T([])], ("ReshapeOptions", {"newShape": [4, -1]})))
  out.append(op_case("RESHAPE_const", "RESHAPE", [T([2, 3, 4]), T([3], "int32", data=[3, 2, 4])], [T([])]))
  out.append(op_case("RESHAPE_input", "RESHAPE", [T([2, 3, 4]), T([2], "int32", value=[-1, 6])], [T([])]))
  out.append(op_case("RESHAPE_scalar", "RESHAPE", [T([1, 1])], [T([])], ("ReshapeOptions", {"newShape": [0]})))
  out.append(op_case("RESHAPE_string", "RESHAPE", [T([2, 3], "string"), T([2], "int32", data=[3, 2])], [T([], "string")]))
  for dt in ["float32", "uint8", "int8", "int32", "int64"]:
    for axis in [0, 1, -1]:
      out.append(op_case(f"PACK_{dt}_{axis}", "PACK", [T([2, 3], dt), T([2, 3], dt), T([2, 3], dt)], [T([], dt)], ("PackOptions", {"valuesCount": 3, "axis": axis})))
      out.append(op_case(f"UNPACK_{dt}_{axis}", "UNPACK", [T([3, 4, 3] if axis != 1 else [4, 3, 2], dt)], [T([], dt) for _ in range(3)],
                         ("UnpackOptions", {"num": 3, "axis": axis})))
    for axis in [0, 1, -1]:
      out.append(op_case(f"CONCATENATION_{dt}_{axis}", "CONCATENATION", [T([2, 3, 2], dt, Q.get(dt)), T([2, 3, 2], dt, Q.get(dt))], [T([], dt, Q.get(dt))],
                         ("ConcatenationOptions", {"axis": axis, "fusedActivationFunction": 0})))
  out.append(op_case("CONCATENATION_uint8_rescale", "CONCATENATION", [T([2, 3], "uint8", Q["uint8"]), T([2, 5], "uint8", Q2["uint8"])], [T([], "uint8", QO["uint8"])],
                     ("ConcatenationOptions", {"axis": 1, "fusedActivationFunction": 0})))
  for dt in ["float32", "int8", "uint8", "int32", "int64", "string"]:
    out.append(op_case(f"GATHER_ND_{dt}", "GATHER_ND", [T([4, 3, 2], dt), T([5, 2], "int32", value=[[0, 0], [3, 2], [1, 1], [2, 0], [3, 1]])], [T([], dt)]))
  out.append(op_case("GATHER_ND_int64_idx", "GATHER_ND", [T([4, 3], "float32"), T([2, 1], "int64", value=[[3], [1]])], [T([])]))
  return out


def reduce_cases():
  out = []
  for op in ["ARG_MAX", "ARG_MIN"]:
    opt = "ArgMaxOptions" if op == "ARG_MAX" else "ArgMinOptions"
    for dt in ["float32", "uint8", "int8", "int32"]:
      for axis in [0, 1, 2, -1]:
        for odt in ["int32", "int64"]:
          out.append(op_case(f"{op}_{dt}_{axis}_{odt}", op, [T([3, 4, 5], dt, gen=("int", 0, 3) if dt != "float32" else None), T([1], "int32", data=[axis])],
                             [T([], odt)], (opt, {"outputType": TT[odt]})))
    out.append(op_case(f"{op}_axis_input", op, [T([3, 4]), T([1], "int32", value=[1])], [T([], "int64")], (opt, {"outputType": TT["int64"]})))
  return out


def unary_cases():
  return [op_case("SQRT_f32", "SQRT", [T([3, 40], gen=("uniform", 0, 100))], [T([3, 40])])]


def all_cases():
  return binary_cases() + quant_cases() + shape_cases() + reduce_cases() + unary_cases()
