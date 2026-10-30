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
  # int16: power-of-two scales (gemmlowp) and general ones (left shift 15)
  i16 = lambda s: (s, 0)
  for op, opt in [("ADD", "AddOptions"), ("SUB", "SubOptions")]:
    for name, sa, sb, so, pot in [("pot_same", 2**-12, 2**-12, 2**-12, True), ("pot_shift", 2**-12, 2**-14, 2**-12, True),
                                  ("pot_shift2", 2**-15, 2**-12, 2**-12, True), ("general", 0.0007, 0.0011, 0.0013, True),
                                  ("pot_off", 2**-12, 2**-12, 2**-11, False)]:
      for k, (a, b) in enumerate(shapes[:3] if op == "SUB" else shapes[:1]):
        out.append(op_case(f"{op}_int16_{name}_{k}", op, [T(a, "int16", i16(sa)), T(b, "int16", i16(sb))], [T([], "int16", i16(so))],
                           (opt, {"fusedActivationFunction": 0, "potScaleInt16": pot})))
  for k, (a, b) in enumerate(shapes[:3]):
    out.append(op_case(f"MUL_int16_{k}", "MUL", [T(a, "int16", i16(0.001)), T(b, "int16", i16(0.002))], [T([], "int16", i16(0.0015))],
                       ("MulOptions", {"fusedActivationFunction": 0})))
  for dt in ["uint8", "int8"]:
    out.append(op_case(f"MUL_int16_to_{dt}", "MUL", [T([3, 20], "int16", i16(2**-15)), T([3, 20], "int16", i16(2**-15))], [T([], dt, (2**-7, 3))],
                       ("MulOptions", {"fusedActivationFunction": 0})))
  bshapes = [([2, 3, 4], [2, 3, 4]), ([2, 3, 4], [4]), ([1, 3, 1], [2, 1, 5]), ([], [3, 2])]
  for op in ["MAXIMUM", "MINIMUM"]:
    for dt in ["float32", "uint8", "int8", "int32", "int64", "int16"]:
      for k, (a, b) in enumerate(bshapes):
        out.append(op_case(f"{op}_{dt}_{k}", op, [T(a, dt, Q.get(dt)), T(b, dt, Q.get(dt))], [T([], dt, Q.get(dt))]))
  for k, (a, b) in enumerate(bshapes):
    out.append(op_case(f"POW_f32_{k}", "POW", [T(a, gen=("uniform", 0.1, 3)), T(b, gen=("uniform", -3, 3))], [T([])]))
    out.append(op_case(f"POW_int32_{k}", "POW", [T(a, "int32", gen=("int", -6, 6)), T(b, "int32", gen=("int", 0, 5))], [T([], "int32")]))
    for dt in ["float32", "int32"]:
      out.append(op_case(f"SQUARED_DIFFERENCE_{dt}_{k}", "SQUARED_DIFFERENCE", [T(a, dt), T(b, dt)], [T([], dt)]))
    out.append(op_case(f"SQUARED_DIFFERENCE_int8_{k}", "SQUARED_DIFFERENCE", [T(a, "int8", Q["int8"]), T(b, "int8", Q2["int8"])], [T([], "int8", (0.5, -100))]))
    for dt in ["float32", "int32", "int64"]:
      gen = ("choice", [-7, -3, -2, 2, 3, 5]) if dt != "float32" else ("choice", [-2.5, -1.5, 1.25, 3.0, -0.75])
      out.append(op_case(f"FLOOR_MOD_{dt}_{k}", "FLOOR_MOD", [T(a, dt, gen=("int", -20, 20) if dt != "float32" else None), T(b, dt, gen=gen)], [T([], dt)]))
    for op in ["LOGICAL_AND", "LOGICAL_OR"]:
      out.append(op_case(f"{op}_{k}", op, [T(a, "bool"), T(b, "bool")], [T([], "bool")]))
  cmps = ["EQUAL", "NOT_EQUAL", "LESS", "LESS_EQUAL", "GREATER", "GREATER_EQUAL"]
  for op in cmps:
    for dt in ["float32", "int32", "int64", "uint8", "int8"] + (["bool", "string"] if op in ("EQUAL", "NOT_EQUAL") else []):
      for k, (a, b) in enumerate(bshapes):
        gen = ("int", 0, 3) if dt in ("int32", "int64") else ("choice", [-1.0, 0.0, 1.0, 0.5]) if dt == "float32" else None
        qa, qb = (Q.get(dt), Q2.get(dt)) if k % 2 else (Q.get(dt), Q.get(dt))
        out.append(op_case(f"{op}_{dt}_{k}", op, [T(a, dt, qa, gen=gen), T(b, dt, qb, gen=gen)], [T([], "bool")]))
  for k, (a, b) in enumerate([([2, 3, 4], [2, 3, 4]), ([2, 3, 4], [4]), ([1, 2, 3, 4], [1, 1, 4])]):
    out.append(op_case(f"PRELU_f32_{k}", "PRELU", [T(a), T(b, gen=("uniform", -1, 1))], [T([])]))
    for dt in ["uint8", "int8"]:
      out.append(op_case(f"PRELU_{dt}_{k}", "PRELU", [T(a, dt, Q[dt]), T(b, dt, Q2[dt])], [T([], dt, QO[dt])]))
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
  out.append(op_case("QUANTIZE_f32_int16", "QUANTIZE", [T([3, 50], gen=("uniform", -40, 40))], [T([3, 50], "int16", (0.002, 0))]))
  # far outside the range: x86 turns the overflowing conversion into the minimum before clamping
  out.append(op_case("QUANTIZE_f32_overflow", "QUANTIZE", [T([6], value=[1e12, -1e12, 3e9, 2147483647.6, 40.0, -40.0])], [T([6], "int8", (1.0, 0))]))
  for dt, q in [("int8", (0.05, 3)), ("int16", (0.0007, 0)), ("int32", (0.0001, 7))]:
    out.append(op_case(f"QUANTIZE_int16_{dt}", "QUANTIZE", [T([4, 100], "int16", Q["int16"])], [T([4, 100], dt, q)]))
  out.append(op_case("DEQUANTIZE_float16", "DEQUANTIZE", [T([4, 25], "float16", data=[[i * 0.37 - 9 for i in range(25)]] * 4)], [T([4, 25])]))
  # ranges with 0 inside, at an end, outside (the zero point clamps), and wide grids
  for k, (mn, mx, bits) in enumerate([(-1.0, 1.0, 8), (-0.3, 5.7, 8), (0.0, 2.0, 4), (-6.0, 0.0, 16), (-2.5, 2.5, 3), (0.1, 3.0, 8), (-7.3, -0.2, 8)]):
    out.append(op_case(f"FAKE_QUANT_{k}", "FAKE_QUANT", [T([4, 50], gen=("uniform", -8, 8))], [T([4, 50])],
                       ("FakeQuantOptions", {"min": mn, "max": mx, "numBits": bits, "narrowRange": False})))
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


def movement_cases():
  out = []
  c = lambda *a, **k: out.append(op_case(*a, **k))
  i32 = lambda shape, v, const=True: T(shape, "int32", data=v) if const else T(shape, "int32", value=v)
  for dt in ["float32", "uint8", "int8", "int16", "int32", "int64", "bool"]:
    q = Q.get(dt)
    c(f"SQUEEZE_{dt}", "SQUEEZE", [T([1, 3, 1, 4], dt, q)], [T([], dt, q)], ("SqueezeOptions", {"squeezeDims": []}))
    c(f"SQUEEZE_{dt}_axis", "SQUEEZE", [T([1, 3, 1, 4], dt, q)], [T([], dt, q)], ("SqueezeOptions", {"squeezeDims": [-2]}))
    c(f"EXPAND_DIMS_{dt}", "EXPAND_DIMS", [T([3, 4], dt, q), i32([1], [-1])], [T([], dt, q)])
    c(f"TRANSPOSE_{dt}", "TRANSPOSE", [T([2, 3, 4], dt, q), i32([3], [2, 0, 1])], [T([], dt, q)])
    c(f"TRANSPOSE_{dt}_5d", "TRANSPOSE", [T([2, 1, 3, 2, 2], dt, q), i32([5], [4, 2, 0, 3, 1])], [T([], dt, q)])
    c(f"SLICE_{dt}", "SLICE", [T([3, 4, 5], dt, q), i32([3], [1, 0, 2]), i32([3], [2, -1, 2])], [T([], dt, q)])
    c(f"REVERSE_V2_{dt}", "REVERSE_V2", [T([3, 4, 2], dt, q), i32([1], [1])], [T([], dt, q)])
    c(f"TILE_{dt}", "TILE", [T([2, 3], dt, q), i32([2], [2, 3])], [T([], dt, q)])
    c(f"GATHER_{dt}", "GATHER", [T([4, 3, 2], dt, q), i32([2, 2], [[2, 0], [1, 1]], False)], [T([], dt, q)], ("GatherOptions", {"axis": 1 if dt != "bool" else 0, "batchDims": 0}))
    c(f"BROADCAST_TO_{dt}", "BROADCAST_TO", [T([3, 1], dt, q), i32([3], [2, 3, 4])], [T([], dt, q)], version=2)
  for dt in ["float32", "uint8", "int8", "int32", "int64"]:
    q = Q.get(dt)
    c(f"PAD_{dt}", "PAD", [T([2, 3], dt, q), i32([2, 2], [[1, 0], [2, 1]])], [T([], dt, q)])
    c(f"PADV2_{dt}", "PADV2", [T([2, 3], dt, q), i32([2, 2], [[0, 2], [1, 1]]), T([], dt, q, data=7)], [T([], dt, q)])
    for mode in [0, 1]:
      c(f"MIRROR_PAD_{dt}_{mode}", "MIRROR_PAD", [T([3, 4], dt, q), i32([2, 2], [[2, 1], [1, 3 - mode]])], [T([], dt, q)], ("MirrorPadOptions", {"mode": mode}))
    c(f"SPACE_TO_DEPTH_{dt}", "SPACE_TO_DEPTH", [T([1, 4, 6, 3], dt, q)], [T([], dt, q)], ("SpaceToDepthOptions", {"blockSize": 2}))
    c(f"DEPTH_TO_SPACE_{dt}", "DEPTH_TO_SPACE", [T([1, 2, 3, 8], dt, q)], [T([], dt, q)], ("DepthToSpaceOptions", {"blockSize": 2}))
    c(f"SPACE_TO_BATCH_ND_{dt}", "SPACE_TO_BATCH_ND", [T([2, 3, 4, 2], dt, q), i32([2], [2, 3]), i32([2, 2], [[1, 0], [1, 1]])], [T([], dt, q)])
    c(f"BATCH_TO_SPACE_ND_{dt}", "BATCH_TO_SPACE_ND", [T([12, 2, 2, 2], dt, q), i32([2], [2, 3]), i32([2, 2], [[1, 0], [0, 2]])], [T([], dt, q)])
    c(f"SPLIT_{dt}", "SPLIT", [i32([], 1), T([2, 6, 3], dt, q)], [T([], dt, q) for _ in range(3)], ("SplitOptions", {"numSplits": 3}))
    c(f"SPLIT_V_{dt}", "SPLIT_V", [T([2, 7, 3], dt, q), i32([3], [2, -1, 1]), i32([], 1)], [T([], dt, q) for _ in range(3)], ("SplitVOptions", {"numSplits": 3}))
    c(f"MATRIX_DIAG_{dt}", "MATRIX_DIAG", [T([2, 3], dt, q)], [T([], dt, q)])
    c(f"MATRIX_SET_DIAG_{dt}", "MATRIX_SET_DIAG", [T([2, 3, 4], dt, q), T([2, 3], dt, q)], [T([], dt, q)])
    c(f"REVERSE_SEQUENCE_{dt}", "REVERSE_SEQUENCE", [T([3, 5, 2], dt, q), T([3], "int32", value=[2, 5, 0])], [T([], dt, q)], ("ReverseSequenceOptions", {"seqDim": 1, "batchDim": 0}))
    c(f"SCATTER_ND_{dt}", "SCATTER_ND", [T([3, 1], "int32", value=[[2], [0], [2]]), T([3, 4], dt, q, gen=("int", 0, 20) if dt != "float32" else None), i32([2], [4, 4])], [T([], dt, q)])
    c(f"UNIQUE_{dt}", "UNIQUE", [T([12], dt, q, gen=("int", 0, 4) if dt != "float32" else ("choice", [1.0, -0.0, 0.0, 2.5, 1.0]))], [T([], dt, q), T([], "int32")], ("UniqueOptions", {"idxOutType": TT["int32"]}))
    c(f"SELECT_{dt}", "SELECT", [T([2, 3], "bool"), T([2, 3], dt, q), T([2, 3], dt, q)], [T([], dt, q)])
    c(f"SELECT_{dt}_rank1", "SELECT", [T([2], "bool"), T([2, 3], dt, q), T([2, 3], dt, q)], [T([], dt, q)])
    c(f"SELECT_V2_{dt}", "SELECT_V2", [T([2, 1], "bool"), T([1, 3], dt, q), T([2, 3], dt, q)], [T([], dt, q)])
    c(f"ONE_HOT_{dt}", "ONE_HOT", [T([2, 3], "int32", gen=("int", -1, 4)), i32([], 4), T([], dt, q, data=5), T([], dt, q, data=1)], [T([], dt, q)], ("OneHotOptions", {"axis": 1}))
    c(f"SPARSE_TO_DENSE_{dt}", "SPARSE_TO_DENSE", [i32([3, 2], [[0, 1], [1, 2], [2, 0]]), i32([2], [3, 4]), T([3], dt, q, value=[1, 2, 3]), T([], dt, q, data=9)], [T([], dt, q)],
      ("SparseToDenseOptions", {"validateIndices": True}))
    c(f"FILL_{dt}", "FILL", [i32([2], [2, 3]), T([], dt, q, data=4)], [T([], dt, q)])
  for k, (b, e, s, masks) in enumerate([([1, 0, 0], [3, 4, 5], [1, 2, 1], (0, 0, 0)), ([0, 3, -1], [3, 0, 0], [1, -1, -2], (0, 0, 0)),
                                         ([1, 1, 1], [2, 3, 4], [1, 1, 1], (5, 2, 0)), ([1, 0, 2], [2, 4, 3], [1, 1, 1], (0, 0, 5)),
                                         ([-2, 0], [-1, 0], [1, 1], (0, 2, 1)), ([0], [2], [1], (0, 0, 0))]):
    for dt in ["float32", "int8", "int32"]:
      c(f"STRIDED_SLICE_{dt}_{k}", "STRIDED_SLICE", [T([3, 4, 5], dt, Q.get(dt)), i32([len(b)], b), i32([len(e)], e), i32([len(s)], s)], [T([], dt, Q.get(dt))],
        ("StridedSliceOptions", {"beginMask": masks[0], "endMask": masks[1], "shrinkAxisMask": masks[2], "ellipsisMask": 0, "newAxisMask": 0}))
  c("STRIDED_SLICE_string", "STRIDED_SLICE", [T([4, 3], "string"), i32([2], [1, 0]), i32([2], [4, 3]), i32([2], [2, 2])], [T([], "string")],
    ("StridedSliceOptions", {"beginMask": 0, "endMask": 0, "shrinkAxisMask": 0, "ellipsisMask": 0, "newAxisMask": 0}))
  c("SLICE_string", "SLICE", [T([3, 4], "string"), i32([2], [1, 1]), i32([2], [2, 2])], [T([], "string")])
  c("TILE_string", "TILE", [T([2], "string"), i32([1], [3])], [T([], "string")])
  c("GATHER_string", "GATHER", [T([5], "string"), i32([3], [4, 0, 4], False)], [T([], "string")], ("GatherOptions", {"axis": 0, "batchDims": 0}))
  c("FILL_string", "FILL", [i32([1], [3]), T([], "string", data=b"ab")], [T([], "string")])
  c("GATHER_batch", "GATHER", [T([2, 4, 3]), i32([2, 2], [[0, 3], [1, 1]], False)], [T([])], ("GatherOptions", {"axis": 1, "batchDims": 1}))
  c("GATHER_int64_idx", "GATHER", [T([4, 3]), T([2], "int64", value=[3, 0])], [T([])], ("GatherOptions", {"axis": 0, "batchDims": 0}))
  # shapes from non-constant inputs: sized while running
  c("RESHAPE_like_SLICE_dynamic", "SLICE", [T([3, 4]), i32([2], [1, 1], False), i32([2], [2, -1], False)], [T([])])
  c("TRANSPOSE_dynamic", "TRANSPOSE", [T([2, 3]), i32([2], [1, 0], False)], [T([])])
  c("PAD_dynamic", "PAD", [T([2, 2]), i32([2, 2], [[1, 1], [0, 2]], False)], [T([])])
  c("FILL_dynamic", "FILL", [i32([3], [2, 1, 3], False), T([], value=2.5)], [T([])])
  for dt in ["float32", "int32"]:
    c(f"RANGE_{dt}", "RANGE", [T([], dt, data=1), T([], dt, data=10), T([], dt, data=3 if dt == "int32" else 0.7)], [T([], dt)])
    c(f"RANGE_{dt}_neg", "RANGE", [T([], dt, value=5), T([], dt, value=-3), T([], dt, value=-2 if dt == "int32" else -0.3)], [T([], dt)])
    c(f"ADD_N_{dt}", "ADD_N", [T([3, 4], dt), T([3, 4], dt), T([3, 4], dt)], [T([], dt)])
  c("ADD_N_f32_zero", "ADD_N", [T([4], value=[-0.0, 0.0, -1.0, 2.0]), T([4], value=[-0.0, -0.0, 1.0, -2.0])], [T([])])
  for dt in ["float32", "int8", "string"]:
    c(f"SHAPE_{dt}", "SHAPE", [T([2, 3, 4], dt, Q.get(dt))], [T([], "int32")], ("ShapeOptions", {"outType": TT["int32"]}))
  c("SHAPE_int64", "SHAPE", [T([5, 1])], [T([], "int64")], ("ShapeOptions", {"outType": TT["int64"]}))
  c("RANK", "RANK", [T([2, 3, 4])], [T([], "int32")])
  c("WHERE_bool", "WHERE", [T([3, 4], "bool")], [T([], "int64")])
  c("WHERE_1d", "WHERE", [T([6], "bool")], [T([], "int64")])
  return out


def reduce_cases():
  out = []
  red = lambda name, op, x, axis, keep, o_q=None, const=True: out.append(op_case(
    name, op, [x, T([len(axis)], "int32", data=axis) if const else T([len(axis)], "int32", value=axis)], [T([], x.dtype, o_q if o_q is not None else x.quant)],
    ("ReducerOptions", {"keepDims": keep})))
  for op in ["MEAN", "SUM", "REDUCE_PROD", "REDUCE_MAX", "REDUCE_MIN"]:
    for dt in ["float32", "int32", "int64", "uint8", "int8", "int16"]:
      q = (Q[dt][0], 0) if dt == "int16" else Q.get(dt)
      gen = ("int", 1, 3) if op == "REDUCE_PROD" and dt not in ("float32",) else ("uniform", 0.5, 1.5) if op == "REDUCE_PROD" else None
      for k, (shape, axis, keep) in enumerate([([2, 3, 4], [1], False), ([2, 3, 4], [-1], True), ([2, 3, 4], [0, 2], False),
                                                ([2, 4, 5, 3], [1, 2], True), ([2, 4, 5, 3], [2, 1], False), ([3, 4], [1, 1], False)]):
        red(f"{op}_{dt}_{k}", op, T(shape, dt, q, gen=gen), axis, keep)
      if dt in ("uint8", "int8") and op in ("MEAN", "SUM"):
        for k, (shape, axis, keep) in enumerate([([2, 3, 4], [1], False), ([2, 4, 5, 3], [1, 2], True), ([2, 4, 5, 3], [1, 2], False)]):
          red(f"{op}_{dt}_rescale_{k}", op, T(shape, dt, Q[dt]), axis, keep, QO[dt])
      if dt == "int16" and op == "MEAN":
        red("MEAN_int16_rescale_hw", "MEAN", T([2, 4, 5, 3], dt, (0.001, 0)), [1, 2], True, (0.0007, 0))
  for shape in [[3, 7], [2, 19], [5, 33], [1, 8]]:
    red(f"MEAN_f32_last_{shape[1]}", "MEAN", T(shape), [1], False)
    red(f"MEAN_f32_last_{shape[1]}_neg", "MEAN", T(shape), [-1], False)
  red("SUM_f32_dynamic_axis", "SUM", T([3, 4, 5]), [0, 2], False, const=False)
  for k, (shape, axis, keep) in enumerate([([3, 4], [1], False), ([2, 3, 4], [0, 2], True)]):
    red(f"REDUCE_ANY_{k}", "REDUCE_ANY", T(shape, "bool"), axis, keep)
  for op in ["ARG_MAX", "ARG_MIN"]:
    opt = "ArgMaxOptions" if op == "ARG_MAX" else "ArgMinOptions"
    for dt in ["float32", "uint8", "int8", "int32"]:
      for axis in [0, 1, 2, -1]:
        for odt in ["int32", "int64"]:
          out.append(op_case(f"{op}_{dt}_{axis}_{odt}", op, [T([3, 4, 5], dt, gen=("int", 0, 3) if dt != "float32" else None), T([1], "int32", data=[axis])],
                             [T([], odt)], (opt, {"outputType": TT[odt]})))
    out.append(op_case(f"{op}_axis_input", op, [T([3, 4]), T([1], "int32", value=[1])], [T([], "int64")], (opt, {"outputType": TT["int64"]})))
  for dt in ["float32", "int32", "int64"]:
    for k, (shape, axis) in enumerate([([3, 5], 1), ([3, 5], 0), ([2, 3, 4], -2), ([7], 0)]):
      for ex, rev in [(False, False), (True, False), (False, True), (True, True)]:
        out.append(op_case(f"CUMSUM_{dt}_{k}_{int(ex)}{int(rev)}", "CUMSUM", [T(shape, dt), T([], "int32", data=axis)], [T([], dt)],
                           ("CumsumOptions", {"exclusive": ex, "reverse": rev})))
  out.append(op_case("CUMSUM_f32_signed_zero", "CUMSUM", [T([2, 3], value=[[-0.0, -0.0, 1.0], [-0.0, 2.0, -2.0]]), T([], "int32", data=1)], [T([])],
                     ("CumsumOptions", {"exclusive": False, "reverse": False})))
  for dt in ["float32", "int32"]:
    ids = [0, 0, 1, 2, 2, 2]
    out.append(op_case(f"SEGMENT_SUM_{dt}", "SEGMENT_SUM", [T([6, 3], dt), T([6], "int32", data=ids)], [T([], dt)]))
    out.append(op_case(f"SEGMENT_SUM_{dt}_ids_input", "SEGMENT_SUM", [T([6, 2, 2], dt), T([6], "int32", value=[0, 1, 1, 1, 2, 3])], [T([], dt)]))
    out.append(op_case(f"SEGMENT_SUM_{dt}_const", "SEGMENT_SUM", [T([4, 2], dt, data=[[1, 2], [3, 4], [5, 6], [7, 8]]), T([4], "int32", data=[0, 0, 0, 1])], [T([], dt)]))
  out.append(op_case("SEGMENT_SUM_f32_signed_zero", "SEGMENT_SUM", [T([2, 2], value=[[-0.0, 1.0], [-0.0, -1.0]]), T([2], "int32", data=[0, 1])], [T([])]))
  for dt in ["float32", "uint8", "int8", "int32", "int64"]:
    # few distinct values, so ties and their index order show
    gen = ("int", 0, 4) if dt != "float32" else ("choice", [1.0, -1.0, 0.5, 2.0, -0.0, 0.0])
    for k, (shape, kk) in enumerate([([3, 8], 3), ([2, 2, 5], 5), ([6], 1), ([4, 7], 0)]):
      out.append(op_case(f"TOPK_V2_{dt}_{k}", "TOPK_V2", [T(shape, dt, gen=gen), T([], "int32", data=kk)], [T([], dt), T([], "int32")]))
  out.append(op_case("TOPK_V2_k_input", "TOPK_V2", [T([3, 6]), T([], "int32", value=2)], [T([]), T([], "int32")]))
  return out


def unary_cases():
  out = []
  f = lambda name, op, gen=None, opts=None, shape=(4, 75): out.append(op_case(name, op, [T(list(shape), gen=gen)], [T(list(shape))], opts))
  f("ABS_f32", "ABS")
  f("NEG_f32", "NEG")
  f("EXP_f32", "EXP", ("uniform", -20, 20))
  f("LOG_f32", "LOG", ("uniform", 0.001, 100))
  f("SIN_f32", "SIN", ("uniform", -10, 10))
  f("COS_f32", "COS", ("uniform", -10, 10))
  f("SQRT_f32", "SQRT", ("uniform", 0, 100))
  f("RSQRT_f32", "RSQRT", ("uniform", 0.01, 100))
  f("SQUARE_f32", "SQUARE")
  for op in ["CEIL", "FLOOR"]:
    f(f"{op}_f32", op)
    # a length that leaves a tail after the packets of 4, and signed zeros
    f(f"{op}_f32_tail", op, ("choice", [-0.0, 0.0, -0.3, -0.7, 0.3, 0.7, -1.5, 2.5, 1e9, -1e9]), shape=(3, 7))
  f("ROUND_f32", "ROUND", ("choice", [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, -3.49, 7.51, 0.0, -0.0, 1e10, 2.4]))
  f("ROUND_f32_random", "ROUND")
  for op in ["RELU", "RELU_N1_TO_1", "RELU6", "HARD_SWISH", "ELU"]:
    f(f"{op}_f32", op, ("uniform", -8, 8))
  for alpha in [0.2, -0.5, 1.5]:
    f(f"LEAKY_RELU_f32_{alpha}", "LEAKY_RELU", None, ("LeakyReluOptions", {"alpha": alpha}))
  f("TANH_f32", "TANH", ("uniform", -10, 10))
  f("TANH_f32_small", "TANH", ("uniform", -0.001, 0.001))
  f("LOGISTIC_f32", "LOGISTIC", ("uniform", -20, 20))
  f("LOGISTIC_f32_far", "LOGISTIC", ("uniform", -100, -9))
  q = lambda name, op, dt, qi, qo, opts=None, gen=None: out.append(op_case(name, op, [T([4, 75], dt, qi, gen=gen)], [T([4, 75], dt, qo)], opts))
  for dt in ["int8", "int16"]:
    q(f"ABS_{dt}_same", "ABS", dt, Q[dt], Q[dt])
    q(f"ABS_{dt}_rescale", "ABS", dt, Q[dt], QO[dt])
  q("RSQRT_int8", "RSQRT", "int8", (0.05, -100), (0.02, -128), gen=("int", -100, 127))
  for dt in ["uint8", "int8", "int16"]:
    for op in ["RELU", "RELU6"] + (["RELU_N1_TO_1"] if dt != "int16" else []):
      q(f"{op}_{dt}", op, dt, Q[dt], QO[dt])
      q(f"{op}_{dt}_same", op, dt, Q[dt], Q[dt])
    for alpha in [0.2, -0.5]:
      q(f"LEAKY_RELU_{dt}_{alpha}", "LEAKY_RELU", dt, Q[dt], QO[dt], ("LeakyReluOptions", {"alpha": alpha}))
  for dt in ["uint8", "int8"]:
    for k, (qi, qo) in enumerate([(Q[dt], QO[dt]), (Q2[dt], Q[dt]), ((0.1, Q[dt][1]), (0.02, QO[dt][1]))]):
      q(f"HARD_SWISH_{dt}_{k}", "HARD_SWISH", dt, qi, qo)
    q(f"TANH_{dt}", "TANH", dt, Q[dt], (1 / 128, 128 if dt == "uint8" else 0))
    q(f"LOGISTIC_{dt}", "LOGISTIC", dt, Q[dt], (1 / 256, 0 if dt == "uint8" else -128))
  q("ELU_int8", "ELU", "int8", Q["int8"], QO["int8"])
  for name, s in [("pot", 1 / 4096), ("pot2", 1 / 2048), ("other", 0.0007)]:
    for op in ["TANH", "LOGISTIC"]:
      q(f"{op}_int16_{name}", op, "int16", (s, 0), (1 / 32768, 0))
  for dt in ["int32", "int64"]:
    out.append(op_case(f"NEG_{dt}", "NEG", [T([4, 30], dt)], [T([4, 30], dt)]))
  out.append(op_case("LOGICAL_NOT", "LOGICAL_NOT", [T([4, 30], "bool")], [T([4, 30], "bool")]))
  for dt in ["float32", "int32", "int64"]:
    out.append(op_case(f"ZEROS_LIKE_{dt}", "ZEROS_LIKE", [T([3, 5], dt)], [T([3, 5], dt)]))
  return out


def norm_cases():
  out = []
  c = lambda *a, **k: out.append(op_case(*a, **k))
  # row lengths with and without a tail after Eigen's packets of 4, and rows that start unaligned
  shapes = [[2, 3, 7], [5, 16], [1, 1, 1, 33], [3], [2, 2, 2, 2, 5], [6, 2]]
  for k, shape in enumerate(shapes):
    for beta in [1.0, 0.3, 2.5]:
      c(f"SOFTMAX_f32_{k}_{beta}", "SOFTMAX", [T(shape, gen=("uniform", -8, 8))], [T([])], ("SoftmaxOptions", {"beta": beta}))
    c(f"LOG_SOFTMAX_f32_{k}", "LOG_SOFTMAX", [T(shape, gen=("uniform", -8, 8))], [T([])])
    if len(shape) <= 4:
      c(f"L2_NORMALIZATION_f32_{k}", "L2_NORMALIZATION", [T(shape)], [T([])], ("L2NormOptions", {"fusedActivationFunction": 0}))
  for dt in ["uint8", "int8"]:
    for k, shape in enumerate([[2, 3, 7], [4, 40]]):
      for beta in [1.0, 0.7]:
        c(f"SOFTMAX_{dt}_{k}_{beta}", "SOFTMAX", [T(shape, dt, Q[dt])], [T([], dt, (1 / 256, 0 if dt == "uint8" else -128))],
          ("SoftmaxOptions", {"beta": beta}))
      c(f"SOFTMAX_{dt}_to_int16_{k}", "SOFTMAX", [T(shape, dt, Q[dt])], [T([], "int16", (1 / 32768, 0))], ("SoftmaxOptions", {"beta": 1.0}))
      # on int8, rows that span at most 127 (2.5 reads outside its table beyond that)
      gen = ("int", -60, 60) if dt == "int8" else None
      c(f"LOG_SOFTMAX_{dt}_{k}", "LOG_SOFTMAX", [T(shape, dt, Q[dt], gen=gen)], [T([], dt, (16 / 256, 255 if dt == "uint8" else 127))])
      c(f"LOG_SOFTMAX_{dt}_{k}_coarse", "LOG_SOFTMAX", [T(shape, dt, (0.3, Q[dt][1]), gen=gen)], [T([], dt, (16 / 256, 255 if dt == "uint8" else 127))])
      c(f"L2_NORMALIZATION_{dt}_{k}", "L2_NORMALIZATION", [T(shape, dt, Q[dt])], [T([], dt, (1 / 128, 128 if dt == "uint8" else 0))],
        ("L2NormOptions", {"fusedActivationFunction": 0}))
    # an output scale with room to round, and a zero point (which 2.5's uint8 output ignores)
    c(f"SOFTMAX_{dt}_coarse", "SOFTMAX", [T([3, 10], dt, (0.2, 3))], [T([], dt, (0.01, 5))], ("SoftmaxOptions", {"beta": 1.0}))
  for k, (shape, s, beta) in enumerate([([2, 3, 7], 0.001, 1.0), ([4, 40], 0.0003, 1.0), ([2, 2, 3, 9], 0.002, 0.5), ([5], 0.01, 2.0)]):
    c(f"SOFTMAX_int16_{k}", "SOFTMAX", [T(shape, "int16", (s, 0))], [T([], "int16", (1 / 32768, 0))], ("SoftmaxOptions", {"beta": beta}))
  c("L2_NORMALIZATION_f32_zero", "L2_NORMALIZATION", [T([2, 4], value=[[0, 0, 0, 0], [1e-8, 0, 0, 0]])], [T([])],
    ("L2NormOptions", {"fusedActivationFunction": 0}))
  c("L2_NORMALIZATION_uint8_zero", "L2_NORMALIZATION", [T([2, 4], "uint8", Q["uint8"], value=[[128] * 4, [129, 128, 128, 128]])],
    [T([], "uint8", (1 / 128, 128))], ("L2NormOptions", {"fusedActivationFunction": 0}))
  for k, (radius, bias, alpha, beta) in enumerate([(2, 1.0, 1e-4, 0.75), (1, 2.0, 0.5, 1.0), (3, 0.5, 0.2, 0.5), (0, 1.0, 1.0, 0.75), (5, 1.0, 0.01, 2.0)]):
    for shape in [[1, 3, 3, 5], [2, 2, 2, 16]]:
      c(f"LOCAL_RESPONSE_NORMALIZATION_{k}_{shape[3]}", "LOCAL_RESPONSE_NORMALIZATION", [T(shape)], [T([])],
        ("LocalResponseNormalizationOptions", {"radius": radius, "bias": bias, "alpha": alpha, "beta": beta}))
  return out


def all_cases():
  return binary_cases() + quant_cases() + shape_cases() + movement_cases() + reduce_cases() + unary_cases() + norm_cases()
