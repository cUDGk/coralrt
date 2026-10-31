"""Compares two results.npz of run_device.py (pycoral 2.0 and coralrt) output by output. Also shows, for each, which
pipeline results equal the whole model's: on one stick only the first does, with pycoral 2.0 too (the later ones
are all zeros, and so are those of a by-hand run of the segments, one image after another). The pipeline expects
a stick per segment; the segments of one compile share a parameter caching token, so on one stick they are taken
to be cached side by side, and most likely the parameters of each overwrite the other's.
  python compare_device.py <official out dir> <coralrt out dir>"""
import pathlib
import sys

import numpy as np

a, b = (np.load(pathlib.Path(d) / "results.npz") for d in sys.argv[1:3])
bad = 0
for k in sorted(a.files):
  same = k in b.files and a[k].shape == b[k].shape and np.array_equal(a[k], b[k])
  bad += not same
  if not same:
    print("DIFF", k, a[k].reshape(-1)[:8], b[k].reshape(-1)[:8] if k in b.files else None)
print(len(a.files), "outputs,", bad, "differ")
for name, r in (("official", a), ("coralrt", b)):
  pipe = sorted(k for k in r.files if k.startswith("pipe_"))
  print(name, "pipeline results equal to the whole model:", [k for k in pipe if np.array_equal(r[k], r["full_" + k[5:]])])
sys.exit(1 if bad else 0)
