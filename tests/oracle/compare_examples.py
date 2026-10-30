"""Run Google's pycoral example scripts (pycoral_examples/, unmodified) with the official stack and with coralrt's
compatibility packages, and compare what they print (inference times excepted). The images they draw are not
compared: the two environments have different Pillow versions, which render the label text differently.

  python compare_examples.py <official python> <coralrt python> <test_data dir> <work dir>
"""
import os, re, subprocess, sys

OFFICIAL, CORALRT, DATA, WORK = sys.argv[1:5]
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pycoral_examples")
d = lambda f: os.path.join(DATA, f)
os.makedirs(WORK, exist_ok=True)
# Pillow 13 resizes parrot.jpg differently from the official environment's 11.3 (18% of the values, by up to 6; the
# decoded JPEG and grace_hopper.bmp resized are the same). Both runs get it at the model's 224x224, resized by the
# official Pillow, so the example's own resize is a copy.
parrot = os.path.join(WORK, "parrot_224.png")
subprocess.run([OFFICIAL, "-I", "-c", "import sys; from PIL import Image; "
                "Image.open(sys.argv[1]).convert('RGB').resize((224, 224), Image.LANCZOS).save(sys.argv[2])", d("parrot.jpg"), parrot], check=True)
RUNS = {
  "classify": ["classify_image.py", "-m", d("mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite"), "-l", d("inat_bird_labels.txt"),
               "-i", parrot, "-k", "5", "-c", "3"],
  "detect": ["detect_image.py", "-m", d("ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"), "-l", d("coco_labels.txt"),
             "-i", d("grace_hopper.bmp"), "-c", "3", "-o", "{out}.png"],
  "lightning": ["movenet_pose_estimation.py", "-m", d("movenet_single_pose_lightning_ptq_edgetpu.tflite"), "-i", d("grace_hopper.bmp"),
                "--output", "{out}.png"],
  "thunder": ["movenet_pose_estimation.py", "-m", d("movenet_single_pose_thunder_ptq_edgetpu.tflite"), "-i", d("grace_hopper.bmp"),
              "--output", "{out}.png"],
}
failed = 0
for name, args in RUNS.items():
  printed = {}
  for who, py in [("official", OFFICIAL), ("coralrt", CORALRT)]:
    out = os.path.join(WORK, f"{name}_{who}")
    cmd = [py, "-I", os.path.join(HERE, args[0])] + [a.replace("{out}", out) for a in args[1:]]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    # drop the timing lines ("12.3ms", "12.34 ms") and the note about the first inference; the image's path differs
    lines = [l.replace(out, "{out}") for l in r.stdout.splitlines() if not re.fullmatch(r"\s*[\d.]+\s*ms", l) and not l.startswith("Note:")]
    printed[who] = (r.returncode, lines, r.stderr.strip().splitlines()[-3:])
  same = printed["official"][:2] == printed["coralrt"][:2]
  print(f"{name}: {'same output' if same else 'DIFFERENT output'}")
  if not same:
    failed += 1
    for who in printed:
      print(f"  {who}: exit {printed[who][0]}", *printed[who][1], *printed[who][2], sep="\n    ")
  else:
    print("   ", "\n    ".join(printed["coralrt"][1][-6:]))
sys.exit(1 if failed else 0)
