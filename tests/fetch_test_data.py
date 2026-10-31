"""Downloads the models, labels and images the tests use from pycoral's google-coral/test_data, into test_data next
to the coralrt folder (where tests/run_device_tests.ps1 and tests/linux look for it) or the folder given.

  python fetch_test_data.py [out dir]
"""
import pathlib
import sys
import urllib.request

BASE = "https://github.com/google-coral/test_data/raw/master/"
NAMES = [
    "mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite",
    "mobilenet_v2_1.0_224_inat_bird_quant.tflite",
    "inat_bird_labels.txt",
    "parrot.jpg",
    "ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite",
    "ssd_mobilenet_v2_coco_quant_postprocess.tflite",
    "coco_labels.txt",
    "grace_hopper.bmp",
    # MoveNet, 17 keypoints: lightning (192 px, fast) and thunder (256 px, accurate)
    "movenet_single_pose_lightning_ptq_edgetpu.tflite",
    "movenet_single_pose_lightning_ptq.tflite",
    "movenet_single_pose_thunder_ptq_edgetpu.tflite",
    "movenet_single_pose_thunder_ptq.tflite",
    # pycoral.learn: imprinting (rewrites the last layer's weights) and backprop (a layer after the embeddings)
    "mobilenet_v1_1.0_224_l2norm_quant.tflite",
    "mobilenet_v1_1.0_224_l2norm_quant_edgetpu.tflite",
    "mobilenet_v1_1.0_224_quant_embedding_extractor.tflite",
    "mobilenet_v1_1.0_224_quant_embedding_extractor_edgetpu.tflite",
    # pycoral.pipeline: inception_v3 in two segments, and whole to compare with
    "pipeline/inception_v3_299_quant_segment_0_of_2_edgetpu.tflite",
    "pipeline/inception_v3_299_quant_segment_1_of_2_edgetpu.tflite",
    "inception_v3_299_quant_edgetpu.tflite",
]

out = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(__file__).resolve().parents[2] / "test_data"
for n in NAMES:
    p = out / n
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        urllib.request.urlretrieve(BASE + n, p)
    print(n, p.stat().st_size)
