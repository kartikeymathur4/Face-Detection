"""Download the pretrained ONNX models from the official OpenCV Model Zoo.

Usage:  python download_models.py

Models and licences
-------------------
* YuNet face detector (MIT)      https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
* SFace face recogniser (Apache-2.0)  https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface
"""

from __future__ import annotations

import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://github.com/opencv/opencv_zoo/raw/main/models"
MODELS = {
    "face_detection_yunet_2023mar.onnx": f"{BASE}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx": f"{BASE}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}
MIN_BYTES = 100_000  # sanity check: reject HTML error pages / truncated files


def main() -> int:
    models_dir = Path(__file__).resolve().parent / "models"
    models_dir.mkdir(exist_ok=True)
    status = 0
    for name, url in MODELS.items():
        target = models_dir / name
        if target.is_file() and target.stat().st_size >= MIN_BYTES:
            print(f"[skip] {name} already present")
            continue
        print(f"[download] {name}")
        tmp = target.with_suffix(".part")
        try:
            urllib.request.urlretrieve(url, tmp)
            if tmp.stat().st_size < MIN_BYTES:
                raise OSError("downloaded file is too small to be a valid model")
            tmp.replace(target)
            print(f"[ok] saved to {target}")
        except (urllib.error.URLError, OSError) as exc:
            tmp.unlink(missing_ok=True)
            print(f"[error] could not download {name}: {exc}")
            print(f"        Download it manually from {url}\n        and place it in {models_dir}")
            status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
