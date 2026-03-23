from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from urllib.request import urlretrieve

from app.core.settings import MODELS_DIR


class FaceEncoder:
    def __init__(self) -> None:
        if not hasattr(cv2, "FaceDetectorYN_create") or not hasattr(cv2, "FaceRecognizerSF_create"):
            raise RuntimeError(
                "Current OpenCV build lacks FaceDetectorYN/FaceRecognizerSF. Install opencv-contrib-python>=4.10."
            )

        detector_model, recognizer_model = self._ensure_models(MODELS_DIR)
        self.detector = cv2.FaceDetectorYN_create(
            str(detector_model),
            "",
            (320, 320),
            0.9,
            0.3,
            5000,
        )
        self.recognizer = cv2.FaceRecognizerSF_create(str(recognizer_model), "")

    def encode_from_image(self, image: np.ndarray) -> list[float] | None:
        if image is None or image.size == 0:
            return None
        h, w = image.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(image)
        if faces is None or len(faces) == 0:
            return None

        face = max(faces, key=lambda f: float(f[2] * f[3]))
        aligned = self.recognizer.alignCrop(image, face)
        feat = self.recognizer.feature(aligned).astype(np.float32).flatten()
        feat = feat / (np.linalg.norm(feat) + 1e-8)
        return feat.tolist()

    def _ensure_models(self, model_dir: Path) -> tuple[Path, Path]:
        detector_path = model_dir / "face_detection_yunet_2023mar.onnx"
        recognizer_path = model_dir / "face_recognition_sface_2021dec.onnx"

        if not detector_path.exists():
            urlretrieve(
                "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
                detector_path,
            )
        if not recognizer_path.exists():
            urlretrieve(
                "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
                recognizer_path,
            )
        return detector_path, recognizer_path


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    a = np.array(v1, dtype=np.float32)
    b = np.array(v2, dtype=np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)
