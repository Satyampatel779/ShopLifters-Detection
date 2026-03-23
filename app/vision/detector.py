from __future__ import annotations

import numpy as np
from ultralytics import YOLO


class PersonDetector:
    def __init__(self, model_name: str = "yolov8x.pt", conf: float = 0.45) -> None:
        self.model = YOLO(model_name)
        self.conf = conf

    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        results: list[tuple[int, int, int, int, float]] = []
        pred = self.model.predict(frame, conf=self.conf, verbose=False)
        if not pred:
            return results
        boxes = pred[0].boxes
        if boxes is None:
            return results
        for b in boxes:
            cls_id = int(b.cls.item())
            if cls_id != 0:
                continue
            conf = float(b.conf.item())
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
            results.append((x1, y1, x2, y2, conf))
        return self._non_max_suppression(results)

    @staticmethod
    def _non_max_suppression(
        boxes: list[tuple[int, int, int, int, float]], iou_thresh: float = 0.3
    ) -> list[tuple[int, int, int, int, float]]:
        if not boxes:
            return []
        b = np.array([[x1, y1, x2, y2, s] for x1, y1, x2, y2, s in boxes], dtype=np.float32)
        x1 = b[:, 0]
        y1 = b[:, 1]
        x2 = b[:, 2]
        y2 = b[:, 3]
        scores = b[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep: list[int] = []

        while order.size > 0:
            i = int(order[0])
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
            inds = np.where(iou <= iou_thresh)[0]
            order = order[inds + 1]

        return [boxes[idx] for idx in keep]
