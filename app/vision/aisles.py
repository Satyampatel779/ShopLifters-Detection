from __future__ import annotations

import cv2
import numpy as np


class AisleDetector:
    def __init__(self, min_aisles: int = 3, max_aisles: int = 6) -> None:
        self.min_aisles = min_aisles
        self.max_aisles = max_aisles

    def auto_detect(self, frame: np.ndarray) -> list[dict[str, int | str]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 180)
        projection = edges.sum(axis=0)
        smooth = cv2.GaussianBlur(projection.reshape(1, -1), (1, 21), 0).flatten()

        width = frame.shape[1]
        candidates = np.argwhere(smooth > np.percentile(smooth, 75)).flatten().tolist()
        if not candidates:
            return self._fallback_zones(width)

        bins = np.array_split(sorted(candidates), self.min_aisles)
        cuts = [int(np.mean(b)) for b in bins if len(b) > 0]
        cuts = sorted(set([0] + cuts + [width - 1]))
        zones: list[dict[str, int | str]] = []
        zone_id = 1
        for left, right in zip(cuts[:-1], cuts[1:]):
            if right - left < width * 0.08:
                continue
            zones.append({"name": f"Aisle-{zone_id}", "x1": int(left), "x2": int(right)})
            zone_id += 1
            if zone_id > self.max_aisles:
                break

        if len(zones) < self.min_aisles:
            return self._fallback_zones(width)
        return zones

    def map_centroid(self, x: int, zones: list[dict[str, int | str]]) -> str:
        for zone in zones:
            if int(zone["x1"]) <= x <= int(zone["x2"]):
                return str(zone["name"])
        return "Unknown"

    def _fallback_zones(self, width: int) -> list[dict[str, int | str]]:
        split = np.linspace(0, width - 1, self.min_aisles + 1, dtype=int)
        zones: list[dict[str, int | str]] = []
        for idx in range(self.min_aisles):
            zones.append(
                {
                    "name": f"Aisle-{idx + 1}",
                    "x1": int(split[idx]),
                    "x2": int(split[idx + 1]),
                }
            )
        return zones
