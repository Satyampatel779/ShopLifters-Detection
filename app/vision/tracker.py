from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class TrackState:
    track_id: int
    bbox: tuple[int, int, int, int]
    centroid: tuple[int, int]
    missed: int = 0
    history: list[tuple[int, int]] = field(default_factory=list)


class CentroidTracker:
    def __init__(self, max_distance: float = 80.0, max_missed: int = 8) -> None:
        self.max_distance = max_distance
        self.max_missed = max_missed
        self.next_track_id = 1
        self.tracks: dict[int, TrackState] = {}

    def update(self, detections: list[tuple[int, int, int, int, float]]) -> dict[int, TrackState]:
        centroids = [((x1 + x2) // 2, (y1 + y2) // 2) for x1, y1, x2, y2, _ in detections]
        boxes = [(x1, y1, x2, y2) for x1, y1, x2, y2, _ in detections]

        if not self.tracks:
            for box, c in zip(boxes, centroids):
                self._create_track(box, c)
            return self.tracks

        unmatched_tracks = set(self.tracks.keys())
        unmatched_dets = set(range(len(centroids)))
        pairs: list[tuple[float, int, int]] = []

        for track_id, track in self.tracks.items():
            for i, c in enumerate(centroids):
                d = math.dist(track.centroid, c)
                pairs.append((d, track_id, i))

        for d, track_id, det_idx in sorted(pairs, key=lambda x: x[0]):
            if d > self.max_distance:
                continue
            if track_id not in unmatched_tracks or det_idx not in unmatched_dets:
                continue
            unmatched_tracks.remove(track_id)
            unmatched_dets.remove(det_idx)
            track = self.tracks[track_id]
            track.bbox = boxes[det_idx]
            track.centroid = centroids[det_idx]
            track.history.append(track.centroid)
            track.missed = 0

        for track_id in list(unmatched_tracks):
            track = self.tracks[track_id]
            track.missed += 1
            if track.missed > self.max_missed:
                del self.tracks[track_id]

        for det_idx in unmatched_dets:
            self._create_track(boxes[det_idx], centroids[det_idx])

        return self.tracks

    def _create_track(self, box: tuple[int, int, int, int], centroid: tuple[int, int]) -> None:
        tid = self.next_track_id
        self.next_track_id += 1
        self.tracks[tid] = TrackState(track_id=tid, bbox=box, centroid=centroid, history=[centroid])
