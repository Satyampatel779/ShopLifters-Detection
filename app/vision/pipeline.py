from __future__ import annotations

import math
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import cv2

from app.core.storage import Storage
from app.vision.aisles import AisleDetector
from app.vision.detector import PersonDetector
from app.vision.face import FaceEncoder, cosine_similarity
from app.vision.tracker import CentroidTracker


class VideoPipeline:
    def __init__(self, storage: Storage, snapshot_dir: Path, processed_dir: Path) -> None:
        self.storage = storage
        self.snapshot_dir = snapshot_dir
        self.processed_dir = processed_dir
        self.detector = PersonDetector()
        self.tracker = CentroidTracker()
        self.face_encoder = FaceEncoder()
        self.aisle_detector = AisleDetector()

    def process(
        self,
        video_id: str,
        video_path: Path,
        alert_callback: Callable[[dict], None],
        frame_callback: Callable[[str, bytes], None] | None = None,
        frame_skip: int = 3,
        match_threshold: float = 0.72,
    ) -> None:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError("Failed to open video file")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

        output_path = self.processed_dir / f"{video_id}.mp4"
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            max(1.0, fps / frame_skip),
            (width, height),
        )
        if not writer.isOpened():
            output_path = self.processed_dir / f"{video_id}.avi"
            writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*"XVID"),
                max(1.0, fps / frame_skip),
                (width, height),
            )
        if not writer.isOpened():
            cap.release()
            raise RuntimeError("Failed to initialize processed video writer")

        watchlist = self.storage.list_watchlist()

        tracks_payload: dict[int, dict] = {}
        track_last_alert_ts: dict[int, float] = defaultdict(lambda: -9999.0)
        zones: list[dict[str, int | str]] = []

        frame_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_index == 0:
                zones = self.aisle_detector.auto_detect(frame)

            if frame_index % frame_skip != 0:
                frame_index += 1
                continue

            detections = self.detector.detect(frame)
            active_tracks = self.tracker.update(detections)
            timestamp_sec = frame_index / fps
            display_frame = frame.copy()

            for zone in zones:
                x1 = int(zone["x1"])
                x2 = int(zone["x2"])
                cv2.rectangle(display_frame, (x1, 5), (x2, 36), (65, 145, 250), 2)
                cv2.putText(
                    display_frame,
                    str(zone["name"]),
                    (x1 + 6, 27),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (65, 145, 250),
                    2,
                )

            for track_id, track in active_tracks.items():
                x1, y1, x2, y2 = track.bbox
                centroid_x, centroid_y = track.centroid
                aisle = self.aisle_detector.map_centroid(centroid_x, zones)
                det_conf = 0.0
                for d in detections:
                    if (d[0], d[1], d[2], d[3]) == (x1, y1, x2, y2):
                        det_conf = float(d[4])
                        break

                payload = tracks_payload.setdefault(
                    track_id,
                    {
                        "track_id": track_id,
                        "start_time_sec": timestamp_sec,
                        "end_time_sec": timestamp_sec,
                        "avg_speed_px_per_sec": 0.0,
                        "aisles_visited": [],
                        "events": [],
                        "best_face_embedding": None,
                    },
                )
                payload["end_time_sec"] = timestamp_sec
                if aisle not in payload["aisles_visited"]:
                    payload["aisles_visited"].append(aisle)
                payload["events"].append(
                    {
                        "frame_index": frame_index,
                        "timestamp_sec": timestamp_sec,
                        "bbox": [x1, y1, x2, y2],
                        "centroid": [centroid_x, centroid_y],
                        "aisle": aisle,
                        "confidence": det_conf,
                    }
                )

                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (50, 220, 90), 2)
                cv2.putText(
                    display_frame,
                    f"ID {track_id} | {aisle}",
                    (x1, max(16, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (50, 220, 90),
                    2,
                )

                crop = frame[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]
                if crop.size == 0:
                    continue
                embedding = self.face_encoder.encode_from_image(crop)
                if embedding is None:
                    continue
                payload["best_face_embedding"] = embedding

                if not watchlist:
                    continue

                best_match = None
                best_score = -1.0
                for candidate in watchlist:
                    score = cosine_similarity(embedding, candidate["embedding"])
                    if score > best_score:
                        best_score = score
                        best_match = candidate

                # Cooldown suppresses repeated alerts for the same tracked person.
                cooldown_sec = 20.0
                if (
                    best_match
                    and best_score >= match_threshold
                    and (timestamp_sec - track_last_alert_ts[track_id]) >= cooldown_sec
                ):
                    alert = {
                        "id": str(uuid.uuid4()),
                        "video_id": video_id,
                        "watchlist_person_id": best_match["id"],
                        "watchlist_person_name": best_match["name"],
                        "track_id": track_id,
                        "match_score": round(float(best_score), 4),
                        "timestamp_sec": round(timestamp_sec, 2),
                        "aisle": aisle,
                        "status": "pending",
                    }
                    self.storage.add_alert(alert)
                    self._save_snapshot(video_id, frame_index, frame)
                    alert_callback(alert)
                    track_last_alert_ts[track_id] = timestamp_sec

                    cv2.putText(
                        display_frame,
                        f"ALERT: {best_match['name']} ({best_score:.2f})",
                        (x1, min(height - 20, y2 + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                    )

            writer.write(display_frame)
            if frame_callback:
                ok_img, jpg = cv2.imencode(".jpg", display_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                if ok_img:
                    frame_callback(video_id, jpg.tobytes())

            progress = min(frame_index / total_frames, 1.0)
            self.storage.update_video_job(
                video_id,
                status="processing",
                progress=progress,
                summary={
                    "live": {
                        "frame_index": frame_index,
                        "timestamp_sec": round(timestamp_sec, 2),
                        "detections": len(detections),
                        "active_tracks": len(active_tracks),
                    },
                    "aisles": zones,
                },
            )
            frame_index += 1

        for payload in tracks_payload.values():
            events = payload["events"]
            if len(events) > 1:
                dists = []
                for a, b in zip(events[:-1], events[1:]):
                    c1 = a["centroid"]
                    c2 = b["centroid"]
                    dt = max(0.001, b["timestamp_sec"] - a["timestamp_sec"])
                    dists.append(math.dist(c1, c2) / dt)
                payload["avg_speed_px_per_sec"] = round(sum(dists) / len(dists), 2)

        self.storage.upsert_profiles(video_id, list(tracks_payload.values()))
        summary = {
            "processed_frames": frame_index,
            "profiles_count": len(tracks_payload),
            "alerts_count": len(self.storage.list_alerts(video_id=video_id)),
            "aisles": zones,
            "output_video": str(output_path),
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
        self.storage.update_video_job(video_id, status="completed", progress=1.0, summary=summary)
        writer.release()
        cap.release()

    def _save_snapshot(self, video_id: str, frame_index: int, frame) -> None:
        out = self.snapshot_dir / f"{video_id}_{frame_index}.jpg"
        cv2.imwrite(str(out), frame)
