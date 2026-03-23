from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class WatchlistPerson:
    id: str
    name: str
    image_path: str
    embedding: list[float]
    created_at: str


@dataclass
class TrackEvent:
    frame_index: int
    timestamp_sec: float
    bbox: list[int]
    centroid: list[int]
    aisle: str
    confidence: float


@dataclass
class PersonProfile:
    track_id: int
    start_time_sec: float
    end_time_sec: float
    avg_speed_px_per_sec: float
    aisles_visited: list[str] = field(default_factory=list)
    events: list[TrackEvent] = field(default_factory=list)
    best_face_embedding: list[float] | None = None


@dataclass
class AlertRecord:
    id: str
    video_id: str
    watchlist_person_id: str
    watchlist_person_name: str
    track_id: int
    match_score: float
    timestamp_sec: float
    aisle: str
    status: str


@dataclass
class VideoJob:
    id: str
    file_path: str
    original_filename: str
    status: str
    progress: float
    summary: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
