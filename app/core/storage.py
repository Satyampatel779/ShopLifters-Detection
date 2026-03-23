from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any


class Storage:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS watchlist (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    image_path TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS videos (
                    id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    original_filename TEXT NOT NULL,
                    status TEXT NOT NULL,
                    progress REAL NOT NULL,
                    summary TEXT,
                    error TEXT
                );

                CREATE TABLE IF NOT EXISTS profiles (
                    id TEXT PRIMARY KEY,
                    video_id TEXT NOT NULL,
                    track_id INTEGER NOT NULL,
                    payload TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS alerts (
                    id TEXT PRIMARY KEY,
                    video_id TEXT NOT NULL,
                    watchlist_person_id TEXT NOT NULL,
                    watchlist_person_name TEXT NOT NULL,
                    track_id INTEGER NOT NULL,
                    match_score REAL NOT NULL,
                    timestamp_sec REAL NOT NULL,
                    aisle TEXT NOT NULL,
                    status TEXT NOT NULL
                );
                """
            )

    def add_watchlist(self, payload: dict[str, Any]) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO watchlist(id, name, image_path, embedding, created_at) VALUES (?, ?, ?, ?, ?)",
                (
                    payload["id"],
                    payload["name"],
                    payload["image_path"],
                    json.dumps(payload["embedding"]),
                    payload["created_at"],
                ),
            )

    def list_watchlist(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM watchlist ORDER BY created_at DESC").fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            result.append(
                {
                    "id": row["id"],
                    "name": row["name"],
                    "image_path": row["image_path"],
                    "embedding": json.loads(row["embedding"]),
                    "created_at": row["created_at"],
                }
            )
        return result

    def add_video_job(self, payload: dict[str, Any]) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO videos(id, file_path, original_filename, status, progress, summary, error) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    payload["id"],
                    payload["file_path"],
                    payload["original_filename"],
                    payload["status"],
                    payload["progress"],
                    json.dumps(payload.get("summary", {})),
                    payload.get("error"),
                ),
            )

    def update_video_job(self, video_id: str, **fields: Any) -> None:
        if not fields:
            return
        if "summary" in fields:
            fields["summary"] = json.dumps(fields["summary"])
        keys = list(fields.keys())
        values = [fields[k] for k in keys]
        set_clause = ", ".join([f"{k} = ?" for k in keys])
        with self._lock, self._connect() as conn:
            conn.execute(f"UPDATE videos SET {set_clause} WHERE id = ?", (*values, video_id))

    def get_video_job(self, video_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM videos WHERE id = ?", (video_id,)).fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "file_path": row["file_path"],
            "original_filename": row["original_filename"],
            "status": row["status"],
            "progress": row["progress"],
            "summary": json.loads(row["summary"] or "{}"),
            "error": row["error"],
        }

    def upsert_profiles(self, video_id: str, profiles: list[dict[str, Any]]) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM profiles WHERE video_id = ?", (video_id,))
            for profile in profiles:
                key = f"{video_id}:{profile['track_id']}"
                conn.execute(
                    "INSERT INTO profiles(id, video_id, track_id, payload) VALUES (?, ?, ?, ?)",
                    (key, video_id, profile["track_id"], json.dumps(profile)),
                )

    def list_profiles(self, video_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM profiles WHERE video_id = ? ORDER BY track_id ASC", (video_id,)
            ).fetchall()
        return [json.loads(row["payload"]) for row in rows]

    def add_alert(self, payload: dict[str, Any]) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO alerts(id, video_id, watchlist_person_id, watchlist_person_name, track_id, match_score, timestamp_sec, aisle, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["id"],
                    payload["video_id"],
                    payload["watchlist_person_id"],
                    payload["watchlist_person_name"],
                    payload["track_id"],
                    payload["match_score"],
                    payload["timestamp_sec"],
                    payload["aisle"],
                    payload["status"],
                ),
            )

    def list_alerts(self, video_id: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM alerts"
        args: tuple[Any, ...] = ()
        if video_id:
            query += " WHERE video_id = ?"
            args = (video_id,)
        query += " ORDER BY timestamp_sec DESC"
        with self._connect() as conn:
            rows = conn.execute(query, args).fetchall()
        return [dict(row) for row in rows]
