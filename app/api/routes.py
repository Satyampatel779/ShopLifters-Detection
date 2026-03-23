from __future__ import annotations

import shutil
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse

from app.core.settings import PROCESSED_DIR, SNAPSHOTS_DIR, UPLOADS_DIR, WATCHLIST_DIR
from app.core.storage import Storage
from app.vision.face import FaceEncoder
from app.vision.pipeline import VideoPipeline


class RealtimeBus:
    def __init__(self) -> None:
        self._clients = set()
        self._lock = threading.Lock()

    async def register(self, websocket) -> None:
        await websocket.accept()
        with self._lock:
            self._clients.add(websocket)

    def unregister(self, websocket) -> None:
        with self._lock:
            if websocket in self._clients:
                self._clients.remove(websocket)

    async def broadcast(self, payload: dict) -> None:
        dead = []
        with self._lock:
            clients = list(self._clients)
        for ws in clients:
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        if dead:
            with self._lock:
                for ws in dead:
                    self._clients.discard(ws)


class RouterFactory:
    def __init__(self, storage: Storage, bus: RealtimeBus) -> None:
        self.storage = storage
        self.bus = bus
        self.face_encoder = FaceEncoder()
        self.pipeline = VideoPipeline(
            storage=storage,
            snapshot_dir=SNAPSHOTS_DIR,
            processed_dir=PROCESSED_DIR,
        )
        self._preview_frames: dict[str, bytes] = {}
        self._preview_lock = threading.Lock()

    def build(self) -> APIRouter:
        router = APIRouter(prefix="/api")

        @router.get("/health")
        def health() -> dict[str, str]:
            return {"status": "ok"}

        @router.post("/watchlist")
        async def add_watchlist(name: str = Form(...), image: UploadFile = File(...)) -> dict:
            ext = Path(image.filename or "face.jpg").suffix.lower() or ".jpg"
            item_id = str(uuid.uuid4())
            target = WATCHLIST_DIR / f"{item_id}{ext}"
            with target.open("wb") as f:
                shutil.copyfileobj(image.file, f)

            img = cv2.imread(str(target))
            if img is None:
                raise HTTPException(status_code=400, detail="Invalid image file")

            embedding = self.face_encoder.encode_from_image(img)
            if embedding is None:
                raise HTTPException(status_code=400, detail="No face detected in uploaded image")

            payload = {
                "id": item_id,
                "name": name,
                "image_path": str(target),
                "embedding": embedding,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            self.storage.add_watchlist(payload)
            return payload

        @router.get("/watchlist")
        def list_watchlist() -> list[dict]:
            return self.storage.list_watchlist()

        @router.post("/videos/upload")
        async def upload_video(video: UploadFile = File(...)) -> dict:
            if not video.filename:
                raise HTTPException(status_code=400, detail="Video file is required")
            ext = Path(video.filename).suffix.lower()
            if ext not in {".mp4", ".mov", ".avi", ".mkv"}:
                raise HTTPException(status_code=400, detail="Unsupported video format")

            video_id = str(uuid.uuid4())
            target = UPLOADS_DIR / f"{video_id}{ext}"
            with target.open("wb") as f:
                shutil.copyfileobj(video.file, f)

            job = {
                "id": video_id,
                "file_path": str(target),
                "original_filename": video.filename,
                "status": "queued",
                "progress": 0.0,
                "summary": {},
                "error": None,
            }
            self.storage.add_video_job(job)

            thread = threading.Thread(
                target=self._run_job,
                args=(video_id, target),
                daemon=True,
            )
            thread.start()

            return {"video_id": video_id, "status": "queued"}

        @router.get("/videos/{video_id}/status")
        def video_status(video_id: str) -> dict:
            job = self.storage.get_video_job(video_id)
            if not job:
                raise HTTPException(status_code=404, detail="Video job not found")
            return job

        @router.get("/videos/{video_id}/profiles")
        def video_profiles(video_id: str) -> list[dict]:
            return self.storage.list_profiles(video_id)

        @router.get("/videos/{video_id}/preview")
        def video_preview(video_id: str) -> Response:
            with self._preview_lock:
                img = self._preview_frames.get(video_id)
            if not img:
                return Response(status_code=204)
            return Response(content=img, media_type="image/jpeg")

        @router.get("/videos/{video_id}/stream")
        def video_stream(video_id: str) -> StreamingResponse:
            return StreamingResponse(
                self._preview_stream(video_id),
                media_type="multipart/x-mixed-replace; boundary=frame",
            )

        @router.get("/videos/{video_id}/result")
        def video_result(video_id: str) -> FileResponse:
            job = self.storage.get_video_job(video_id)
            if not job:
                raise HTTPException(status_code=404, detail="Video job not found")

            output = job.get("summary", {}).get("output_video")
            path = Path(output) if output else (PROCESSED_DIR / f"{video_id}.mp4")
            if not path.exists():
                # Backward compatibility for earlier jobs.
                alt = PROCESSED_DIR / f"{video_id}.avi"
                if alt.exists():
                    path = alt
            if not path.exists():
                raise HTTPException(status_code=404, detail="Processed video not available yet")
            media_type = "video/mp4" if path.suffix.lower() == ".mp4" else "video/x-msvideo"
            return FileResponse(path, media_type=media_type, filename=path.name)

        @router.get("/alerts")
        def list_alerts(video_id: str | None = None) -> list[dict]:
            return self.storage.list_alerts(video_id=video_id)

        @router.post("/videos/{video_id}/aisles/auto")
        def aisles(video_id: str) -> dict:
            job = self.storage.get_video_job(video_id)
            if not job:
                raise HTTPException(status_code=404, detail="Video job not found")
            return {"video_id": video_id, "aisles": job.get("summary", {}).get("aisles", [])}

        return router

    def _run_job(self, video_id: str, video_path: Path) -> None:
        import asyncio

        try:
            self.storage.update_video_job(video_id, status="processing", progress=0.01)
            self.pipeline.process(
                video_id=video_id,
                video_path=video_path,
                alert_callback=lambda a: asyncio.run(self.bus.broadcast(a)),
                frame_callback=self._on_preview_frame,
            )
        except Exception as exc:
            self.storage.update_video_job(
                video_id,
                status="error",
                error=str(exc),
                progress=1.0,
            )

    def _on_preview_frame(self, video_id: str, jpeg: bytes) -> None:
        with self._preview_lock:
            self._preview_frames[video_id] = jpeg

    def _preview_stream(self, video_id: str):
        last_frame: bytes | None = None
        idle_ticks = 0
        while True:
            with self._preview_lock:
                frame = self._preview_frames.get(video_id)

            if frame is not None:
                last_frame = frame
                idle_ticks = 0
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii")
                    + frame
                    + b"\r\n"
                )
            elif last_frame is not None:
                idle_ticks += 1
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(last_frame)}\r\n\r\n".encode("ascii")
                    + last_frame
                    + b"\r\n"
                )
            else:
                idle_ticks += 1

            job = self.storage.get_video_job(video_id)
            if job and job.get("status") in {"completed", "error"} and idle_ticks > 15:
                break
            time.sleep(0.15)
