# Retail Vision Intelligence Platform

Portfolio-grade computer vision project for retail safety operations.

This project demonstrates how to design and deliver an end-to-end CV system: from detection and face matching to real-time operator UX, evidence generation, and alert workflows.

## Why this project matters
- Solves a real retail problem: identifying watchlist persons across CCTV footage quickly.
- Demonstrates applied ML engineering, backend systems design, and product-focused frontend execution.
- Shows recruiter-relevant skills across AI, APIs, real-time streaming, and operational dashboards.

## Key capabilities
- Upload watchlist face photos and generate reusable facial embeddings.
- Upload CCTV footage and run people detection + track profiling.
- Assign people movement to detected aisle zones.
- Trigger automatic alerts on watchlist face match.
- Stream live detection preview during processing.
- Generate a processed output video with overlays for demo and evidence.

## Technical stack
- Backend: FastAPI, asynchronous job threading, WebSocket alerts
- Vision: YOLOv8x (person detection), YuNet + SFace (face detection/recognition), centroid tracking
- Data: SQLite for jobs, profiles, watchlist, alerts
- Frontend: vanilla JS dashboard with real-time telemetry and live stream preview

## Architecture overview
1. Video upload API stores CCTV input in local storage.
2. Pipeline reads frames and performs person detection.
3. Tracker links detections to persistent track ids.
4. Face encoder extracts embeddings per valid face crop.
5. Matcher compares embeddings against watchlist vectors.
6. Alert service writes incidents and pushes real-time events.
7. UI displays live preview stream, alerts, and processed output video.

## Recruiter-facing engineering highlights
- Built a complete vision workflow, not just model inference.
- Implemented real-time preview streaming and post-process evidence rendering.
- Added robust output handling with codec fallback for cross-environment reliability.
- Structured code into API/core/vision modules for maintainability.
- Delivered a product-style UI with operational and showcase layers.

## Quick start (Windows)
1. Create and activate environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run server:

```powershell
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

4. Open dashboard:
- http://localhost:8000

On first run, YuNet/SFace ONNX models are auto-downloaded to `data/models`.

## Demo flow (for interviews)
1. Add 2-3 watchlist profiles.
2. Upload a short CCTV clip.
3. Show live detection feed while processing is running.
4. Open processed output video and explain overlays.
5. Walk through profile cards, aisles visited, and alert events.

## API surface
- `GET /api/health`
- `POST /api/watchlist`
- `GET /api/watchlist`
- `POST /api/videos/upload`
- `GET /api/videos/{video_id}/status`
- `GET /api/videos/{video_id}/profiles`
- `GET /api/videos/{video_id}/stream`
- `GET /api/videos/{video_id}/result`
- `GET /api/alerts`
- `WS /ws/alerts`

## Skills demonstrated
- Computer vision pipeline design
- Model integration and threshold-based decisioning
- API design and async processing
- Real-time frontend visualization
- Fault-tolerant media output handling
- Product storytelling for technical demos

## Resume bullets you can use
- Designed and implemented a real-time retail vision intelligence platform using YOLOv8x, YuNet/SFace, and FastAPI.
- Built an end-to-end CCTV processing pipeline that generated person tracks, aisle movement analytics, and automated watchlist alerts.
- Developed a live operator dashboard with WebSocket-driven alerting, stream preview, and processed evidence playback.
- Improved reliability by adding codec-aware output fallback and runtime telemetry for easier operations debugging.

## Current limitations
- Optimized for uploaded videos, not multi-camera RTSP production scale yet.
- Aisle mapping is heuristic and may require scene-specific tuning.
- Accuracy varies with occlusion, low lighting, and camera angle.

## Next upgrades
1. ByteTrack/DeepSORT for stronger multi-person identity continuity.
2. RTSP live ingestion and multi-camera orchestration.
3. Role-based access, audit logging, and retention policies.
4. Evaluation dashboard with precision/recall and threshold calibration.
