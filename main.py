from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import RealtimeBus, RouterFactory
from app.core.settings import DB_PATH
from app.core.storage import Storage

app = FastAPI(title="ShopLifters Detection Demo", version="0.1.0")

storage = Storage(DB_PATH)
bus = RealtimeBus()
router = RouterFactory(storage=storage, bus=bus).build()
app.include_router(router)

web_dir = Path("web")
app.mount("/assets", StaticFiles(directory=web_dir), name="assets")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(web_dir / "index.html")


@app.websocket("/ws/alerts")
async def alerts_socket(websocket: WebSocket) -> None:
    await bus.register(websocket)
    try:
        while True:
            # Keep connection alive. Client does not need to send business payload.
            await websocket.receive_text()
    except WebSocketDisconnect:
        bus.unregister(websocket)
    except Exception:
        bus.unregister(websocket)
