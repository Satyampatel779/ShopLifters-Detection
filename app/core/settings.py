from __future__ import annotations

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
WATCHLIST_DIR = DATA_DIR / "watchlist"
SNAPSHOTS_DIR = DATA_DIR / "snapshots"
MODELS_DIR = DATA_DIR / "models"
PROCESSED_DIR = DATA_DIR / "processed"
DB_PATH = DATA_DIR / "app.db"

for path in [DATA_DIR, UPLOADS_DIR, WATCHLIST_DIR, SNAPSHOTS_DIR, MODELS_DIR, PROCESSED_DIR]:
    path.mkdir(parents=True, exist_ok=True)
