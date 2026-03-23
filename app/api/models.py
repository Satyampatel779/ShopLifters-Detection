from __future__ import annotations

from pydantic import BaseModel


class UploadVideoResponse(BaseModel):
    video_id: str
    status: str
