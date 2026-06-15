import base64
import os
from io import BytesIO
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from PIL import Image
from ultralytics import YOLO

MODEL_NAME = os.getenv("YOLO_MODEL", "yolov8n-seg.pt")
CONFIDENCE = float(os.getenv("YOLO_CONF", "0.25"))
IOU = float(os.getenv("YOLO_IOU", "0.45"))

app = FastAPI(title="yolo-seg-service", version="1.0.0")
model: Optional[YOLO] = None


class SegmentRequest(BaseModel):
    image: str = Field(..., description="Data URL or raw base64 JPEG/PNG")
    clickX: Optional[float] = Field(default=None, ge=0, le=1)
    clickY: Optional[float] = Field(default=None, ge=0, le=1)


class SegmentResponse(BaseModel):
    mask: list
    label: str
    score: float


def decode_image(image: str) -> np.ndarray:
    encoded = image.split(",", 1)[1] if "," in image else image
    try:
        raw = base64.b64decode(encoded, validate=True)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Invalid base64 image payload") from exc

    try:
        rgb = Image.open(BytesIO(raw)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Unsupported image format") from exc

    return np.array(rgb)


def select_instance(req: SegmentRequest, masks: np.ndarray, scores: np.ndarray) -> int:
    if req.clickX is None or req.clickY is None:
        return int(scores.argmax())

    mask_h, mask_w = masks.shape[1:]
    px = min(mask_w - 1, max(0, int(req.clickX * mask_w)))
    py = min(mask_h - 1, max(0, int(req.clickY * mask_h)))

    hit_indices = np.where(masks[:, py, px] > 0.5)[0]
    if hit_indices.size == 0:
        return int(scores.argmax())

    hit_scores = scores[hit_indices]
    best_local_index = int(hit_scores.argmax())
    return int(hit_indices[best_local_index])


@app.on_event("startup")
def startup() -> None:
    global model
    model = YOLO(MODEL_NAME)


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "service": "yolo-seg-service",
        "model": MODEL_NAME,
        "ready": model is not None,
    }


@app.post("/segment", response_model=SegmentResponse)
def segment(req: SegmentRequest) -> SegmentResponse:
    if model is None:
        raise HTTPException(status_code=503, detail="Model is still loading")

    image_np = decode_image(req.image)
    result = model.predict(image_np, conf=CONFIDENCE, iou=IOU, verbose=False)[0]

    if result.masks is None or result.boxes is None or len(result.boxes) == 0:
        return SegmentResponse(mask=[], label="none", score=0.0)

    masks = result.masks.data.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)

    idx = select_instance(req, masks, scores)
    mask = masks[idx]
    label = result.names[int(classes[idx])]
    score = float(scores[idx])

    return SegmentResponse(mask=(mask > 0.5).astype(int).tolist(), label=label, score=score)
