import base64
import os
from io import BytesIO
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from PIL import Image
from ultralytics import YOLO

MODEL_NAME = os.getenv("YOLO_MODEL", "yolov8m-seg.pt")
CONFIDENCE = float(os.getenv("YOLO_CONF", "0.25"))
IOU = float(os.getenv("YOLO_IOU", "0.45"))
ROI_WIDTH_RATIO = float(os.getenv("YOLO_ROI_WIDTH_RATIO", "1.0"))
ROI_HEIGHT_RATIO = float(os.getenv("YOLO_ROI_HEIGHT_RATIO", "1.0"))

app = FastAPI(title="yolo-seg-service", version="1.0.0")
model: Optional[YOLO] = None

# Mount the sofa_scan sub-app under /api to handle /api/startup-scan-sofa and avoid wildcard collisions
import sofa_scan
app.mount("/api", sofa_scan.app)


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


def select_instance(
    click_x: Optional[float],
    click_y: Optional[float],
    masks: np.ndarray,
    scores: np.ndarray,
    boxes_norm: np.ndarray,
) -> int:
    if click_x is None or click_y is None:
        return int(scores.argmax())

    mask_h, mask_w = masks.shape[1:]
    px = min(mask_w - 1, max(0, int(click_x * mask_w)))
    py = min(mask_h - 1, max(0, int(click_y * mask_h)))

    y_slice = slice(max(0, py - 1), min(mask_h, py + 2))
    x_slice = slice(max(0, px - 1), min(mask_w, px + 2))
    neighborhood = masks[:, y_slice, x_slice]
    hit_indices = np.where(np.any(neighborhood > 0.5, axis=(1, 2)))[0]
    if hit_indices.size == 0:
        # If the exact mask pixel misses, use normalized boxes as a robust fallback:
        # 1) instance boxes that contain the click, else
        # 2) nearest box center to the click.
        cx = click_x
        cy = click_y
        x1 = boxes_norm[:, 0]
        y1 = boxes_norm[:, 1]
        x2 = boxes_norm[:, 2]
        y2 = boxes_norm[:, 3]

        containing = np.where((cx >= x1) & (cx <= x2) & (cy >= y1) & (cy <= y2))[0]
        if containing.size > 0:
            containing_scores = scores[containing]
            return int(containing[int(containing_scores.argmax())])

        centers_x = (x1 + x2) / 2.0
        centers_y = (y1 + y2) / 2.0
        dist2 = (centers_x - cx) ** 2 + (centers_y - cy) ** 2
        return int(dist2.argmin())

    hit_scores = scores[hit_indices]
    best_local_index = int(hit_scores.argmax())
    return int(hit_indices[best_local_index])


def center_roi_bounds(height: int, width: int) -> tuple[int, int, int, int]:
    roi_w_ratio = min(1.0, max(0.1, ROI_WIDTH_RATIO))
    roi_h_ratio = min(1.0, max(0.1, ROI_HEIGHT_RATIO))

    roi_w = max(1, int(round(width * roi_w_ratio)))
    roi_h = max(1, int(round(height * roi_h_ratio)))

    x0 = (width - roi_w) // 2
    y0 = (height - roi_h) // 2
    x1 = x0 + roi_w
    y1 = y0 + roi_h
    return x0, y0, x1, y1


def map_click_to_roi(
    click_x: Optional[float],
    click_y: Optional[float],
    width: int,
    height: int,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> tuple[Optional[float], Optional[float]]:
    if click_x is None or click_y is None:
        return None, None

    px = click_x * width
    py = click_y * height
    if px < x0 or px >= x1 or py < y0 or py >= y1:
        return None, None

    roi_w = x1 - x0
    roi_h = y1 - y0
    roi_click_x = (px - x0) / roi_w
    roi_click_y = (py - y0) / roi_h
    return roi_click_x, roi_click_y


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
    img_h, img_w = image_np.shape[:2]
    x0, y0, x1, y1 = center_roi_bounds(img_h, img_w)
    roi_image = image_np[y0:y1, x0:x1]

    # Inference runs on the ROI crop to suppress background clutter.
    result = model.predict(roi_image, conf=CONFIDENCE, iou=IOU, imgsz=1024, verbose=False)[0]

    if result.masks is None or result.boxes is None or len(result.boxes) == 0:
        return SegmentResponse(mask=[], label="none", score=0.0)

    masks = result.masks.data.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    boxes_norm = result.boxes.xyxyn.cpu().numpy()

    roi_click_x, roi_click_y = map_click_to_roi(req.clickX, req.clickY, img_w, img_h, x0, y0, x1, y1)
    idx = select_instance(roi_click_x, roi_click_y, masks, scores, boxes_norm)
    roi_mask = masks[idx]
    label = result.names[int(classes[idx])]
    score = float(scores[idx])
    full_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    roi_h_size = y1 - y0
    roi_w_size = x1 - x0
    roi_mask_resized = np.array(
        Image.fromarray(roi_mask).resize((roi_w_size, roi_h_size), resample=Image.BILINEAR)
    )
    full_mask[y0:y1, x0:x1] = (roi_mask_resized > 0.5).astype(np.uint8)

    return SegmentResponse(mask=full_mask.astype(int).tolist(), label=label, score=score)
