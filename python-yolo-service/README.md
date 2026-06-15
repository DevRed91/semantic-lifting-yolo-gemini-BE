# YOLO Segmentation Microservice

Lightweight FastAPI service for running YOLO segmentation and returning a binary mask.

## Endpoints

- `GET /health`: service status and loaded model
- `POST /segment`: run segmentation

## Request

```json
{
  "image": "data:image/jpeg;base64,...",
  "clickX": 0.52,
  "clickY": 0.41
}
```

- `image`: required, data URL or raw base64 image
- `clickX`, `clickY`: optional normalized coordinates (`0..1`) to prefer the instance under the click

## Response

```json
{
  "mask": [[0, 0, 1], [0, 1, 1]],
  "label": "chair",
  "score": 0.87
}
```

When no object is detected:

```json
{
  "mask": [],
  "label": "none",
  "score": 0.0
}
```

## Local run

```bash
cd python-yolo-service
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Config

- `YOLO_MODEL` (default: `yolov8n-seg.pt`)
- `YOLO_CONF` (default: `0.25`)
- `YOLO_IOU` (default: `0.45`)
