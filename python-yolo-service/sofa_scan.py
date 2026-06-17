import base64
from io import BytesIO
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image
from ultralytics import YOLO

# ============================================================================
# FastAPI Setup
# ============================================================================
app = FastAPI(
    title="Sofa Startup Scanner",
    description="Ultra-fast sofa scanner targeting COCO Class 57",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Schemas & Validation
# ============================================================================
class SofaScanRequest(BaseModel):
    image: str = Field(
        ..., 
        description="Base64 encoded image frame or data URL captured from viewport"
    )

class SofaScanResponse(BaseModel):
    sofaMasks: list[list[list[int]]] = Field(
        ..., 
        description="List of downscaled (128x128) binary sofa masks"
    )

# Global model reference
startup_model: YOLO | None = None

@app.on_event("startup")
def load_model() -> None:
    """Loads the Nano segmentation model on startup to prevent import-time blockages."""
    global startup_model
    startup_model = YOLO("yolov8n-seg.pt")

# ============================================================================
# Endpoints
# ============================================================================
@app.post(
    "/startup-scan-sofa", 
    response_model=SofaScanResponse,
    summary="Scan frame for sofas and return resized binary masks"
)
def startup_scan_sofa(payload: SofaScanRequest) -> dict[str, Any]:
    """
    Scans a frame for sofa instances (COCO class 57) and returns downscaled masks.
    
    NOTE: Executed as a standard synchronous function (def) to allow FastAPI 
    to automatically offload the CPU-bound YOLO prediction to a thread pool, 
    preventing blocking the main event loop.
    """
    if startup_model is None:
        raise HTTPException(status_code=503, detail="YOLO Model is still loading")

    # 1. Parse and decode base64 image data URL robustly
    image_str = payload.image
    encoded = image_str.split(",", 1)[1] if "," in image_str else image_str
    
    try:
        img_bytes = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 image encoding") from exc

    try:
        # Load and convert image to RGB using Pillow (consistent with main.py)
        img_pil = Image.open(BytesIO(img_bytes)).convert("RGB")
        img_np = np.array(img_pil)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Unsupported or corrupted image format") from exc

    # 2. Run prediction targeting ONLY COCO Class 57 (Sofa) at a lower resolution
    results = startup_model.predict(
        img_np, 
        imgsz=480, 
        classes=[57], 
        conf=0.3, 
        verbose=False
    )[0]
    
    sofa_instances = []
    
    # 3. Extract and resize masks if a sofa is found in the frame
    if results.masks is not None:
        for mask in results.masks.data:
            # Convert PyTorch tensor to 2D NumPy binary mask
            mask_np = mask.cpu().numpy().astype(np.uint8)
            
            # Downscale mask to 128x128 for network speed efficiency
            # Using Pillow NEAREST interpolation to align with main.py patterns
            mask_pil = Image.fromarray(mask_np)
            mask_resized = np.array(mask_pil.resize((128, 128), resample=Image.NEAREST))
            
            # Convert NumPy array to nested list for JSON serialization
            sofa_instances.append(mask_resized.tolist())
            
    return {"sofaMasks": sofa_instances}
