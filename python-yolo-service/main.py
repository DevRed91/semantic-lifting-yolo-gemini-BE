from fastapi import FastAPI, Request
from ultralytics import YOLO
import cv2
import numpy as np
import base64

app = FastAPI()
model = YOLO("yolov8n-seg.pt") # Small, fast segmentation model

@app.post("/segment")
async def get_yolo_mask(info: Request):
    data = await info.json()
    # 1. Decode Image
    img_bytes = base64.b64decode(data['image'].split(',')[1])
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 2. Run YOLOv8-seg
    results = model(img)
    
    # 3. Extract masks for specific objects
    # This returns a binary mask of the same size as the image
    if results[0].masks is not None:
        # Get the first detected object's mask
        mask = results[0].masks.data[0].cpu().numpy() 
        # Resize to original image size
        mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))
        return {"mask": mask_resized.tolist(), "label": results[0].names[int(results[0].boxes.cls[0])]}
    
    return {"mask": [], "label": "none"}