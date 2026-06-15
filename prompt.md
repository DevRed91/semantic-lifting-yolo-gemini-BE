#INSTRUCTIONS:
High-Performance Backend (YOLO & Gemini Pipelines) Check:
1.YOLO Pipeline: Set up a lightweight FastAPI microservice to run YOLO segmentations.
2.Dual-Channel Processing: Update the backend orchestrator: when a click occurs, it forwards the snapshot to YOLO (fast) and Gemini (slow). YOLO mask data is pushed immediately to the client via Socket.io (mask_ready), followed by Gemini's text descriptions (description_ready).
3.Structured Outputs: Fix the Gemini response schema in server.ts to separate the coordinates from the detailed description prompt, resolving the current structure mismatch