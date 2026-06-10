# Semantic Lifting Backend for 3DGS

This backend service acts as the vision-intelligence engine for a Gaussian Splatting viewer. It processes 2D snapshots of a 3D scene and utilizes Google Gemini to perform semantic object detection (identifying chairs and photo frames).

## Overview
This service provides an API endpoint that:
1. Receives a Base64 image snapshot from the frontend.
2. Applies spatial context (bounding boxes/click coordinates).
3. Queries Gemini (Vision) to segment/identify objects.
4. Returns structured JSON to the frontend for 3D "Lifting" (mapping 2D boxes to 3D world space).

## Tech Stack
*   **Runtime:** Node.js
*   **Framework:** Express.js
*   **AI SDK:** `@google/generative-ai`
*   **Language:** TypeScript/JavaScript

## Prerequisites
*   Node.js (v18+)
*   An API Key from [Google AI Studio](https://aistudio.google.com/)

## Installation

1. Clone the repository and install dependencies:
   ```bash
   npm install
   ```

2. Create a `.env` file in the root directory:
   ```text
   GEMINI_API_KEY=your_actual_api_key_here
   ```

3. Start the development server:
   ```bash
   npm run dev
   # OR
   npx ts-node server.ts
   ```

## API Documentation

### `POST /api/annotate`

Analyzes an image snapshot to detect chairs and photo frames within a specific region.

**Request Body:**
```json
{
  "image": "data:image/jpeg;base64,...",
  "userBox": [ymin, xmin, ymax, xmax] 
}
```

**Response:**
```json
{
  "chairs": [{"box": [0.2, 0.3, 0.5, 0.4]}],
  "frames": [{"box": [0.1, 0.8, 0.2, 0.9]}]
}
```

## Troubleshooting & FAQ

### 1. "404 Not Found" Errors
If you see a 404 in the terminal, it usually means your API Key does not have access to the model string provided in the code.
*   **Solution:** Run a model discovery script to see your authorized models. Use the exact model name from your authorized list (e.g., `models/gemini-3.5-flash`).

### 2. "No JSON found" Errors
This happens when Gemini returns empty results or chatty text.
*   **Solution:** Check your `server.ts` regex logic. Ensure the prompt explicitly tells the model: *"No markdown, no backticks, return ONLY raw JSON."*

### 3. Image Size Issues
If images are coming through as "invalid format," ensure your `express.json` limit is set to at least `20mb`. Base64 snapshots are large.

## Current Configuration
*   **Model:** `models/gemini-3.5-flash` (Update this in `server.ts` if your project's model list changes).
*   **Classes Detected:** Chairs, Frames.