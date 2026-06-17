### 1. The Implementation Flow
In [server.ts](file:///c:/Projects/Agents/gemini-test/gemini-test/server.ts), the core integration happens inside the **WebSocket connection handler** (`io.on("connection", ...)`):

1. **User Action:** The client sends a `"request_annotation"` event containing the image data and the click coordinates (`clickX`, `clickY`).
2. **Step 1 — Instant YOLO Response:** 
   * The server immediately forwards the image and click coordinates to a local YOLO segmentation service ([server.ts:L298](file:///c:/Projects/Agents/gemini-test/gemini-test/server.ts#L298)).
   * The YOLO service performs fast instance segmentation and returns a visual segmentation `mask` and a class `label` (e.g., `"sofa"` or `"chair"`).
   * The server emits a `"mask_ready"` event to the client instantly, allowing the UI to render the outline without delay.
3. **Step 2 — Gemini Enrichment (Asynchronous):**
   * The server then calls Gemini (`callGeminiDescription`) asynchronously ([server.ts:L302](file:///c:/Projects/Agents/gemini-test/gemini-test/server.ts#L302)).
   * Gemini takes the image and the YOLO-detected label, analyzes the context, and generates an engaging, museum-style description of the object.
   * Once Gemini finishes, the server emits `"description_ready"` to update the UI with the detailed description.

---

### 2. Benefits of Combining YOLO & Gemini

| Feature | YOLO Service | Gemini (VLM) | Combined Benefit |
| :--- | :--- | :--- | :--- |
| **Speed / Latency** | **Extremely Fast** (Milliseconds) | **Slower** (Seconds) | **Immediate User Feedback:** The user sees the visual selection mask instantly (YOLO), while the detailed text description loads progressively (Gemini). |
| **Spatial Precision** | **High Precision** (Pixel-perfect mask/contours) | **Low Precision** (Coordinates/rough boxes) | **Perfect Contours:** VLMs struggle with pixel-perfect visual segmentation. YOLO handles the spatial boundaries perfectly, while Gemini handles the conceptual context. |
| **Semantic Depth** | **Limited** (Discrete class names, e.g., "sofa") | **Unlimited** (Rich natural language, museum descriptions) | **Rich Contextual UX:** YOLO identifies *what* and *where* the object is, while Gemini describes the *details*, *aesthetics*, and *story* behind it. |
