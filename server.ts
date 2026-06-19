import express from "express";
import cors from "cors";
import {
  GoogleGenerativeAI,
  SchemaType,
  type Schema,
} from "@google/generative-ai";
import dotenv from "dotenv";
import fs from "fs";

dotenv.config();

// ============================================================================
// Constants & Configuration
// ============================================================================
const MODEL_NAME = "gemini-3-pro-image";
const PORT = process.env.PORT || 3000;

type NormalizedBox = [number, number, number, number];

interface SingleAnnotation {
  label: string;
  box: NormalizedBox;
}

const annotationSchema: Schema = {
  type: SchemaType.OBJECT,
  properties: {
    label: { type: SchemaType.STRING },
    box: {
      type: SchemaType.ARRAY,
      items: { type: SchemaType.NUMBER },
      minItems: 4,
      maxItems: 4,
    },
  },
  required: ["label", "box"],
};

// ============================================================================
// CORS Configuration
// ============================================================================
const normalizeOrigin = (origin: string): string =>
  origin.trim().replace(/\/$/, "");

const envAllowedOrigins = (process.env.CORS_ALLOWED_ORIGINS || "")
  .split(",")
  .map(normalizeOrigin)
  .filter(Boolean);

const defaultAllowedOrigins = ["https://auto-annotate.netlify.app"];

const devOrigins = ["http://localhost:5173", "http://127.0.0.1:5173"];
const isProduction = process.env.NODE_ENV === "production";

const effectiveAllowedOrigins = Array.from(
  new Set(
    (isProduction
      ? [...envAllowedOrigins, ...defaultAllowedOrigins]
      : [...envAllowedOrigins, ...defaultAllowedOrigins, ...devOrigins]
    ).map(normalizeOrigin),
  ),
);

const isAllowedOrigin = (origin?: string): boolean => {
  if (!origin) return true;
  if (effectiveAllowedOrigins.length === 0) return true;
  return effectiveAllowedOrigins.includes(normalizeOrigin(origin));
};

const corsOptions: cors.CorsOptions = {
  origin: (origin, callback) => callback(null, isAllowedOrigin(origin)),
  methods: ["GET", "POST", "OPTIONS"],
  allowedHeaders: [
    "Content-Type",
    "Authorization",
    "bypass-tunnel-reminder",
    "ngrok-skip-browser-warning",
  ],
  optionsSuccessStatus: 204,
};

// ============================================================================
// Express Setup
// ============================================================================
const app = express();

app.use(cors(corsOptions));
app.use(express.json({ limit: "20mb" }));

// ============================================================================
// Helpers
// ============================================================================
function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function sanitizeBox(box: unknown): NormalizedBox | null {
  if (!Array.isArray(box) || box.length !== 4) return null;
  const numeric = box.map((value) => Number(value));
  if (numeric.some((value) => !Number.isFinite(value))) return null;

  const clamped = numeric.map(clamp01) as NormalizedBox;
  const [xmin, ymin, xmax, ymax] = clamped;
  if (xmax <= xmin || ymax <= ymin) return null;

  return clamped;
}

function sanitizeDetectionPayload(payload: any): SingleAnnotation | null {
  const candidate = payload?.objects?.[0] ?? payload;
  if (!candidate || typeof candidate.label !== "string") return null;

  const box = sanitizeBox(candidate.box);
  if (!box) return null;

  return {
    label: candidate.label.trim(),
    box,
  };
}

function buildPrompt(
  clickX: number | undefined,
  clickY: number | undefined,
  focusBox: NormalizedBox | null,
): string {
  const clickInfo =
    typeof clickX === "number" && typeof clickY === "number"
      ? `(x: ${clickX.toFixed(4)}, y: ${clickY.toFixed(4)})`
      : "(x: unknown, y: unknown)";

  const focusInfo = focusBox
    ? `Focus region [xmin, ymin, xmax, ymax]: [${focusBox.join(", ")}].`
    : "No focus region available.";

  return `Analyze this 3D scene snapshot.

The user clicked near normalized image coordinates ${clickInfo}.
${focusInfo}

Task:
- Identify the single object at the click target.
- Return one concise object label.
- Return one tight normalized bounding box [xmin, ymin, xmax, ymax] in [0,1].

Return ONLY raw JSON in this exact shape:
{
  "label": "string",
  "box": [xmin, ymin, xmax, ymax]
}

Rules:
- No markdown and no additional keys.
- Bounding box must tightly follow visible object boundaries.
- Coordinates must be numeric and normalized to [0,1].`;
}

// ============================================================================
// Routes
// ============================================================================
app.get("/health", (_req: express.Request, res: express.Response) => {
  res.status(200).json({
    ok: true,
    service: "gemini-annotate-backend",
    hasApiKey: Boolean(process.env.GEMINI_API_KEY),
    timestamp: new Date().toISOString(),
  });
});

app.post(
  "/api/annotate",
  async (req: express.Request, res: express.Response) => {
    console.log("--- REQUEST RECEIVED ---");
    console.log("Headers:", req.headers);
    console.log("Body Keys:", Object.keys(req.body));

    try {
      if (!process.env.GEMINI_API_KEY) {
        return res.status(500).json({ error: "Missing GEMINI_API_KEY" });
      }

      const { image, userBox, clickX, clickY } = req.body;

      if (!image || !image.startsWith("data:image/jpeg")) {
        return res.status(400).json({ error: "Invalid image format" });
      }

      const clickXNum = Number.isFinite(Number(clickX)) ? Number(clickX) : undefined;
      const clickYNum = Number.isFinite(Number(clickY)) ? Number(clickY) : undefined;

      let focusBox = sanitizeBox(userBox);
      if (!focusBox && clickXNum !== undefined && clickYNum !== undefined) {
        const size = 0.1;
        focusBox = sanitizeBox([
          clickXNum - size,
          clickYNum - size,
          clickXNum + size,
          clickYNum + size,
        ]);
      }

      console.log("AI focus box:", focusBox);

      // Write a debug snapshot
      const base64Data = image.includes(",") ? image.split(",")[1] : image;
      fs.writeFileSync("debug_snapshot.jpg", Buffer.from(base64Data, "base64"));

      // Call Gemini
      const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
      const model = genAI.getGenerativeModel({
        model: MODEL_NAME,
        generationConfig: {
          responseMimeType: "application/json",
          responseSchema: annotationSchema,
        },
      });

      const result = await model.generateContent([
        buildPrompt(clickXNum, clickYNum, focusBox),
        { inlineData: { data: base64Data, mimeType: "image/jpeg" } },
      ]);

      const text = result.response.text();
      console.log("Raw Response:", text);

      const parsed = JSON.parse(text);
      const detection = sanitizeDetectionPayload(parsed);

      if (!detection) {
        return res.status(502).json({ error: "Invalid Gemini annotation payload" });
      }

      res.json(detection);
    } catch (error: any) {
      console.error("SERVER ERROR:", error.message);
      res.status(500).json({ error: "Failed to annotate" });
    }
  },
);

// ============================================================================
// Start Server
// ============================================================================
app.listen(PORT, () =>
  console.log(`Annotation server running on port ${PORT}`),
);
