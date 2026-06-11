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
const MODEL_NAME = "gemini-3-pro-image"; // Use the 'models/' prefix
const PORT = process.env.PORT || 3000;

const annotationSchema: Schema = {
  type: SchemaType.OBJECT,
  properties: {
    objects: {
      type: SchemaType.ARRAY,
      items: {
        type: SchemaType.OBJECT,
        properties: {
          label: { type: SchemaType.STRING },
          box: { type: SchemaType.ARRAY, items: { type: SchemaType.NUMBER } },
        },
        required: ["label", "box"],
      },
    },
  },
  required: ["objects"],
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
function buildPrompt(box: number[]): string {
  return `
    Analyze this image. Identify ALL significant physical objects (furniture, decorations, etc).
    
    For each object found, perform classification.
    Return ONLY raw JSON with an "objects" key containing an array of objects.
    
    Format:
    {
      "objects": [
        {"label": "sofa", "box": [ymin, xmin, ymax, xmax]},
        {"label": "frame", "box": [ymin, xmin, ymax, xmax]}
      ]
    }

    Rules:
    1. 'label' MUST be either 'sofa' or 'frame' based on visual features. Do NOT use the word 'objects' as a label.
    2. If no objects found, return {"objects": []}.
    3. No markdown, no backticks.
    4. Use normalized 0-1 coordinates.
    5. Focus on the region: ${JSON.stringify(box)}.

    IMPORTANT: Perform visual analysis to differentiate. If it has pillows and back cushions, label it 'sofa'. If it has a visible border on a wall, label it 'frame'.
  `;
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

      // Determine the focus box for the prompt
      let box = userBox;
      if (!box && clickX !== undefined && clickY !== undefined) {
        const size = 0.1;
        box = [
          Math.max(0, clickY - size),
          Math.max(0, clickX - size),
          Math.min(1, clickY + size),
          Math.min(1, clickX + size),
        ];
      }

      console.log("AI is focusing on box:", box);

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
        buildPrompt(box),
        { inlineData: { data: base64Data, mimeType: "image/jpeg" } },
      ]);

      const text = result.response.text();
      console.log("Raw Response:", text);

      const parsed = JSON.parse(text);
      res.json(parsed);
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
