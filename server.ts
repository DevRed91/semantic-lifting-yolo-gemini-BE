import express from "express";
import cors from "cors";
import { createServer } from "http";
import { Server } from "socket.io";
import {
  GoogleGenerativeAI,
  SchemaType,
  type Schema,
} from "@google/generative-ai";
import dotenv from "dotenv";
import { promises as fs } from "fs";

dotenv.config();

// ============================================================================
// Constants & Configuration
// ============================================================================
const MODEL_NAME = "gemini-3-pro-image"; // Use the 'models/' prefix
const PORT = process.env.PORT || 3000;
const YOLO_SERVICE_URL =
  process.env.YOLO_SERVICE_URL || "http://127.0.0.1:8000/segment";

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
const httpServer = createServer(app);
const io = new Server(httpServer, {
  cors: {
    origin: (origin, callback) => callback(null, isAllowedOrigin(origin)),
  },
});

app.use(cors(corsOptions));
app.use(express.json({ limit: "20mb" }));

// ============================================================================
// Helpers
// ============================================================================
const relevantYoloClasses = ["chair", "sofa", "bed", "dining table", "potted plant"];
function buildPrompt(clickX: any, clickY: any): string {
  return `
    Analyze this 3D scene snapshot. The user has clicked on the coordinates (x: ${clickX}, y: ${clickY}).
    
    TASK:
    1. Identify the object. You MUST prioritize labels from this list: [${relevantYoloClasses.join(", ")}].
    2. If the object matches one of the prioritized labels, use it directly (e.g., if the object is a chair, the label MUST be "chair").
    3. Provide an engaging, museum-style description of the object (2-3 sentences).
    
    Return ONLY raw JSON in this format:
    {
      "label": "string",
      "description": "string"
    }

    Rules:
      1. 'label' MUST be descriptive based on the structure (e.g., 'house', 'apartment', 'skyscraper', 'commercial_building').
      2. If no structure found, return {"objects": []}.
      3. No markdown, no backticks, no explanatory text.
      4. Use normalized 0-1 coordinates.
      5. CRITICAL: Do not label the sky, streets, vegetation, or general 'background' as a building. Only include the actual architectural structure.
      6. If a building is partially blocked by a tree or pole, still include it and draw the box around the visible structure.
  `;
}

function buildDescriptionPrompt(label: string): string {
  return `
    You are writing museum wall text.
    The highlighted object label is "${label}".
    
    Write an engaging museum-style description in 2-3 sentences.
    Return ONLY raw JSON:
    {
      "description": "string"
    }
    No markdown or backticks.
  `;
}

async function callYoloService(
  image: string,
  clickX?: number,
  clickY?: number,
): Promise<{ mask: unknown; label: string }> {
  const response = await fetch(YOLO_SERVICE_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image, clickX, clickY }),
  });

  if (!response.ok) {
    throw new Error(`YOLO service failed with status ${response.status}`);
  }

  const yolo = (await response.json()) as { mask?: unknown; label?: string };
  return {
    mask: yolo.mask ?? [],
    label: yolo.label ?? "unknown",
  };
}

async function callGeminiDescription(
  imageBase64: string,
  label: string,
): Promise<{ description: string }> {
  if (!process.env.GEMINI_API_KEY) {
    throw new Error("Missing GEMINI_API_KEY");
  }

  const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
  const model = genAI.getGenerativeModel({
    model: MODEL_NAME,
    generationConfig: {
      responseMimeType: "application/json",
      responseSchema: {
        type: SchemaType.OBJECT,
        properties: {
          description: { type: SchemaType.STRING },
        },
        required: ["description"],
      },
    },
  });

  const result = await model.generateContent([
    buildDescriptionPrompt(label),
    { inlineData: { data: imageBase64, mimeType: "image/jpeg" } },
  ]);

  const parsed = JSON.parse(result.response.text()) as { description?: string };
  return { description: parsed.description ?? "" };
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
      await fs.writeFile(
        "debug_snapshot.jpg",
        Buffer.from(base64Data, "base64"),
      );

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
        buildPrompt(clickX, clickY),
        { inlineData: { data: base64Data, mimeType: "image/jpeg" } },
      ]);

      const text = result.response.text();
      console.log("Raw Response:", text);

      const parsed = JSON.parse(text);
      res.json(Array.isArray(parsed) ? parsed : (parsed.objects ?? []));
    } catch (error: any) {
      console.error("SERVER ERROR:", error.message);
      res.status(500).json({ error: "Failed to annotate" });
    }
  },
);

io.on("connection", (socket) => {
  socket.on(
    "request_annotation",
    async ({
      image,
      clickX,
      clickY,
    }: {
      image: string;
      box?: number[];
      clickX?: number;
      clickY?: number;
    }) => {
      try {
        if (!image || !image.startsWith("data:image/jpeg")) {
          throw new Error("Invalid image format");
        }

        const base64Data = image.includes(",") ? image.split(",")[1] : image;
        await fs.writeFile(
          "debug_snapshot.jpg",
          Buffer.from(base64Data, "base64"),
        );

        // 1. Instant YOLO response
        const yolo = await callYoloService(image, clickX, clickY);
        socket.emit("mask_ready", { mask: yolo.mask, label: yolo.label });

        // 2. Gemini response with museum-style description
        const gemini = await callGeminiDescription(base64Data, yolo.label);
        socket.emit("description_ready", {
          description: gemini.description,
          clickX,
          clickY,
        });
      } catch (err) {
        const message =
          err instanceof Error ? err.message : "Vision pipeline failed";
        socket.emit("error", message);
      }
    },
  );
});

// ============================================================================
// Start Server
// ============================================================================
httpServer.listen(PORT, () =>
  console.log(`Annotation server (HTTP + WebSockets) running on port ${PORT}`),
);
