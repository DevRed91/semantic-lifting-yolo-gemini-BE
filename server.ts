import express from "express";
import cors from "cors";
import { GoogleGenerativeAI, SchemaType, type Schema } from "@google/generative-ai";
import dotenv from "dotenv";

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
import fs from "fs";

dotenv.config();

const app = express();
const normalizeOrigin = (origin: string): string =>
  origin.trim().replace(/\/$/, "");
const envAllowedOrigins = (process.env.CORS_ALLOWED_ORIGINS || "")
  .split(",")
  .map((origin) => normalizeOrigin(origin))
  .filter(Boolean);
const defaultAllowedOrigins = [
  "https://auto-annotate.netlify.app",
];
const isProduction = process.env.NODE_ENV === "production";
const devOrigins = ["http://localhost:5173", "http://127.0.0.1:5173"];
const effectiveAllowedOrigins = Array.from(
  new Set(
    (isProduction
      ? [...envAllowedOrigins, ...defaultAllowedOrigins]
      : [...envAllowedOrigins, ...defaultAllowedOrigins, ...devOrigins]
    ).map((origin) => normalizeOrigin(origin)),
  ),
);
const isAllowedOrigin = (origin?: string): boolean => {
  if (!origin) return true;
  if (effectiveAllowedOrigins.length === 0) return true;
  const normalizedOrigin = normalizeOrigin(origin);
  return effectiveAllowedOrigins.includes(normalizedOrigin);
};
const resolveOriginHeader = (origin?: string): string => {
  if (!origin) return "*";
  return isAllowedOrigin(origin) ? origin : "null";
};
const resolveRequestedHeaders = (requestHeaders?: string): string => {
  const defaults = [
    "Content-Type",
    "Authorization",
    "bypass-tunnel-reminder",
    "ngrok-skip-browser-warning",
  ];
  if (!requestHeaders) return defaults.join(", ");
  const requested = requestHeaders
    .split(",")
    .map((header) => header.trim())
    .filter(Boolean);
  return Array.from(new Set([...defaults, ...requested])).join(", ");
};
const corsOptions: cors.CorsOptions = {
  origin: (origin, callback) => {
    return callback(null, isAllowedOrigin(origin));
  },
  methods: ["GET", "POST", "OPTIONS"],
  allowedHeaders: [
    "Content-Type",
    "Authorization",
    "bypass-tunnel-reminder",
    "ngrok-skip-browser-warning",
  ],
  optionsSuccessStatus: 204,
};

app.use(cors(corsOptions));
app.options(/.*/, cors(corsOptions));
app.use(
  (req: express.Request, res: express.Response, next: express.NextFunction) => {
    const origin = req.headers.origin;
    const requestedHeaders = req.headers["access-control-request-headers"];
    const requestedHeadersValue = Array.isArray(requestedHeaders)
      ? requestedHeaders.join(",")
      : requestedHeaders;

    res.header("Access-Control-Allow-Origin", resolveOriginHeader(origin));
    res.header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.header(
      "Access-Control-Allow-Headers",
      resolveRequestedHeaders(requestedHeadersValue),
    );
    res.header("Vary", "Origin");

    if (req.method === "OPTIONS") {
      return res.status(204).end();
    }

    return next();
  },
);
// Set one limit and stick to it
app.use(express.json({ limit: "20mb" }));

const MODEL_NAME = "gemini-3-pro-image"; // Use the 'models/' prefix

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

      // Logic to prepare the box for the prompt
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
      //     const prompt = `
      //     Analyze this image. Find ALL 'buildings' AND specifically identify 'landmark_buildings' (e.g., iconic hotels, historic sites, famous architectural structures like the Biltmore Hotel).

      //     For each object found, perform classification.
      //     Return ONLY raw JSON with an "objects" key containing an array of objects.

      //     Format:
      //     {
      //       "objects": [
      //         {"label": "Biltmore Hotel", "box": [ymin, xmin, ymax, xmax]},
      //         {"label": "commercial building", "box": [ymin, xmin, ymax, xmax]}
      //       ]
      //     }

      //     Rules:
      //     1. 'label' MUST be descriptive. Use specific names if recognized (e.g., 'Biltmore Hotel'), otherwise use generic categories like 'skyscraper' or 'residential_building'.
      //     2. If no structures found, return {"objects": []}.
      //     3. No markdown, no backticks, no explanatory text.
      //     4. Use normalized 0-1 coordinates.
      //     5. Focus on the region: ${JSON.stringify(box)}.
      //     6. CRITICAL: Do not label the sky, streets, vegetation, or general 'background' as a building. Only include the actual architectural structure.
      //     7. Prioritize identifying the Biltmore Hotel if present in the view.
      //     8. If a building is partially blocked, still include it and draw the box around the visible structure.
      // `;
      const prompt = `
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

      console.log("AI is focusing on box:", box);
      const base64Data = image.includes(",") ? image.split(",")[1] : image;
      fs.writeFileSync("debug_snapshot.jpg", Buffer.from(base64Data, "base64"));
      const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
      const model = genAI.getGenerativeModel({
        model: MODEL_NAME,
        generationConfig: {
          responseMimeType: "application/json",
          responseSchema: annotationSchema,
        },
      });

      const result = await model.generateContent([
        prompt,
        { inlineData: { data: base64Data, mimeType: "image/jpeg" } },
      ]);

      const text = result.response.text();
      console.log("Raw Response:", text);

      const parsed = JSON.parse(text);
      // Send the standardized format
      res.json(parsed);
    } catch (error: any) {
      console.error("SERVER ERROR:", error.message);
      res.status(500).json({ error: "Failed to annotate" });
    }
  },
);

app.listen(3000, () => console.log("Annotation server running on port 3000"));
