import { GoogleGenerativeAI } from "@google/generative-ai";

type Event = {
  httpMethod: string;
  headers: Record<string, string | undefined>;
  body: string | null;
};

type Result = {
  statusCode: number;
  headers?: Record<string, string>;
  body: string;
};

const normalizeOrigin = (origin: string): string => origin.trim().replace(/\/$/, "");
const envAllowedOrigins = (process.env.CORS_ALLOWED_ORIGINS || "")
  .split(",")
  .map((origin) => normalizeOrigin(origin))
  .filter(Boolean);
const defaultAllowedOrigins = [
  "https://auto-annotate.netlify.app",
  "https://www.auto-annotate.netlify.app",
  "https://main--auto-annotate.netlify.app",
];
const allowedOrigins = Array.from(
  new Set([...envAllowedOrigins, ...defaultAllowedOrigins].map((origin) => normalizeOrigin(origin))),
);

const isOriginAllowed = (origin: string | undefined): boolean => {
  if (!origin) return true;
  if (allowedOrigins.length === 0) return true;
  return allowedOrigins.includes(normalizeOrigin(origin));
};

const getHeader = (headers: Record<string, string | undefined>, key: string): string | undefined => {
  const target = key.toLowerCase();
  for (const [name, value] of Object.entries(headers)) {
    if (name.toLowerCase() === target) return value;
  }
  return undefined;
};

const resolveRequestedHeaders = (requestHeaders: string | undefined): string => {
  const defaults = ["Content-Type", "Authorization", "bypass-tunnel-reminder", "ngrok-skip-browser-warning"];
  if (!requestHeaders) return defaults.join(", ");
  const requested = requestHeaders
    .split(",")
    .map((header) => header.trim())
    .filter(Boolean);
  return Array.from(new Set([...defaults, ...requested])).join(", ");
};

const corsHeaders = (origin: string | undefined, requestHeaders: string | undefined): Record<string, string> => {
  const resolvedOrigin = isOriginAllowed(origin) ? origin || "*" : "null";
  return {
    "Access-Control-Allow-Origin": resolvedOrigin,
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Access-Control-Allow-Headers": resolveRequestedHeaders(requestHeaders),
    "Content-Type": "application/json",
    Vary: "Origin",
  };
};

const extractJsonFromResponse = (text: string): unknown => {
  const jsonMatch = text.match(/(\[[\s\S]*\]|\{[\s\S]*\})/);
  if (!jsonMatch) throw new Error("No JSON found in model response");
  return JSON.parse(jsonMatch[0]);
};

export const handler = async (event: Event): Promise<Result> => {
  const origin = getHeader(event.headers, "origin");
  const requestHeaders = getHeader(event.headers, "access-control-request-headers");
  const headers = corsHeaders(origin, requestHeaders);

  if (event.httpMethod === "OPTIONS") {
    return { statusCode: 204, headers, body: "" };
  }

  if (!isOriginAllowed(origin)) {
    return {
      statusCode: 403,
      headers,
      body: JSON.stringify({ error: `CORS blocked for origin: ${origin}` }),
    };
  }

  if (event.httpMethod !== "POST") {
    return {
      statusCode: 405,
      headers,
      body: JSON.stringify({ error: "Method Not Allowed" }),
    };
  }

  if (!process.env.GEMINI_API_KEY) {
    return {
      statusCode: 500,
      headers,
      body: JSON.stringify({ error: "Missing GEMINI_API_KEY" }),
    };
  }

  try {
    const payload = event.body ? JSON.parse(event.body) : {};
    const { image, userBox, clickX, clickY } = payload as {
      image?: string;
      userBox?: number[];
      clickX?: number;
      clickY?: number;
    };

    if (!image || !image.startsWith("data:image/jpeg")) {
      return {
        statusCode: 400,
        headers,
        body: JSON.stringify({ error: "Invalid image format" }),
      };
    }

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

    const prompt = `
Analyze this image. Find ALL 'buildings' (houses, skyscrapers, commercial structures, etc.).

For each building found, perform classification.
Return ONLY raw JSON with an "objects" key containing an array of objects.

Format:
{
  "objects": [
    {"label": "house", "box": [ymin, xmin, ymax, xmax]},
    {"label": "skyscraper", "box": [ymin, xmin, ymax, xmax]}
  ]
}

Rules:
1. 'label' MUST be descriptive based on the structure (e.g., 'house', 'apartment', 'skyscraper', 'commercial_building').
2. If no structures found, return {"objects": []}.
3. No markdown, no backticks, no explanatory text.
4. Use normalized 0-1 coordinates.
5. Focus on the region: ${JSON.stringify(box)}.
6. CRITICAL: Do not label the sky, streets, vegetation, or general 'background' as a building. Only include the actual architectural structure.
7. If a building is partially blocked by a tree or pole, still include it and draw the box around the visible structure.
`;

    const base64Data = image.includes(",") ? image.split(",")[1] : image;
    const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
    const model = genAI.getGenerativeModel({ model: "gemini-3-pro-image" });

    const result = await model.generateContent([
      prompt,
      { inlineData: { data: base64Data, mimeType: "image/jpeg" } },
    ]);

    const parsed = extractJsonFromResponse(result.response.text());
    return { statusCode: 200, headers, body: JSON.stringify(parsed) };
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    return {
      statusCode: 500,
      headers,
      body: JSON.stringify({ error: "Failed to annotate", details: message }),
    };
  }
};
