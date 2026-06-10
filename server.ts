import express from "express";
import cors from "cors";
import { GoogleGenerativeAI } from "@google/generative-ai";
import dotenv from "dotenv";

dotenv.config();

const app = express();
app.use(cors());
// Set one limit and stick to it
app.use(express.json({ limit: "20mb" }));

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY!);
const MODEL_NAME = "models/gemini-3.5-flash"; // Use the 'models/' prefix

app.post("/api/annotate", async (req: express.Request, res: express.Response) => {
  try {
    const { image, userBox, clickX, clickY } = req.body;

    if (!image || !image.startsWith('data:image/jpeg')) {
      return res.status(400).json({ error: "Invalid image format" });
    }

    // Logic to prepare the box for the prompt
    let box = userBox;
    if (!box && clickX !== undefined && clickY !== undefined) {
      const size = 0.1;
      box = [
        Math.max(0, clickY - size), Math.max(0, clickX - size),
        Math.min(1, clickY + size), Math.min(1, clickX + size)
      ];
    }

    const prompt = `
  Analyze the entire image. 
  List every chair and every photo frame you can see.
  Return valid JSON: {"chairs": [...], "frames": [...]}.
  If you find nothing, look harder.
`;

    const base64Data = image.includes(",") ? image.split(",")[1] : image;
    const model = genAI.getGenerativeModel({ model: MODEL_NAME });

    const result = await model.generateContent([
      prompt,
      { inlineData: { data: base64Data, mimeType: "image/jpeg" } },
    ]);

    const text = result.response.text();
    console.log("Raw Response:", text);

    // Robust Regex: Matches an array [...] OR an object {...}
    const jsonMatch = text.match(/(\[[\s\S]*\]|\{[\s\S]*\})/);
    if (!jsonMatch) throw new Error("No JSON found in response");

    const parsed = JSON.parse(jsonMatch[0]);

    // Ensure frontend always receives { chairs: [...] }
    const responseData = Array.isArray(parsed)
      ? { chairs: parsed }
      : (parsed.chairs ? parsed : { chairs: [parsed] });

    res.json(responseData);

  } catch (error: any) {
    console.error("SERVER ERROR:", error.message);
    res.status(500).json({ error: "Failed to annotate" });
  }
});

app.listen(3000, () => console.log("Annotation server running on port 3000"));