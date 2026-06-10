import express from "express";
import cors from "cors";
import { GoogleGenerativeAI } from "@google/generative-ai";
import dotenv from "dotenv";
import fs from 'fs';

dotenv.config();

const app = express();
app.use(cors());
// Set one limit and stick to it
app.use(express.json({ limit: "20mb" }));

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY!);
const MODEL_NAME = "gemini-3-pro-image"; // Use the 'models/' prefix

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
    console.log("AI is focusing on box:", box);
    const base64Data = image.includes(",") ? image.split(",")[1] : image;
    fs.writeFileSync('debug_snapshot.jpg', Buffer.from(base64Data, 'base64'));
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
    console.log("Parsed Response:", parsed);

    // Iterate through the keys in the parsed object (sofa, chairs, etc)
    // Object.keys(parsed).forEach(key => {
    //   const items = parsed[key];
    //   if (Array.isArray(items)) {
    //     items.forEach(item => {
    //       normalizedObjects.push({
    //         label: key, // The key name becomes the label
    //         box: item.box
    //       });
    //     });
    //   }
    // });

    // Send the standardized format
    res.json(parsed);

  } catch (error: any) {
    console.error("SERVER ERROR:", error.message);
    res.status(500).json({ error: "Failed to annotate" });
  }
});

app.listen(3000, () => console.log("Annotation server running on port 3000"));