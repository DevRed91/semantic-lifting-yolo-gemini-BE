import dotenv from "dotenv";
dotenv.config();

const API_KEY = process.env.GEMINI_API_KEY;
const MODEL = "gemini-3.5-flash";
const URL = `https://generativelanguage.googleapis.com/v1beta/models/${MODEL}:generateContent?key=${API_KEY}`;

async function debugRequest() {
  console.log("Testing URL:", URL);
  try {
    const response = await fetch(URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        contents: [{ parts: [{ text: "Hello" }] }],
      }),
    });

    const data = await response.json();

    if (response.ok) {
      console.log("SUCCESS! API is working.");
    } else {
      console.error("FAILURE! Server returned status:", response.status);
      console.error("RESPONSE BODY:", JSON.stringify(data, null, 2));
    }
  } catch (err) {
    console.error("Fetch failed:", err);
  }
}

debugRequest();
