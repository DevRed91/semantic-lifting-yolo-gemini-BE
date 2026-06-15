PHASE 0: REFRACTORING & SOCKET.IO SETUP      
- Add Socket.io server to backend server.ts  
- Replace sync file writing with async paths 
- Establish client-side Socket.io listener 

import { createServer } from "http";
import { Server } from "socket.io";

const httpServer = createServer(app);
const io = new Server(httpServer, {
  cors: { origin: "*" }
});

io.on("connection", (socket) => {
  socket.on("request_annotation", async ({ image, box, clickX, clickY }) => {
    try {
      // 1. Instant YOLO response (The Glow)
      const yolo = await callYoloService(image); 
      socket.emit("mask_ready", { mask: yolo.mask, label: yolo.label });

      // 2. Later Gemini response (The Museum Description)
      const gemini = await callGemini(image, yolo.label);
      socket.emit("description_ready", { description: gemini.description });
      
    } catch (err) {
      socket.emit("error", "Vision pipeline failed");
    }
  });
});

// Replace app.listen with httpServer.listen
httpServer.listen(3000, () => console.log("Server with WebSockets on 3000"));