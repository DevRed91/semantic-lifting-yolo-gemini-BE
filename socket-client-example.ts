import { io } from "socket.io-client";

const socket = io("http://localhost:3000", {
  transports: ["websocket"],
});

socket.on("connect", () => {
  console.log("Connected:", socket.id);
});

socket.on("mask_ready", (payload: { mask: unknown; label: string }) => {
  console.log("YOLO result:", payload.label);
  // Render payload.mask immediately in the UI.
});

socket.on("description_ready", (payload: { description: string }) => {
  console.log("Museum description:", payload.description);
  // Update the text panel with payload.description.
});

socket.on("error", (message: string) => {
  console.error("Vision pipeline error:", message);
});

export function requestAnnotation(input: {
  image: string;
  box?: number[];
  clickX?: number;
  clickY?: number;
}) {
  socket.emit("request_annotation", input);
}
