import "dotenv/config";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export const ROOT_DIR = path.resolve(__dirname, "..", "..");
export const CHUNKS_PATH = path.join(ROOT_DIR, "data", "processed", "chunks.json");
export const INDEX_PATH = path.join(ROOT_DIR, "data", "processed", "vector_index.json");

export const config = {
  openaiApiKey: process.env.OPENAI_API_KEY,
  embeddingModel: process.env.EMBEDDING_MODEL || "text-embedding-3-small",
  chatModel: process.env.CHAT_MODEL || "gpt-4o-mini",
  port: Number(process.env.PORT) || 3001,
  topK: Number(process.env.TOP_K) || 6,
};

if (!config.openaiApiKey) {
  throw new Error(
    "OPENAI_API_KEY is not set. Copy backend/.env.example to backend/.env and add your key."
  );
}
