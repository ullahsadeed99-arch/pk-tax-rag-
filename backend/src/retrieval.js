import fs from "node:fs";
import { INDEX_PATH } from "./config.js";

let index = null;

function loadIndex() {
  if (!index) {
    if (!fs.existsSync(INDEX_PATH)) {
      throw new Error(
        `Vector index not found at ${INDEX_PATH}. Run "npm run build-index" first.`
      );
    }
    index = JSON.parse(fs.readFileSync(INDEX_PATH, "utf-8"));
    console.log(`Loaded vector index: ${index.length} chunks`);
  }
  return index;
}

function cosineSimilarity(a, b) {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

export function search(queryEmbedding, topK) {
  const chunks = loadIndex();
  const scored = chunks.map((chunk) => ({
    chunk,
    score: cosineSimilarity(queryEmbedding, chunk.embedding),
  }));
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, topK);
}
