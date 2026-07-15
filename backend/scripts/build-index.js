import fs from "node:fs";
import { CHUNKS_PATH, INDEX_PATH, config } from "../src/config.js";
import { openai } from "../src/openaiClient.js";

const BATCH_SIZE = 100;

async function embedBatch(texts) {
  const res = await openai.embeddings.create({
    model: config.embeddingModel,
    input: texts,
  });
  return res.data.map((d) => d.embedding);
}

async function main() {
  if (!fs.existsSync(CHUNKS_PATH)) {
    throw new Error(`${CHUNKS_PATH} not found. Run "python scripts/ingest.py" first.`);
  }

  const chunks = JSON.parse(fs.readFileSync(CHUNKS_PATH, "utf-8"));
  console.log(`Loaded ${chunks.length} chunks from ${CHUNKS_PATH}`);

  const indexed = [];
  for (let i = 0; i < chunks.length; i += BATCH_SIZE) {
    const batch = chunks.slice(i, i + BATCH_SIZE);
    const embeddings = await embedBatch(batch.map((c) => c.text));
    batch.forEach((chunk, j) => indexed.push({ ...chunk, embedding: embeddings[j] }));
    console.log(`Embedded ${Math.min(i + BATCH_SIZE, chunks.length)}/${chunks.length}`);
  }

  fs.writeFileSync(INDEX_PATH, JSON.stringify(indexed), "utf-8");
  console.log(`Wrote ${indexed.length} vectors to ${INDEX_PATH}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
