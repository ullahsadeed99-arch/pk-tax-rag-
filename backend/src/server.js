import cors from "cors";
import express from "express";
import { config } from "./config.js";
import { openai } from "./openaiClient.js";
import { buildMessages } from "./prompt.js";
import { search } from "./retrieval.js";

const app = express();
app.use(cors());
app.use(express.json());

app.get("/api/health", (req, res) => {
  res.json({ status: "ok" });
});

app.post("/api/chat", async (req, res) => {
  const { question } = req.body;
  if (!question || typeof question !== "string" || !question.trim()) {
    return res.status(400).json({ error: "question is required" });
  }

  try {
    const embedRes = await openai.embeddings.create({
      model: config.embeddingModel,
      input: question,
    });
    const queryEmbedding = embedRes.data[0].embedding;

    const retrieved = search(queryEmbedding, config.topK);

    const completion = await openai.chat.completions.create({
      model: config.chatModel,
      messages: buildMessages(question, retrieved),
      temperature: 0,
    });

    res.json({
      answer: completion.choices[0].message.content,
      sources: retrieved.map(({ chunk, score }) => ({
        doc_name: chunk.doc_name,
        section: chunk.section,
        section_title: chunk.section_title,
        page_number: chunk.page_number,
        is_table: chunk.is_table,
        score,
      })),
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message || "internal error" });
  }
});

app.listen(config.port, () => {
  console.log(`pk-tax-rag backend listening on http://localhost:${config.port}`);
});
