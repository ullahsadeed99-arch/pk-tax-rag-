function describeChunk(chunk) {
  const sectionLabel = chunk.section ? `Section ${chunk.section}` : "Unnumbered text";
  const title = chunk.section_title ? ` - ${chunk.section_title}` : "";
  const kind = chunk.is_table ? " [table]" : "";
  return `${chunk.doc_name}, ${sectionLabel}${title}${kind} (p.${chunk.page_number})`;
}

export function buildMessages(question, retrieved) {
  const context = retrieved
    .map(({ chunk }, i) => `[${i + 1}] ${describeChunk(chunk)}\n${chunk.text}`)
    .join("\n\n---\n\n");

  const system = [
    "You are a Pakistani tax law assistant.",
    "Answer questions using ONLY the provided excerpts from the Income Tax Ordinance 2001,",
    "Income Tax Rules 2002, Sales Tax Act 1990, and Sales Tax Rules 2006.",
    "Cite the source for every claim using the matching [n] marker from the excerpts below.",
    "If the excerpts don't contain enough information to answer, say so plainly instead of guessing.",
  ].join(" ");

  const user = `Context excerpts:\n\n${context}\n\nQuestion: ${question}`;

  return [
    { role: "system", content: system },
    { role: "user", content: user },
  ];
}
