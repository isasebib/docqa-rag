"""Prompt versions. The eval compares them on the same questions, so keep each one fixed
once results are recorded; add a new version instead of editing an old one.

Every template receives:
  {context}  - numbered source passages, e.g. "[1] (p. 12) ..."
  {question} - the user's question
"""

REFUSAL = "I don't know based on the provided document."

PROMPTS = {
    # v1: the minimal baseline most tutorials start with.
    "v1_basic": """Use the following context to answer the question.

Context:
{context}

Question: {question}
Answer:""",

    # v2: adds grounding rules, an explicit refusal, and citations by passage number.
    "v2_grounded": f"""You answer questions about the NIST AI Risk Management Framework (AI RMF 1.0).
Use ONLY the numbered passages below. Do not use outside knowledge.

Rules:
- If the passages do not contain the answer, reply exactly: "{REFUSAL}"
- Cite the passages you used with their numbers in square brackets, e.g. [1] or [2][3].
- Be concise: at most 4 sentences.

Passages:
{{context}}

Question: {{question}}
Answer:""",

    # v3: v2 plus a short "find the evidence first" step, to reduce answers that sound
    # right but are not supported by the passages.
    "v3_quote_first": f"""You answer questions about the NIST AI Risk Management Framework (AI RMF 1.0).
Use ONLY the numbered passages below. Do not use outside knowledge.

First, on a line starting with "Evidence:", copy the shortest exact phrase from the passages
that answers the question, with its passage number. If no passage answers it, write "Evidence: none".
Then, on a line starting with "Answer:", answer in at most 4 sentences, citing passage numbers
like [1]. If the evidence is "none", the answer must be exactly: "{REFUSAL}"

Passages:
{{context}}

Question: {{question}}
""",
}

DEFAULT_PROMPT = "v2_grounded"
