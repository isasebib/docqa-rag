"""The RAG step: retrieve passages, put them in a prompt, ask the model, return answer + sources.

Run:  python -m docqa.answer "What are the four functions of the AI RMF core?"
"""

import re

from docqa.llm import generate
from docqa.prompts import DEFAULT_PROMPT, PROMPTS, REFUSAL
from docqa.retrieve import cite, load_index


def format_context(chunks: list[dict]) -> str:
    """Number the passages so the model can cite them as [1], [2], ..."""
    return "\n\n".join(f"[{i}] ({cite(c)}) {c['text']}" for i, c in enumerate(chunks, start=1))


def clean_answer(raw: str) -> str:
    """v3 replies with 'Evidence: ... Answer: ...'; keep only the answer part for the user."""
    match = re.search(r"Answer:\s*(.*)", raw, flags=re.S)
    return match.group(1).strip() if match else raw.strip()


def answer(
    question: str,
    k: int = 4,
    prompt_name: str = DEFAULT_PROMPT,
    chunk_size: int = 200,
    overlap: int = 40,
) -> dict:
    """Answer a question from the document. Returns the answer, its sources and the raw output."""
    chunks = load_index(chunk_size, overlap).search(question, k=k)
    prompt = PROMPTS[prompt_name].format(context=format_context(chunks), question=question)
    raw = generate(prompt)
    text = clean_answer(raw)

    # Only report the passages the model actually cited, so sources match the answer.
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", text)}
    sources = [
        {"n": i, "page": cite(c), "score": round(c["score"], 3), "text": c["text"]}
        for i, c in enumerate(chunks, start=1)
        if i in cited
    ]
    return {
        "question": question,
        "answer": text,
        "refused": text.strip().strip('"') == REFUSAL,
        "sources": sources,
        "retrieved_pages": [cite(c) for c in chunks],
        "prompt": prompt_name,
        "raw": raw,
    }


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "What are the four functions of the AI RMF core?"
    result = answer(q)
    print(f"Q: {q}\n\nA: {result['answer']}\n")
    for s in result["sources"]:
        print(f"  [{s['n']}] {s['page']}")
