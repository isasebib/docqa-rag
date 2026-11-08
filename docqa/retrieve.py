"""Vector search over the saved index.

All vectors are L2-normalised, so cosine similarity is just a dot product:
one matrix-vector multiply scores every chunk at once. For ~100-1,000 chunks
this brute-force search takes well under a millisecond; a vector database
(FAISS, Chroma) only starts to pay off at much larger sizes.
"""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from docqa.ingest import index_dir
from docqa.llm import embed

QUERY_PREFIX = "search_query: "  # nomic-embed-text's prefix for queries (see DOC_PREFIX)


class Index:
    def __init__(self, folder: Path):
        if not (folder / "embeddings.npy").exists():
            raise FileNotFoundError(f"No index in {folder}. Run: python -m docqa.ingest")
        self.vectors = np.load(folder / "embeddings.npy")
        with open(folder / "chunks.jsonl") as f:
            self.chunks = [json.loads(line) for line in f]

    def search(self, question: str, k: int = 4) -> list[dict]:
        """Return the k chunks most similar to the question, best first, with a score."""
        query = np.array(embed([QUERY_PREFIX + question])[0], dtype=np.float32)
        query /= np.linalg.norm(query)
        scores = self.vectors @ query  # cosine similarity with every chunk
        top = np.argsort(scores)[::-1][:k]
        return [{**self.chunks[i], "score": float(scores[i])} for i in top]


@lru_cache(maxsize=8)
def load_index(chunk_size: int = 200, overlap: int = 40) -> Index:
    """Load an index once and reuse it (the API calls this on every request)."""
    return Index(index_dir(chunk_size, overlap))


def cite(chunk: dict) -> str:
    """Human-readable source: the printed page if there is one, else the PDF page."""
    if chunk["page_label"]:
        return f"p. {chunk['page_label']}"
    return f"PDF p. {chunk['pdf_page']}"


if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]) or "What are the characteristics of trustworthy AI?"
    print(f"Q: {question}\n")
    for hit in load_index().search(question):
        print(f"[{hit['score']:.3f}] {cite(hit)}: {hit['text'][:160]}...")
