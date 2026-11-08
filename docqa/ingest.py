"""Build the search index: PDF -> clean text -> chunks -> embeddings -> files on disk.

Run:  python -m docqa.ingest                 (default: 200-word chunks, 40-word overlap)
      python -m docqa.ingest --chunk-size 400 --overlap 80
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
from pypdf import PdfReader

from docqa.llm import EMBED_MODEL, embed

PDF_PATH = Path("data/raw/nist-ai-rmf-1.0.pdf")
INDEX_ROOT = Path("data/index")

# nomic-embed-text was trained with task prefixes: documents and queries get
# different prefixes so the model can place them well relative to each other.
DOC_PREFIX = "search_document: "

HEADER_RE = re.compile(r"^NIST AI 100-1\s*(AI RMF 1\.0)?\s*\n")  # repeated page header
FOOTER_RE = re.compile(r"\n\s*Page (\d+)\s*$")  # printed page number at the bottom


def load_pages(pdf_path: Path) -> list[dict]:
    """One dict per PDF page: the PDF page index and the raw extracted text."""
    reader = PdfReader(pdf_path)
    return [
        {"pdf_page": i, "text": page.extract_text() or ""}
        for i, page in enumerate(reader.pages, start=1)
    ]


def clean_page(text: str) -> tuple[str, str | None]:
    """Strip the header/footer and repair line breaks.

    Returns (clean_text, printed_page_label). The label is the "Page N" a reader
    sees in the PDF, which differs from the PDF index (PDF page 13 = "Page 8").
    """
    text = HEADER_RE.sub("", text)
    label = None
    footer = FOOTER_RE.search(text)
    if footer:
        label = footer.group(1)
        text = text[: footer.start()]

    # Word split across lines ("envi-\nronment") -> join it. But keep real hyphens
    # like "high-\nor low-probability": skip the join when the next word is or/and/to.
    text = re.sub(r"(\w)-\n(?!(?:or|and|to)\b)([a-z])", r"\1\2", text)
    text = text.replace("\n", " ")  # remaining line breaks are just layout
    text = re.sub(r"\s+", " ", text).strip()
    return text, label


def chunk_words(text: str, size: int, overlap: int) -> list[str]:
    """Split text into windows of `size` words; consecutive windows share `overlap` words.

    The overlap keeps a sentence that straddles a boundary intact in at least one chunk.
    """
    words = text.split()
    if not words:
        return []
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start : start + size]))
        if start + size >= len(words):
            break
    return chunks


def build_chunks(pages: list[dict], size: int, overlap: int, min_words: int = 25) -> list[dict]:
    """Chunk page by page so every chunk maps to exactly one page (simple, exact citations)."""
    chunks = []
    for page in pages:
        text, label = clean_page(page["text"])
        if len(text.split()) < min_words:  # cover page, blank pages, section dividers
            continue
        for part in chunk_words(text, size, overlap):
            chunks.append(
                {
                    "id": len(chunks),
                    "pdf_page": page["pdf_page"],
                    "page_label": label,
                    "text": part,
                }
            )
    return chunks


def embed_chunks(chunks: list[dict], batch_size: int = 32) -> np.ndarray:
    """Embed all chunks and L2-normalise them, so cosine similarity = a dot product."""
    vectors = []
    for i in range(0, len(chunks), batch_size):
        batch = [DOC_PREFIX + c["text"] for c in chunks[i : i + batch_size]]
        vectors.extend(embed(batch))
        print(f"  embedded {min(i + batch_size, len(chunks))}/{len(chunks)} chunks")
    matrix = np.array(vectors, dtype=np.float32)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def index_dir(size: int, overlap: int) -> Path:
    """Each chunking config gets its own folder, so evals can compare them side by side."""
    return INDEX_ROOT / f"size{size}_overlap{overlap}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the DocQA search index.")
    parser.add_argument("--chunk-size", type=int, default=200, help="words per chunk")
    parser.add_argument("--overlap", type=int, default=40, help="words shared by neighbours")
    args = parser.parse_args()
    if not 0 <= args.overlap < args.chunk_size:
        parser.error("overlap must be >= 0 and smaller than chunk size")

    pages = load_pages(PDF_PATH)
    chunks = build_chunks(pages, args.chunk_size, args.overlap)
    print(f"{len(pages)} pages -> {len(chunks)} chunks ({args.chunk_size} words, {args.overlap} overlap)")

    vectors = embed_chunks(chunks)

    out = index_dir(args.chunk_size, args.overlap)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "embeddings.npy", vectors)
    with open(out / "chunks.jsonl", "w") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    (out / "meta.json").write_text(
        json.dumps(
            {"embed_model": EMBED_MODEL, "chunk_size": args.chunk_size, "overlap": args.overlap,
             "num_chunks": len(chunks), "dim": int(vectors.shape[1])},
            indent=2,
        )
    )
    print(f"Saved index to {out}/ (vectors: {vectors.shape})")


if __name__ == "__main__":
    main()
