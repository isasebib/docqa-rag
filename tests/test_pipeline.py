"""Fast unit tests for the parts that don't need Ollama. Run: pytest -q"""

from docqa.answer import clean_answer, format_context
from docqa.ingest import chunk_words, clean_page
from docqa.retrieve import cite
from evals.run_eval import is_correct, page_labels


def test_clean_page_removes_header_and_footer_and_keeps_label():
    raw = "NIST AI 100-1 AI RMF 1.0\nExecutive Summary\nSome text here.\nPage 1"
    text, label = clean_page(raw)
    assert text == "Executive Summary Some text here."
    assert label == "1"


def test_clean_page_joins_split_words_but_keeps_real_hyphens():
    text, _ = clean_page("to the envi-\nronment and high-\nor low-probability risks")
    assert "environment" in text
    assert "high- or low-probability" in text


def test_chunk_words_overlap_and_coverage():
    words = [f"w{i}" for i in range(10)]
    chunks = chunk_words(" ".join(words), size=4, overlap=1)
    assert chunks[0] == "w0 w1 w2 w3"
    assert chunks[1].startswith("w3")  # one shared word with the previous chunk
    assert chunks[-1].endswith("w9")  # nothing at the end is dropped


def test_chunk_words_empty_text():
    assert chunk_words("", size=4, overlap=1) == []


def test_cite_prefers_printed_page():
    assert cite({"page_label": "12", "pdf_page": 17}) == "p. 12"
    assert cite({"page_label": None, "pdf_page": 3}) == "PDF p. 3"


def test_format_context_numbers_passages():
    ctx = format_context([{"page_label": "8", "pdf_page": 13, "text": "Stop the system."}])
    assert ctx == "[1] (p. 8) Stop the system."


def test_clean_answer_strips_evidence_line():
    assert clean_answer('Evidence: "cease" [1]\nAnswer: It should stop. [1]') == "It should stop. [1]"
    assert clean_answer("Plain answer.") == "Plain answer."


def test_is_correct_requires_every_group():
    groups = [["govern"], ["map"]]
    assert is_correct("GOVERN and MAP", groups)
    assert not is_correct("Only GOVERN", groups)


def test_page_labels_ignores_pdf_pages():
    assert page_labels(["p. 12", "PDF p. 5"]) == {"12"}
