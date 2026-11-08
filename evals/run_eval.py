"""Score DocQA configurations on the eval set and write a results table.

Run:  python -m evals.run_eval                       (all configs below, ~15 min on an M2 Pro)
      python -m evals.run_eval --retrieval-only      (no LLM calls, a few seconds)

Scoring is deliberately simple and deterministic (no LLM judge), so results are
repeatable and every point can be checked by hand:
- retrieval hit:   at least one expected page is among the k retrieved chunks
- correct:         the answer contains a phrase from EVERY `must_include` group
- refusal correct: unanswerable questions get the refusal sentence
- false refusal:   an answerable question was refused
Keyword matching can miss a correct paraphrase, so failures are saved for manual review.
"""

import argparse
import json
import time
from pathlib import Path

from docqa.answer import answer
from docqa.retrieve import cite, load_index

EVAL_FILE = Path("evals/questions.jsonl")
OUT_DIR = Path("evals/results")

# (chunk_size, overlap, prompt) combinations to compare.
CONFIGS = [
    (200, 40, "v1_basic"),
    (200, 40, "v2_grounded"),
    (200, 40, "v3_quote_first"),
    (400, 80, "v2_grounded"),
]
K = 4


def load_questions() -> list[dict]:
    with open(EVAL_FILE) as f:
        return [json.loads(line) for line in f]


def page_labels(pages: list[str]) -> set[str]:
    """'p. 12' -> '12' so retrieved pages can be compared with expected_pages."""
    return {p.removeprefix("p. ") for p in pages if p.startswith("p. ")}


def is_correct(text: str, must_include: list[list[str]]) -> bool:
    low = text.lower()
    return all(any(phrase in low for phrase in group) for group in must_include)


def eval_retrieval(questions: list[dict], size: int, overlap: int) -> float:
    index = load_index(size, overlap)
    answerable = [q for q in questions if q["type"] == "answerable"]
    hits = sum(
        bool(page_labels([cite(c) for c in index.search(q["question"], k=K)]) & set(q["expected_pages"]))
        for q in answerable
    )
    return hits / len(answerable)


def eval_config(questions: list[dict], size: int, overlap: int, prompt: str) -> dict:
    rows = []
    started = time.time()
    for q in questions:
        result = answer(q["question"], k=K, prompt_name=prompt, chunk_size=size, overlap=overlap)
        row = {"id": q["id"], "type": q["type"], "question": q["question"], "answer": result["answer"],
               "refused": result["refused"], "retrieved_pages": result["retrieved_pages"],
               "cited": bool(result["sources"])}
        if q["type"] == "answerable":
            row["retrieval_hit"] = bool(page_labels(result["retrieved_pages"]) & set(q["expected_pages"]))
            row["correct"] = (not result["refused"]) and is_correct(result["answer"], q["must_include"])
        else:
            row["correct"] = result["refused"]
        rows.append(row)
        print(f"  {q['id']} {'OK ' if row['correct'] else 'MISS'} {result['answer'][:70]!r}")

    ans = [r for r in rows if r["type"] == "answerable"]
    una = [r for r in rows if r["type"] == "unanswerable"]
    summary = {
        "config": f"chunk {size}/{overlap}, {prompt}",
        "retrieval_hit": sum(r["retrieval_hit"] for r in ans) / len(ans),
        "correct": sum(r["correct"] for r in ans),
        "answerable": len(ans),
        "refusals_correct": sum(r["correct"] for r in una),
        "unanswerable": len(una),
        "false_refusals": sum(r["refused"] for r in ans),
        "cited": sum(r["cited"] for r in ans),
        "seconds_per_q": round((time.time() - started) / len(rows), 1),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    name = f"size{size}_{prompt}"
    with open(OUT_DIR / f"{name}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return summary


def write_table(summaries: list[dict]) -> str:
    lines = [
        "| Config | Retrieval hit@4 | Correct answers | Correct refusals | False refusals | Answers with citation | Sec/question |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for s in summaries:
        lines.append(
            f"| {s['config']} | {s['retrieval_hit']:.0%} | {s['correct']}/{s['answerable']} | "
            f"{s['refusals_correct']}/{s['unanswerable']} | {s['false_refusals']} | "
            f"{s['cited']}/{s['answerable']} | {s['seconds_per_q']} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-only", action="store_true")
    args = parser.parse_args()
    questions = load_questions()

    if args.retrieval_only:
        for size, overlap in sorted({(s, o) for s, o, _ in CONFIGS}):
            print(f"chunk {size}/{overlap}: retrieval hit@{K} = {eval_retrieval(questions, size, overlap):.0%}")
        return

    summaries = []
    for size, overlap, prompt in CONFIGS:
        print(f"\n== chunk {size}/{overlap}, {prompt}")
        summaries.append(eval_config(questions, size, overlap, prompt))
    table = write_table(summaries)
    (OUT_DIR / "summary.md").write_text(table + "\n")
    (OUT_DIR / "summary.json").write_text(json.dumps(summaries, indent=2))
    print("\n" + table)


if __name__ == "__main__":
    main()
