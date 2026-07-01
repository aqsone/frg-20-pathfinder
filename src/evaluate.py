import csv
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai

from src.judge import judge_answer
from src.rag.rag_query import generate_answer
from src.rag.retriever import embed_query, load_corpus, retrieve_base, retrieve_mmr

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
EVAL_DIR = PROCESSED_DIR / "eval"
QA_PATH = RAW_DIR / "Q_A.csv"

load_dotenv()

_URL_TO_SOURCE = {
    "f-cp090601":       "f-cp090601_text.txt",
    "BEA2017-0568":     "BEA2017-0568_text.txt",
    "cs-e-amendment-7": "CS-E__Amendment_7_0_text.txt",
    "00_afh_full":      "00_afh_full_text.txt",
}


def _source_from_url(pdf_url: str) -> str | None:
    for key, source in _URL_TO_SOURCE.items():
        if key in pdf_url:
            return source
    return None


def load_questions(qa_path: Path = QA_PATH) -> list[dict]:
    with qa_path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _evaluate_question(
    question: str,
    reference: str,
    source: str | None,
    client: genai.Client,
    corpus: list[dict],
) -> dict:
    sources = [source] if source else None
    query_vec = embed_query(question, client)

    results = {}
    for strategy, retrieve_fn in [("base", retrieve_base), ("mmr", retrieve_mmr)]:
        chunks = retrieve_fn(question, client, corpus, query_embedding=query_vec, sources=sources)
        answer = generate_answer(question, chunks, client)
        verdict = judge_answer(question, reference, answer, client)
        results[f"rag_{strategy}"] = {
            "retrieved_chunks": [c["chunk_id"] for c in chunks],
            "generated_answer": answer,
            "score": verdict["score"],
            "justification": verdict["justification"],
        }

    # À terme : résultats PageIndex à ajouter ici
    # results["page_index"] = run_page_index(question, source, client)

    return results


def run_evaluation(
    client: genai.Client,
    corpus: list[dict],
    questions: list[dict],
    output_path: Path,
) -> list[dict]:
    all_results = []
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as out:
        for row in questions:
            n = row["N°"]
            hop = row["Hop"]
            question = row["Q"]
            reference = row["A"]
            source = _source_from_url(row["pdf"])

            print(f"\n[Q{n} | Hop={hop} | {source}]")
            print(f"  {question[:80]}...")

            try:
                strategies = _evaluate_question(question, reference, source, client, corpus)
            except Exception as e:
                print(f"  [ERREUR] {e}")
                strategies = {}

            for key, data in strategies.items():
                print(f"  {key} → score {data['score']}/5 — {data['justification'][:70]}")

            result = {
                "id": n,
                "hop": hop,
                "question": question,
                "source": source,
                "reference_answer": reference,
                **strategies,
            }
            all_results.append(result)
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()

    return all_results


def print_summary(results: list[dict]) -> None:
    strategies = [k for k in results[0] if k.startswith("rag_") or k == "page_index"]
    print("\n=== Résumé ===")

    for strategy in strategies:
        scores = [r[strategy]["score"] for r in results if strategy in r]
        print(f"\n{strategy} — score moyen : {sum(scores)/len(scores):.2f}/5 ({len(scores)} questions)")

        for hop in sorted({r["hop"] for r in results}):
            subset = [r[strategy]["score"] for r in results if r["hop"] == hop and strategy in r]
            label = "multi-hop" if hop == "1" else "single-hop"
            print(f"  Hop={hop} ({label}) : {sum(subset)/len(subset):.2f}/5 ({len(subset)} questions)")


def main() -> None:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Erreur : GEMINI_API_KEY non configurée.")
        return

    client = genai.Client()

    print("Chargement du corpus...")
    corpus = load_corpus()
    if not corpus:
        print("Aucun embedding trouvé. Lancez d'abord `python -m src.rag.embedder`.")
        return
    print(f"Corpus chargé : {len(corpus)} chunks.")

    questions = load_questions()
    print(f"{len(questions)} questions chargées depuis {QA_PATH.name}.")

    output_path = EVAL_DIR / "eval_results.jsonl"
    results = run_evaluation(client, corpus, questions, output_path)
    print(f"\nRésultats sauvegardés : {output_path}")
    print_summary(results)


if __name__ == "__main__":
    main()