import csv
import json
import os
import re
from difflib import SequenceMatcher
from pathlib import Path

from dotenv import load_dotenv
from google import genai

from src.rag.retriever import embed_query, load_corpus, retrieve_base, retrieve_mmr

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
EVAL_DIR = PROCESSED_DIR / "eval"
QA_PATH = RAW_DIR / "Q_A.csv"

RELEVANCE_THRESHOLD = 0.3

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


def _parse_ground_truth(gt_str: str) -> list[str]:
    """Split ground truth on ||| to support multi-hop questions."""
    if not gt_str or not gt_str.strip():
        return []
    return [p.strip() for p in gt_str.split("|||") if p.strip()]


def _normalize(text: str) -> str:
    # Collapse Unicode dash/hyphen variants to ASCII hyphen
    for ch in "‐‑‒–—―−":
        text = text.replace(ch, "-")
    # Collapse Unicode space variants to ASCII space
    for ch in "      　":
        text = text.replace(ch, " ")
    # Normalize quotes
    for ch in "‘’ʼ":
        text = text.replace(ch, "'")
    for ch in "“”":
        text = text.replace(ch, '"')
    return re.sub(r"\s+", " ", text.lower()).strip()


def _gt_coverage(gt_passage: str, item_text: str) -> float:
    """Fraction of gt characters covered by matching blocks in item.

    Uses SequenceMatcher so the metric is fully normalized by gt length:
    a threshold of 0.3 always means '30% of the gt text appears in the item',
    regardless of how long gt or item are.
    """
    gt_norm = _normalize(gt_passage)
    item_norm = _normalize(item_text)
    if not gt_norm:
        return 0.0
    matcher = SequenceMatcher(None, gt_norm, item_norm, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    return matched / len(gt_norm)


def _is_relevant(gt_passage: str, item_text: str) -> bool:
    return _gt_coverage(gt_passage, item_text) >= RELEVANCE_THRESHOLD


def compute_metrics(gt_passages: list[str], retrieved_items: list[dict]) -> dict:
    """Compute Precision@K and Recall for one retrieval result.

    Precision@K — fraction of K retrieved items relevant to any gt passage.
    Recall      — fraction of gt passages covered by at least one retrieved item.
    """
    k = len(retrieved_items)

    precision = (
        sum(
            1
            for item in retrieved_items
            if any(_is_relevant(gt, item["text"]) for gt in gt_passages)
        )
        / k
        if k > 0
        else 0.0
    )

    recall = (
        sum(
            1
            for gt in gt_passages
            if any(_is_relevant(gt, item["text"]) for item in retrieved_items)
        )
        / len(gt_passages)
        if gt_passages
        else 0.0
    )

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
    }


def _evaluate_question(
    question: str,
    gt_passages: list[str],
    source: str | None,
    client: genai.Client,
    corpus: list[dict],
) -> dict:
    sources = [source] if source else None
    query_vec = embed_query(question, client)
    results = {}

    for strategy, retrieve_fn in [("rag_base", retrieve_base), ("rag_mmr", retrieve_mmr)]:
        chunks = retrieve_fn(question, client, corpus, query_embedding=query_vec, sources=sources)
        metrics = compute_metrics(gt_passages, chunks)
        results[strategy] = {
            "retrieved_ids": [c["chunk_id"] for c in chunks],
            **metrics,
        }

    # PageIndex will be added here once implemented:
    # pages = retrieve_pages(question, client, page_corpus, query_embedding=query_vec, sources=sources)
    # results["page_index"] = {"retrieved_ids": [p["page_id"] for p in pages], **compute_metrics(gt_passages, pages)}

    return results


def load_questions(qa_path: Path = QA_PATH) -> list[dict]:
    with qa_path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f, restval=""))


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
            source = _source_from_url(row["pdf"])
            gt_passages = _parse_ground_truth(row.get("ground_truth", ""))

            if not gt_passages:
                print(f"[Q{n}] Pas de ground_truth, ignoré.")
                continue

            print(f"\n[Q{n} | Hop={hop} | {source}]")
            print(f"  {question[:80]}...")
            print(f"  Ground truth : {len(gt_passages)} passage(s)")

            try:
                strategies = _evaluate_question(question, gt_passages, source, client, corpus)
            except Exception as e:
                print(f"  [ERREUR] {e}")
                continue

            for key, data in strategies.items():
                print(f"  {key:<12} → P={data['precision']:.2f}  R={data['recall']:.2f}")

            result = {
                "id": n,
                "hop": hop,
                "question": question,
                "source": source,
                "n_gt_passages": len(gt_passages),
                **strategies,
            }
            all_results.append(result)
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()

    return all_results


def print_summary(results: list[dict]) -> None:
    if not results:
        print("Aucun résultat.")
        return

    strategies = [k for k in results[0] if k.startswith("rag_") or k == "page_index"]

    print("\n=== Résumé évaluation retrieval ===")
    for strategy in strategies:
        subset = [r for r in results if strategy in r]
        if not subset:
            continue
        p_vals = [r[strategy]["precision"] for r in subset]
        r_vals = [r[strategy]["recall"] for r in subset]
        print(f"\n{strategy}  ({len(subset)} questions)")
        print(f"  precision : {sum(p_vals)/len(p_vals):.4f}")
        print(f"  recall    : {sum(r_vals)/len(r_vals):.4f}")

        for hop in sorted({r["hop"] for r in results}):
            hop_sub = [r for r in subset if r["hop"] == hop]
            if not hop_sub:
                continue
            label = "multi-hop" if hop == "1" else "single-hop"
            p = [r[strategy]["precision"] for r in hop_sub]
            rv = [r[strategy]["recall"] for r in hop_sub]
            print(f"  Hop={hop} ({label}, {len(hop_sub)} Q)")
            print(f"    precision : {sum(p)/len(p):.4f}")
            print(f"    recall    : {sum(rv)/len(rv):.4f}")


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
    with_gt = [q for q in questions if q.get("ground_truth", "").strip()]
    print(f"{len(questions)} questions chargées, {len(with_gt)} avec ground_truth.")

    if not with_gt:
        print(
            "Aucune question avec ground_truth.\n"
            "Ajoutez une colonne 'ground_truth' dans Q_A.csv avec le texte exact du PDF.\n"
            "Pour les questions multi-hop (Hop=1), séparez les passages avec '|||'."
        )
        return

    output_path = EVAL_DIR / "eval_retrieval_results.jsonl"
    results = run_evaluation(client, corpus, with_gt, output_path)
    print(f"\nRésultats sauvegardés : {output_path}")
    print_summary(results)


if __name__ == "__main__":
    main()