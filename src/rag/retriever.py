import argparse
import json
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

from .utils import call_with_retry

PROCESSED_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
EMBEDDINGS_DIR = PROCESSED_DIR / "embeddings"

EMBEDDING_MODEL = "gemini-embedding-2"
TOP_K = 5
MMR_LAMBDA = 0.5
MMR_CANDIDATES = TOP_K * 3

load_dotenv()

DEFAULT_QUESTION = "Quelles sont les recommandations du BEA concernant les sondes Pitot ?"


def load_corpus(embeddings_dir: Path = EMBEDDINGS_DIR) -> list[dict]:
    corpus = []
    for path in sorted(embeddings_dir.glob("*_embeddings.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    corpus.append(json.loads(line))
    return corpus


def embed_query(question: str, client: genai.Client, model: str = EMBEDDING_MODEL) -> list[float]:
    response = call_with_retry(
        client.models.embed_content,
        model=model,
        contents=[question],
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    return response.embeddings[0].values


def _filter_and_score(
    question: str,
    client: genai.Client,
    corpus: list[dict],
    query_embedding: list[float] | None,
    sources: list[str] | None,
) -> tuple[np.ndarray, list[dict]]:
    filtered = [c for c in corpus if sources is None or c["source"] in sources]

    if query_embedding is None:
        query_embedding = embed_query(question, client)

    query_vec = np.array(query_embedding)
    query_vec = query_vec / np.linalg.norm(query_vec)

    matrix = np.array([c["embedding"] for c in filtered])
    matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)

    scores = matrix @ query_vec
    return query_vec, filtered, scores


def retrieve_base(
    question: str,
    client: genai.Client,
    corpus: list[dict],
    top_k: int = TOP_K,
    query_embedding: list[float] | None = None,
    sources: list[str] | None = None,
) -> list[dict]:
    query_vec, filtered, scores = _filter_and_score(question, client, corpus, query_embedding, sources)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [{**filtered[i], "score": float(scores[i])} for i in top_indices]


def retrieve_mmr(
    question: str,
    client: genai.Client,
    corpus: list[dict],
    top_k: int = TOP_K,
    query_embedding: list[float] | None = None,
    sources: list[str] | None = None,
) -> list[dict]:
    query_vec, filtered, scores = _filter_and_score(question, client, corpus, query_embedding, sources)

    n_candidates = min(MMR_CANDIDATES, len(filtered))
    top_indices = np.argsort(scores)[::-1][:n_candidates]
    candidates = [{**filtered[i], "score": float(scores[i])} for i in top_indices]

    if len(candidates) <= top_k:
        return candidates

    embeddings = np.array([c["embedding"] for c in candidates])
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    selected = []
    remaining = list(range(len(candidates)))

    for _ in range(top_k):
        rel_scores = embeddings[remaining] @ query_vec

        if not selected:
            best = remaining[int(np.argmax(rel_scores))]
        else:
            sel_emb = embeddings[selected]
            diversity_penalty = (embeddings[remaining] @ sel_emb.T).max(axis=1)
            mmr_scores = MMR_LAMBDA * rel_scores - (1 - MMR_LAMBDA) * diversity_penalty
            best = remaining[int(np.argmax(mmr_scores))]

        selected.append(best)
        remaining.remove(best)

    return [candidates[i] for i in selected]


def main() -> None:
    parser = argparse.ArgumentParser(description="Teste le retriever RAG sur une question.")
    parser.add_argument("--question", "-q", default=DEFAULT_QUESTION, help="Question à tester")
    parser.add_argument("--sources", "-s", nargs="+", default=None, metavar="SOURCE",
                        help="Filtrer par source (ex: f-cp090601_text.txt). Sans filtre = corpus entier.")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Erreur : GEMINI_API_KEY non configurée.")
        return

    client = genai.Client()

    print("Chargement du corpus...")
    corpus = load_corpus()
    print(f"Corpus chargé : {len(corpus)} chunks.")
    if args.sources:
        print(f"Filtre sources : {args.sources}")
    print(f"\nQuestion : {args.question}\n")

    query_vec = embed_query(args.question, client)

    for label, results in [
        ("BASE", retrieve_base(args.question, client, corpus, query_embedding=query_vec, sources=args.sources)),
        ("MMR ", retrieve_mmr(args.question, client, corpus, query_embedding=query_vec, sources=args.sources)),
    ]:
        print(f"=== {label} ===")
        for i, chunk in enumerate(results, 1):
            print(f"  [{i}] score={chunk['score']:.4f} | {chunk['chunk_id']}")
            print(f"       {chunk['text'][:130]}...")
        print()


if __name__ == "__main__":
    main()