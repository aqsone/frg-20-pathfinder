import argparse
import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from google import genai
from google.genai import types

from .retriever import embed_query, load_corpus, retrieve_base, retrieve_mmr
from .utils import call_with_retry

GENERATION_MODEL = "gemini-2.5-flash"

load_dotenv()

DEFAULT_QUESTION = "Quelles sont les recommandations du BEA concernant les sondes Pitot ?"

_PROMPT = """\
Tu es un assistant expert en aéronautique. Réponds à la question en te basant uniquement sur les extraits fournis.
Sois précis et complet. Si les extraits ne contiennent pas l'information, réponds : "Je ne sais pas à partir des documents fournis."

Extraits :
{context}

Question : {question}

Réponse :"""


def generate_answer(question: str, chunks: list[dict], client: genai.Client, model: str = GENERATION_MODEL) -> str:
    context = "\n\n---\n\n".join(f"[{c['source']}]\n{c['text']}" for c in chunks)
    prompt = _PROMPT.format(context=context, question=question)
    response = call_with_retry(client.models.generate_content, model=model, contents=prompt)
    return response.text.strip()


def query(
    question: str,
    client: genai.Client,
    corpus: list[dict],
    strategy: Literal["base", "mmr"] = "mmr",
    sources: list[str] | None = None,
) -> tuple[str, list[dict]]:
    retrieve_fn = retrieve_mmr if strategy == "mmr" else retrieve_base
    chunks = retrieve_fn(question, client, corpus, sources=sources)
    answer = generate_answer(question, chunks, client)
    return answer, chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Interroge le corpus RAG avec une question.")
    parser.add_argument("--question", "-q", default=DEFAULT_QUESTION)
    parser.add_argument("--sources", "-s", nargs="+", default=None, metavar="SOURCE",
                        help="Filtrer par source (ex: f-cp090601_text.txt).")
    parser.add_argument("--strategy", choices=["base", "mmr"], default="mmr",
                        help="Stratégie de retrieval (défaut : mmr).")
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
    print(f"Stratégie : {args.strategy}")
    print(f"\nQuestion : {args.question}\n")

    answer, chunks = query(args.question, client, corpus, strategy=args.strategy, sources=args.sources)

    print("=== Chunks récupérés ===")
    for i, chunk in enumerate(chunks, 1):
        print(f"  [{i}] score={chunk['score']:.4f} | {chunk['chunk_id']}")
    print()
    print("=== Réponse générée ===")
    print(answer)


if __name__ == "__main__":
    main()