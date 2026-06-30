import json
import os
from pathlib import Path

from google import genai
from google.genai import types
from dotenv import load_dotenv

import time

PROCESSED_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
CHUNKS_DIR = PROCESSED_DIR / "chunks"
EMBEDDINGS_DIR = PROCESSED_DIR / "embeddings"

load_dotenv()

EMBEDDING_MODEL = "gemini-embedding-2"

def embed_chunks(chunks: list[dict], client: genai.Client, model: str = EMBEDDING_MODEL) -> list[dict]:
    """Prend une liste de chunks, extrait le texte, appelle l'API Gemini
    par batchs et ajoute le vecteur dans une nouvelle clé 'embedding'.
    Gère les limites de taux (Rate Limits) via un système de retry.
    """
    if not chunks:
        return []

    texts = [chunk["text"] for chunk in chunks]
    embeddings = []

    batch_size = 50
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    print(f"  -> Début du traitement : {len(texts)} chunks découpés en {total_batches} batch(s).")

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        current_batch = (i // batch_size) + 1

        success = False
        while not success:
            try:
                response = client.models.embed_content(
                    model=model,
                    contents=batch_texts,
                    config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
                )

                batch_embeddings = [e.values for e in response.embeddings]
                embeddings.extend(batch_embeddings)
                
                print(f"  -> [OK] Batch {current_batch}/{total_batches} traité.")
                success = True

            except Exception as e:
                if "429" in str(e):
                    print(f"  -> [!] Quota atteint (Batch {current_batch}). Mise en pause de 60 secondes...")
                    time.sleep(60)
                else:
                    raise e

        if current_batch < total_batches:
            print("  -> Pause d'espacement de 5 secondes...")
            time.sleep(5)

    embedded_chunks = []
    for chunk, embedding in zip(chunks, embeddings):
        embedded_chunk = chunk.copy()
        embedded_chunk["embedding"] = embedding
        embedded_chunks.append(embedded_chunk)

    return embedded_chunks

def main() -> None:
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Erreur : La variable d'environnement GEMINI_API_KEY n'est pas configurée.")
        return
    
    
    client = genai.Client()

    chunk_files = sorted(CHUNKS_DIR.glob("*_chunks.jsonl"))
    if not chunk_files:
        print(f"Aucun fichier de chunks trouvé dans {CHUNKS_DIR}")
        return

    for chunk_path in chunk_files:
        doc_name = chunk_path.stem.removesuffix("_chunks")
        output_path = EMBEDDINGS_DIR / f"{doc_name}_embeddings.jsonl"

        if output_path.exists():
            print(f"Cache trouvé pour {doc_name} ({output_path.name}). Passage au suivant.")
            continue

        print(f"Calcul des embeddings pour {chunk_path.name}...")

        chunks = []
        with chunk_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line))

        try:
            embedded_chunks = embed_chunks(chunks, client)

            with output_path.open("w", encoding="utf-8") as f:
                for chunk in embedded_chunks:
                    f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

            print(f"Embeddings créés : {chunk_path.name} -> {output_path.name} ({len(embedded_chunks)} chunks)")

        except Exception as e:
            print(f"Erreur lors du traitement des embeddings pour {chunk_path.name} : {e}")


if __name__ == "__main__":
    main()