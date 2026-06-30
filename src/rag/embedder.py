import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

PROCESSED_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
CHUNKS_DIR = PROCESSED_DIR / "chunks"
EMBEDDINGS_DIR = PROCESSED_DIR / "embeddings"

load_dotenv()

EMBEDDING_MODEL = "gemini-embedding-2"

BATCH_SIZE = 50


def embed_chunks(chunks: list[dict], client: genai.Client, model: str = EMBEDDING_MODEL) -> list[dict]:
    if not chunks:
        return []

    total = len(chunks)
    total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"  -> Début du traitement : {total} chunks en {total_batches} batch(s).")

    embedded_chunks = []
    for i in range(0, total, BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        current_batch = i // BATCH_SIZE + 1

        contents = [
            types.Content(parts=[types.Part.from_text(text=f"title: none | text: {c['text']}")])
            for c in batch
        ]

        success = False
        while not success:
            try:
                response = client.models.embed_content(model=model, contents=contents)
                for chunk, emb in zip(batch, response.embeddings):
                    ec = chunk.copy()
                    ec["embedding"] = emb.values
                    embedded_chunks.append(ec)
                print(f"  -> [OK] Batch {current_batch}/{total_batches} traité.")
                success = True

            except Exception as e:
                if "429" in str(e):
                    print(f"  -> [!] Quota atteint (Batch {current_batch}). Pause de 60 secondes...")
                    time.sleep(60)
                else:
                    raise e

        if current_batch < total_batches:
            time.sleep(5)

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
            n_chunks = sum(1 for l in chunk_path.open(encoding="utf-8") if l.strip())
            n_embedded = sum(1 for l in output_path.open(encoding="utf-8") if l.strip())
            if n_chunks == n_embedded:
                print(f"Cache OK pour {doc_name} ({n_embedded}/{n_chunks} chunks). Passage au suivant.")
                continue
            print(f"Cache invalide pour {doc_name} ({n_embedded}/{n_chunks} chunks). Recalcul...")

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