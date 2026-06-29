import json
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

PROCESSED_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
RAW_TEXT_DIR = PROCESSED_DIR / "raw_text"
CHUNKS_DIR = PROCESSED_DIR / "chunks"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)


def chunk_file(text_path: Path) -> list[dict]:
    text = text_path.read_text(encoding="utf-8")
    doc_name = text_path.stem.removesuffix("_text")

    chunks = splitter.split_text(text)
    return [
        {
            "chunk_id": f"{doc_name}_{i:04d}",
            "source": text_path.name,
            "chunk_index": i,
            "text": chunk,
            "n_chars": len(chunk),
        }
        for i, chunk in enumerate(chunks)
    ]


def main() -> None:
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    for text_path in sorted(RAW_TEXT_DIR.glob("*_text.txt")):
        chunks = chunk_file(text_path)

        doc_name = text_path.stem.removesuffix("_text")
        output_path = CHUNKS_DIR / f"{doc_name}_chunks.jsonl"
        with output_path.open("w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

        print(f"Chunke: {text_path.name} -> {output_path.name} ({len(chunks)} chunks)")


if __name__ == "__main__":
    main()