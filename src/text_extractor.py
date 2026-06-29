from pathlib import Path

from pypdf import PdfReader

PDF_FILES = [
    "00_afh_full.pdf",
    "601262main_ModelingFlight-ebook.pdf",
    "601333main_NASAsContributionsToAeronauticsVolume2-ebook.pdf",
    "BEA2017-0568.pdf",
    "CS-25_Amendment_28____correction_page_1171_.pdf",
    "CS-E__Amendment_7_0.pdf",
    "f-cp090601.pdf",
]

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW_TEXT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "raw_text"


def extract_text(pdf_path: Path) -> str:
    reader = PdfReader(pdf_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def main() -> None:
    RAW_TEXT_DIR.mkdir(parents=True, exist_ok=True)

    for filename in PDF_FILES:
        pdf_path = RAW_DIR / filename
        if not pdf_path.exists():
            print(f"Fichier manquant, ignore: {pdf_path}")
            continue

        text = extract_text(pdf_path)

        output_path = RAW_TEXT_DIR / f"{pdf_path.stem}_text.txt"
        output_path.write_text(text, encoding="utf-8")
        print(f"Extrait: {filename} -> {output_path.name}")


if __name__ == "__main__":
    main()