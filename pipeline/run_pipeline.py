import argparse
from pathlib import Path

from pipeline import build_index, chunk_standards, extract_pdf


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full SP21 ingestion pipeline.")
    parser.add_argument("--pdf", required=True, help="Path to SP21 PDF")
    parser.add_argument("--work-dir", default="data", help="Output working directory")
    parser.add_argument("--engine", choices=["pdfplumber", "pypdf"], default="pdfplumber")
    parser.add_argument("--embeddings", choices=["google", "sbert", "none"], default="google")
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    md_path = work_dir / "sp21_raw.md"
    chunks_path = work_dir / "chunks.json"
    index_dir = work_dir / "index"

    extract_pdf.extract_pdf_to_md(Path(args.pdf), md_path, args.engine)
    count = chunk_standards.write_chunks(md_path, chunks_path)
    build_index.build_indexes(chunks_path, index_dir, args.embeddings)

    print(f"Pipeline complete. Chunks: {count}")


if __name__ == "__main__":
    main()
