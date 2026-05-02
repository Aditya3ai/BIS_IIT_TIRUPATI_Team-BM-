import argparse
from pathlib import Path
from typing import List


def _table_to_markdown(rows: List[List[str]]) -> str:
    if not rows:
        return ""
    cleaned = [["" if c is None else str(c).strip() for c in row] for row in rows]
    header = cleaned[0]
    sep = ["---" for _ in header]
    body = cleaned[1:] if len(cleaned) > 1 else []
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _extract_with_pdfplumber(pdf_path: Path) -> List[str]:
    import pdfplumber

    pages_text: List[str] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for idx, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            tables = []
            try:
                for table in page.extract_tables() or []:
                    md = _table_to_markdown(table)
                    if md:
                        tables.append(md)
            except Exception:
                tables = tables
            chunks = [text]
            if tables:
                chunks.append("\n\n".join(tables))
            page_text = "\n\n".join([c for c in chunks if c.strip()])
            if not page_text.strip():
                page_text = "[[NO_TEXT_EXTRACTED]]"
            pages_text.append(page_text)
    return pages_text


def _extract_with_pypdf(pdf_path: Path) -> List[str]:
    try:
        import pypdf
    except Exception:
        import PyPDF2 as pypdf

    pages_text: List[str] = []
    reader = pypdf.PdfReader(str(pdf_path))
    for page in reader.pages:
        text = page.extract_text() or ""
        if not text.strip():
            text = "[[NO_TEXT_EXTRACTED]]"
        pages_text.append(text)
    return pages_text


def extract_pdf_to_md(pdf_path: Path, out_path: Path, engine: str = "pdfplumber") -> None:
    if engine == "pdfplumber":
        try:
            pages = _extract_with_pdfplumber(pdf_path)
        except Exception as exc:
            raise SystemExit(f"pdfplumber failed: {exc}")
    else:
        pages = _extract_with_pypdf(pdf_path)

    avg_len = sum(len(p) for p in pages) / max(1, len(pages))
    if avg_len < 50:
        print("Warning: low text extraction density; PDF may be scanned.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for idx, text in enumerate(pages, start=1):
            f.write(f"=== Page {idx} ===\n")
            f.write(text)
            f.write("\n\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract PDF to markdown with page markers.")
    parser.add_argument("--pdf", required=True, help="Path to SP21 PDF.")
    parser.add_argument("--out", required=True, help="Output markdown file path.")
    parser.add_argument("--engine", choices=["pdfplumber", "pypdf"], default="pdfplumber")
    args = parser.parse_args()

    extract_pdf_to_md(Path(args.pdf), Path(args.out), args.engine)


if __name__ == "__main__":
    main()
