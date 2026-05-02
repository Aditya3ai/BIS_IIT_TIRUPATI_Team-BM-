import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pipeline.utils import lines_with_page_numbers, normalize_text


_IS_LINE_RE = re.compile(r"^IS\s+\d{1,5}\b", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_PART_RE = re.compile(r"\(\s*Part\s*\d+\s*\)", re.IGNORECASE)


def _extract_standard_id(text: str) -> Optional[str]:
    text = normalize_text(text)
    if not _IS_LINE_RE.search(text):
        return None
    is_match = re.search(r"IS\s+\d{1,5}", text, re.IGNORECASE)
    if not is_match:
        return None
    is_number = is_match.group(0).upper().replace("  ", " ")
    part_match = _PART_RE.search(text)
    year_match = _YEAR_RE.search(text)
    if not year_match:
        return None
    part = f" {part_match.group(0)}" if part_match else ""
    return f"{is_number}{part}: {year_match.group(0)}"


def _parse_header(lines: List[str], start: int) -> Optional[Tuple[str, str, int]]:
    lookahead = " ".join(l.strip() for l in lines[start : start + 3] if l.strip())
    standard_id = _extract_standard_id(lookahead)
    if not standard_id:
        return None
    title = normalize_text(lookahead)
    title = title.replace(standard_id, "").replace(":", " ")
    title = normalize_text(title)
    return standard_id, title, start


def chunk_standards(md_text: str) -> List[Dict]:
    lines = [line for line, _ in lines_with_page_numbers(md_text)]
    page_lines = list(lines_with_page_numbers(md_text))

    chunks: List[Dict] = []
    current: Optional[Dict] = None

    for idx, (line, page) in enumerate(page_lines):
        if not line.strip():
            if current is not None:
                current["content"].append("")
            continue

        header = _parse_header(lines, idx)
        if header:
            if current is not None:
                current["page_end"] = current["page_end"] or page
                current["content"] = "\n".join(current["content"]).strip()
                chunks.append(current)
            standard_id, title, _ = header
            current = {
                "standard_id": standard_id,
                "title": title,
                "page_start": page,
                "page_end": page,
                "content": [line],
            }
            continue

        if current is not None:
            current["page_end"] = page
            current["content"].append(line)

    if current is not None:
        current["content"] = "\n".join(current["content"]).strip()
        chunks.append(current)

    return chunks


def write_chunks(md_path: Path, out_path: Path) -> int:
    md_text = md_path.read_text(encoding="utf-8")
    chunks = chunk_standards(md_text)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Split SP21 markdown into one chunk per IS standard.")
    parser.add_argument("--md", required=True, help="Input markdown from extract_pdf.py")
    parser.add_argument("--out", required=True, help="Output chunks JSON file")
    args = parser.parse_args()

    count = write_chunks(Path(args.md), Path(args.out))
    print(f"Wrote {count} chunks to {args.out}")


if __name__ == "__main__":
    main()
