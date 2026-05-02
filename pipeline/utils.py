import re
from typing import Iterable, List


_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_IS_NUMBER_RE = re.compile(r"\bis\s+(\d{1,5})\b")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    # Keep BIS standard IDs as a compact token too, e.g. "IS 6909" -> "is6909".
    text = _IS_NUMBER_RE.sub(r"is\1", text)
    text = _NON_ALNUM_RE.sub(" ", text)
    return [t for t in text.split() if t]


def lines_with_page_numbers(md_text: str) -> Iterable[tuple[str, int]]:
    page = 0
    for line in md_text.splitlines():
        if line.startswith("=== Page ") and line.endswith(" ==="):
            try:
                page = int(line.replace("=== Page ", "").replace(" ===", ""))
            except ValueError:
                page = page
            continue
        yield line, page
