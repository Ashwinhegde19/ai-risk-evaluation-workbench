"""Dependency-free minimal PDF writer for text-based compliance reports.

This module emits a valid PDF 1.4 document containing one or more pages of
left-aligned Helvetica text. It is intentionally self-contained (no
``reportlab`` / ``weasyprint`` dependency) so report generation works in any
environment without extra installs. The output is a normal PDF that standard
viewers can open; it is not intended for rich layout (tables, images, etc.).
"""

from __future__ import annotations

from pathlib import Path
from typing import List

# US Letter page geometry, in PDF points (72 pt == 1 inch).
PAGE_WIDTH: float = 612.0
PAGE_HEIGHT: float = 792.0
MARGIN_X: float = 54.0
MARGIN_TOP: float = 54.0
MARGIN_BOTTOM: float = 54.0
LINE_HEIGHT: float = 14.0
FONT_SIZE: float = 11.0


def _escape(text: str) -> str:
    """Escape a string for use inside a PDF literal text object.

    Args:
        text: Raw text line.

    Returns:
        The text with ``(``, ``)`` and ``\\`` escaped and non-ASCII characters
        replaced with ``?`` (PDF literal strings are Latin-1 in this writer).
    """
    out: List[str] = []
    for ch in text:
        if ch in ("(", ")", "\\"):
            out.append("\\" + ch)
        elif ord(ch) < 32 or ord(ch) > 126:
            out.append("?")
        else:
            out.append(ch)
    return "".join(out)


def write_pdf(path: str | Path, title: str, lines: List[str]) -> Path:
    """Write a list of text lines to a minimal multi-page PDF.

    Long inputs are automatically paginated; the supplied ``title`` is prefixed
    to the first page only.

    Args:
        path: Destination ``.pdf`` path.
        title: Report title rendered at the top of the first page.
        lines: Body lines (one entry per visual line).

    Returns:
        The path the PDF was written to.
    """
    path = Path(path)
    usable_height = PAGE_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM
    max_lines_per_page = max(1, int(usable_height // LINE_HEIGHT))

    body: List[str] = [title, ""] + list(lines)
    pages: List[List[str]] = [
        body[i : i + max_lines_per_page]
        for i in range(0, len(body), max_lines_per_page)
    ]
    if not pages:
        pages = [[]]

    # Object numbering plan:
    #   1 -> Catalog, 2 -> Pages, 3 -> Font, then per page: Page, Content.
    font_obj = 3
    page_specs: List[tuple[int, int]] = []  # (page_obj_num, content_obj_num)
    next_num = 4
    for _ in pages:
        page_specs.append((next_num, next_num + 1))
        next_num += 2
    total_objs = next_num

    contents: List[bytes] = []
    for chunk in pages:
        stream_lines: List[str] = []
        y = PAGE_HEIGHT - MARGIN_TOP
        for line in chunk:
            y -= LINE_HEIGHT
            stream_lines.append(
                f"BT /F1 {FONT_SIZE:.1f} Tf {MARGIN_X:.1f} {y:.2f} Td "
                f"({_escape(line)}) Tj ET"
            )
        contents.append("\n".join(stream_lines).encode("latin-1", "replace"))

    objects: dict[int, bytes] = {}
    objects[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    kids = " ".join(f"{p} 0 R" for p, _ in page_specs)
    objects[2] = (
        f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode()
    )
    objects[3] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    for idx, (page_num, content_num) in enumerate(page_specs):
        content = contents[idx]
        objects[page_num] = (
            f"<< /Type /Page /Parent 2 0 R "
            f"/MediaBox [0 0 {PAGE_WIDTH:.0f} {PAGE_HEIGHT:.0f}] "
            f"/Resources << /Font << /F1 {font_obj} 0 R >> >> "
            f"/Contents {content_num} 0 R >>"
        ).encode()
        objects[content_num] = (
            b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n"
            + content
            + b"\nendstream"
        )

    out = bytearray()
    out += b"%PDF-1.4\n"
    offsets: dict[int, int] = {}
    for num in range(1, total_objs):
        offsets[num] = len(out)
        out += f"{num} 0 obj\n".encode() + objects[num] + b"\nendobj\n"

    xref_pos = len(out)
    out += f"xref\n0 {total_objs}\n".encode()
    out += b"0000000000 65535 f \n"
    for num in range(1, total_objs):
        out += f"{offsets[num]:010d} 00000 n \n".encode()
    out += b"trailer\n"
    out += f"<< /Size {total_objs} /Root 1 0 R >>\n".encode()
    out += f"startxref\n{xref_pos}\n%%EOF\n".encode()

    path.write_bytes(out)
    return path
