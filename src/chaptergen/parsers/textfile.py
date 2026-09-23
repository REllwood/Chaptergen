"""Read transcript files in the encodings common subtitle and text tools save them in."""

from __future__ import annotations

import codecs
from pathlib import Path


def read_text(path: Path) -> str:
    """Return the file's text with newlines normalised to ``\\n``.

    Tries UTF-8 (with or without a byte-order mark), UTF-16 when a BOM says so
    (Notepad's "Unicode" option), then Windows-1252 for subtitles saved by older
    Windows software.
    """
    data = path.read_bytes()

    if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        encodings: tuple[str, ...] = ("utf-16",)
    else:
        encodings = ("utf-8-sig", "cp1252")

    for encoding in encodings:
        try:
            text = data.decode(encoding)
        except UnicodeDecodeError:
            continue
        return text.replace("\r\n", "\n").replace("\r", "\n")

    raise ValueError("the file isn't UTF-8, UTF-16 or Windows-1252 text. Re-save it as UTF-8 and try again.")
