"""Parse and format timestamps (``SS``, ``MM:SS``, ``H:MM:SS``)."""

from __future__ import annotations

import re

_CLOCK = re.compile(r"^(?:(?P<h>\d+):)?(?P<m>\d{1,2}):(?P<s>\d{1,2}(?:\.\d+)?)$")


def parse_timestamp(value: str) -> float:
    """Return the number of seconds in ``"SS"``, ``"MM:SS"`` or ``"H:MM:SS"``.

    Fractional seconds are allowed. Raises :class:`ValueError` for anything else.
    """
    text = value.strip()
    match = _CLOCK.match(text)
    if match:
        return int(match.group("h") or 0) * 3600 + int(match.group("m")) * 60 + float(match.group("s"))
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        return float(text)
    raise ValueError(f"Not a timestamp: {value!r}")


def format_timestamp(seconds: float) -> str:
    """Format seconds as ``MM:SS``, or ``H:MM:SS`` from an hour, as YouTube chapter lists use."""
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"
