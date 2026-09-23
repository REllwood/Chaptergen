"""Parse the timestamp strings people and models write (``SS``, ``MM:SS``, ``H:MM:SS``)."""

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
