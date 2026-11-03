"""Console helpers shared by the command-line entry points."""

from __future__ import annotations

import sys


def utf8_stdio() -> None:
    """Write command-line output as UTF-8 (Å, Δ, ±, →) whatever the console code page.

    Windows writes redirected output (CI logs, pipes) in the ANSI code page, e.g.
    cp1252, which cannot encode these characters and would raise
    ``UnicodeEncodeError`` in the middle of a report.
    """
    for stream in (sys.stdout, sys.stderr):
        encoding = (getattr(stream, "encoding", None) or "").lower().replace("-", "").replace("_", "")
        if encoding != "utf8" and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # detached or non-reconfigurable stream: leave it alone
                pass
