"""
Transcript ingestion: accept text/file, normalize, extract metadata.

Supports:
  - Plain text strings
  - File paths (.txt, .md, .srt, .vtt)
  - URL strings (fetched via httpx)

Metadata extraction uses heuristics + optional caller-supplied overrides.
"""

from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from truthbot.models import Transcript


# ── Helpers ───────────────────────────────────────────────────────────────────

_DATE_PATTERNS = [
    r"\b(\w+ \d{1,2},\s*\d{4})\b",          # "January 20, 2025"
    r"\b(\d{4}-\d{2}-\d{2})\b",             # "2025-01-20"
    r"\b(\d{1,2}/\d{1,2}/\d{4})\b",         # "1/20/2025"
    r"\b(\d{1,2} \w+ \d{4})\b",             # "20 January 2025"
]

_VENUE_PATTERNS = [
    r"(?:delivered at|speaking at|address(?:ed)? (?:to|at)|remarks? (?:at|before|to))\s+(.+?)(?:\.|,|$)",
    r"(?:State of the Union|Oval Office|Rose Garden|United Nations|Congress)",
]

_HEADER_RE = re.compile(
    r"^(?:speaker|by|from|date|location|venue|event):\s*(.+)$",
    re.IGNORECASE | re.MULTILINE,
)


def _strip_srt(text: str) -> str:
    """Remove SRT/VTT subtitle formatting, leaving only spoken words."""
    # Remove numeric sequence markers
    text = re.sub(r"^\d+\s*$", "", text, flags=re.MULTILINE)
    # Remove timestamp lines  00:00:00,000 --> 00:00:05,000
    text = re.sub(r"\d{2}:\d{2}:\d{2}[,\.]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[,\.]\d{3}", "", text)
    # Remove WEBVTT header
    text = re.sub(r"^WEBVTT.*$", "", text, flags=re.MULTILINE)
    return text


def _normalize_whitespace(text: str) -> str:
    """Collapse multiple blank lines; normalize line endings."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_date_from_text(text: str) -> Optional[datetime]:
    """Try to detect a date mentioned near the beginning of the transcript."""
    sample = text[:500]
    for pattern in _DATE_PATTERNS:
        m = re.search(pattern, sample)
        if m:
            raw = m.group(1)
            # Try several parse strategies
            for fmt in ("%B %d, %Y", "%Y-%m-%d", "%m/%d/%Y", "%d %B %Y", "%d %b %Y"):
                try:
                    return datetime.strptime(raw, fmt)
                except ValueError:
                    continue
    return None


def _extract_venue_from_text(text: str) -> Optional[str]:
    """Heuristic venue extraction from transcript header area."""
    sample = text[:800]
    for pattern in _VENUE_PATTERNS:
        m = re.search(pattern, sample, re.IGNORECASE)
        if m:
            return m.group(0).strip() if not m.lastindex else m.group(1).strip()
    return None


# ── Public interface ──────────────────────────────────────────────────────────


@dataclass
class IngestResult:
    """Result of an ingestion operation."""

    transcript: Transcript
    warnings: list[str]


class TranscriptIngester:
    """
    Normalize and ingest a transcript from multiple input sources.

    Usage::

        ingester = TranscriptIngester()
        result = ingester.ingest_text("Unemployment is at a 50-year low.",
                                      speaker="President", date=datetime(2025,1,20))
        transcript = result.transcript
    """

    def ingest_text(
        self,
        text: str,
        *,
        speaker: str = "Unknown",
        date: Optional[datetime] = None,
        venue: Optional[str] = None,
        source_url: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> IngestResult:
        """
        Ingest a plain-text transcript string.

        Parameters
        ----------
        text:
            Raw transcript text (any whitespace / encoding).
        speaker:
            Speaker name or title. Will attempt heuristic extraction if
            left as "Unknown".
        date:
            Explicit speech date; auto-detected from text if not provided.
        venue:
            Venue or event name; auto-detected from text if not provided.
        source_url:
            Optional URL where the transcript was obtained.
        metadata:
            Any extra key-value pairs to attach to the transcript.

        Returns
        -------
        IngestResult
            Contains the normalized Transcript and any warnings.
        """
        warnings: list[str] = []

        if not text or not text.strip():
            raise ValueError("Transcript text is empty")

        # Strip subtitle formatting if present
        if re.search(r"\d{2}:\d{2}:\d{2}[,\.]\d{3}\s*-->", text):
            text = _strip_srt(text)
            warnings.append("Detected and stripped subtitle timing markers.")

        normalized = _normalize_whitespace(text)

        # Auto-detect metadata if not provided
        detected_date = date or _extract_date_from_text(normalized)
        detected_venue = venue or _extract_venue_from_text(normalized)

        if not detected_date:
            warnings.append("Could not detect a date in the transcript.")

        transcript = Transcript(
            text=normalized,
            speaker=speaker,
            date=detected_date,
            venue=detected_venue,
            source_url=source_url,
            metadata=metadata or {},
        )

        return IngestResult(transcript=transcript, warnings=warnings)

    def ingest_file(
        self,
        path: str | Path,
        *,
        speaker: str = "Unknown",
        date: Optional[datetime] = None,
        venue: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> IngestResult:
        """
        Read a transcript from a file and ingest it.

        Supported formats: .txt, .md, .srt, .vtt (any UTF-8 text file).

        Parameters
        ----------
        path:
            Path to the transcript file.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Transcript file not found: {p}")
        if not p.is_file():
            raise ValueError(f"Path is not a file: {p}")

        text = p.read_text(encoding="utf-8", errors="replace")
        return self.ingest_text(
            text,
            speaker=speaker,
            date=date,
            venue=venue,
            source_url=p.as_uri(),
            metadata=metadata,
        )

    def ingest_url(
        self,
        url: str,
        *,
        speaker: str = "Unknown",
        date: Optional[datetime] = None,
        venue: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> IngestResult:
        """
        Fetch a transcript from a URL and ingest it.

        This performs a simple HTTP GET and extracts visible text. For
        JavaScript-heavy pages, a browser-based fetcher will be needed.

        Parameters
        ----------
        url:
            HTTP/HTTPS URL to fetch.
        """
        import httpx

        try:
            response = httpx.get(url, follow_redirects=True, timeout=30.0)
            response.raise_for_status()
            text = response.text
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Failed to fetch transcript from {url}: {exc}") from exc

        return self.ingest_text(
            text,
            speaker=speaker,
            date=date,
            venue=venue,
            source_url=url,
            metadata=metadata,
        )

    def ingest(
        self,
        source: str | Path,
        *,
        speaker: str = "Unknown",
        date: Optional[datetime] = None,
        venue: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> IngestResult:
        """
        Dispatch ingestion based on source type.

        - If source is a URL string → ingest_url
        - If source looks like a file path → ingest_file
        - Otherwise → treat as raw transcript text

        Parameters
        ----------
        source:
            URL, file path, or raw transcript text.
        """
        s = str(source)

        if s.startswith("http://") or s.startswith("https://"):
            return self.ingest_url(s, speaker=speaker, date=date, venue=venue, metadata=metadata)

        p = Path(s)
        if p.exists() and p.is_file():
            return self.ingest_file(p, speaker=speaker, date=date, venue=venue, metadata=metadata)

        return self.ingest_text(s, speaker=speaker, date=date, venue=venue, metadata=metadata)
