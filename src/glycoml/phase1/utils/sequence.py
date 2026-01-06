"""Sequence utilities for N-glycosylation motif handling."""

from __future__ import annotations

from typing import Iterable, List, Optional
import re


NG_MOTIF_REGEX = re.compile(r"N[^P][ST]")


def find_nglyco_motifs(sequence: str) -> List[int]:
    """Return 0-based indices of N in N-X-S/T motifs (X != P)."""
    sequence = sequence.strip().upper()
    indices: List[int] = []
    for i in range(0, max(len(sequence) - 2, 0)):
        if sequence[i] == "N" and sequence[i + 1] != "P" and sequence[i + 2] in {"S", "T"}:
            indices.append(i)
    return indices


def parse_site_list(site_str: Optional[str], chain: Optional[str] = None) -> List[int]:
    """Parse a string of 1-based site positions into 0-based indices.

    Accepts delimiters ';', ',', or whitespace. Supports chain prefixes like 'H:297'.
    """
    if not site_str:
        return []

    items = re.split(r"[;,\s]+", site_str.strip())
    positions: List[int] = []
    for item in items:
        if not item:
            continue
        token = item
        if ":" in token:
            prefix, pos = token.split(":", 1)
            if chain and prefix.strip().upper() != chain.upper():
                continue
            token = pos

        match = re.search(r"(\d+)", token)
        if match:
            pos = int(match.group(1)) - 1
            if pos >= 0:
                positions.append(pos)

    return sorted(set(positions))


def format_site_list(positions: Iterable[int]) -> str:
    """Format 0-based positions as a semicolon-delimited 1-based string."""
    return ";".join(str(pos + 1) for pos in sorted(set(positions)))

