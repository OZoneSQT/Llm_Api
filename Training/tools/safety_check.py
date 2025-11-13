import re
from pathlib import Path
from typing import List, Tuple

# Safety checker with simple regex patterns and severity scoring.
# - load_blocklist returns list of literal phrases
# - scan_text returns list of (phrase, span, severity)
# - contains_blocked(text, blocklist) remains for backwards compatibility


def load_blocklist(path: str | Path) -> List[str]:
    p = Path(path)
    if not p.exists():
        return []
    lines = [l.strip() for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]
    return lines


def _compile_patterns(blocklist: List[str]) -> List[Tuple[str, re.Pattern, int]]:
    patterns = []
    for phrase in blocklist:
        # simple heuristic: longer phrases = higher severity
        severity = min(5, max(1, len(phrase) // 10))
        try:
            pat = re.compile(re.escape(phrase), re.IGNORECASE)
        except Exception:
            pat = re.compile(re.escape(phrase))
        patterns.append((phrase, pat, severity))
    return patterns


def scan_text(text: str, blocklist: List[str]) -> List[Tuple[str, Tuple[int, int], int]]:
    """Return list of matches (phrase, (start, end), severity) found in text."""
    if not blocklist:
        return []
    patterns = _compile_patterns(blocklist)
    matches = []
    for phrase, pat, sev in patterns:
        for m in pat.finditer(text):
            matches.append((phrase, (m.start(), m.end()), sev))
    return matches


def contains_blocked(text: str, blocklist: List[str]) -> bool:
    return len(scan_text(text, blocklist)) > 0


if __name__ == '__main__':
    bl = load_blocklist(Path(__file__).resolve().parents[1] / 'data' / 'banned_words.txt')
    sample = "This is a harmless sentence."
    print('blocklist size', len(bl))
    print('sample clean?', not contains_blocked(sample, bl))
