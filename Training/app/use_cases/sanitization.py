from __future__ import annotations

import csv
import datetime
import json
import re
from pathlib import Path
from typing import Iterable

from Training.tools.hf_imports import load_datasets_module

from Training.domain.entities import SanitizationRequest, SanitizationResult

_CONTEXT_KEYWORDS = {
    'news', 'report', 'court', 'case', 'trial', 'conviction', 'arrest', 'charge',
    'allegation', 'investigation', 'police', 'law', 'legal', 'study', 'research',
    'statistics', 'prevent', 'awareness', 'education', 'survivor', 'victim',
    'advocacy', 'justice', 'prosecut', 'prosecutor', 'age of consent'
}

_FLAG_PHRASES = (
    r"\bnon-?consensual (?:sex|sexual|intercourse|encounter|activity)\b",
    r"\bwithout consent (?:sex|sexual|intercourse)\b",
    r"\bdid not consent (?:sex|sexual|intercourse)\b",
    r"\bdidn't consent (?:sex|sexual|intercourse)\b",
    r"\bno consent (?:sex|sexual|intercourse)\b",
    r"\bforced (?:sex|sexual|intercourse|oral|anal|penetration)\b",
    r"\bforce (?:sex|sexual|intercourse|oral|anal|penetration)\b",
    r"\bmade (?:me|him|her|them|someone) (?:to )?(?:have sex|perform sexual acts)\b",
    r"\brape\s+(?:victim|scene|fantasy|roleplay|simulation)\b",
    r"\babuse\s+(?:sexual|intimate|physical|domestic)\b",
    r"\bmolest(?:ation|ed)?\b",
    r"\bsexual (?:coercion)\b",
    r"\b(?:forced|violent|brutal)\s+(?:sex|rape|abuse)\b",
    r"\bsexual (?:assault|abuse|exploitation|battery)\b",
    r"\bagainst (?:his|her|their) will\s+(?:sex|sexual|intimate|rape|abuse)\b",
    r"\bunable to consent\s+(?:to|for)\s+(?:sex|sexual|intimate)\b",
    r"\b(?:cannot|can't|cant) consent\s+(?:to|for)\s+(?:sex|sexual|intimate)\b",
    r"\brefuse(?:d|s)?\s+(?:sex|sexual|intimate|advances)\b",
    r"\bresist(?:ed|ing)?\s+(?:sex|sexual|rape|advances|abuse)\b",
    r"\bprotest(?:ed|ing)?\s+(?:sex|sexual|rape|advances|abuse)\b",
    r"\bboundaries\s+(?:ignored|violated|crossed)\b",
    r"\bconsent\s+(?:not|wasn't|isn't)\s+respected\b",
    r"\bunderage\s+(?:sex|sexual|intimate)\b",
    r"\bchild\s+(?:abuse|exploitation|sexual)\b",
    r"\bminor\s+(?:sexual|abuse|exploitation|intimate)\b",
    r"\bincest\s+(?:for\s+)?(?:sexual|intimate)\b",
    r"\bpedophil(?:e|ia)\s+(?:for\s+)?(?:sexual|intimate)\b",
    r"\bgrooming\s+(?:for\s+)?(?:sexual|intimate)\b",
    r"\bchild\s+(?:pornography|porn)\b",
    r"\bunderage\s+(?:pornography|porn|prostitution)\b",
    r"\babduct(?:ed|ing)\s+(?:for|sexual|rape|abuse)\b",
    r"\bkidnap(?:ped|ping)\s+(?:for|sexual|rape|abuse)\b",
    r"\b\d{1,2}\s*(?:yo|yrs?|years? old)\s+(?:sex|sexual|intimate|naked|nude|porn)\b",
    r"\btoo young\s+(?:for|sex|sexual|intimate)\b",
    r"\bsex traffick(?:ing|ed)\b",
    r"\bhuman traffick(?:ing|ed)\s+(?:for|sexual)\b",
    r"\bdrugg?ed\s+(?:and|for|sexual|rape|abuse)\b",
    r"\btoo drunk\s+(?:to|for|sex|consent)\b",
    r"\bunconsci(?:ous|ously)\s+(?:sex|sexual|rape)\b",
    r"\bmentally (?:disabled|impaired)\s+(?:sex|sexual|abuse)\b",
    r"\bblackmail(?:ed|ing)?\s+(?:for|into|sex|sexual)\b",
    r"\bthreaten(?:ed|ing)?\s+(?:to|into|sex|sexual|rape)\b",
)

_COMPILED_PATTERNS = tuple(re.compile(pattern, re.IGNORECASE) for pattern in _FLAG_PHRASES)

_hf_datasets = load_datasets_module()
Dataset = _hf_datasets.Dataset
load_from_disk = _hf_datasets.load_from_disk


def _excerpt(text: str, length: int = 300) -> str:
    return text.replace('\n', ' ')[:length]


def _context_is_safe(text: str, match_start: int, window: int = 120) -> bool:
    start = max(0, match_start - window)
    context = text[start:match_start].lower()
    return any(keyword in context for keyword in _CONTEXT_KEYWORDS)


def _iter_dataset(dataset: Dataset) -> Iterable[tuple[int, str]]:
    for index, record in enumerate(dataset):
        if isinstance(record, dict):
            value = record.get('text')
        else:
            value = record
        if not value:
            yield index, ''
            continue
        yield index, str(value)


def sanitize_dataset(request: SanitizationRequest) -> SanitizationResult:
    dataset = load_from_disk(str(request.dataset_path))
    flagged: list[tuple[int, list[str], str]] = []
    safe_indices: list[int] = []

    for index, text in _iter_dataset(dataset):
        if not text:
            safe_indices.append(index)
            continue
        matches: list[str] = []
        for pattern in _COMPILED_PATTERNS:
            for found in pattern.finditer(text):
                if _context_is_safe(text, found.start()):
                    continue
                matches.append(found.group(0))
        if matches:
            flagged.append((index, matches, _excerpt(text)))
        else:
            safe_indices.append(index)

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = request.dataset_path.parent
    quarantine_csv = request.quarantine_csv or base_dir / f'quarantine_{timestamp}.csv'
    quarantine_csv.parent.mkdir(parents=True, exist_ok=True)
    with quarantine_csv.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['index', 'matched_phrases', 'excerpt'])
        for row_index, phrases, excerpt in flagged:
            writer.writerow([row_index, '|'.join(phrases), excerpt])

    sanitized_path: Path | None = None
    if request.save_sanitized:
        sanitized = dataset.select(safe_indices)
        sanitized_path = base_dir / f'sanitized_dataset_{timestamp}'
        sanitized.save_to_disk(str(sanitized_path))
        metadata = {
            'source_dataset': str(request.dataset_path),
            'created_at': timestamp,
            'quarantine_report': str(quarantine_csv),
            'flagged_examples': len(flagged),
            'retained_examples': len(sanitized),
        }
        metadata_path = sanitized_path / 'sanitization_metadata.json'
        metadata_path.write_text(
            json.dumps(metadata, indent=2),
            encoding='utf-8',
        )

    return SanitizationResult(
        flagged=flagged,
        quarantine_csv=quarantine_csv,
        sanitized_path=sanitized_path,
    )
