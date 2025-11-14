from __future__ import annotations

import csv
import datetime
import glob
import json
from pathlib import Path
from typing import List, TYPE_CHECKING

try:  # pragma: no cover - optional dependency
    import PyPDF2  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    PyPDF2 = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import docx  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    docx = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import ebooklib  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    ebooklib = None  # type: ignore

if TYPE_CHECKING:  # pragma: no cover - hints only
    from ebooklib import epub as _epub_mod  # type: ignore
else:
    _epub_mod = None

try:  # pragma: no cover - optional dependency
    from ebooklib import epub  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    epub = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import mobi  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    mobi = None  # type: ignore

from Training.tools.hf_imports import load_datasets_module

from Training.app.use_cases.sanitization import sanitize_dataset
from Training.domain.entities import (
    CustomDatasetRequest,
    CustomDatasetResult,
    SanitizationRequest,
)

try:  # Optional dependency used for legacy DOC files
    import textract  # type: ignore
except Exception:  # pragma: no cover - optional dependency may not exist
    textract = None


_hf_datasets = load_datasets_module()
Dataset = _hf_datasets.Dataset


def _collect_csv_samples(source_dir: Path) -> list[str]:
    samples: list[str] = []
    for csv_path in glob.glob(str(source_dir / '*.csv')) + glob.glob(str(source_dir / '*.CSV')):
        try:
            with open(csv_path, newline='', encoding='utf-8') as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames:
                    lowered = [header.lower() for header in reader.fieldnames]
                    if 'text' in lowered:
                        text_field = reader.fieldnames[lowered.index('text')]
                        for row in reader:
                            text = (row.get(text_field) or '').strip()
                            if text:
                                samples.append(text)
                    else:
                        for row in reader:
                            parts = [str(value).strip() for value in row.values() if value is not None and str(value).strip()]
                            if parts:
                                samples.append(' | '.join(parts))
                else:
                    handle.seek(0)
                    raw_reader = csv.reader(handle)
                    for row in raw_reader:
                        parts = [cell.strip() for cell in row if cell and cell.strip()]
                        if parts:
                            samples.append(' | '.join(parts))
        except Exception as exc:  # pragma: no cover - best effort ingestion
            print(f'Failed to read {csv_path}: {exc}')
    return samples


def _collect_txt_samples(source_dir: Path) -> list[str]:
    samples: list[str] = []
    for txt_path in glob.glob(str(source_dir / '*.txt')) + glob.glob(str(source_dir / '*.TXT')):
        with open(txt_path, 'r', encoding='utf-8') as handle:
            text = handle.read().strip()
            if text:
                samples.append(text)
    return samples


def _collect_pdf_samples(source_dir: Path) -> list[str]:
    samples: list[str] = []
    if PyPDF2 is None:
        print('Skipping PDF ingestion: install PyPDF2 to enable this format.')
        return samples
    for pdf_path in glob.glob(str(source_dir / '*.pdf')) + glob.glob(str(source_dir / '*.PDF')):
        with open(pdf_path, 'rb') as handle:
            reader = PyPDF2.PdfReader(handle)
            text = ''.join(page.extract_text() or '' for page in reader.pages)
            if text.strip():
                samples.append(text.strip())
    return samples


def _collect_epub_samples(source_dir: Path) -> list[str]:
    samples: list[str] = []
    if ebooklib is None or epub is None:
        print('Skipping EPUB ingestion: install ebooklib to enable this format.')
        return samples
    for epub_path in glob.glob(str(source_dir / '*.epub')) + glob.glob(str(source_dir / '*.EPUB')):
        book = epub.read_epub(epub_path)
        text_parts: List[str] = []
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                text_parts.append(item.get_body_content_str())
        combined = ''.join(text_parts).strip()
        if combined:
            samples.append(combined)
    return samples


def _collect_mobi_samples(source_dir: Path) -> list[str]:
    samples: list[str] = []
    if mobi is None:
        print('Skipping MOBI ingestion: install mobi to enable this format.')
        return samples
    for mobi_path in glob.glob(str(source_dir / '*.mobi')) + glob.glob(str(source_dir / '*.MOBI')):
        book = mobi.Mobi(mobi_path)
        book.parse()
        if getattr(book, 'content', None):
            try:
                text = ''.join(str(chunk) for chunk in book.content if chunk).strip()
                if text:
                    samples.append(text)
            except Exception:  # pragma: no cover - mobi parsing is best effort
                pass
    return samples


def _collect_doc_samples(source_dir: Path) -> list[str]:
    samples: list[str] = []
    if textract is None:
        print('Skipping DOC ingestion: install textract to enable this format.')
        return samples
    for doc_path in glob.glob(str(source_dir / '*.doc')) + glob.glob(str(source_dir / '*.DOC')):
        try:
            text = textract.process(doc_path).decode('utf-8').strip()
        except Exception as exc:  # pragma: no cover - best effort ingestion
            print(f'Failed to read {doc_path} with textract: {exc}')
            continue
        if text:
            samples.append(text)
    return samples


def _collect_docx_samples(source_dir: Path) -> list[str]:
    samples: list[str] = []
    if docx is None:
        print('Skipping DOCX ingestion: install python-docx to enable this format.')
        return samples
    for docx_path in glob.glob(str(source_dir / '*.docx')) + glob.glob(str(source_dir / '*.DOCX')):
        try:
            document = docx.Document(docx_path)
        except Exception as exc:  # pragma: no cover - best effort ingestion
            print(f'Failed to read {docx_path}: {exc}')
            continue
        text = '\n'.join(paragraph.text for paragraph in document.paragraphs).strip()
        if text:
            samples.append(text)
    return samples


def _collect_samples(source_dir: Path) -> list[str]:
    samples: list[str] = []
    samples.extend(_collect_csv_samples(source_dir))
    samples.extend(_collect_txt_samples(source_dir))
    samples.extend(_collect_pdf_samples(source_dir))
    samples.extend(_collect_epub_samples(source_dir))
    samples.extend(_collect_mobi_samples(source_dir))
    samples.extend(_collect_doc_samples(source_dir))
    samples.extend(_collect_docx_samples(source_dir))
    return samples


def build_custom_dataset(request: CustomDatasetRequest) -> CustomDatasetResult:
    request.dataset_root.mkdir(parents=True, exist_ok=True)
    samples = _collect_samples(request.source_dir)
    if not samples:
        return CustomDatasetResult(
            raw_dataset_path=None,
            sanitized_dataset_path=None,
            metadata_path=None,
            document_count=0,
            flagged_count=0,
            quarantine_report=None,
        )

    unique_samples = list(dict.fromkeys(samples))
    dataset = Dataset.from_dict({'text': unique_samples})
    timestamp = request.timestamp or datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = request.dataset_root / f'custom_dataset_{timestamp}'
    dataset.save_to_disk(str(output_path))

    metadata = {
        'created_at': timestamp,
        'source_documents_dir': str(request.source_dir),
        'document_count': len(unique_samples),
    }

    flagged_count = 0
    sanitized_path: Path | None = None
    quarantine_report: Path | None = None

    if request.sanitize:
        sanitize_result = sanitize_dataset(SanitizationRequest(dataset_path=output_path))
        flagged_count = len(sanitize_result.flagged)
        sanitized_path = sanitize_result.sanitized_path
        quarantine_report = sanitize_result.quarantine_csv
        if sanitized_path:
            metadata['sanitized_dataset'] = str(sanitized_path)
        metadata['flagged_count'] = flagged_count
        if quarantine_report:
            metadata['quarantine_report'] = str(quarantine_report)

    metadata_path = output_path / 'dataset_metadata.json'
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')

    return CustomDatasetResult(
        raw_dataset_path=output_path,
        sanitized_dataset_path=sanitized_path,
        metadata_path=metadata_path,
        document_count=len(unique_samples),
        flagged_count=flagged_count,
        quarantine_report=quarantine_report,
    )
