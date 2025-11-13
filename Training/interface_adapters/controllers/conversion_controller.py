from __future__ import annotations

from pathlib import Path
from typing import List

from Training.app.use_cases import conversion
from Training.domain.entities import ConversionRequest


def detect_formats(path: Path) -> List[str]:
    detection = conversion.detect_formats(path)
    messages = [f'Format detection for {detection.root}']
    for key, value in detection.formats.items():
        messages.append(f'  {key}: {value}')
    if detection.formats.get('gguf') or detection.formats.get('ggml'):
        messages.append('Hint: GGUF/GGML artifacts detected. Consult Training/README.md for conversion steps.')
    if detection.formats.get('gptq'):
        messages.append('Hint: GPTQ artifacts detected. Consider auto_gptq conversion or compatible loaders.')
    return messages


def run_llama_conversion(request: ConversionRequest) -> bool:
    return conversion.run_llama_conversion(request)


def run_auto_gptq_conversion(request: ConversionRequest) -> bool:
    return conversion.run_auto_gptq_conversion(request)


def ensure_auto_gptq_available() -> bool:
    return conversion.is_auto_gptq_available()
