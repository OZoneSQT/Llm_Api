r"""Download selected Hugging Face repos into the configured MODEL_ROOT/DATA_ROOT.

This CLI mirrors the legacy Training/tools/download_runner_e.py script while delegating
path discovery to the Clean Architecture domain layer.
"""
import argparse
import glob
import itertools
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from shutil import which
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from huggingface_hub import snapshot_download

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

from Training.domain.path_config import PathConfig
from Training.frameworks_drivers.logging import StructuredCsvLogger, get_csv_logger


LOGGER: StructuredCsvLogger | None = None


def _log_info(message: str, **context) -> None:
    if LOGGER:
        LOGGER.info(message, **context)


def _log_warning(message: str, **context) -> None:
    if LOGGER:
        LOGGER.warning(message, **context)


def _log_error(message: str, **context) -> None:
    if LOGGER:
        LOGGER.error(message, **context)


def _log_exception(message: str, **context) -> None:
    if LOGGER:
        LOGGER.exception(message, **context)


def _find_incomplete_cache_markers(cache_snapshot_dir: Path) -> List[Path]:
    if not cache_snapshot_dir.exists():
        return []

    markers: List[Path] = []
    markers.extend(cache_snapshot_dir.glob('*.lock'))
    markers.extend(cache_snapshot_dir.glob('*.incomplete'))

    for entry in cache_snapshot_dir.iterdir():
        if entry.name.endswith('.incomplete'):
            markers.append(entry)
            continue
        if entry.is_file() and entry.name.endswith('.lock'):
            markers.append(entry)
            continue
        if not entry.is_dir():
            continue
        try:
            if not any(entry.iterdir()):
                markers.append(entry)
                continue
        except OSError:
            markers.append(entry)
            continue
        for pattern in ('*.partial', '*.incomplete', '*.lock', '*.tmp'):
            markers.extend(entry.rglob(pattern))

    downloads_dir = cache_snapshot_dir.parent / 'downloads'
    if downloads_dir.exists():
        for pattern in ('*.partial', '*.lock', '*.tmp'):
            markers.extend(downloads_dir.rglob(pattern))

    unique: List[Path] = []
    seen: set[str] = set()
    for marker in markers:
        key = str(marker)
        if key in seen:
            continue
        seen.add(key)
        unique.append(marker)
    return unique


def _classify_cache_state(cache_snapshot_dir: Path) -> tuple[str, List[Path]]:
    if not cache_snapshot_dir.exists():
        return 'missing', []

    try:
        entries = list(cache_snapshot_dir.iterdir())
    except OSError:
        return 'incomplete', []

    if not entries:
        return 'incomplete', []

    markers = _find_incomplete_cache_markers(cache_snapshot_dir)
    if markers:
        return 'incomplete', markers

    snapshot_dirs = [entry for entry in entries if entry.is_dir() and not entry.name.endswith('.incomplete')]
    if not snapshot_dirs:
        return 'incomplete', []

    refs_dir = cache_snapshot_dir.parent / 'refs'
    if not refs_dir.exists():
        return 'incomplete', []
    try:
        if not any(refs_dir.iterdir()):
            return 'incomplete', []
    except OSError:
        return 'incomplete', []

    return 'complete', []


def _cleanup_incomplete_markers(markers: List[Path]) -> List[str]:
    removed: List[str] = []
    for marker in sorted(markers, key=lambda path: len(str(path)), reverse=True):
        marker_str = str(marker)
        try:
            if marker.is_dir():
                shutil.rmtree(marker, ignore_errors=True)
                if not marker.exists():
                    removed.append(marker_str)
            else:
                marker.unlink(missing_ok=True)
                if not marker.exists():
                    removed.append(marker_str)
        except FileNotFoundError:
            removed.append(marker_str)
        except Exception as cleanup_error:  # pragma: no cover - best effort cleanup
            _log_warning('cleanup_incomplete_marker_failed', path=marker_str, error=str(cleanup_error))
    return removed

#####################################################################################################
#####################################################################################################

# Explocit list of datasets to download
DATASETS = []

# Instruct

# https://huggingface.co/datasets/HuggingFaceH4/helpful-instructions
DATASETS.append("HuggingFaceH4/helpful-instructions")

# https://huggingface.co/datasets/meta-llama/Llama-3.2-1B-Instruct-evals
DATASETS.append("meta-llama/Llama-3.2-1B-Instruct-evals")

# https://huggingface.co/datasets/meta-llama/Llama-3.2-3B-Instruct-evals
DATASETS.append("meta-llama/Llama-3.2-3B-Instruct-evals")

# https://huggingface.co/datasets/crumb/Clean-Instruct-440k
DATASETS.append("crumb/Clean-Instruct-440k")

# https://huggingface.co/datasets/yizhongw/self_instruct
DATASETS.append("yizhongw/self_instruct")

# https://huggingface.co/datasets/iamtarun/code_instructions_120k_alpaca
DATASETS.append("iamtarun/code_instructions_120k_alpaca")

# https://huggingface.co/datasets/Muennighoff/natural-instructions
DATASETS.append("Muennighoff/natural-instructions")

# https://huggingface.co/datasets/HuggingFaceH4/instruction-dataset
DATASETS.append("HuggingFaceH4/instruction-dataset")

# https://huggingface.co/datasets/mrm8488/unnatural-instructions-full
DATASETS.append("mrm8488/unnatural-instructions-full")

# https://huggingface.co/datasets/HuggingFaceH4/instruction-dataset
DATASETS.append("HuggingFaceH4/instruction-dataset")

# https://huggingface.co/datasets/HuggingFaceH4/helpful_instructions
DATASETS.append("HuggingFaceH4/helpful_instructions")


# MATH

# https://huggingface.co/datasets/qwedsacf/grade-school-math-instructions
DATASETS.append("qwedsacf/grade-school-math-instructions")


# Code

# https://huggingface.co/datasets/bugdaryan/sql-create-context-instruction
DATASETS.append("bugdaryan/sql-create-context-instruction")

# https://huggingface.co/datasets/nickrosh/Evol-Instruct-Code-80k-v1
DATASETS.append("nickrosh/Evol-Instruct-Code-80k-v1")

# https://huggingface.co/datasets/m-a-p/CodeFeedback-Filtered-Instruction
DATASETS.append("m-a-p/CodeFeedback-Filtered-Instruction")

# https://huggingface.co/datasets/bigcode/the-stack
DATASETS.append("bigcode/the-stack")


# NSFW

# https://huggingface.co/datasets/Fizzarolli/lima-nsfw-spin
DATASETS.append("Fizzarolli/lima-nsfw-spin")

# https://huggingface.co/datasets/Maxx0/sexting-nsfw-adultconten
DATASETS.append("Maxx0/sexting-nsfw-adultconten")

# https://huggingface.co/datasets/CanadianGamer/female-nsfwcaht
DATASETS.append("CanadianGamer/female-nsfwcaht")

# https://huggingface.co/datasets/utsavm/NSFW_Chat_Dataset
DATASETS.append("utsavm/NSFW_Chat_Dataset")

# https://huggingface.co/datasets/baiango/NSFW-dirty-talk
DATASETS.append("baiango/NSFW-dirty-talk")


#####################################################################################################


# Remove duplicate dataset entries while preserving order


def _deduplicate(repos: List[str]) -> Tuple[List[str], List[str]]:
    seen = set()
    unique: List[str] = []
    duplicates: List[str] = []
    for repo in repos:
        key = repo.strip()
        if key in seen:
            duplicates.append(repo)
        else:
            seen.add(key)
            unique.append(repo)
    return unique, duplicates


DATASETS, _DATASET_DUPLICATES = _deduplicate(DATASETS)


#####################################################################################################

# Explicit list of models to download
MODELS = []

# https://huggingface.co/deepseek-ai/collections
# https://huggingface.co/dphn/models
# https://huggingface.co/mistralai/models
# https://huggingface.co/Qwen
# https://huggingface.co/google
# https://huggingface.co/meta-llama
# https://huggingface.co/openai
# https://huggingface.co/microsoft

# https://huggingface.co/datasets?sort=trending&search=instruct


# “Normal” Models - Example: A raw GPT or LLaMA model before fine-tuning.
# These are base models trained on large datasets without special alignment for instructions or safety.
# They are good at predicting the next token and generating text, but:
# - They don’t inherently follow instructions well.
# - They may produce unsafe or off-topic outputs because they lack strong guardrails.

# https://huggingface.co/dphn/Dolphin3.0-Llama3.2-1B
MODELS.append("dphn/Dolphin3.0-Llama3.2-1B")

# https://huggingface.co/dphn/Dolphin3.0-Llama3.2-3B
MODELS.append("dphn/Dolphin3.0-Llama3.2-3B")

# https://huggingface.co/Qwen/Qwen3-0.6B
MODELS.append("Qwen/Qwen3-0.6B")

# https://huggingface.co/Qwen/Qwen3-0.6B-GPTQ-Int8
MODELS.append("Qwen/Qwen3-0.6B-GPTQ-Int8")

# https://huggingface.co/Qwen/Qwen3-1.7B
MODELS.append("Qwen/Qwen3-1.7B")

# https://huggingface.co/Qwen/Qwen3-1.7B-GPTQ-Int8
MODELS.append("Qwen/Qwen3-1.7B-GPTQ-Int8")

# https://huggingface.co/Qwen/Qwen3-4B
MODELS.append("Qwen/Qwen3-4B")

# https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507
MODELS.append("Qwen/Qwen3-4B-Thinking-2507")

# https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507-FP8
MODELS.append("Qwen/Qwen3-4B-Thinking-2507-FP8")

# https://huggingface.co/google/gemma-2b
MODELS.append("google/gemma-2b")

# https://huggingface.co/google/gemma-7b
MODELS.append("google/gemma-7b")

# https://huggingface.co/meta-llama/Llama-3.2-1B
MODELS.append("meta-llama/Llama-3.2-1B")

# https://huggingface.co/meta-llama/Llama-3.2-3B
MODELS.append("meta-llama/Llama-3.2-3B")


# Instruct Models Example: GPT-3.5-Instruct, LLaMA-2-Instruct.
# These are instruction-tuned models.
# They are optimized for conversational and task-oriented use cases.
# Fine-tuned using supervised learning and reinforcement learning from human feedback (RLHF) to:
# - Follow user instructions more reliably.
# - Provide helpful, relevant, and coherent answers.

# https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
MODELS.append("meta-llama/Llama-3.2-1B-Instruct")

# https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
MODELS.append("meta-llama/Llama-3.2-3B-Instruct")

# https://huggingface.co/google/gemma-2b-it
MODELS.append("google/gemma-2b-it")

# https://huggingface.co/google/gemma-7b-it-pytorch
MODELS.append("google/gemma-7b-it-pytorch")


# Guarded Models - Example: Anthropic’s Claude with Constitutional AI principles, or OpenAI models with moderation baked in.
# They are designed for enterprise or regulated environments where compliance and safety are critical.
# These models include additional safety layers or guardrails.
# Often integrated with:
# - Content filtering (to block harmful or disallowed outputs).
# - Policy enforcement (e.g., avoiding disallowed topics).


#Specials

# Embedding Models - Example: OpenAI’s text-embedding-ada-002, or Sentence Transformers.
# These models are optimized to convert text into high-dimensional vector representations (embeddings).
# They are used for tasks like semantic search, clustering, and recommendation systems.
# They capture semantic meaning and relationships between texts rather than generating text.

# https://huggingface.co/google/embeddinggemma-300m
MODELS.append("google/embeddinggemma-300m")

# https://huggingface.co/google/embeddinggemma-300m-qat-q4_0-unquantized
MODELS.append("google/embeddinggemma-300m-qat-q4_0-unquantized")

# https://huggingface.co/google/embeddinggemma-300m-qat-q8_0-unquantized
MODELS.append("google/embeddinggemma-300m-qat-q8_0-unquantized")

# https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
MODELS.append("Qwen/Qwen3-Embedding-0.6B")

# https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF
MODELS.append("Qwen/Qwen3-Embedding-0.6B-GGUF")

# https://huggingface.co/Qwen/Qwen3-Embedding-4B
MODELS.append("Qwen/Qwen3-Embedding-4B")

# https://huggingface.co/Qwen/Qwen3-Embedding-8B
MODELS.append("Qwen/Qwen3-Embedding-8B")


# Code Models - Example: OpenAI’s code-davinci-002, or Meta’s Code Llama.
# These models are specifically trained on programming languages and code-related tasks.
# They assist with code generation, completion, and understanding.
# They are fine-tuned to understand syntax, semantics, and common coding patterns.

# https://huggingface.co/google/codegemma-2b
MODELS.append("google/codegemma-2b")

# https://huggingface.co/google/codegemma-2b-GGUF"
MODELS.append("google/codegemma-2b-GGUF")

# https://huggingface.co/google/codegemma-7b-it
MODELS.append("google/codegemma-7b-it")

# https://huggingface.co/google/codegemma-7b-pytorch
MODELS.append("google/codegemma-7b-pytorch")


#####################################################################################################

# Media Models - Example: Meta’s Make-A-Video, or OpenAI’s DALL·E.
MEDIAMODELS = []

## Text-to-Video
# https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B
MEDIAMODELS.append("Wan-AI/Wan2.2-TI2V-5B")

# https://huggingface.co/enhanceaiteam/Flux-uncensored?not-for-all-audiences=true
MEDIAMODELS.append("enhanceaiteam/Flux-uncensored")


## Text-To-Image
# https://huggingface.co/enhanceaiteam/Flux-Uncensored-V2
MEDIAMODELS.append("enhanceaiteam/Flux-Uncensored-V2")  

# https://huggingface.co/second-state/FLUX.1-dev-GGUF
MEDIAMODELS.append("second-state/FLUX.1-dev-GGUF")


# Text-to-speech
# https://huggingface.co/microsoft/VibeVoice-1.5B
MEDIAMODELS.append("microsoft/VibeVoice-1.5B")


# Image-to-Video
# https://huggingface.co/Phr00t/WAN2.2-14B-Rapid-AllInOne
MEDIAMODELS.append("Phr00t/WAN2.2-14B-Rapid-AllInOne")


## Image-to-Image
# https://huggingface.co/nvidia/ChronoEdit-14B-Diffusers-Upscaler-Lora
MEDIAMODELS.append("nvidia/ChronoEdit-14B-Diffusers-Upscaler-Lora")

# https://huggingface.co/dx8152/Qwen-Edit-2509-Multiple-angles
MEDIAMODELS.append("dx8152/Qwen-Edit-2509-Multiple-angles")

# https://huggingface.co/dx8152/Qwen-Image-Edit-2509-Light_restoration
MEDIAMODELS.append("dx8152/Qwen-Image-Edit-2509-Light_restoration")


# Video-to-Video
# https://huggingface.co/Wan-AI/Wan2.2-Animate-14B
MEDIAMODELS.append("Wan-AI/Wan2.2-Animate-14B")



#####################################################################################################
#####################################################################################################
# Filtering step

# Remove duplicate entries from MODELS while preserving order (ignore leading/trailing whitespace)
MODELS, _MODEL_DUPLICATES = _deduplicate(MODELS)
MEDIAMODELS, _MEDIAMODEL_DUPLICATES = _deduplicate(MEDIAMODELS)

_config = PathConfig.from_env()
HF_CACHE_DIR = _config.cache_dir
MODEL_ROOT = _config.model_root
DATA_ROOT = _config.data_root
LOG_DIR = _config.log_dir
MEDIA_ROOT = _config.media_root

# Expose the names used by the rest of this script for backward compatibility
CACHE_DIR = HF_CACHE_DIR
DEST_ROOT = MODEL_ROOT
DATA_DEST_ROOT = DATA_ROOT
# Use the same central cache for datasets and models
DATA_CACHE_DIR = CACHE_DIR



def _dereference_windows_symlinks(target: Path, cache_dir: Path):
    """Replace symlink/reparse-point files under `target` with real copies from the cache."""
    if not target.exists():
        return
    for p in sorted(target.rglob('*'), key=lambda x: len(str(x).split(os.sep)), reverse=True):
        try:
            if not p.exists() and not p.is_symlink():
                continue
            if not p.is_symlink():
                continue
            try:
                rel = os.readlink(str(p))
            except Exception:
                continue

            resolved = os.path.normpath(os.path.join(str(p.parent), rel))

            source = None
            if os.path.exists(resolved):
                source = resolved
            else:
                leaf = os.path.basename(rel)
                matches = glob.glob(str(cache_dir / '**' / leaf), recursive=True)
                if matches:
                    source = matches[0]

            if source and os.path.exists(source):
                tmp = str(p) + '.copytmp'
                if p.is_dir():
                    shutil.copytree(source, tmp)
                    try:
                        os.rmdir(str(p))
                    except Exception:
                        try:
                            p.unlink()
                        except Exception:
                            pass
                    shutil.move(tmp, str(p))
                else:
                    shutil.copy2(source, tmp)
                    try:
                        p.unlink()
                    except Exception:
                        try:
                            os.remove(str(p))
                        except Exception:
                            pass
                    shutil.move(tmp, str(p))
        except Exception as error:
            _log_warning('symlink_replace_failed', path=str(p), error=str(error))
            print(f"Warning: failed to replace symlink {p}: {error}")


def _find_symlink_blobs(target: Path, cache_dir: Path) -> List[Tuple[Path, str, int]]:
    results: List[Tuple[Path, str, int]] = []
    for p in target.rglob('*'):
        try:
            if not p.is_symlink():
                continue
            try:
                rel = os.readlink(str(p))
            except Exception:
                continue
            resolved = os.path.normpath(os.path.join(str(p.parent), rel))
            source = ''
            size = 0
            if os.path.exists(resolved):
                source = resolved
            else:
                leaf = os.path.basename(rel)
                matches = glob.glob(str(cache_dir / '**' / leaf), recursive=True)
                if matches:
                    source = matches[0]

            if source and os.path.exists(source):
                try:
                    if os.path.isdir(source):
                        total = 0
                        for root, _, files in os.walk(source):
                            for f in files:
                                total += os.path.getsize(os.path.join(root, f))
                        size = total
                    else:
                        size = os.path.getsize(source)
                except Exception:
                    size = 0
                results.append((p, source, size))
        except Exception:
            continue
    return results


def _chunk_iterable(items, size=4):
    it = iter(items)
    while True:
        chunk = list(itertools.islice(it, size))
        if not chunk:
            break
        yield chunk


def _resolve_default_python() -> str:
    candidates = ['python', 'python3']
    for cand in candidates:
        resolved = which(cand)
        if resolved:
            return resolved
    return sys.executable


def _ensure_login(executable: str) -> None:
    cmd = [executable, '-m', 'huggingface_hub.cli.login']
    print('Invoking Hugging Face login helper:', ' '.join(cmd))
    _log_info('login_helper_invoked', executable=executable)
    subprocess.run(cmd, check=False)


class DownloadStats:
    def __init__(self):
        self.lock = threading.Lock()
        self.success = []
        self.failed = []
        self.skipped = []

    def add_success(self, item):
        with self.lock:
            self.success.append(item)

    def add_failed(self, item, error):
        with self.lock:
            self.failed.append((item, error))

    def add_skipped(self, item):
        with self.lock:
            self.skipped.append(item)

    def summarize(self):
        print('\nSummary:')
        print('  Success:', len(self.success))
        print('  Failed :', len(self.failed))
        print('  Skipped:', len(self.skipped))
        _log_info(
            'download_summary',
            success=len(self.success),
            failed=len(self.failed),
            skipped=len(self.skipped),
        )
        if self.failed:
            print('\nFailures:')
            for item, error in self.failed:
                print(' -', item, ':', error)
            _log_error('download_failures', failures=[{'repo_id': item, 'error': str(error)} for item, error in self.failed])


def download_repo(
    repo_id: str,
    stats: DownloadStats,
    dest_root: Path,
    cache_dir: Path,
    yes: bool,
    no_spinner: bool,
    repo_type: str = 'model',
    cache_only: bool = False,
) -> None:
    destination = dest_root / repo_id.replace('/', '_')
    cache_snapshot_dir = _config.cache_snapshot_dir(repo_id, repo_type)

    force_download = False
    if cache_only:
        cache_state, markers = _classify_cache_state(cache_snapshot_dir)
        if cache_state == 'complete':
            print('Cached snapshot already present, skipping:', cache_snapshot_dir)
            stats.add_skipped(repo_id)
            _log_info(
                'download_skipped_existing_cache',
                repo_id=repo_id,
                cache_path=str(cache_snapshot_dir),
                repo_type=repo_type,
            )
            return
        if cache_state == 'incomplete':
            print('Cached snapshot appears incomplete, retrying download:', cache_snapshot_dir)
            _log_warning(
                'download_incomplete_cache_detected',
                repo_id=repo_id,
                cache_path=str(cache_snapshot_dir),
                repo_type=repo_type,
                markers=[str(marker) for marker in markers],
            )
            if markers:
                removed = _cleanup_incomplete_markers(markers)
                if removed:
                    _log_info(
                        'download_incomplete_cache_cleaned',
                        repo_id=repo_id,
                        repo_type=repo_type,
                        removed=removed,
                    )
            force_download = True
    else:
        if destination.exists():
            print('Target already exists, skipping:', destination)
            stats.add_skipped(repo_id)
            _log_info('download_skipped_existing', repo_id=repo_id, destination=str(destination), repo_type=repo_type)
            return

    auth_warning = False
    if 'HUGGINGFACE_HUB_TOKEN' not in os.environ:
        auth_warning = True

    target_path = cache_snapshot_dir if cache_only else destination
    _log_info(
        'download_started',
        repo_id=repo_id,
        destination=str(target_path),
        cache_only=cache_only,
        repo_type=repo_type,
        force_download=force_download,
    )

    try:
        kwargs = {
            'repo_id': repo_id,
            'revision': None,
            'cache_dir': str(cache_dir),
            'allow_patterns': None,
            'ignore_patterns': None,
            'repo_type': repo_type,
            'resume_download': True,
        }
        if not cache_only:
            kwargs['local_dir'] = str(destination)
            kwargs['local_dir_use_symlinks'] = False
        if force_download:
            kwargs['force_download'] = True

        if tqdm and not no_spinner:
            with tqdm(desc=f'Downloading {repo_id}', unit='file') as progress:
                snapshot_path = snapshot_download(**kwargs)
                progress.update(1)
        else:
            snapshot_path = snapshot_download(**kwargs)

        resolved_path = Path(snapshot_path) if cache_only else destination
        print('SUCCESS:', repo_id, '->', resolved_path)
        _log_info(
            'download_completed',
            repo_id=repo_id,
            destination=str(resolved_path),
            repo_type=repo_type,
            cache_only=cache_only,
        )
        stats.add_success(repo_id)
    except Exception as error:
        stats.add_failed(repo_id, error)
        print('FAILED', repo_id, ':', error)
        _log_error(
            'download_failed',
            repo_id=repo_id,
            error=str(error),
            repo_type=repo_type,
            cache_only=cache_only,
        )
        if auth_warning:
            print('Warning: no HUGGINGFACE_HUB_TOKEN detected; download may require authentication.')
            _log_warning('download_auth_warning', repo_id=repo_id)


def download_dataset(repo_id: str, stats: DownloadStats, dest_root: Path, cache_dir: Path, yes: bool, no_spinner: bool, cache_only: bool):
    download_repo(repo_id, stats, dest_root, cache_dir, yes, no_spinner, repo_type='dataset', cache_only=cache_only)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Download selected Hugging Face models/datasets to local storage.')
    parser.add_argument('--datasets', action='store_true', help='Download datasets list instead of models.')
    parser.add_argument('--models', action='store_true', help='Download model list explicitly.')
    parser.add_argument('--media', action='store_true', help='Download media model list.')
    parser.add_argument('--all', action='store_true', help='Download models, datasets, and media models.')
    parser.add_argument('--mirror-dirs', action='store_true', help='Mirror snapshots into destination folders (legacy behaviour). Default is cache-only downloads.')
    parser.add_argument('--yes', action='store_true', help='Skip confirmations.')
    parser.add_argument('--no-spinner', action='store_true', help='Disable tqdm progress display.')
    parser.add_argument('--max-workers', type=int, default=2, help='Number of parallel download threads.')
    parser.add_argument('--python-exe', type=str, default='', help='Python executable to use for auth helpers.')
    args = parser.parse_args(argv)

    global LOGGER
    LOGGER = get_csv_logger('download_runner_cli')
    _log_info('download_runner_invoked', argv=list(argv) if argv is not None else [])
    _log_info('log_directory_resolved', path=str(LOG_DIR))

    try:
        if _DATASET_DUPLICATES:
            print('Removed duplicate DATASETS:', ', '.join(_DATASET_DUPLICATES))
            _log_warning('duplicate_datasets_removed', duplicates=_DATASET_DUPLICATES)
        if _MODEL_DUPLICATES:
            print('Removed duplicate MODELS:', ', '.join(_MODEL_DUPLICATES))
            _log_warning('duplicate_models_removed', duplicates=_MODEL_DUPLICATES)
        if _MEDIAMODEL_DUPLICATES:
            print('Removed duplicate MEDIAMODELS:', ', '.join(_MEDIAMODEL_DUPLICATES))
            _log_warning('duplicate_media_models_removed', duplicates=_MEDIAMODEL_DUPLICATES)

        if not any((args.datasets, args.models, args.media)):
            print('No mode specified; defaulting to --models')
            args.models = True
        if args.all:
            args.datasets = args.models = args.media = True

        python_exe = args.python_exe or _resolve_default_python()
        _log_info('download_modes_resolved', datasets=bool(args.datasets), models=bool(args.models), media=bool(args.media), max_workers=args.max_workers)

        token_path = Path.home() / '.cache' / 'huggingface' / 'token'
        if not token_path.exists() and not os.environ.get('HUGGINGFACE_HUB_TOKEN'):
            print('No Hugging Face token detected; running login helper.')
            _ensure_login(python_exe)

        stats = DownloadStats()

        cache_only = not args.mirror_dirs
        if cache_only:
            print('Cache-only mode active (default). Use --mirror-dirs to mirror into destination folders.')

        if args.datasets:
            dataset_target_desc = DATA_CACHE_DIR if cache_only else DATA_DEST_ROOT
            print('Downloading datasets to', dataset_target_desc)
            _log_info('dataset_download_started', count=len(DATASETS), destination=str(dataset_target_desc), cache_only=cache_only)
            for chunk in _chunk_iterable(DATASETS, size=max(1, args.max_workers)):
                threads = []
                for repo_id in chunk:
                    thread = threading.Thread(target=download_dataset, args=(repo_id, stats, DATA_DEST_ROOT, DATA_CACHE_DIR, args.yes, args.no_spinner, cache_only))
                    thread.start()
                    threads.append(thread)
                for thread in threads:
                    thread.join()

        if args.models:
            model_target_desc = CACHE_DIR if cache_only else DEST_ROOT
            print('Downloading models to', model_target_desc)
            _log_info('model_download_started', count=len(MODELS), destination=str(model_target_desc), cache_only=cache_only)
            for chunk in _chunk_iterable(MODELS, size=max(1, args.max_workers)):
                threads = []
                for repo_id in chunk:
                    thread = threading.Thread(target=download_repo, args=(repo_id, stats, DEST_ROOT, CACHE_DIR, args.yes, args.no_spinner, 'model', cache_only))
                    thread.start()
                    threads.append(thread)
                for thread in threads:
                    thread.join()

        if args.media:
            media_target_desc = CACHE_DIR if cache_only else MEDIA_ROOT
            if not cache_only:
                MEDIA_ROOT.mkdir(parents=True, exist_ok=True)
            print('Downloading media models to', media_target_desc)
            _log_info('media_download_started', count=len(MEDIAMODELS), destination=str(media_target_desc), cache_only=cache_only)
            for chunk in _chunk_iterable(MEDIAMODELS, size=max(1, args.max_workers)):
                threads = []
                for repo_id in chunk:
                    thread = threading.Thread(target=download_repo, args=(repo_id, stats, MEDIA_ROOT, CACHE_DIR, args.yes, args.no_spinner, 'model', cache_only))
                    thread.start()
                    threads.append(thread)
                for thread in threads:
                    thread.join()

        stats.summarize()

        if not cache_only:
            print('\nSymlink audit for models:')
            symlinks = _find_symlink_blobs(DEST_ROOT, CACHE_DIR)
            for path, source, size in symlinks:
                print(f'  {path} -> {source} ({size} bytes)')
            _log_info('model_symlink_audit', count=len(symlinks))

            print('\nSymlink audit for datasets:')
            symlinks = _find_symlink_blobs(DATA_DEST_ROOT, DATA_CACHE_DIR)
            for path, source, size in symlinks:
                print(f'  {path} -> {source} ({size} bytes)')
            _log_info('dataset_symlink_audit', count=len(symlinks))

            if args.media:
                print('\nSymlink audit for media models:')
                symlinks = _find_symlink_blobs(MEDIA_ROOT, CACHE_DIR)
                for path, source, size in symlinks:
                    print(f'  {path} -> {source} ({size} bytes)')
                _log_info('media_symlink_audit', count=len(symlinks))

            print('\nDereferencing symlinks...')
            _dereference_windows_symlinks(DEST_ROOT, CACHE_DIR)
            _dereference_windows_symlinks(DATA_DEST_ROOT, DATA_CACHE_DIR)
            if args.media:
                _dereference_windows_symlinks(MEDIA_ROOT, CACHE_DIR)
            _log_info('symlinks_dereferenced')

        print('Download runner complete.')
        _log_info('download_runner_completed')
        return 0
    except Exception as exc:  # pragma: no cover - defensive guard
        _log_exception('download_runner_failed', error=str(exc))
        print('Download runner failed:', exc)
        return 1
    finally:
        if LOGGER:
            LOGGER.close()
            LOGGER = None


if __name__ == '__main__':
    raise SystemExit(main())