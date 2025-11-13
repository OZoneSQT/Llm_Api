"""Create a `fixed_tokenizer` snapshot for a model folder.

Usage:
    # Example: use configured MODEL_ROOT (see Training/domain/path_config.py)
    python Training/tools/make_fixed_tokenizer.py --model-path <MODEL_ROOT>\Qwen_Qwen3-1.7B [--force]

This script will attempt to load the tokenizer using safe fallbacks (trust_remote_code and use_fast=False)
and save the tokenizer to <model_path>/fixed_tokenizer. The saved snapshot can be used by the adapter
to avoid repeated tokenizer compatibility workarounds.
"""
from pathlib import Path
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description='Create fixed_tokenizer snapshot for a model folder.')
    parser.add_argument('--model-path', required=True, help='Path to model folder')
    parser.add_argument('--force', action='store_true', help='Overwrite existing fixed_tokenizer')
    args = parser.parse_args()

    mp = Path(args.model_path)
    if not mp.exists():
        print('Model path does not exist:', mp)
        sys.exit(2)

    out = mp / 'fixed_tokenizer'
    if out.exists() and not args.force:
        print('fixed_tokenizer already exists at', out, '- use --force to overwrite')
        sys.exit(0)

    # Try to load tokenizer with progressively stronger fallbacks
    from transformers import AutoTokenizer

    tried = []
    tokenizer = None
    # First try: trust_remote_code + use_fast=False
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(mp), local_files_only=True, trust_remote_code=True, use_fast=False)
        print('Loaded tokenizer with trust_remote_code + use_fast=False')
    except Exception as e:
        tried.append(('trust_remote_code+use_fast=False', str(e)))

    if tokenizer is None:
        # Try without trust_remote_code
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(mp), local_files_only=True, use_fast=False)
            print('Loaded tokenizer with use_fast=False')
        except Exception as e:
            tried.append(('use_fast=False', str(e)))

    if tokenizer is None:
        # Last resort: try with default loader (may call fast tokenizer)
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(mp), local_files_only=True)
            print('Loaded tokenizer with default from_pretrained')
        except Exception as e:
            tried.append(('default', str(e)))

    if tokenizer is None:
        print('Failed to load tokenizer with tried strategies:')
        for k, v in tried:
            print(' -', k, ':', v)
        sys.exit(3)

    # Save snapshot
    try:
        if out.exists():
            import shutil
            shutil.rmtree(out)
        tokenizer.save_pretrained(str(out))
        print('Saved fixed_tokenizer to', out)
    except Exception as e:
        print('Failed to save fixed_tokenizer:', e)
        sys.exit(4)

if __name__ == '__main__':
    main()
