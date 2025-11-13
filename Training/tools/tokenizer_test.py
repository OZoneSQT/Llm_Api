import argparse
from pathlib import Path

from transformers import AutoTokenizer

from Training.tools.paths import resolve_model_path
from Training.tools.adapter_utils import load_tokenizer


parser = argparse.ArgumentParser(description='Tokenizer load smoke test')
parser.add_argument('--repo', help='Path to local model snapshot', default=None)
args = parser.parse_args()

default_repo = resolve_model_path('dphn/Dolphin3.0-Llama3.2-1B')
p = Path(args.repo).expanduser().resolve() if args.repo else default_repo
print('Exists:', p.exists())
try:
    t = load_tokenizer(str(p), local_files_only=True, trust_remote_code=True, use_fast=False)
    print('Loaded tokenizer (use_fast=False):', type(t), 'pad_token', t.pad_token, 'vocab_size', getattr(t,'vocab_size', None))
except Exception as e:
    print('Failed use_fast=False:', repr(e))
try:
    t2 = load_tokenizer(str(p), local_files_only=True, trust_remote_code=True)
    print('Loaded tokenizer (fast):', type(t2), 'pad_token', t2.pad_token, 'vocab_size', getattr(t2,'vocab_size', None))
except Exception as e:
    print('Failed fast:', repr(e))
