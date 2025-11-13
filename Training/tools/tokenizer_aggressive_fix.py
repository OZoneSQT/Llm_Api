"""
Aggressive tokenizer repair tool.

Provide --repo pointing at a local model snapshot (defaults to PathConfig model_root / dphn_Dolphin3.0-Llama3.2-1B).
- Tries multiple strategies to make a transformers-compatible tokenizer:
    1. Load tokenizers.Tokenizer.from_file(tokenizer.json) and wrap in PreTrainedTokenizerFast
    2. If vocab.json + merges.txt exist, try ByteLevelBPETokenizer or BPE
    3. Ensure minimal special tokens and model_max_length
    4. Save tokenizer back to repo with save_pretrained and validate AutoTokenizer.from_pretrained
"""
import argparse
import sys
import json
from pathlib import Path

from Training.tools.paths import resolve_model_path

parser = argparse.ArgumentParser(description='Aggressive tokenizer repair helper')
parser.add_argument('--repo', help='Path to local model snapshot', default=None)
args = parser.parse_args()

default_repo = resolve_model_path('dphn/Dolphin3.0-Llama3.2-1B')
repo = Path(args.repo).expanduser().resolve() if args.repo else default_repo

print("Target repo:", repo)
if not repo.exists():
    print("Repo path does not exist:", repo)
    sys.exit(2)

# list files
print("Files in repo:")
for p in sorted(repo.iterdir()):
    print(" -", p.name)

# try to import tokenizers
try:
    from tokenizers import Tokenizer as TK
    from tokenizers import normalizers, pre_tokenizers
    print("tokenizers library available")
except Exception as e:
    TK = None
    print("tokenizers library not available:", e)

from transformers import PreTrainedTokenizerFast
from Training.tools.adapter_utils import load_tokenizer

# helper to try wrapping tokenizer.json
def try_wrap_tokenizer_json(p):
    print("Trying Tokenizer.from_file on:", p)
    try:
        tok = TK.from_file(str(p))
        print("Loaded tokenizers.Tokenizer OK; wrapping into PreTrainedTokenizerFast...")
        t = PreTrainedTokenizerFast(tokenizer_object=tok)
        # ensure basic special tokens
        changed = False
        if t.pad_token is None:
            t.add_special_tokens({"pad_token": "<|pad|>"})
            changed = True
        if getattr(t, 'eos_token', None) is None:
            try:
                t.add_special_tokens({"eos_token": "</s>"})
                changed = True
            except Exception:
                pass
        if getattr(t, 'model_max_length', None) is None:
            # try to read config
            cfgf = repo / 'config.json'
            mlen = None
            if cfgf.exists():
                try:
                    cfg = json.loads(cfgf.read_text(encoding='utf-8'))
                    mlen = cfg.get('max_sequence_length') or cfg.get('max_position_embeddings') or cfg.get('n_positions')
                except Exception:
                    pass
            if mlen is None:
                mlen = 2048
            t.model_max_length = int(mlen)
            changed = True
        # save out safely to "fixed_tokenizer"
        outdir = repo / 'fixed_tokenizer'
        outdir.mkdir(exist_ok=True)
        print("Saving PreTrainedTokenizerFast to:", outdir)
        t.save_pretrained(str(outdir))
        print("Saved tokenizer files:", list(outdir.iterdir()))
        # validate
        try:
            at = load_tokenizer(str(outdir), local_files_only=True)
            print("AutoTokenizer.from_pretrained OK for fixed tokenizer")
            return outdir
        except Exception as e:
            print("AutoTokenizer failed to load fixed tokenizer:", e)
            return None
    except Exception as e:
        print("Tokenizer.from_file failed:", e)
        return None

# try 1: tokenizer.json
tokenizer_json = repo / 'tokenizer.json'
fixed = None
if tokenizer_json.exists() and TK is not None:
    fixed = try_wrap_tokenizer_json(tokenizer_json)
    if fixed is None:
        # attempt to parse tokenizer.json as plain JSON and inspect 'model' wrapper keys
        try:
            print("Attempting to parse tokenizer.json as JSON to inspect model wrapper...")
            raw = json.loads(tokenizer_json.read_text(encoding='utf-8'))
            if 'model' in raw:
                m = raw['model']
                print("tokenizer.json top-level 'model' keys:", list(m.keys())[:20])
                # try to print the model type if present
                ttype = m.get('type') or m.get('model_type') or m.get('model')
                print("model.type/identifiers:", ttype)
                # If model is BPE and contains vocab+merges, try to write them to files and load via ByteLevelBPETokenizer
                if (ttype == 'BPE' or (isinstance(ttype, str) and 'bpe' in ttype.lower())) and 'vocab' in m and 'merges' in m:
                    try:
                        print("Detected BPE model with embedded vocab+merges; attempting to extract files and load ByteLevelBPETokenizer")
                        vocab_obj = m['vocab']
                        merges_obj = m['merges']
                        vocab_file = repo / 'extracted_vocab.json'
                        merges_file = repo / 'extracted_merges.txt'
                        # write vocab
                        if isinstance(vocab_obj, dict):
                            vocab_file.write_text(json.dumps(vocab_obj, ensure_ascii=False), encoding='utf-8')
                        else:
                            # if vocab is list, convert to mapping token->idx
                            if isinstance(vocab_obj, list):
                                mapping = {tok: i for i, tok in enumerate(vocab_obj)}
                                vocab_file.write_text(json.dumps(mapping, ensure_ascii=False), encoding='utf-8')
                            else:
                                raise RuntimeError('Unsupported vocab format in tokenizer.json')
                        # write merges
                        if isinstance(merges_obj, list):
                            merges_file.write_text('\n'.join([m if isinstance(m, str) else ' '.join(m) for m in merges_obj]), encoding='utf-8')
                        elif isinstance(merges_obj, str):
                            merges_file.write_text(merges_obj, encoding='utf-8')
                        else:
                            raise RuntimeError('Unsupported merges format in tokenizer.json')
                        print('Wrote extracted_vocab.json and extracted_merges.txt')
                        # attempt to load via ByteLevelBPETokenizer
                        try:
                            from tokenizers.implementations import ByteLevelBPETokenizer
                            b = ByteLevelBPETokenizer(str(vocab_file), str(merges_file))
                            t = PreTrainedTokenizerFast(tokenizer_object=b)
                            outdir = repo / 'fixed_tokenizer'
                            outdir.mkdir(exist_ok=True)
                            t.save_pretrained(str(outdir))
                            print('Saved extracted BPE tokenizer to', outdir)
                            try:
                                at = load_tokenizer(str(outdir), local_files_only=True)
                                print('AutoTokenizer OK for extracted BPE tokenizer')
                                fixed = outdir
                            except Exception as e:
                                print('AutoTokenizer failed for extracted BPE tokenizer:', e)
                        except Exception as e:
                            print('ByteLevelBPETokenizer import/load failed:', e)
                    except Exception as e:
                        print('Failed to extract/load embedded vocab/merges:', e)
            else:
                print("No 'model' key found at top level of tokenizer.json; keys:", list(raw.keys())[:50])
        except Exception as e:
            print("Parsing tokenizer.json as JSON failed:", e)

# try 2: tokenizer.model
if fixed is None:
    tokenizer_model = repo / 'tokenizer.model'
    if tokenizer_model.exists() and TK is not None:
        fixed = try_wrap_tokenizer_json(tokenizer_model)

# try 3: vocab.json + merges.txt
if fixed is None:
    vocab = repo / 'vocab.json'
    merges = repo / 'merges.txt'
    if vocab.exists() and merges.exists():
        print("Found vocab.json + merges.txt; attempting ByteLevelBPETokenizer load (if available)")
        try:
            from tokenizers.implementations import ByteLevelBPETokenizer
            b = ByteLevelBPETokenizer(str(vocab), str(merges))
            t = PreTrainedTokenizerFast(tokenizer_object=b)
            outdir = repo / 'fixed_tokenizer'
            outdir.mkdir(exist_ok=True)
            t.save_pretrained(str(outdir))
            print("Saved BPE-based tokenizer to", outdir)
            try:
                at = load_tokenizer(str(outdir), local_files_only=True)
                print("AutoTokenizer OK for BPE-based tokenizer")
                fixed = outdir
            except Exception as e:
                print("AutoTokenizer failed for BPE tokenizer:", e)
        except Exception as e:
            print("ByteLevelBPETokenizer path failed:", e)

# final check
if fixed is not None:
    print("SUCCESS: created fixed tokenizer at:", fixed)
    # copy files into repo root so AutoTokenizer.from_pretrained(repo) can pick them up
    for f in fixed.iterdir():
        dst = repo / f.name
        try:
            dst.write_bytes(f.read_bytes())
            print("Wrote", dst)
        except Exception as e:
            print("Failed to write", dst, e)
    # final validation
    try:
        at = load_tokenizer(str(repo), local_files_only=True)
        print("Final AutoTokenizer.from_pretrained(repo) succeeded")
    except Exception as e:
        print("Final validation still failed:", e)
else:
    print("Could not produce a fixed tokenizer via available strategies. Listing repo files again:")
    for p in sorted(repo.iterdir()):
        print(" -", p.name)

print("Done.")
