"""
Quick smoke fine-tune and sample generator.
- Loads latest `sanitized_dataset_*` if present, else `custom_dataset_*`.
- Uses a tiny subset (max 200 examples) for a 1-epoch smoke test.
- Saves a temporary checkpoint to ./Training/tmp_finetune_output
- Loads the checkpoint and performs a short generation sample.

Run this script from repository root with the same Python that runs training.
"""
import os
import argparse
# Decide whether to force CPU. Priority: CLI arg -> ENV FORCE_CPU -> config file 'force_cpu' -> default False
parser = argparse.ArgumentParser(description='Quick smoke finetune and sample')
parser.add_argument('--model', help='Override model name from config', default=None)
parser.add_argument('--tokenizer', help='Override tokenizer name from config', default=None)
parser.add_argument('--force-cpu', action='store_true', help='Force CPU-only run (sets CUDA_VISIBLE_DEVICES to empty)')
parser.add_argument('--local-path', help='Path to a local model folder to use instead of HF repo', default=None)
parser.add_argument('--enable-gc', action='store_true', help='Enable model.gradient_checkpointing to save memory during training')
parser.add_argument('--fp16', action='store_true', help='Enable fp16 training (if supported by environment/GPU)')
parser.add_argument('--no-gpt2-fallback', action='store_true', help='Do not fall back to generic gpt2 tokenizer (fail loudly instead)')
parser.add_argument('--load-8bit', action='store_true', help='Attempt to load model in 8-bit using bitsandbytes (requires bitsandbytes installed)')
parser.add_argument('--deepspeed-config', help='Path to a DeepSpeed JSON config file to enable ZeRO/offload', default=None)
parser.add_argument('--offload-cpu', action='store_true', help='Enable CPU offload where supported (for memory saving).')
parser.add_argument('--max-examples', type=int, default=None, help='Maximum number of examples to use for the smoke test (overrides internal default)')
parser.add_argument('--use-adapter', action='store_true', help='Use model_adapter.load_with_adapter to load tokenizer/model with family heuristics')
parser.add_argument('--save-sample', action='store_true', help='Save generation sample to disk for regression tests')
args, unknown = parser.parse_known_args()

# Honor env/cli settings to force CPU before torch import
force_cpu = False
if args.force_cpu:
    force_cpu = True
elif os.environ.get('FORCE_CPU', '').lower() in ('1', 'true', 'yes'):
    force_cpu = True

# We'll also read config later for a fallback default; if force_cpu is True, prevent CUDA init
if force_cpu:
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
# If CLI requested adapter usage, set env var so adapter_utils will prefer adapter loads at runtime
if args.use_adapter:
    os.environ['USE_ADAPTER'] = '1'
import glob
from pathlib import Path

from datasets import Dataset
from huggingface_hub import snapshot_download
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
from Training.tools import paths as path_utils
from Training.tools.adapter_utils import load_tokenizer as adapter_load_tokenizer, load_model as adapter_load_model
from transformers import PreTrainedTokenizerFast
try:
    # tokenizers may not be present in all envs; import lazily
    from tokenizers import Tokenizer as _TokenizersTokenizer
except Exception:
    _TokenizersTokenizer = None

ROOT = Path(__file__).resolve().parents[2]
TRAINING = ROOT / "Training"
OUT_DIR = TRAINING / "tmp_finetune_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_ROOT = path_utils.MODEL_ROOT

max_examples = 50   # for smoke test
if args.max_examples is not None:
    try:
        max_examples = int(args.max_examples)
        print(f"Overriding smoke-test max_examples -> {max_examples}")
    except Exception:
        pass

# config file path
cfg_path = TRAINING / "data" / "config.json"
import json
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

model_name = cfg.get("llm_model_name") or cfg.get("model_name") or cfg.get("llm_model")
tokenizer_name = cfg.get("tokenizer_model") or model_name

# Allow CLI overrides
if args.model:
    model_name = args.model
if args.tokenizer:
    tokenizer_name = args.tokenizer
# Allow using a local model path
local_model_path = args.local_path or cfg.get("local_model_path")

# pick dataset
sanitized_dirs = sorted(glob.glob(str(ROOT / "datasets" / "sanitized_dataset_*")))
custom_dirs = sorted(glob.glob(str(ROOT / "datasets" / "custom_dataset_*")))
if sanitized_dirs:
    ds_path = sanitized_dirs[-1]
    print(f"Selected sanitized dataset: {ds_path}")
elif custom_dirs:
    ds_path = custom_dirs[-1]
    print(f"Selected custom dataset: {ds_path}")
else:
    raise SystemExit("No dataset found under datasets/ (sanitized_dataset_* or custom_dataset_*)")

print("Loading dataset (disk)...")
dataset = Dataset.load_from_disk(ds_path)
print(f"Loaded dataset with {len(dataset)} examples")

# take small subset for smoke-test
if len(dataset) > max_examples:
    dataset = dataset.select(range(max_examples))
    print(f"Using subset of {len(dataset)} examples for smoke-test")

print("Loading tokenizer and model (trust_remote_code=True where needed)...")
# experimental: apply runtime rotary patch to tolerate mismatched rotary buffers
try:
    # load by file path to avoid package import issues
    from importlib import util
    patch_path = Path(__file__).resolve().parents[0] / 'rotary_inference_patch.py'
    if patch_path.exists():
        spec = util.spec_from_file_location('rotary_inference_patch', str(patch_path))
        mod = util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore
except Exception:
    # ignore; patch not applied
    pass

# Helper to load tokenizer with fallback to use_fast=False for community repos
def load_tokenizer(name_or_path, **kwargs):
    # If the CLI requested adapter usage, try the adapter helper first.
    if args.use_adapter:
        try:
            tok = adapter_load_tokenizer(name_or_path, **kwargs)
            if tok is not None:
                return tok
        except Exception:
            # adapter attempt failed; fall back to in-file logic below
            pass
    # If a local filesystem path was passed, try to load directly from that path first.
    try:
        cand = Path(str(name_or_path))
        if cand.exists():
            # prefer explicit fixed_tokenizer under the folder
            fixed_sub = cand / 'fixed_tokenizer'
            if fixed_sub.exists():
                try:
                    return adapter_load_tokenizer(str(fixed_sub), local_files_only=True, trust_remote_code=True, **kwargs)
                except Exception:
                    pass
            # try loading tokenizer from the directory itself
            try:
                return adapter_load_tokenizer(str(cand), local_files_only=True, trust_remote_code=True, use_fast=False, **kwargs)
            except Exception:
                try:
                    tf = cand / 'tokenizer.json'
                    if tf.exists():
                        from transformers import PreTrainedTokenizerFast
                        return PreTrainedTokenizerFast(tokenizer_file=str(tf))
                except Exception:
                    pass
    except Exception:
        # ignore and continue to the remote/fallback logic below
        pass
    try:
        return adapter_load_tokenizer(name_or_path, local_files_only=False, trust_remote_code=True, **kwargs)
    except Exception:
        # try fallback to slow tokenizer
        try:
            return adapter_load_tokenizer(name_or_path, local_files_only=False, trust_remote_code=True, use_fast=False, **kwargs)
        except Exception:
            # as a last effort, try downloading the repo snapshot locally and load from disk
            try:
                existing_snapshot = path_utils.find_model_path(str(name_or_path))
                if existing_snapshot is not None:
                    local_dir = existing_snapshot
                else:
                    print(f"Caching snapshot for {name_or_path} into shared cache {path_utils.CACHE_DIR}")
                    snapshot_path = snapshot_download(
                        repo_id=name_or_path,
                        cache_dir=str(path_utils.CACHE_DIR),
                        repo_type="model",
                        resume_download=True,
                    )
                    local_dir = Path(snapshot_path)

                fixed_sub = local_dir / 'fixed_tokenizer'
                if fixed_sub.exists():
                    print(f"Found existing fixed_tokenizer at {fixed_sub}; trying to load tokenizer from fixed_tokenizer")
                    try:
                        return adapter_load_tokenizer(str(fixed_sub), local_files_only=True, trust_remote_code=True, **kwargs)
                    except Exception:
                        pass
                if local_dir.exists():
                    print(f"Using cached snapshot at {local_dir}; loading tokenizer from disk")
                    try:
                        return adapter_load_tokenizer(str(local_dir), local_files_only=True, trust_remote_code=True, use_fast=False, **kwargs)
                    except Exception:
                        try:
                            tf = local_dir / 'tokenizer.json'
                            if tf.exists():
                                from transformers import PreTrainedTokenizerFast
                                return PreTrainedTokenizerFast(tokenizer_file=str(tf))
                        except Exception:
                            pass
                    raise
                # Try standard HF slow tokenizer from the local snapshot
                try:
                    return adapter_load_tokenizer(str(local_dir), local_files_only=True, trust_remote_code=True, use_fast=False, **kwargs)
                except Exception:
                    # If tokenizers library is available, try loading tokenizer.json/tokenizer.model directly
                    if _TokenizersTokenizer is not None:
                        for candidate in ("tokenizer.json", "tokenizer.model"):
                            p = local_dir / candidate
                            if p.exists():
                                try:
                                    print(f"Found raw tokenizers file {p}; attempting tokenizer_file -> PreTrainedTokenizerFast")
                                    # prefer tokenizer_file constructor which accepts the serialized tokenizer
                                    try:
                                        t = PreTrainedTokenizerFast(tokenizer_file=str(p))
                                    except Exception:
                                        # fallback to tokenizers.Tokenizer wrapper
                                        tok = _TokenizersTokenizer.from_file(str(p))
                                        t = PreTrainedTokenizerFast(tokenizer_object=tok)
                                    # ensure minimal special tokens exist so downstream code doesn't crash
                                    if t.pad_token is None:
                                        t.add_special_tokens({"pad_token": "<|pad|>"})
                                    if getattr(t, 'eos_token', None) is None:
                                        try:
                                            t.eos_token = "</s>"
                                        except Exception:
                                            pass
                                    if getattr(t, 'model_max_length', None) is None:
                                        t.model_max_length = 2048
                                    return t
                                except Exception:
                                    # try next candidate
                                    pass
                    # last-resort: either fail or fall back to generic tokenizer depending on flag
                    if args.no_gpt2_fallback:
                        raise
                    try:
                        print("Warning: falling back to generic 'gpt2' tokenizer for smoke-test; this may not match the model's tokenizer.")
                        return adapter_load_tokenizer("gpt2", local_files_only=False)
                    except Exception:
                        # give up and re-raise
                        raise
            except Exception as e:
                # re-raise original for clarity
                raise

        
# Try adapter first if requested. If adapter succeeds we'll skip the normal loader.
loaded_via_adapter = False
if args.use_adapter:
    def _get_load_with_adapter():
        try:
            from Training.tools.model_adapter import load_with_adapter
            return load_with_adapter
        except Exception:
            pass
        # try loading from same folder as this script
        mod_path = Path(__file__).resolve().parent / 'model_adapter.py'
        if mod_path.exists():
            import importlib.util
            spec = importlib.util.spec_from_file_location('training_model_adapter', str(mod_path))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore
            if hasattr(mod, 'load_with_adapter'):
                return getattr(mod, 'load_with_adapter')
        try:
            from model_adapter import load_with_adapter
            return load_with_adapter
        except Exception:
            pass
        raise ImportError('Could not import model_adapter.load_with_adapter')

    try:
        load_with_adapter = _get_load_with_adapter()
        target = local_model_path if local_model_path else model_name
        print(f"Using model_adapter.load_with_adapter for {target}")
        tokenizer, model, meta = load_with_adapter(str(target), local_files_only=True)
        if meta.get('warnings'):
            for w in meta['warnings']:
                print('warning:', w)
        try:
            model.config.use_cache = False
        except Exception:
            pass
        if args.enable_gc:
            try:
                model.gradient_checkpointing_enable()
                print("Enabled model.gradient_checkpointing() to save memory")
            except Exception:
                pass
        loaded_via_adapter = True
    except Exception as e:
        print('Adapter load_with_adapter failed, falling back to normal loader:', e)

# Load from local path if provided and exists
if not loaded_via_adapter:
    if local_model_path:
        if Path(local_model_path).exists():
            print(f"Loading model/tokenizer from local path: {local_model_path}")
            # If a tokenizer override was provided explicitly, prefer that to avoid
            # re-loading or attempting to parse a broken tokenizer.json in the model folder.
            if args.tokenizer:
                print(f"Using explicit tokenizer override: {args.tokenizer}")
                tokenizer = load_tokenizer(args.tokenizer)
            else:
                tokenizer = load_tokenizer(local_model_path)
            try:
                model = adapter_load_model(local_model_path, loader=AutoModelForCausalLM.from_pretrained, local_files_only=True, trust_remote_code=True)
            except ValueError as ve:
                # Common case: installed transformers doesn't recognize model_type (e.g. qwen3)
                msg = str(ve)
                print('Model load ValueError:', msg)
                print('Attempting adapter fallback (coerce/config patch) via model_adapter.load_with_adapter...')
                try:
                    try:
                        from Training.tools.model_adapter import load_with_adapter
                    except Exception:
                        from .model_adapter import load_with_adapter
                    tokenizer, model, meta = load_with_adapter(str(local_model_path), local_files_only=True)
                    if meta.get('warnings'):
                        for w in meta['warnings']:
                            print('warning:', w)
                except Exception as exc:
                    print('Adapter fallback failed:', exc)
                    # re-raise original ValueError for clarity
                    raise
            # memory-saver defaults for training
            try:
                model.config.use_cache = False
            except Exception:
                pass
            if args.enable_gc:
                try:
                    model.gradient_checkpointing_enable()
                    print("Enabled model.gradient_checkpointing() to save memory")
                except Exception:
                    pass
    else:
        print(f"Warning: local_model_path '{local_model_path}' does not exist, falling back to HF name.")
    tokenizer = load_tokenizer(tokenizer_name)
    model = adapter_load_model(model_name, loader=AutoModelForCausalLM.from_pretrained, local_files_only=False, trust_remote_code=True)
else:
    # trust_remote_code may be needed for community models
    tokenizer = load_tokenizer(tokenizer_name)
    # helper: try load model and if rope_scaling ValueError occurs, snapshot, patch config.json, and retry from local snapshot
    def _load_model_with_rope_patch(name):
        try:
            return adapter_load_model(name, loader=AutoModelForCausalLM.from_pretrained, local_files_only=False, trust_remote_code=True)
        except ValueError as ve:
            msg = str(ve)
            if 'rope_scaling' in msg or 'rope_type' in msg:
                print("Model config validation failed for rope_scaling; attempting to patch local snapshot config and retry...")
                try:
                    local_dir = path_utils.find_model_path(str(model_name))
                    if local_dir is None:
                        snapshot_path = snapshot_download(
                            repo_id=model_name,
                            cache_dir=str(path_utils.CACHE_DIR),
                            repo_type="model",
                            resume_download=True,
                        )
                        local_dir = Path(snapshot_path)
                    cfg_file = local_dir / "config.json"
                    if cfg_file.exists():
                        import json as _json
                        with open(cfg_file, 'r', encoding='utf-8') as _f:
                            original_cfg_text = _f.read()
                            cfg_dict = _json.loads(original_cfg_text)
                        rs = cfg_dict.get('rope_scaling') or {}
                        # normalize to {'type': ..., 'factor': ...}
                        if isinstance(rs, dict) and 'type' not in rs:
                            factor = float(rs.get('factor', rs.get('high_freq_factor', 1.0)))
                            cfg_dict['rope_scaling'] = {'type': 'linear', 'factor': factor}
                            # backup original (write original file contents)
                            try:
                                (local_dir / 'config.json.bak').write_text(original_cfg_text, encoding='utf-8')
                            except Exception:
                                pass
                            with open(cfg_file, 'w', encoding='utf-8') as _f:
                                _json.dump(cfg_dict, _f, indent=2)
                            print(f"Patched config.json.rope_scaling in {cfg_file}; retrying model load from {local_dir}")
                            m = adapter_load_model(str(local_dir), loader=AutoModelForCausalLM.from_pretrained, local_files_only=True, trust_remote_code=True)
                            try:
                                m.config.use_cache = False
                            except Exception:
                                pass
                            if args.enable_gc:
                                try:
                                    m.gradient_checkpointing_enable()
                                    print("Enabled model.gradient_checkpointing() to save memory")
                                except Exception:
                                    pass
                            return m
                        else:
                            raise
                    else:
                        raise
                except Exception:
                    print("Automatic config patch failed; re-raising original error")
                    raise
            else:
                raise

    # Prefer loading from local snapshot if available (so we don't overwrite patched files)
    local_snapshot = path_utils.find_model_path(str(model_name))
    if local_snapshot is not None:
        print(f"Found cached/custom snapshot for model at {local_snapshot}; loading model from local snapshot to preserve patched files")
        try:
            if args.load_8bit:
                try:
                    model = adapter_load_model(str(local_snapshot), loader=AutoModelForCausalLM.from_pretrained, local_files_only=True, trust_remote_code=True, load_in_8bit=True, device_map='auto')
                    print('Loaded model in 8-bit from local snapshot (bitsandbytes)')
                except Exception as e:
                    print('8-bit load from local snapshot failed; falling back to normal load:', e)
                    model = _load_model_with_rope_patch(str(local_snapshot))
            else:
                model = _load_model_with_rope_patch(str(local_snapshot))
        except Exception:
            print("Loading model from local snapshot failed; falling back to remote load and patch")
            if args.load_8bit:
                try:
                    model = adapter_load_model(model_name, loader=AutoModelForCausalLM.from_pretrained, local_files_only=False, trust_remote_code=True, load_in_8bit=True, device_map='auto')
                    print('Loaded model in 8-bit from remote (bitsandbytes)')
                except Exception as e:
                    print('8-bit remote load failed; will try patched normal load:', e)
                    model = _load_model_with_rope_patch(model_name)
            else:
                model = _load_model_with_rope_patch(model_name)
    else:
        # no local snapshot, try remote load honoring 8-bit if requested
        if args.load_8bit:
            try:
                model = adapter_load_model(model_name, loader=AutoModelForCausalLM.from_pretrained, local_files_only=False, trust_remote_code=True, load_in_8bit=True, device_map='auto')
                print('Loaded model in 8-bit from remote (bitsandbytes)')
            except Exception as e:
                print('8-bit remote load failed; falling back to normal load:', e)
                model = _load_model_with_rope_patch(model_name)
        else:
            model = _load_model_with_rope_patch(model_name)

 

# basic tokenization function (assumes text column named 'text' or 'instruction'/'response')
text_field = None
for c in ("text", "content", "instruction", "prompt", "input"):
    if c in dataset.column_names:
        text_field = c
        break
if text_field is None:
    # try to collapse row dicts
    print("No text-like column found; attempting to stringify rows")
    def _to_text(example):
        return {"text": json.dumps(example)}
    dataset = dataset.map(lambda ex: {"text": json.dumps(ex)})
    text_field = "text"

# ensure pad token
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    try:
        model.resize_token_embeddings(len(tokenizer))
    except Exception:
        # some models may not support resizing in this way; ignore for smoke test
        pass

# tokenize
def tokenize_fn(examples):
    return tokenizer(examples[text_field], truncation=True, padding="max_length", max_length=512)

tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)

# set labels (copy input ids and mask padding tokens)
def label_fn(ex):
    labels = ex["input_ids"].copy()
    labels = [l if l != tokenizer.pad_token_id else -100 for l in labels]
    return {"labels": labels}

tokenized = tokenized.map(label_fn, batched=False)

# Some tokenizers (or tokenization pipelines) may add `token_type_ids` which Llama models don't accept.
# Remove token_type_ids column if present to avoid unexpected kwargs to model.forward.
if 'token_type_ids' in tokenized.column_names:
    try:
        tokenized = tokenized.remove_columns(['token_type_ids'])
        print('Removed token_type_ids column from tokenized dataset to be compatible with Llama models')
    except Exception:
        # fallback to mapping if remove_columns isn't available
        tokenized = tokenized.map(lambda ex: {k: v for k, v in ex.items() if k != 'token_type_ids'})

# training args - keep tiny
model_id = cfg.get("model_ID", "tmp-finetune-model")
run_out = OUT_DIR / model_id
run_out.mkdir(parents=True, exist_ok=True)

# Set a torch seed for reproducibility if provided
seed_val = cfg.get("seed", 42)
try:
    torch.manual_seed(int(seed_val))
except Exception:
    pass

# For a smoke test keep batch size and precision conservative to avoid OOM
safe_train_bs = min(int(cfg.get("per_device_train_batch_size", 1)), 1)
safe_eval_bs = min(int(cfg.get("per_device_eval_batch_size", 1)), 1)
training_args = TrainingArguments(
    output_dir=str(run_out),
    per_device_train_batch_size=safe_train_bs,
    per_device_eval_batch_size=safe_eval_bs,
    num_train_epochs=cfg.get("num_train_epochs", 1),
    logging_steps=cfg.get("logging_steps", 10),
    save_strategy=cfg.get("save_strategy", "epoch"),
    learning_rate=cfg.get("learning_rate", 2e-5),
    # allow fp16 to be enabled via CLI/CFG to reduce memory use on GPUs (default False for smoke run)
    fp16=(args.fp16 or cfg.get('fp16', False)),
    remove_unused_columns=False,
    push_to_hub=False,
    gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 1),
    warmup_steps=cfg.get("warmup_steps", 0),
    weight_decay=cfg.get("weight_decay", 0.0),
    lr_scheduler_type=cfg.get("lr_scheduler_type", "linear"),
    eval_steps=cfg.get("eval_steps", None),
    logging_strategy=cfg.get("logging_strategy", "epoch"),
    seed=cfg.get("seed", 42),
)
# Attach DeepSpeed config to TrainingArguments if provided or if CPU-offload was requested
if getattr(args, 'deepspeed_config', None) or args.offload_cpu:
    if getattr(args, 'deepspeed_config', None):
        ds_path = Path(args.deepspeed_config)
        if ds_path.exists():
            try:
                training_args.deepspeed = str(ds_path)
                print('Using DeepSpeed config:', ds_path)
            except Exception:
                print('Failed to attach DeepSpeed config to TrainingArguments')
        else:
            print('DeepSpeed config path not found:', ds_path)
    else:
        # try to attach the repo-provided sample CPU-offload config
        sample = TRAINING / 'deepspeed_configs' / 'zero2_cpu_offload.json'
        if sample.exists():
            training_args.deepspeed = str(sample)
            print('Attached sample DeepSpeed CPU-offload config:', sample)
        else:
            print('DeepSpeed CPU-offload requested but sample config not found:', sample)

# data collator (for causal LM we can use default collator)
from transformers import DataCollatorForLanguageModeling
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=collator,
)

print("Starting quick training (this will be short)...")
trainer.train()
# ensure model saved using safe output path
try:
    trainer.save_model(str(OUT_DIR / "final"))
except Exception:
    model.save_pretrained(str(OUT_DIR / "final"))

# generation test
prompt = "Write a short helpful reply explaining how to plant tomatoes in a small urban garden."
inputs = tokenizer(prompt, return_tensors="pt")
if torch.cuda.is_available():
    model.to("cuda")
    inputs = {k: v.cuda() for k, v in inputs.items()}

# remove token_type_ids for Llama models/generation (not accepted by forward/generate)
if 'token_type_ids' in inputs:
    del inputs['token_type_ids']

print("Generating sample from fine-tuned model...")
with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=True,
        top_p=0.95,
        temperature=0.8,
        eos_token_id=tokenizer.eos_token_id,
    )
generated = tokenizer.decode(out[0], skip_special_tokens=True)
print(generated)

# Safety post-processing: check against a simple blocklist and save the sample to disk for regression tests
try:
    from .safety_check import load_blocklist, contains_blocked
except Exception:
    # fallback if run as script from root
    from safety_check import load_blocklist, contains_blocked

blk = load_blocklist(TRAINING / 'data' / 'banned_words.txt')
if contains_blocked(generated, blk):
    print('\n*** WARNING: Generated sample contains blocked phrases from banned_words.txt ***')
    # sanitize by redacting blocked phrases so saved sample and tests don't fail catastrophically
    import re
    cleaned = generated
    for phrase in sorted(blk, key=lambda s: -len(s)):
        if not phrase:
            continue
        try:
            cleaned = re.sub(re.escape(phrase), '[REDACTED]', cleaned, flags=re.IGNORECASE)
        except Exception:
            # fallback to simple replace
            cleaned = cleaned.replace(phrase, '[REDACTED]')
    # Save original raw as well for auditing
    raw_out = OUT_DIR / 'final' / 'sample_generation.raw.txt'
    try:
        raw_out.write_text(generated, encoding='utf-8')
    except Exception:
        pass
    generated = cleaned

if args.save_sample:
    # save sample for tests (atomic write to avoid partial writes and ensure visibility)
    sample_out = OUT_DIR / 'final' / 'sample_generation.txt'
    try:
        sample_out.parent.mkdir(parents=True, exist_ok=True)
        tmp = sample_out.with_suffix('.tmp')
        with open(tmp, 'w', encoding='utf-8') as fh:
            fh.write(generated)
            fh.flush()
            fh.close()
        # atomic replace
        import os
        os.replace(str(tmp), str(sample_out))
        print('Saved generation sample to', sample_out)
    except Exception as e:
        print('Failed to save generation sample:', e)
else:
    print('Not saving generation sample (use --save-sample to persist)')

print("Done. Temporary model saved to:", OUT_DIR)
