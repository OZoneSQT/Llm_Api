"""
Diagnostic: load model and tokenizer per Training/data/config.json, print important config
and tokenizer properties, tokenize a sample prompt and attempt a forward pass to expose
shape mismatches (useful for rotary/attention/head-dim issues).
"""
import json
import os
import sys
from pathlib import Path
import traceback

import torch

from Training.tools.adapter_utils import load_tokenizer, load_model
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[2]
CFG_PATH = ROOT / "Training" / "data" / "config.json"

print("Using Python:", sys.executable)
print("Torch:", torch.__version__)

with open(CFG_PATH, "r", encoding="utf-8") as f:
    cfg = json.load(f)

script_dir = Path(__file__).resolve().parents[1]
local_model = cfg.get("local_model_path") or cfg.get("local_model")
model_name_cfg = cfg.get("llm_model_name") or cfg.get("model_name")

try:
    if local_model:
        # resolve relative paths
        p = Path(local_model)
        if not p.is_absolute():
            p = (Path.cwd() / local_model).resolve()
        print("Attempting to load local model from:", str(p))
        model = load_model(str(p), loader=AutoModelForCausalLM.from_pretrained, local_files_only=True, trust_remote_code=True)
        tokenizer = load_tokenizer(str(p), local_files_only=True, trust_remote_code=True)
    else:
        print("No local_model_path set in config, loading model from HF name:", model_name_cfg)
        model = load_model(model_name_cfg, loader=AutoModelForCausalLM.from_pretrained, local_files_only=False, trust_remote_code=True)
        tokenizer = load_tokenizer(model_name_cfg, local_files_only=False, trust_remote_code=True)
except Exception as e:
    print("Failed to load model/tokenizer:")
    traceback.print_exc()
    raise SystemExit(1)

print("\nModel class:", model.__class__)
conf = getattr(model, "config", None)
if conf is not None:
    print("Model config summary:")
    for k in ("model_type", "hidden_size", "n_head", "num_heads", "num_attention_heads", "vocab_size", "max_position_embeddings", "rotary_dim", "rotary_pct", "head_dim"):
        if hasattr(conf, k):
            print(f"  {k}: {getattr(conf, k)}")
    # compute head dim if possible
    try:
        nh = getattr(conf, "num_attention_heads", None) or getattr(conf, "num_heads", None)
        hs = getattr(conf, "hidden_size", None)
        if nh and hs:
            print(f"  computed head_dim = hidden_size/num_heads = {hs}/{nh} = {hs//nh}")
    except Exception:
        pass

print("\nTokenizer class:", tokenizer.__class__)
print("  vocab_size:", getattr(tokenizer, "vocab_size", None))
print("  pad_token_id:", getattr(tokenizer, "pad_token_id", None))
print("  eos_token_id:", getattr(tokenizer, "eos_token_id", None))
print("  model_max_length:", getattr(tokenizer, "model_max_length", None))

# Tokenize a simple prompt
prompt = "Hello, this is a diagnostic prompt to test forward pass."
inputs = tokenizer(prompt, return_tensors="pt")
print("\nTokenized input_ids shape:", inputs["input_ids"].shape)

# Move model to cpu for deterministic output (or cuda if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)
model.to(device)
inputs = {k: v.to(device) for k, v in inputs.items()}

print("\nAttempting a forward pass... (this may raise the same error)")
try:
    with torch.no_grad():
        out = model(**inputs)
    print("Forward pass succeeded. Output keys:", list(out.keys()) if hasattr(out, 'keys') else type(out))
except Exception:
    print("Forward pass failed with exception:")
    traceback.print_exc()
    # Try to run a smaller test: print shapes of q/k/v if accessible by running a forward hook is complex.
    # Instead, attempt to inspect a single attention module's config if present.
    try:
        # try to find first attention module
        import inspect
        attn = None
        for name, module in model.named_modules():
            n = name.lower()
            if "attn" in n and (hasattr(module, 'q_proj') or hasattr(module, 'qkv') or hasattr(module, 'query') or hasattr(module, 'self_attn')):
                print(f"Found attention module: {name} -> {module.__class__}")
                attn = module
                break
        if attn is not None:
            print("Attention module repr:\n", attn)
            # print attributes
            for attr in ("head_dim", "num_heads", "hidden_size", "rotary_dim", "rotary_pct"):
                if hasattr(attn, attr):
                    print(f"  {attr}: {getattr(attn, attr)}")
                # inspect rotary embedding object if present
                try:
                    rotary = getattr(attn, 'rotary_emb', None)
                    if rotary is not None:
                        print("Found rotary_emb object:", rotary.__class__)
                        # print rotary attributes
                        for a in dir(rotary):
                            if a.startswith('_'):
                                continue
                            try:
                                v = getattr(rotary, a)
                                # show tensor shapes
                                if hasattr(v, 'shape'):
                                    print(f"  {a}: tensor shape {getattr(v,'shape')}")
                                else:
                                    print(f"  {a}: {type(v)}")
                            except Exception:
                                pass
                        # attempt to call any available method that returns cos/sin
                        for method_name in ("_build_cache", "get_cos_sin", "forward", "cos_sin", "get_cache"):
                            if hasattr(rotary, method_name):
                                print(f"Trying rotary.{method_name}(...)")
                                try:
                                    m = getattr(rotary, method_name)
                                    # try with a small seq_len
                                    res = m(10) if callable(m) else None
                                    print(f"  {method_name} returned type: {type(res)}")
                                except Exception as e:
                                    print(f"  Calling {method_name} raised: {e}")
                    else:
                        print("No rotary_emb attribute found on attention module.")
                except Exception:
                    pass
    except Exception:
        pass
    raise

print("\nDiagnostic complete.")
