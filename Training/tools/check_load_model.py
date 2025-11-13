# Quick load test for meta-llama_Llama-3.2-1B
import sys
import time

from Training.tools import paths as path_utils

MODEL_PATH = path_utils.find_model_path('meta-llama/Llama-3.2-1B')

if MODEL_PATH is None:
    print("ERROR: cached/custom snapshot for meta-llama/Llama-3.2-1B not found. Run the download CLI first or store a custom copy under E:\\AI\\Models.")
    sys.exit(2)

print(f"Test script starting. Model path: {MODEL_PATH}")

try:
    import torch
    from transformers import AutoModelForCausalLM
    from Training.tools.adapter_utils import load_tokenizer as adapter_load_tokenizer, load_model as adapter_load_model
except Exception as e:
    print("ERROR importing dependencies:", e)
    sys.exit(3)

start = time.time()
try:
    print("Loading tokenizer... (adapter-aware)")
    # Prefer adapter when available; fall back to direct AutoTokenizer
    try:
        # robust import of adapter
        from Training.tools.model_adapter import load_with_adapter
        use_adapter = True
    except Exception:
        use_adapter = False

    if use_adapter:
        try:
            tokenizer, model, meta = load_with_adapter(str(MODEL_PATH), local_files_only=True)
            print("Tokenizer loaded via adapter. vocab_size=", getattr(tokenizer, 'vocab_size', None))
            # we'll skip model load below since adapter returned a model
            adapter_provided_model = True
        except Exception:
            adapter_provided_model = False
            tokenizer = adapter_load_tokenizer(str(MODEL_PATH), local_files_only=True)
            print("Tokenizer loaded. vocab_size=", getattr(tokenizer, 'vocab_size', None))
    else:
        tokenizer = adapter_load_tokenizer(str(MODEL_PATH), local_files_only=True)
        adapter_provided_model = False
        print("Tokenizer loaded. vocab_size=", getattr(tokenizer, 'vocab_size', None))
except Exception as e:
    print("ERROR loading tokenizer:", e)
    sys.exit(4)

# Choose dtype safe for CPU
dtype = torch.float16 if (torch.cuda.is_available()) else torch.float32
print(f"Torch available: cuda={torch.cuda.is_available()}, using dtype={dtype}")

try:
    # Load config first and normalize rope_scaling if present (some HF community configs
    # use different keys; transformers expects {'type':..., 'factor':...}).
    from transformers import AutoConfig
    print("Loading config and normalizing rope_scaling if needed...")
    # Workaround: patch the on-disk config.json if it contains a non-standard rope_scaling
    import json
    cfg_path = MODEL_PATH / "config.json"
    backup_path = MODEL_PATH / "config.json.bak"
    patched = False
    if cfg_path.exists():
        try:
            with open(cfg_path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            rs = cfg.get('rope_scaling')
            if isinstance(rs, dict) and (('type' not in rs) or ('factor' not in rs)):
                factor = rs.get('factor') or rs.get('high_freq_factor') or rs.get('low_freq_factor') or 1.0
                rope_type = (rs.get('rope_type') or rs.get('type') or 'rope')
                # Normalize known community variants to transformers-expected values
                rt = str(rope_type).lower()
                if 'llama' in rt:
                    typ = 'dynamic'
                elif rt in ('linear', 'dynamic'):
                    typ = rt
                else:
                    typ = 'linear'
                cfg['rope_scaling'] = {'type': typ, 'factor': float(factor)}
                # backup and write patched config
                backup_path.write_bytes(cfg_path.read_bytes())
                with open(cfg_path, 'w', encoding='utf-8') as f:
                    json.dump(cfg, f, indent=2)
                patched = True
                print("Patched on-disk config.json rope_scaling ->", cfg['rope_scaling'])
        except Exception as e:
            print("Warning: failed to patch config.json:", e)

    print("Loading model (may take a while...)")
    try:
        if adapter_provided_model:
            model = model  # from adapter
        else:
            try:
                model = adapter_load_model(str(MODEL_PATH), loader=AutoModelForCausalLM.from_pretrained, local_files_only=True, torch_dtype=dtype, low_cpu_mem_usage=True)
            except ValueError as ve:
                # e.g. unknown model_type (qwen3) -> try adapter fallback
                print('Model load ValueError:', ve)
                try:
                    from Training.tools.model_adapter import load_with_adapter
                    tokenizer, model, meta = load_with_adapter(str(MODEL_PATH), local_files_only=True)
                    print('Loaded model via adapter fallback')
                except Exception as exc:
                    print('Adapter fallback failed:', exc)
                    raise
    finally:
        # restore original config if we patched it
        if patched:
            try:
                cfg_path.write_bytes(backup_path.read_bytes())
                backup_path.unlink()
                print("Restored original config.json from backup")
            except Exception:
                pass
    print("Model loaded. num_parameters=", sum(p.numel() for p in model.parameters()))
except Exception as e:
    print("ERROR loading model:", e)
    sys.exit(5)

# Run a tiny forward pass to ensure model is functional
try:
    print("Running tiny forward pass...")
    inputs = tokenizer("Hello world", return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)
    print("Forward pass OK. logits shape:", out.logits.shape)
except Exception as e:
    print("ERROR during forward pass:", e)
    sys.exit(6)

elapsed = time.time() - start
print(f"SUCCESS: load+forward completed in {elapsed:.1f}s")
sys.exit(0)
