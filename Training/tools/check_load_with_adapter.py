# Quick test harness for model_adapter
from pathlib import Path
import time
import sys

# Ensure repo root is on sys.path so `Training.tools` can be imported when run directly
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from Training.tools import paths as path_utils
from Training.tools.model_adapter import load_with_adapter

MODELS = [
    path_utils.find_model_path('meta-llama/Llama-3.2-1B'),
    path_utils.find_model_path('Qwen/Qwen3-1.7B'),
]

for m in MODELS:
    if m is None:
        print("\nSkipping test because cached/custom snapshot missing")
        continue
    print("\nTesting:", m)
    if not m.exists():
        print("  SKIP: path not found", m)
        continue
    start = time.time()
    try:
        tokenizer, model, meta = load_with_adapter(str(m), family=None, commit_patch=False)
        warns = meta.get('warnings', []) if isinstance(meta, dict) else []
        print(f"  OK: loaded tokenizer (vocab_size={getattr(tokenizer,'vocab_size',None)}), model parameters={sum(p.numel() for p in model.parameters())}")
        for w in warns:
            print("   warning:", w)
        # tiny forward
        try:
            import torch
            device = next(model.parameters()).device
            inputs = tokenizer("Hello world", return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model(**inputs)
            print("  forward pass OK; logits shape=", out.logits.shape)
        except Exception as e:
            print("  forward pass FAILED:", e)
    except Exception as e:
        print("  FAILED to load with adapter:", e)
    finally:
        print(f"  elapsed: {time.time()-start:.1f}s")
