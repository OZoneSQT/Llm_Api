"""Runtime rotary-embedding patch (inference-only, experimental).

This module monkey-patches transformers' StableLm apply_rotary_pos_emb to tolerate
mis-sized cos/sin buffers by padding or trimming them to match q/k's last dim.

WARNING: This is an experimental, inference-only workaround. It may produce
incorrect results for training and should not be used for production training.
"""
from types import ModuleType
import importlib
import torch

def _safe_rotate_half(x: torch.Tensor) -> torch.Tensor:
    # rotate half: split last dim in two and swap with sign
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)

def _pad_or_trim(t: torch.Tensor, target: int) -> torch.Tensor:
    if t is None:
        return None
    if t.shape[-1] == target:
        return t
    if t.shape[-1] < target:
        pad_shape = list(t.shape)
        pad_shape[-1] = target - t.shape[-1]
        pad = t.new_zeros(*pad_shape)
        return torch.cat([t, pad], dim=-1)
    # trim
    return t[..., :target]

def apply_patch():
    try:
        mod = importlib.import_module('transformers.models.stablelm.modeling_stablelm')
    except Exception:
        # nothing to patch
        return False

    # keep original if present
    orig = getattr(mod, 'apply_rotary_pos_emb', None)

    def _patched_apply_rotary_pos_emb(q, k, cos, sin, *args, **kwargs):
        # q/k expected shape: (..., rotary_width) where last dim is head-part
        try:
            q_last = q.shape[-1]
            # pad/trim cos/sin to match q's last dim
            cos_safe = _pad_or_trim(cos, q_last)
            sin_safe = _pad_or_trim(sin, q_last)
            # If original exists, try to call it. If it errors, fall back to local implementation.
            if orig is not None:
                try:
                    # preserve any additional args (e.g., position_ids)
                    return orig(q, k, cos_safe, sin_safe, *args, **kwargs)
                except Exception:
                    pass

            # local implementation: q_embed = (q * cos) + (rotate_half(q) * sin)
            q_embed = (q * cos_safe) + (_safe_rotate_half(q) * sin_safe)
            k_embed = (k * cos_safe) + (_safe_rotate_half(k) * sin_safe)
            return q_embed, k_embed
        except Exception:
            # if anything goes wrong, re-raise to preserve original behavior
            raise

    try:
        setattr(mod, 'apply_rotary_pos_emb', _patched_apply_rotary_pos_emb)
        return True
    except Exception:
        return False

# Auto-apply on import to make it easy for the quick smoke test
_applied = apply_patch()

if _applied:
    print("rotary_inference_patch: applied patched apply_rotary_pos_emb (experimental)")
else:
    print("rotary_inference_patch: failed to apply patch or module not present")
