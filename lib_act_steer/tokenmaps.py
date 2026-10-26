"""The word maps' attention reduction: one call per cross-attention layer for all words.

token_maps(q, k, sel, heads, scale) -> [n, L, C]: per pixel, the head-averaged softmax attention summed
over each column's tokens (sel [n, T, C] is 1 where token t belongs to word c).

The fused Triton kernel lives in amd_tuned_torch (attn_token_maps_ops, ~/optimiert/amd_tools/source/
cmp_ext_turing). The webui's venv_amd holds a read-only copy of that package that may predate it, so
it is taken from (1) the installed amd_tuned_torch, else (2) the source tree's file (only torch and
triton are needed; ACT_STEER_TOKEN_MAPS_SRC overrides the path, "" disables), else (3) the plain torch
version below (same result, a few more launches).
"""

import importlib.util
import os

import torch

SOURCE = os.environ.get("ACT_STEER_TOKEN_MAPS_SRC", os.path.expanduser(
    "~/optimiert/amd_tools/source/cmp_ext_turing/amd_tuned_torch/attn_token_maps_ops.py"))


def _torch_maps(q, k, sel, heads, scale):
    n, L, HD = q.shape
    T = k.shape[1]
    D = HD // heads
    qh = q.reshape(n, L, heads, D).transpose(1, 2)
    kh = k.to(q.dtype).reshape(n, T, heads, D).transpose(1, 2)
    probs = (qh @ kh.transpose(-1, -2)).float().mul_(scale).softmax(-1).mean(1)
    return probs @ sel.to(device=probs.device, dtype=torch.float32)


def _load():
    try:
        from amd_tuned_torch import attn_token_maps_ops as ops
        return ops.attention_token_maps, "amd_tuned_torch"
    except Exception:
        pass
    if SOURCE and os.path.isfile(SOURCE):
        try:
            spec = importlib.util.spec_from_file_location("act_steer_attn_token_maps_ops", SOURCE)
            ops = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(ops)
            if ops.available():
                return ops.attention_token_maps, SOURCE
        except Exception as e:
            print(f"[ActSteer] fused word-map kernel not loaded from {SOURCE}: {type(e).__name__}: {e}")
    return _torch_maps, "torch"


token_maps, backend = _load()
