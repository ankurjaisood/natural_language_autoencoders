"""Capture activations at every decoder layer in one forward pass, and save them.

WHY ALL LAYERS
  The read-site probe needs one layer, but the forward pass that produces it computes all of
  them. Hooking every layer costs almost nothing extra on top of a pass we have to run anyway,
  and it turns the layer sweep from a second GPU job into a CPU job over a file.

WHY NOT output_hidden_states=True
  It works, but it materialises [B, T, d] for every layer before we reduce to one token per
  sequence, which is ~3 GB per batch at max_length 1024 on this model. Hooking each layer and
  taking the last real token inside the hook keeps peak memory flat in the layer count.

THE READ SITE IS READ FROM THE CHECKPOINT, NOT ASSUMED
  `extraction_layer_index` in the AV's nla_meta.yaml is the contract. The convention is that
  layer_index K means block K's OUTPUT, which HF exposes as hidden_states[K+1] — see
  bench/capture.py. Slicing this array at block K therefore reproduces what
  capture_activations(block_index=K) returns, and `verify_matches_single_layer` asserts it.
"""

from __future__ import annotations

import numpy as np
import torch

MAX_LENGTH = 1024
BATCH_SIZE = 8


def read_site_from_checkpoint(av_repo: str) -> int:
    """The layer the NLA actually reads, from its own sidecar."""
    from pathlib import Path

    import yaml
    from huggingface_hub import hf_hub_download
    p = Path(hf_hub_download(av_repo, "nla_meta.yaml"))
    meta = yaml.safe_load(p.read_text())
    idx = meta.get("extraction_layer_index")
    if idx is None:                       # v1 sidecars nest it under the critic block
        idx = (meta.get("critic") or {}).get("extraction_layer_index")
    if idx is None:
        raise ValueError(f"{av_repo}: no extraction_layer_index in nla_meta.yaml")
    return int(idx)


def capture_all_layers(model, tok, texts, *, batch_size: int = BATCH_SIZE,
                       max_length: int = MAX_LENGTH) -> np.ndarray:
    """[N, n_blocks, d_model] fp32, the last real token of each text at every block output."""
    from nla.arch_adapters import resolve_text_config

    from bench.capture import _decoder_layers
    layers = _decoder_layers(model)
    d_model = resolve_text_config(model.config).hidden_size
    n_blocks = len(layers)
    out = np.empty((len(texts), n_blocks, d_model), dtype=np.float32)

    # the hook reduces to one token immediately; holding [B, T, d] for every block is what
    # makes the naive version expensive
    state: dict = {"last": None, "buf": None}

    def make_hook(bi):
        def hook(_m, _i, output):
            h = output[0] if isinstance(output, tuple) else output
            picked = h[torch.arange(h.shape[0], device=h.device), state["last"]]
            state["buf"][bi] = picked.float().cpu().numpy()
        return hook

    handles = [layers[i].register_forward_hook(make_hook(i)) for i in range(n_blocks)]
    try:
        for start in range(0, len(texts), batch_size):
            chunk = list(texts[start:start + batch_size])
            enc = tok(chunk, return_tensors="pt", padding=True,
                      truncation=True, max_length=max_length).to(model.device)
            state["last"] = enc["attention_mask"].sum(dim=1) - 1
            state["buf"] = np.empty((n_blocks, len(chunk), d_model), dtype=np.float32)
            with torch.no_grad():
                model(**enc)
            out[start:start + len(chunk)] = state["buf"].transpose(1, 0, 2)
    finally:
        for h in handles:
            h.remove()
    return out


def verify_matches_single_layer(model, tok, texts, all_acts, block_index, *,
                                n_check: int = 24, tol: float = 1e-4) -> float:
    """Assert the block-index slice equals what the one-layer path returns.

    Without this the sweep could silently read a different tensor than every published number
    was computed from, and the read-site comparison would be against the wrong thing.
    """
    from bench.capture import capture_activations
    ref = capture_activations(model, tok, list(texts[:n_check]), block_index=block_index,
                              max_length=MAX_LENGTH)
    got = all_acts[:n_check, block_index, :]
    dev = float(np.abs(ref - got).max())
    assert dev < tol, f"block {block_index} slice differs from single-layer capture: {dev:.2e}"
    return dev
