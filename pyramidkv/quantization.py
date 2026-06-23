"""Utilities for KV cache quantization runner configuration."""

from __future__ import annotations

from typing import Any, Dict, Optional


KIVI_AXIS_KEY = 1
KIVI_AXIS_VALUE = 0
DEFAULT_QUANT_BACKEND = "hqq"
SUPPORTED_QUANT_METHODS = ("kivi", "kvquant")


def normalize_quant_method(method: Optional[str]) -> Optional[str]:
    if method is None:
        return None
    normalized = method.strip().lower()
    if normalized in {"", "none", "null"}:
        return None
    if normalized not in SUPPORTED_QUANT_METHODS:
        raise ValueError(f"Unsupported quant_method {method!r}; choose one of {SUPPORTED_QUANT_METHODS}")
    return normalized


def build_quantized_cache_config(
    quant_method: Optional[str],
    *,
    nbits: int,
    residual_length: int,
    device: str = "cuda",
    backend: str = DEFAULT_QUANT_BACKEND,
    q_group_size: int = 64,
    axis_key: Optional[int] = None,
    axis_value: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Build the Hugging Face `generate(..., cache_config=...)` dictionary.

    KIVI quantizes keys per-channel and values per-token. With the HQQ cache
    layout used by Transformers, this is represented by `axis_key=1` and
    `axis_value=0`. `kvquant` intentionally uses the same runner config but
    swaps the HQQ cache class for the local outlier-preserving implementation.
    """

    method = normalize_quant_method(quant_method)
    if method is None:
        return None
    if nbits <= 0:
        raise ValueError(f"nbits must be positive, got {nbits}")
    if residual_length <= 0:
        raise ValueError(f"residual_length must be positive, got {residual_length}")
    if q_group_size <= 0:
        raise ValueError(f"q_group_size must be positive, got {q_group_size}")

    return {
        "nbits": nbits,
        "backend": backend.lower(),
        "device": device,
        "residual_length": residual_length,
        "axis_key": KIVI_AXIS_KEY if axis_key is None else axis_key,
        "axis_value": KIVI_AXIS_VALUE if axis_value is None else axis_value,
        "q_group_size": q_group_size,
    }


def patch_quantized_cache(quant_method: Optional[str]) -> None:
    """Install local cache overrides required by selected quantization methods."""

    method = normalize_quant_method(quant_method)
    if method != "kvquant":
        return

    from pyramidkv.quantcache import KVQuantizedCache
    from transformers import cache_utils

    cache_utils.HQQQuantizedCache = KVQuantizedCache
