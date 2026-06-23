"""NACL-style one-shot KV cache eviction utilities.

NACL combines Proxy-Tokens Eviction with Random Eviction during the encoding
phase. This module implements the shared selection contract in plain PyTorch:
reduce attention statistics from proxy query tokens, keep protected tokens,
select high-score tokens, and optionally diversify retained tokens with
probability sampling.
"""

from __future__ import annotations

import torch


def select_nacl_proxy_indices(
    seq_len: int,
    *,
    proxy_size: int,
    mode: str = "suffix",
    sink_size: int = 0,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Choose proxy token positions for NACL scoring.

    `suffix` matches the common long-context QA layout where the user question
    is placed at the end. `edges` protects both beginning and ending context
    when the question span is unknown.
    """

    if seq_len < 0:
        raise ValueError(f"seq_len must be non-negative, got {seq_len}")
    if proxy_size < 0:
        raise ValueError(f"proxy_size must be non-negative, got {proxy_size}")
    if sink_size < 0:
        raise ValueError(f"sink_size must be non-negative, got {sink_size}")
    if seq_len == 0 or proxy_size == 0:
        return torch.empty((0,), dtype=torch.long, device=device)

    keep = min(seq_len, proxy_size)
    if mode == "suffix":
        return torch.arange(seq_len - keep, seq_len, dtype=torch.long, device=device)
    if mode == "prefix":
        return torch.arange(0, keep, dtype=torch.long, device=device)
    if mode == "edges":
        prefix = torch.arange(0, min(seq_len, sink_size), dtype=torch.long, device=device)
        suffix = torch.arange(seq_len - keep, seq_len, dtype=torch.long, device=device)
        return torch.unique(torch.cat([prefix, suffix]), sorted=True)
    raise ValueError(f"Unsupported proxy mode {mode!r}; choose suffix, prefix, or edges")


def reduce_nacl_proxy_scores(
    attn_scores: torch.Tensor,
    proxy_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reduce attention scores column-wise using NACL proxy tokens.

    Args:
        attn_scores: Either `[batch, heads, query, key]` raw attention scores or
            `[batch, heads, key]` already-reduced token scores.
        proxy_indices: Query positions used as proxy tokens when `attn_scores`
            has a query dimension.
    """

    if attn_scores.dim() == 3:
        return attn_scores
    if attn_scores.dim() != 4:
        raise ValueError("attn_scores must have shape [batch, heads, key] or [batch, heads, query, key]")

    query_len = attn_scores.shape[-2]
    if proxy_indices is None:
        proxy_indices = torch.arange(query_len, dtype=torch.long, device=attn_scores.device)
    proxy_indices = proxy_indices.to(device=attn_scores.device, dtype=torch.long)
    if proxy_indices.numel() == 0:
        return attn_scores.new_zeros((*attn_scores.shape[:2], attn_scores.shape[-1]))
    proxy_indices = proxy_indices.clamp(min=0, max=query_len - 1)
    return attn_scores.index_select(dim=-2, index=proxy_indices).sum(dim=-2)


def _normalise_indices(indices: torch.Tensor | None, key_len: int, device: torch.device) -> torch.Tensor:
    if indices is None:
        return torch.empty((0,), dtype=torch.long, device=device)
    if indices.numel() == 0:
        return indices.to(device=device, dtype=torch.long)
    return torch.unique(indices.to(device=device, dtype=torch.long).clamp(min=0, max=key_len - 1), sorted=True)


def _sample_without_replacement(
    scores: torch.Tensor,
    candidates: torch.Tensor,
    sample_count: int,
    *,
    generator: torch.Generator | None,
    random_temperature: float,
) -> torch.Tensor:
    if sample_count <= 0 or candidates.numel() == 0:
        return candidates.new_empty((0,))
    count = min(sample_count, candidates.numel())
    candidate_scores = scores.index_select(dim=0, index=candidates)
    if random_temperature <= 0:
        weights = torch.ones_like(candidate_scores, dtype=torch.float)
    else:
        weights = torch.softmax(candidate_scores.float() / random_temperature, dim=0)
    if torch.isnan(weights).any() or weights.sum() <= 0:
        weights = torch.ones_like(candidate_scores, dtype=torch.float)
    sampled = torch.multinomial(weights, num_samples=count, replacement=False, generator=generator)
    return candidates.index_select(dim=0, index=sampled)


def select_nacl_tokens(
    attn_scores: torch.Tensor,
    *,
    token_budget: int,
    proxy_indices: torch.Tensor | None = None,
    protected_indices: torch.Tensor | None = None,
    random_budget: int = 0,
    generator: torch.Generator | None = None,
    random_temperature: float = 1.0,
) -> torch.Tensor:
    """Select retained KV token indices with NACL proxy/random eviction.

    Returned indices have shape `[batch, heads, retained]` and are sorted in
    ascending token order so they can be used for chronological KV gathering.
    """

    if token_budget <= 0:
        raise ValueError(f"token_budget must be positive, got {token_budget}")
    if random_budget < 0:
        raise ValueError(f"random_budget must be non-negative, got {random_budget}")

    scores = reduce_nacl_proxy_scores(attn_scores, proxy_indices)
    if scores.dim() != 3:
        raise ValueError("reduced NACL scores must have shape [batch, heads, key]")

    bsz, num_heads, key_len = scores.shape
    if token_budget >= key_len:
        return torch.arange(key_len, device=scores.device, dtype=torch.long).expand(
            bsz, num_heads, key_len
        )

    proxy_protection = _normalise_indices(proxy_indices, key_len, scores.device)
    explicit_protection = _normalise_indices(protected_indices, key_len, scores.device)
    protected = torch.unique(torch.cat([proxy_protection, explicit_protection]), sorted=True)
    if protected.numel() >= token_budget:
        return protected[-token_budget:].expand(bsz, num_heads, token_budget)

    random_keep = min(random_budget, token_budget - protected.numel())
    top_keep = token_budget - protected.numel() - random_keep
    all_indices = torch.arange(key_len, device=scores.device, dtype=torch.long)
    retained = torch.empty((bsz, num_heads, token_budget), dtype=torch.long, device=scores.device)

    for batch_idx in range(bsz):
        for head_idx in range(num_heads):
            head_scores = scores[batch_idx, head_idx]
            candidate_mask = torch.ones(key_len, dtype=torch.bool, device=scores.device)
            if protected.numel():
                candidate_mask[protected] = False
            candidates = all_indices[candidate_mask]

            if top_keep > 0 and candidates.numel() > 0:
                _, top_positions = torch.topk(
                    head_scores.index_select(dim=0, index=candidates),
                    k=min(top_keep, candidates.numel()),
                    dim=0,
                )
                top_indices = candidates.index_select(dim=0, index=top_positions)
            else:
                top_indices = candidates.new_empty((0,))

            random_mask = candidate_mask.clone()
            if top_indices.numel():
                random_mask[top_indices] = False
            random_candidates = all_indices[random_mask]
            random_indices = _sample_without_replacement(
                head_scores,
                random_candidates,
                random_keep,
                generator=generator,
                random_temperature=random_temperature,
            )
            chosen = torch.unique(torch.cat([top_indices, random_indices, protected]), sorted=True)
            if chosen.numel() < token_budget:
                fill_mask = torch.ones(key_len, dtype=torch.bool, device=scores.device)
                fill_mask[chosen] = False
                fill_candidates = all_indices[fill_mask]
                fill_count = min(token_budget - chosen.numel(), fill_candidates.numel())
                chosen = torch.cat([chosen, fill_candidates[:fill_count]]).sort().values
            retained[batch_idx, head_idx] = chosen[:token_budget]

    return retained
