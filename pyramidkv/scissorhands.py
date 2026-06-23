"""Scissorhands-style persistence-of-importance KV selection utilities."""

from __future__ import annotations

import torch


def reduce_scissorhands_scores(
    attn_scores: torch.Tensor,
    *,
    history_window: int | None = None,
) -> torch.Tensor:
    """Reduce recent attention history into per-token importance scores.

    `attn_scores` may be raw attention with shape `[batch, heads, query, key]`
    or already-reduced scores `[batch, heads, key]`.
    """

    if attn_scores.dim() == 3:
        return attn_scores
    if attn_scores.dim() != 4:
        raise ValueError("attn_scores must have shape [batch, heads, key] or [batch, heads, query, key]")
    if history_window is not None:
        if history_window <= 0:
            raise ValueError(f"history_window must be positive, got {history_window}")
        attn_scores = attn_scores[..., -history_window:, :]
    return attn_scores.sum(dim=-2)


def update_scissorhands_importance(
    previous_importance: torch.Tensor | None,
    attn_scores: torch.Tensor,
    *,
    history_window: int | None = None,
    decay: float = 1.0,
) -> torch.Tensor:
    """Update persistent token importance with optional exponential decay."""

    if decay < 0:
        raise ValueError(f"decay must be non-negative, got {decay}")
    current = reduce_scissorhands_scores(attn_scores, history_window=history_window)
    if previous_importance is None:
        return current
    if previous_importance.shape != current.shape:
        raise ValueError(
            f"previous/current importance shapes must match, got {previous_importance.shape} and {current.shape}"
        )
    return previous_importance * decay + current


def _protected_indices(
    key_len: int,
    *,
    sink_size: int,
    recent_size: int,
    device: torch.device,
) -> torch.Tensor:
    if sink_size < 0:
        raise ValueError(f"sink_size must be non-negative, got {sink_size}")
    if recent_size < 0:
        raise ValueError(f"recent_size must be non-negative, got {recent_size}")
    sinks = torch.arange(0, min(key_len, sink_size), dtype=torch.long, device=device)
    recent_count = min(key_len, recent_size)
    recent = torch.arange(key_len - recent_count, key_len, dtype=torch.long, device=device)
    return torch.unique(torch.cat([sinks, recent]), sorted=True)


def _sample_by_importance(
    scores: torch.Tensor,
    candidates: torch.Tensor,
    count: int,
    *,
    generator: torch.Generator | None,
    temperature: float,
) -> torch.Tensor:
    if count <= 0 or candidates.numel() == 0:
        return candidates.new_empty((0,))
    count = min(count, candidates.numel())
    candidate_scores = scores.index_select(dim=0, index=candidates).float()
    if temperature <= 0:
        weights = torch.ones_like(candidate_scores)
    else:
        shifted = candidate_scores - candidate_scores.max()
        weights = torch.softmax(shifted / temperature, dim=0)
    if torch.isnan(weights).any() or weights.sum() <= 0:
        weights = torch.ones_like(candidate_scores)
    sampled = torch.multinomial(weights, num_samples=count, replacement=False, generator=generator)
    return candidates.index_select(dim=0, index=sampled)


def select_scissorhands_tokens(
    importance_scores: torch.Tensor,
    *,
    token_budget: int,
    recent_size: int = 0,
    sink_size: int = 0,
    selection: str = "topk",
    generator: torch.Generator | None = None,
    random_temperature: float = 1.0,
) -> torch.Tensor:
    """Select persistent pivotal tokens under a fixed KV budget.

    Returned indices have shape `[batch, heads, retained]` and are sorted in
    chronological order for direct KV gathering.
    """

    if token_budget <= 0:
        raise ValueError(f"token_budget must be positive, got {token_budget}")
    if importance_scores.dim() != 3:
        raise ValueError("importance_scores must have shape [batch, heads, key]")

    bsz, num_heads, key_len = importance_scores.shape
    if token_budget >= key_len:
        return torch.arange(key_len, device=importance_scores.device, dtype=torch.long).expand(
            bsz, num_heads, key_len
        )

    protected = _protected_indices(
        key_len,
        sink_size=sink_size,
        recent_size=recent_size,
        device=importance_scores.device,
    )
    if protected.numel() >= token_budget:
        return protected[-token_budget:].expand(bsz, num_heads, token_budget)

    selection_budget = token_budget - protected.numel()
    all_indices = torch.arange(key_len, device=importance_scores.device, dtype=torch.long)
    retained = torch.empty((bsz, num_heads, token_budget), dtype=torch.long, device=importance_scores.device)

    for batch_idx in range(bsz):
        for head_idx in range(num_heads):
            candidate_mask = torch.ones(key_len, dtype=torch.bool, device=importance_scores.device)
            if protected.numel():
                candidate_mask[protected] = False
            candidates = all_indices[candidate_mask]
            head_scores = importance_scores[batch_idx, head_idx]

            mode = selection.lower()
            if mode in {"topk", "deterministic"}:
                _, positions = torch.topk(
                    head_scores.index_select(dim=0, index=candidates),
                    k=min(selection_budget, candidates.numel()),
                    dim=0,
                )
                selected = candidates.index_select(dim=0, index=positions)
            elif mode in {"prob", "probabilistic", "sample"}:
                selected = _sample_by_importance(
                    head_scores,
                    candidates,
                    selection_budget,
                    generator=generator,
                    temperature=random_temperature,
                )
            else:
                raise ValueError(f"Unsupported Scissorhands selection mode {selection!r}")

            retained[batch_idx, head_idx] = torch.cat([selected, protected]).sort().values[:token_budget]

    return retained
