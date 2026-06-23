"""Quest-style query-aware KV cache retrieval utilities.

Quest scores KV cache pages with per-page key minima/maxima and the current
query. The utilities here implement the page metadata and selection contract in
plain PyTorch so runners and runtime ports can share the same semantics before
adding specialized kernels.
"""

from __future__ import annotations

import math

import torch


def _last_query(query_states: torch.Tensor) -> torch.Tensor:
    if query_states.dim() == 4:
        return query_states[:, :, -1, :]
    if query_states.dim() == 3:
        return query_states
    raise ValueError("query_states must have shape [batch, heads, dim] or [batch, heads, query, dim]")


def build_quest_page_metadata(
    key_states: torch.Tensor,
    *,
    page_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build per-page key minima and maxima.

    Args:
        key_states: Tensor with shape `[batch, heads, seq_len, head_dim]`.
        page_size: Number of tokens represented by one Quest page.
    """

    if key_states.dim() != 4:
        raise ValueError("key_states must have shape [batch, heads, seq_len, head_dim]")
    if page_size <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}")

    bsz, num_heads, seq_len, head_dim = key_states.shape
    if seq_len == 0:
        empty = key_states.new_empty((bsz, num_heads, 0, head_dim))
        return empty, empty

    num_pages = math.ceil(seq_len / page_size)
    padded_len = num_pages * page_size
    if padded_len != seq_len:
        pad_len = padded_len - seq_len
        pad = key_states[:, :, -1:, :].expand(-1, -1, pad_len, -1)
        key_states = torch.cat([key_states, pad], dim=2)

    pages = key_states.view(bsz, num_heads, num_pages, page_size, head_dim)
    return pages.amin(dim=3), pages.amax(dim=3)


def score_quest_pages(
    query_states: torch.Tensor,
    page_min: torch.Tensor,
    page_max: torch.Tensor,
) -> torch.Tensor:
    """Estimate query-aware criticality for each page.

    The score is the upper bound of `query @ key` over the page's min/max box:
    positive query dimensions use the page maximum and negative dimensions use
    the page minimum.
    """

    if page_min.shape != page_max.shape:
        raise ValueError(f"page_min/page_max shapes must match, got {page_min.shape} and {page_max.shape}")
    if page_min.dim() != 4:
        raise ValueError("page metadata must have shape [batch, heads, pages, head_dim]")

    query = _last_query(query_states)
    if query.shape[:2] != page_min.shape[:2] or query.shape[-1] != page_min.shape[-1]:
        raise ValueError(
            "query_states and page metadata must share batch, head, and head_dim dimensions"
        )

    query = query.unsqueeze(-2)
    chosen_bounds = torch.where(query >= 0, page_max, page_min)
    return (query * chosen_bounds).sum(dim=-1)


def select_quest_pages(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    *,
    page_size: int,
    page_budget: int,
    recent_size: int = 0,
) -> torch.Tensor:
    """Return top Quest page indices for the non-recent prefix.

    Recent tokens are protected outside the page score path and should be
    appended by `select_quest_tokens` or a runtime-specific cache policy.
    """

    if page_budget <= 0:
        raise ValueError(f"page_budget must be positive, got {page_budget}")
    if recent_size < 0:
        raise ValueError(f"recent_size must be non-negative, got {recent_size}")
    if key_states.dim() != 4:
        raise ValueError("key_states must have shape [batch, heads, seq_len, head_dim]")

    prefix_len = max(key_states.shape[2] - recent_size, 0)
    prefix_keys = key_states[:, :, :prefix_len, :]
    page_min, page_max = build_quest_page_metadata(prefix_keys, page_size=page_size)
    if page_min.shape[2] == 0:
        return torch.empty((*key_states.shape[:2], 0), dtype=torch.long, device=key_states.device)

    scores = score_quest_pages(query_states, page_min, page_max)
    keep_pages = min(page_budget, scores.shape[-1])
    _, page_indices = torch.topk(scores, k=keep_pages, dim=-1)
    return page_indices.sort(dim=-1).values


def _expand_pages_to_tokens(
    page_indices: torch.Tensor,
    *,
    page_size: int,
    prefix_len: int,
    token_budget: int,
) -> torch.Tensor:
    offsets = torch.arange(page_size, device=page_indices.device, dtype=torch.long)
    tokens = page_indices.unsqueeze(-1) * page_size + offsets
    tokens = tokens.reshape(*page_indices.shape[:-1], -1)
    tokens = tokens.masked_fill(tokens >= prefix_len, prefix_len)
    tokens = tokens.sort(dim=-1).values
    return tokens[..., :token_budget]


def select_quest_tokens(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    *,
    page_size: int,
    token_budget: int,
    recent_size: int = 0,
) -> torch.Tensor:
    """Return Quest-selected token indices with an exact token budget.

    The prefix selection is page-based, then expanded to ascending token
    indices. The protected recent window is always appended last.
    """

    if token_budget <= 0:
        raise ValueError(f"token_budget must be positive, got {token_budget}")
    if recent_size < 0:
        raise ValueError(f"recent_size must be non-negative, got {recent_size}")
    if key_states.dim() != 4:
        raise ValueError("key_states must have shape [batch, heads, seq_len, head_dim]")

    bsz, num_heads, seq_len, _ = key_states.shape
    if token_budget >= seq_len:
        return torch.arange(seq_len, device=key_states.device, dtype=torch.long).expand(
            bsz, num_heads, seq_len
        )

    recent_count = min(seq_len, recent_size, token_budget)
    prefix_len = seq_len - recent_count
    prefix_budget = token_budget - recent_count
    recent = torch.arange(prefix_len, seq_len, device=key_states.device, dtype=torch.long).expand(
        bsz, num_heads, recent_count
    )
    if prefix_budget == 0:
        return recent

    page_budget = math.ceil(prefix_budget / page_size)
    num_prefix_pages = math.ceil(prefix_len / page_size)
    if prefix_len % page_size and page_budget < num_prefix_pages:
        page_budget += 1
    page_indices = select_quest_pages(
        query_states,
        key_states,
        page_size=page_size,
        page_budget=page_budget,
        recent_size=recent_count,
    )
    prefix_tokens = _expand_pages_to_tokens(
        page_indices,
        page_size=page_size,
        prefix_len=prefix_len,
        token_budget=prefix_budget,
    )
    return torch.cat([prefix_tokens, recent], dim=-1)
