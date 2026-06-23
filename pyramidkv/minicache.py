"""MiniCache-style cross-layer KV cache compression utilities."""

from __future__ import annotations

import math

import torch


def _check_pair(current_states: torch.Tensor, previous_states: torch.Tensor) -> None:
    if current_states.shape != previous_states.shape:
        raise ValueError(f"current/previous shapes must match, got {current_states.shape} and {previous_states.shape}")
    if current_states.dim() != 4:
        raise ValueError("MiniCache states must have shape [batch, heads, seq_len, head_dim]")


def _unit_and_magnitude(states: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    magnitude = states.norm(dim=-1, keepdim=True).clamp_min(eps)
    return states / magnitude, magnitude


def minicache_slerp(
    current_states: torch.Tensor,
    previous_states: torch.Tensor,
    *,
    interpolation: float = 0.6,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Merge adjacent-layer states with MiniCache SLERP direction sharing.

    Args:
        current_states: KV states from layer `l`, shape `[batch, heads, seq, dim]`.
        previous_states: KV states from layer `l - 1`, same shape.
        interpolation: MiniCache `t`; `0.6` follows the paper default.

    Returns:
        `(shared_direction, current_magnitude, previous_magnitude, angle)`.
    """

    _check_pair(current_states, previous_states)
    if not 0.0 <= interpolation <= 1.0:
        raise ValueError(f"interpolation must be in [0, 1], got {interpolation}")
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")

    current_unit, current_magnitude = _unit_and_magnitude(current_states, eps)
    previous_unit, previous_magnitude = _unit_and_magnitude(previous_states, eps)
    cosine = (current_unit * previous_unit).sum(dim=-1, keepdim=True).clamp(min=-1.0 + eps, max=1.0 - eps)
    angle = torch.acos(cosine)
    sin_angle = torch.sin(angle)

    previous_weight = torch.sin((1.0 - interpolation) * angle) / sin_angle
    current_weight = torch.sin(interpolation * angle) / sin_angle
    slerp = previous_weight * previous_unit + current_weight * current_unit
    lerp = (1.0 - interpolation) * previous_unit + interpolation * current_unit
    shared_direction = torch.where(sin_angle.abs() > eps, slerp, lerp)
    shared_direction = shared_direction / shared_direction.norm(dim=-1, keepdim=True).clamp_min(eps)
    return shared_direction, current_magnitude, previous_magnitude, angle.squeeze(-1)


def select_minicache_retention_indices(
    angular_distance: torch.Tensor,
    *,
    retention_ratio: float = 0.05,
    retention_count: int | None = None,
) -> torch.Tensor:
    """Select high-angular-distance token positions to keep unmerged.

    `angular_distance` is expected to be normalized by pi and shaped
    `[batch, heads, seq_len]`. Returned indices are sorted for chronological
    gather/scatter.
    """

    if angular_distance.dim() != 3:
        raise ValueError("angular_distance must have shape [batch, heads, seq_len]")
    if not 0.0 <= retention_ratio <= 1.0:
        raise ValueError(f"retention_ratio must be in [0, 1], got {retention_ratio}")
    seq_len = angular_distance.shape[-1]
    if retention_count is None:
        retention_count = math.ceil(seq_len * retention_ratio)
    if retention_count < 0:
        raise ValueError(f"retention_count must be non-negative, got {retention_count}")
    keep = min(seq_len, retention_count)
    if keep == 0:
        return torch.empty((*angular_distance.shape[:2], 0), dtype=torch.long, device=angular_distance.device)

    _, indices = torch.topk(angular_distance, k=keep, dim=-1)
    return indices.sort(dim=-1).values


def _gather_tokens(states: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if indices.numel() == 0:
        return states.new_empty((*states.shape[:2], 0, states.shape[-1]))
    gather_index = indices.unsqueeze(-1).expand(*indices.shape, states.shape[-1])
    return states.gather(dim=2, index=gather_index)


def _scatter_tokens(states: torch.Tensor, indices: torch.Tensor | None, tokens: torch.Tensor | None) -> torch.Tensor:
    if indices is None or tokens is None or indices.numel() == 0:
        return states
    scatter_index = indices.unsqueeze(-1).expand_as(tokens)
    return states.scatter(dim=2, index=scatter_index, src=tokens)


def compress_minicache_pair(
    current_states: torch.Tensor,
    previous_states: torch.Tensor,
    *,
    interpolation: float = 0.6,
    retention_ratio: float = 0.05,
    retention_count: int | None = None,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compress one adjacent-layer KV pair using the MiniCache core contract."""

    shared_direction, current_magnitude, previous_magnitude, angle = minicache_slerp(
        current_states,
        previous_states,
        interpolation=interpolation,
        eps=eps,
    )
    angular_distance = angle / math.pi
    retained_indices = select_minicache_retention_indices(
        angular_distance,
        retention_ratio=retention_ratio,
        retention_count=retention_count,
    )
    current_retained = _gather_tokens(current_states, retained_indices)
    previous_retained = _gather_tokens(previous_states, retained_indices)
    return (
        shared_direction,
        current_magnitude,
        previous_magnitude,
        angular_distance,
        current_retained,
        previous_retained,
        retained_indices,
    )


def restore_minicache_pair(
    shared_direction: torch.Tensor,
    current_magnitude: torch.Tensor,
    previous_magnitude: torch.Tensor,
    *,
    current_retained: torch.Tensor | None = None,
    previous_retained: torch.Tensor | None = None,
    retained_indices: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Restore approximate current/previous states from a MiniCache pair."""

    if shared_direction.dim() != 4:
        raise ValueError("shared_direction must have shape [batch, heads, seq_len, head_dim]")
    if current_magnitude.shape != previous_magnitude.shape or current_magnitude.shape != shared_direction.shape[:-1] + (1,):
        raise ValueError("magnitudes must both have shape [batch, heads, seq_len, 1]")

    unit_direction = shared_direction / shared_direction.norm(dim=-1, keepdim=True).clamp_min(eps)
    current = unit_direction * current_magnitude
    previous = unit_direction * previous_magnitude
    current = _scatter_tokens(current, retained_indices, current_retained)
    previous = _scatter_tokens(previous, retained_indices, previous_retained)
    return current, previous
