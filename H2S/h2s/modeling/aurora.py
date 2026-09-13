"""Training-free AURORA: embedding routing, mask refinement, and D-only scoring."""

import torch


H2S_TRACK_ADMISSION_THRESHOLD = 0.3


def route_local_queries(clip_embeddings, frame_embeddings, query_indices):
    """Route selected video queries to H2S frame queries."""
    if clip_embeddings.ndim != 2:
        raise ValueError("clip_embeddings must have shape [queries, channels]")
    if frame_embeddings.ndim != 3:
        raise ValueError(
            "frame_embeddings must have shape [frames, queries, channels]"
        )
    if query_indices.ndim != 1:
        raise ValueError("query_indices must be one-dimensional")

    similarity = torch.einsum(
        "qc,tfc->tqf", clip_embeddings.float(), frame_embeddings.float()
    )
    selected = similarity[:, query_indices]
    route_values, local_indices = selected.max(dim=-1)

    return {
        "local_query_indices": local_indices.transpose(0, 1),
        "positive_matches": (route_values > 0.0).transpose(0, 1),
    }


def refine_routed_masks(global_logits, local_logits, positive_matches):
    """Fuse routed masks and retain frames with shared foreground support."""
    if global_logits.shape != local_logits.shape or global_logits.ndim != 4:
        raise ValueError(
            "global_logits and local_logits must share [tracks, frames, H, W]"
        )
    if positive_matches.shape != global_logits.shape[:2]:
        raise ValueError("positive_matches must have shape [tracks, frames]")

    spatial_support = (
        (global_logits > 0.0) & (local_logits > 0.0)
    ).flatten(2).any(-1)
    fused = 0.5 * (global_logits + local_logits)
    refined = torch.where(
        positive_matches[:, :, None, None], fused, global_logits
    )
    refined = refined.masked_fill(
        ~spatial_support[:, :, None, None], -20.0
    )
    return refined, spatial_support


def routed_dice_statistics(global_logits, local_logits, support):
    """Accumulate binary routed-mask statistics on supported frames."""
    if global_logits.shape != local_logits.shape or global_logits.ndim != 4:
        raise ValueError(
            "global_logits and local_logits must share [tracks, frames, H, W]"
        )
    if support.shape != global_logits.shape[:2]:
        raise ValueError("support must have shape [tracks, frames]")

    global_foreground = global_logits > 0.0
    local_foreground = local_logits > 0.0
    valid = support[:, :, None, None]
    return {
        "hard_intersection": (
            global_foreground & local_foreground & valid
        ).sum(dim=(1, 2, 3)).float(),
        "global_area": (
            global_foreground & valid
        ).sum(dim=(1, 2, 3)).float(),
        "local_area": (
            local_foreground & valid
        ).sum(dim=(1, 2, 3)).float(),
    }


def aurora_score(canonical_score, hard_dice):
    """Multiply canonical confidence by routed Dice reliability."""
    return canonical_score * hard_dice
