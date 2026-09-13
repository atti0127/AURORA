"""AURORA: embedding routing, mask refinement, and D-only scoring."""

import torch


def route_local_queries(clip_embeddings, frame_embeddings, query_indices):
    """Route selected video queries to their strongest local query per frame."""
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
    selected_similarity = similarity[:, query_indices]
    route_values, local_indices = selected_similarity.max(dim=-1)
    return local_indices.transpose(0, 1), (route_values > 0.0).transpose(0, 1)


def routed_mask_statistics(global_logits, local_logits, support):
    """Accumulate binary routed-mask agreement for track re-ranking."""
    if global_logits.shape != local_logits.shape or global_logits.ndim != 4:
        raise ValueError(
            "global_logits and local_logits must share [tracks, frames, H, W]"
        )
    if support.shape != global_logits.shape[:2]:
        raise ValueError("support must have shape [tracks, frames]")

    global_foreground = global_logits > 0.0
    local_foreground = local_logits > 0.0
    valid = support[:, :, None, None]

    hard_intersection = (
        global_foreground & local_foreground & valid
    ).sum(dim=(1, 2, 3)).float()
    global_area = (global_foreground & valid).sum(dim=(1, 2, 3)).float()
    local_area = (local_foreground & valid).sum(dim=(1, 2, 3)).float()

    return {
        "hard_intersection": hard_intersection,
        "global_area": global_area,
        "local_area": local_area,
    }


def aurora_score(canonical_score, hard_dice):
    """Multiply canonical confidence by routed Dice reliability."""
    return canonical_score * hard_dice
