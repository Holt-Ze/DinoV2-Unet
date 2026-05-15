"""Checkpoint loading helpers shared by training, evaluation, and export."""

from typing import Any, Dict, Mapping, Tuple


PROFILE_SUFFIXES = ("total_ops", "total_params")


def clean_state_dict(
    state: Mapping[str, Any],
    drop_aux_heads: bool = True,
) -> Dict[str, Any]:
    """Normalize checkpoint keys before loading them into a model."""
    cleaned = {}
    for key, value in state.items():
        if key.startswith("module."):
            key = key.replace("module.", "", 1)
        if key.endswith(PROFILE_SUFFIXES):
            continue
        if drop_aux_heads and key.startswith("aux_heads"):
            continue
        cleaned[key] = value
    return cleaned


def load_checkpoint_state(
    checkpoint_path: str,
    drop_aux_heads: bool = True,
) -> Dict[str, Any]:
    """Load a checkpoint and return a cleaned model state dict."""
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    if not isinstance(state, Mapping):
        raise TypeError(f"Checkpoint does not contain a state dict: {checkpoint_path}")
    return clean_state_dict(state, drop_aux_heads=drop_aux_heads)


def load_model_state(
    model,
    checkpoint_path: str,
    drop_aux_heads: bool = True,
    strict: bool = False,
) -> Tuple[list, list]:
    """Load a cleaned checkpoint state into a model."""
    state = load_checkpoint_state(checkpoint_path, drop_aux_heads=drop_aux_heads)
    return model.load_state_dict(state, strict=strict)
