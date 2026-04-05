"""Metrics tracking and logging infrastructure.

This module provides structured metric collection capabilities:
- MetricsHistory: Per-epoch metrics tracking with JSON persistence
- GradientTracker: Optional gradient flow monitoring
- ActivationTracker: Optional activation statistics tracking
"""

import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import numpy as np


class MetricsHistory:
    """Track and persist metrics across training epochs.

    Stores metrics for train and validation phases, with automatic JSON
    serialization after each epoch. Enables post-hoc analysis and plotting.

    Args:
        save_dir: Directory where metrics will be saved.
    """

    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        self.history: Dict[int, Dict[str, Dict[str, float]]] = {}
        self.gradient_history: List[Dict[str, Any]] = []
        self.activation_history: List[Dict[str, Any]] = []

    def record_epoch(self, epoch: int, phase: str, metrics: Dict[str, float]) -> None:
        """Record metrics for a specific phase (train/val) in an epoch.

        Args:
            epoch: Epoch number (int).
            phase: Phase identifier ('train' or 'val').
            metrics: Dictionary of metric_name -> metric_value.
        """
        if epoch not in self.history:
            self.history[epoch] = {}
        self.history[epoch][phase] = metrics.copy()

    def record_gradients(self, step: int, gradient_stats: Dict[str, Dict]) -> None:
        """Record gradient statistics for a training step.

        Args:
            step: Global training step number.
            gradient_stats: Dict of layer_name -> {norm, mean, std, min, max}.
        """
        entry = {"step": step, **gradient_stats}
        self.gradient_history.append(entry)

    def record_activations(self, step: int, activation_stats: Dict[str, Dict]) -> None:
        """Record activation statistics for a training step.

        Args:
            step: Global training step number.
            activation_stats: Dict of layer_name -> {mean, std, sparsity, saturation}.
        """
        entry = {"step": step, **activation_stats}
        self.activation_history.append(entry)

    def to_dict(self) -> Dict[str, Any]:
        """Convert history to a serializable dictionary."""
        result = {
            "metrics": self.history,
        }
        if self.gradient_history:
            result["gradients"] = self.gradient_history
        if self.activation_history:
            result["activations"] = self.activation_history
        return result

    def save_json(self, path: str) -> None:
        """Persist metrics to a JSON file.

        Args:
            path: Output file path.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # Convert epoch keys (int) to strings for JSON serialization
        data = {
            "metrics": {str(k): v for k, v in self.history.items()},
        }
        if self.gradient_history:
            data["gradients"] = self.gradient_history
        if self.activation_history:
            data["activations"] = self.activation_history

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load_json(self, path: str) -> None:
        """Load metrics from a JSON file.

        Args:
            path: Input file path.
        """
        with open(path, "r") as f:
            data = json.load(f)

        # Convert string keys back to int
        self.history = {int(k): v for k, v in data.get("metrics", {}).items()}
        self.gradient_history = data.get("gradients", [])
        self.activation_history = data.get("activations", [])

    def get_best_epoch(self, metric_name: str, phase: str = "val") -> Tuple[int, float]:
        """Find the epoch with best metric value.

        Args:
            metric_name: Name of metric to search.
            phase: Phase to search in ('val' or 'train').

        Returns:
            Tuple of (epoch_number, metric_value).
        """
        best_epoch = -1
        best_value = -np.inf

        for epoch, phases in self.history.items():
            if phase in phases and metric_name in phases[phase]:
                value = phases[phase][metric_name]
                if value > best_value:
                    best_value = value
                    best_epoch = epoch

        return best_epoch, best_value

    def get_metric_curve(self, metric_name: str, phase: str = "val") -> Tuple[List[int], List[float]]:
        """Extract a single metric's evolution across epochs.

        Args:
            metric_name: Name of metric to extract.
            phase: Phase to extract from.

        Returns:
            Tuple of (epoch_numbers, metric_values).
        """
        epochs = []
        values = []

        for epoch in sorted(self.history.keys()):
            if phase in self.history[epoch]:
                if metric_name in self.history[epoch][phase]:
                    epochs.append(epoch)
                    values.append(self.history[epoch][phase][metric_name])

        return epochs, values


class GradientTracker:
    """Monitor gradient flow during training.

    Tracks gradient statistics (norm, mean, std, min, max) for specified
    layers. Useful for debugging gradient vanishing/explosion issues.

    Args:
        model: PyTorch model to track.
        tracked_layers: List of layer names to monitor. If None, all layers.
    """

    def __init__(self, model: nn.Module, tracked_layers: Optional[List[str]] = None):
        self.model = model
        self.tracked_layers = tracked_layers
        self.grad_history = []

    def capture_grads(self) -> Dict[str, Dict[str, float]]:
        """Capture gradient statistics for tracked layers.

        Returns:
            Dict mapping layer_name -> {norm, mean, std, min, max}.
        """
        grads = {}

        for name, param in self.model.named_parameters():
            # Filter by tracked layers
            if self.tracked_layers is not None:
                if not any(layer in name for layer in self.tracked_layers):
                    continue

            if param.grad is not None:
                g = param.grad.detach().flatten()
                norm = g.norm().item()

                grads[name] = {
                    "norm": norm,
                    "mean": g.mean().item(),
                    "std": g.std().item() if len(g) > 1 else 0.0,
                    "min": g.min().item(),
                    "max": g.max().item(),
                }

        return grads

    def record_step(self, step: int, grads: Dict[str, Dict]) -> None:
        """Record gradient statistics for a training step.

        Args:
            step: Global training step number.
            grads: Gradient statistics from capture_grads().
        """
        entry = {"step": step, **grads}
        self.grad_history.append(entry)

    def detect_dying_neurons(self) -> Dict[str, float]:
        """Detect percentage of dying neurons (near-zero activations).

        Note: This requires hooks on activation layers; returns dummy
        data for now.

        Returns:
            Dict mapping layer_name -> dead_fraction (0 to 1).
        """
        # TODO: Implement with forward hooks
        return {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert history to dictionary."""
        return {"gradients": self.grad_history}


class ActivationTracker:
    """Monitor activation statistics during training.

    Uses forward hooks to track mean, std, sparsity, and saturation
    for specified layers.

    Args:
        model: PyTorch model to track.
        tracked_layers: List of layer names or patterns to monitor.
    """

    def __init__(self, model: nn.Module, tracked_layers: Optional[List[str]] = None):
        self.model = model
        self.tracked_layers = tracked_layers or []
        self.act_history = []
        self.activations = {}
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks to capture activations."""
        for name, module in self.model.named_modules():
            # Filter by tracked layers
            if self.tracked_layers:
                if not any(layer in name for layer in self.tracked_layers):
                    continue

            # Only hook on recognized layer types
            if not isinstance(module, (nn.Linear, nn.Conv2d, nn.BatchNorm2d)):
                continue

            def hook_fn(module, input, output, layer_name=name):
                if isinstance(output, torch.Tensor):
                    self.activations[layer_name] = output.detach()

            module.register_forward_hook(hook_fn)

    def capture_activations(self) -> Dict[str, Dict[str, float]]:
        """Capture activation statistics for hooked layers.

        Returns:
            Dict mapping layer_name -> {mean, std, sparsity, saturation}.
        """
        act_stats = {}

        for name, acts in self.activations.items():
            if acts is None:
                continue

            acts_flat = acts.detach().flatten()

            # Compute statistics
            mean = acts_flat.mean().item()
            std = acts_flat.std().item() if len(acts_flat) > 1 else 0.0

            # Sparsity: fraction of zero or near-zero activations
            sparsity = (acts_flat.abs() < 1e-6).float().mean().item()

            # Saturation: fraction of values near ReLU saturation (>10) or negative
            saturation = ((acts_flat > 10) | (acts_flat < -10)).float().mean().item()

            act_stats[name] = {
                "mean": mean,
                "std": std,
                "sparsity": sparsity,
                "saturation": saturation,
            }

        self.activations.clear()  # Reset for next capture
        return act_stats

    def record_step(self, step: int, act_stats: Dict[str, Dict]) -> None:
        """Record activation statistics for a training step.

        Args:
            step: Global training step number.
            act_stats: Activation statistics from capture_activations().
        """
        entry = {"step": step, **act_stats}
        self.act_history.append(entry)

    def to_dict(self) -> Dict[str, Any]:
        """Convert history to dictionary."""
        return {"activations": self.act_history}
