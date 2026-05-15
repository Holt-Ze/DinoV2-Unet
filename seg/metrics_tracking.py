"""Per-epoch metrics history persistence."""

import json
import os
from typing import Any, Dict, List, Tuple


class MetricsHistory:
    """Track and persist train/validation metrics across epochs."""

    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        self.history: Dict[int, Dict[str, Dict[str, float]]] = {}

    def record_epoch(self, epoch: int, phase: str, metrics: Dict[str, float]) -> None:
        if epoch not in self.history:
            self.history[epoch] = {}
        self.history[epoch][phase] = metrics.copy()

    def to_dict(self) -> Dict[str, Any]:
        return {"metrics": self.history}

    def save_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {"metrics": {str(k): v for k, v in self.history.items()}}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_json(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.history = {int(k): v for k, v in data.get("metrics", {}).items()}

    def get_best_epoch(self, metric_name: str, phase: str = "val") -> Tuple[int, float]:
        best_epoch = -1
        best_value = -float("inf")
        for epoch, phases in self.history.items():
            if phase in phases and metric_name in phases[phase]:
                value = phases[phase][metric_name]
                if value > best_value:
                    best_value = value
                    best_epoch = epoch
        return best_epoch, best_value

    def get_metric_curve(
        self,
        metric_name: str,
        phase: str = "val",
    ) -> Tuple[List[int], List[float]]:
        epochs = []
        values = []
        for epoch in sorted(self.history.keys()):
            if phase in self.history[epoch] and metric_name in self.history[epoch][phase]:
                epochs.append(epoch)
                values.append(self.history[epoch][phase][metric_name])
        return epochs, values
