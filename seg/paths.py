"""Default path configuration for data and output directories.

Paths can be overridden via environment variables:
- DATA_ROOT: Root directory for datasets (default: ./data).
- RUNS_ROOT: Root directory for training outputs (default: ./runs).
"""

import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_ROOT = os.path.abspath(
    os.environ.get("DATA_ROOT", os.path.join(SCRIPT_DIR, "..", "data"))
)
DEFAULT_RUNS_ROOT = os.path.abspath(
    os.environ.get("RUNS_ROOT", os.path.join(SCRIPT_DIR, "..", "runs"))
)
