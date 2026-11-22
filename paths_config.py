#!/usr/bin/env python3
"""
Portable path configuration for the training pipeline.

All modules must import paths from here instead of hardcoding absolute
Windows paths. The project root resolves as:

1. ``GEMMA4_ROOT`` environment variable (for Kaggle/Linux deployments)
2. Parent directory of ``training/`` (auto-detected from this file's location)
"""

import os
from pathlib import Path


def _resolve_project_root() -> Path:
    env_root = os.environ.get("GEMMA4_ROOT", "").strip()
    if env_root:
        return Path(env_root).resolve()
    # this file lives at <root>/training/src/paths_config.py
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT = _resolve_project_root()

# Harness / mini-eval locations (read-only inputs)
HARNESS_DIR = PROJECT_ROOT / "harness_v1_20261119"
MINI_EVAL_DIR = PROJECT_ROOT / "mini_eval"
RECON_PATH = MINI_EVAL_DIR / "reconstructed.json"
REPOS_DIR = MINI_EVAL_DIR / "repos"
RESULTS_DIR = MINI_EVAL_DIR / "results"

# Training pipeline locations
TRAINING_DIR = PROJECT_ROOT / "training"
SRC_DIR = TRAINING_DIR / "src"
DATA_DIR = TRAINING_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
SNAPSHOT_DIR = DATA_DIR / "snapshots"
PROCESSED_DIR = DATA_DIR / "processed"
ADAPTERS_DIR = TRAINING_DIR / "adapters"

# Standard processed artifacts
CONTEXTS_DIR = PROCESSED_DIR / "contexts"
VALIDATION_SUMMARY = PROCESSED_DIR / "validation_summary.json"
AUDIT_REPORT = PROCESSED_DIR / "gates" / "audit_report.json"
ELIGIBLE_SAMPLES = PROCESSED_DIR / "gates" / "eligible_samples.jsonl"

SFT_DATA_PATH = DATA_DIR / "sft_verified_patches.jsonl"
DPO_DATA_PATH = DATA_DIR / "dpo_verified_patches.jsonl"


def repo_dir_for(instance_id: str) -> Path:
    """Repo snapshot directory for a task instance."""
    return REPOS_DIR / instance_id


__all__ = [
    "PROJECT_ROOT", "HARNESS_DIR", "MINI_EVAL_DIR", "RECON_PATH", "REPOS_DIR",
    "RESULTS_DIR", "TRAINING_DIR", "SRC_DIR", "DATA_DIR", "RAW_DIR",
    "SNAPSHOT_DIR", "PROCESSED_DIR", "ADAPTERS_DIR", "CONTEXTS_DIR",
    "VALIDATION_SUMMARY", "AUDIT_REPORT", "ELIGIBLE_SAMPLES",
    "SFT_DATA_PATH", "DPO_DATA_PATH", "repo_dir_for",
]
