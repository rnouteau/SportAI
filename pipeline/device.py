"""Resolve inference device from config / env (single source of truth)."""

from __future__ import annotations

import os

import torch
from loguru import logger


def resolve_device(requested: str | None = None) -> str:
    """Return a usable device string.

    Priority: ``SPORTAI_DEVICE`` env (if set) overrides ``requested``.
    Falls back to CPU when CUDA is requested but unavailable.
    """
    env = os.environ.get("SPORTAI_DEVICE")
    device = str(env if env else (requested or "cpu"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA is not available, using CPU instead")
        return "cpu"
    return device
