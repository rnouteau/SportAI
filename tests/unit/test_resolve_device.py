"""Unit tests for pipeline.device.resolve_device."""

from __future__ import annotations

import pytest

from pipeline.device import resolve_device


@pytest.mark.unit
def test_resolve_device_defaults_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SPORTAI_DEVICE", raising=False)
    assert resolve_device(None) == "cpu"
    assert resolve_device("cpu") == "cpu"


@pytest.mark.unit
def test_resolve_device_env_overrides_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPORTAI_DEVICE", "cpu")
    assert resolve_device("cuda") == "cpu"


@pytest.mark.unit
def test_resolve_device_cuda_falls_back_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SPORTAI_DEVICE", raising=False)
    monkeypatch.setattr("pipeline.device.torch.cuda.is_available", lambda: False)
    assert resolve_device("cuda") == "cpu"
    assert resolve_device("cuda:0") == "cpu"


@pytest.mark.unit
def test_resolve_device_keeps_cuda_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SPORTAI_DEVICE", raising=False)
    monkeypatch.setattr("pipeline.device.torch.cuda.is_available", lambda: True)
    assert resolve_device("cuda") == "cuda"
    assert resolve_device("cuda:1") == "cuda:1"
