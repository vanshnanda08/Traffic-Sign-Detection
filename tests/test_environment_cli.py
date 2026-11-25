"""Device precedence regressions without modifying the working config or training."""

import importlib.util

import pytest
import torch

import p57.environment as environment


@pytest.fixture
def cli():
    spec = importlib.util.spec_from_file_location("validate_cli", environment.ROOT / "scripts/validate_environment.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("configured,flags,expected_argument", [
    ("cpu", [], None),
    ("mps", ["--device", "cpu"], "cpu"),
    ("cpu", ["--device", "auto"], "auto"),
])
def test_cli_respects_config_unless_device_explicit(cli, monkeypatch, tmp_path, configured, flags, expected_argument):
    cfg = {**environment.load_config(), "device": configured}
    monkeypatch.setattr(environment, "load_config", lambda: cfg)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    calls = []
    original = environment.validate_environment

    def capture(requested):
        calls.append(requested)
        result = original(requested)
        assert result["device"] == "cpu"
        return result

    monkeypatch.setattr(cli, "validate_environment", capture)
    cli.main([*flags, "--report-dir", str(tmp_path)])
    assert calls == [expected_argument]
    assert (tmp_path / "environment.json").is_file()


def test_explicit_unavailable_mps_still_raises(cli, monkeypatch, tmp_path):
    cfg = {**environment.load_config(), "device": "cpu"}
    monkeypatch.setattr(environment, "load_config", lambda: cfg)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="MPS was requested but is unavailable"):
        cli.main(["--device", "mps", "--report-dir", str(tmp_path)])
    assert not (tmp_path / "environment.json").exists()
