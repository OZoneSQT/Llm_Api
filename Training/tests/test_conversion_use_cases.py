from pathlib import Path

import pytest

from Training.app.use_cases import conversion
from Training.domain.entities import ConversionRequest


def test_detect_formats_identifies_known_extensions(tmp_path: Path) -> None:
    (tmp_path / "model.gguf").write_text("", encoding="utf-8")
    (tmp_path / "weights.ggml").write_text("", encoding="utf-8")
    (tmp_path / "quantized-GPTQ.pt").write_text("", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "params.safetensors").write_text("", encoding="utf-8")

    detection = conversion.detect_formats(tmp_path)

    assert detection.formats["gguf"] is True
    assert detection.formats["ggml"] is True
    assert detection.formats["gptq"] is True
    assert detection.formats["safetensors"] is True


def test_detect_formats_missing_root_raises(tmp_path: Path) -> None:
    missing = tmp_path / "absent"
    with pytest.raises(FileNotFoundError):
        conversion.detect_formats(missing)


def test_run_llama_conversion_executes_when_confirmed(monkeypatch, tmp_path: Path) -> None:
    script = tmp_path / "convert_llama.py"
    script.write_text("print('ok')", encoding="utf-8")

    monkeypatch.setattr(conversion, "_find_transformers_llama_converter", lambda: script)

    recorded = {}

    def fake_run(cmd, check):
        recorded["cmd"] = cmd

        class Result:
            returncode = 0

        return Result()

    monkeypatch.setattr(conversion.subprocess, "run", fake_run)

    request = ConversionRequest(
        input_dir=tmp_path / "input",
        output_dir=tmp_path / "output",
        model_size="7B",
        llama_version="3",
        confirm=True,
    )

    assert conversion.run_llama_conversion(request) is True
    assert recorded["cmd"][0] == conversion.sys.executable
    assert "--model_size" in recorded["cmd"]
    assert str(request.input_dir) in recorded["cmd"]


def test_run_llama_conversion_respects_dry_run(monkeypatch, tmp_path: Path) -> None:
    script = tmp_path / "convert_llama.py"
    script.write_text("print('ok')", encoding="utf-8")

    monkeypatch.setattr(conversion, "_find_transformers_llama_converter", lambda: script)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("subprocess.run should not be triggered during dry-run")

    monkeypatch.setattr(conversion.subprocess, "run", fail_if_called)

    request = ConversionRequest(
        input_dir=tmp_path / "input",
        output_dir=tmp_path / "output",
        dry_run=True,
        confirm=True,
    )

    assert conversion.run_llama_conversion(request) is True


def test_run_llama_conversion_returns_false_when_script_missing(tmp_path: Path) -> None:
    request = ConversionRequest(input_dir=tmp_path / "input", output_dir=tmp_path / "output", confirm=True)
    assert conversion.run_llama_conversion(request) is False


def test_run_auto_gptq_conversion_requires_availability(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(conversion, "is_auto_gptq_available", lambda: False)

    request = ConversionRequest(input_dir=tmp_path / "input", output_dir=tmp_path / "output", confirm=True)

    assert conversion.run_auto_gptq_conversion(request) is False


def test_run_auto_gptq_conversion_executes_when_available(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(conversion, "is_auto_gptq_available", lambda: True)

    recorded = {}

    def fake_run(cmd, check):
        recorded["cmd"] = cmd

        class Result:
            returncode = 0

        return Result()

    monkeypatch.setattr(conversion.subprocess, "run", fake_run)

    request = ConversionRequest(
        input_dir=tmp_path / "input",
        output_dir=tmp_path / "output",
        device="cuda",
        confirm=True,
    )

    assert conversion.run_auto_gptq_conversion(request) is True
    assert "--device" in recorded["cmd"]
    assert "cuda" in recorded["cmd"]


def test_run_auto_gptq_conversion_dry_run(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(conversion, "is_auto_gptq_available", lambda: True)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("subprocess.run should not be triggered during dry-run")

    monkeypatch.setattr(conversion.subprocess, "run", fail_if_called)

    request = ConversionRequest(
        input_dir=tmp_path / "input",
        output_dir=tmp_path / "output",
        dry_run=True,
        confirm=True,
    )

    assert conversion.run_auto_gptq_conversion(request) is True


def test_run_auto_gptq_conversion_requires_confirmation(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(conversion, "is_auto_gptq_available", lambda: True)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("subprocess.run should not be triggered without confirmation")

    monkeypatch.setattr(conversion.subprocess, "run", fail_if_called)

    request = ConversionRequest(
        input_dir=tmp_path / "input",
        output_dir=tmp_path / "output",
        confirm=False,
    )

    assert conversion.run_auto_gptq_conversion(request) is False