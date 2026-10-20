"""Public benchmark records must not identify the builder's host or checkout."""

from __future__ import annotations

import importlib.util
import json
import platform
from argparse import Namespace
from pathlib import Path

import pytest


def test_runtime_report_omits_host_and_git_identity(tmp_path, monkeypatch):
    pytest.importorskip("numba")
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "bench_runtime_audit.py"
    spec = importlib.util.spec_from_file_location("review_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    private_host = "private-compute-host.invalid"
    private_commit = "a" * 40
    monkeypatch.setattr(platform, "node", lambda: private_host)

    def command_output(command, **kwargs):
        return private_commit if command[0] == "git" else "Generic CPU information"

    monkeypatch.setattr(module.subprocess, "check_output", command_output)
    monkeypatch.setattr(module.os, "sched_getaffinity", lambda _: {0}, raising=False)
    monkeypatch.setattr(module.os, "getloadavg", lambda: (0.0, 0.0, 0.0), raising=False)
    output = tmp_path / "audit.json"
    module.Audit(Namespace(section="variants", repeats=1, delta=0.2, output=output))
    text = output.read_text(encoding="utf-8")
    assert private_host not in text
    assert private_commit not in text
    assert json.loads(text)["source_sha256"]
