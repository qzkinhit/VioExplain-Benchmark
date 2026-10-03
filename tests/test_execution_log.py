"""Execution evidence must survive a failed child and refuse accidental reuse."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

from run_vioexplain.logged_runner import run_recorded


def test_failure_keeps_both_streams_and_exit_status(tmp_path, monkeypatch):
    monkeypatch.setenv("VIO_TEST_SECRET", "do-not-record-this-value")
    directory = tmp_path / "failed"
    code = "import sys; print('out', flush=True); print('err', file=sys.stderr); sys.exit(7)"
    status = run_recorded([sys.executable, "-c", code], directory, cwd=tmp_path)
    assert status == 7
    log = (directory / "console.log").read_bytes()
    assert b"out" in log and b"err" in log
    record = json.loads((directory / "execution.json").read_text())
    assert record["exit_code"] == 7
    assert record["console_sha256"] == hashlib.sha256(log).hexdigest()
    assert record["scientific_validation"] == "not_performed_by_process_wrapper"
    assert "do-not-record-this-value" not in json.dumps(record)
    assert not (directory / "COMPLETE.json").exists()


def test_existing_run_is_never_overwritten(tmp_path):
    directory = tmp_path / "existing"
    directory.mkdir()
    marker = directory / "console.log"
    marker.write_text("earlier evidence")
    with pytest.raises(FileExistsError):
        run_recorded([sys.executable, "-c", "print('new')"], directory, cwd=tmp_path)
    assert marker.read_text() == "earlier evidence"
