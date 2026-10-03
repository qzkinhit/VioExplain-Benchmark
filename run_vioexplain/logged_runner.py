"""Run a public entry point with a separate, durable execution log.

Examples:
    python -m run_vioexplain.logged_runner --log-root log formal statistical ...
    python -m run_vioexplain.logged_runner --log-root log smoke

The child keeps its own result protocol and completion marker. This wrapper
records process execution only; a zero exit code does not validate a score.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
ENTRY_MODULES = {
    "formal": "run_vioexplain.formal_runner",
    "smoke": "vioexplain.cli",
    "prototype-smoke": "vioexplain.cli",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def source_hashes(root: Path) -> dict[str, str]:
    """Hash executable source, excluding data, logs and the user's environment."""
    files = []
    for directory in ("src", "run_vioexplain", "benchmark"):
        files.extend((root / directory).rglob("*.py"))
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(files) if path.is_file() and not path.is_symlink()
    }


def run_recorded(argv: list[str], directory: Path, *, cwd: Path = ROOT) -> int:
    """Capture combined stdout/stderr and retain failure evidence without retry."""
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = {
        "schema_version": 1,
        "started_at_utc": utc_now(),
        "argv": argv,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "source_sha256": source_hashes(cwd),
        "environment": {key: os.environ[key] for key in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "CUDA_VISIBLE_DEVICES",
        ) if key in os.environ},
        "status": "started",
        "scientific_validation": "not_performed_by_process_wrapper",
    }
    path = directory / "execution.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    log_path = directory / "console.log"
    try:
        with log_path.open("xb", buffering=0) as log:
            process = subprocess.Popen(
                argv, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            assert process.stdout is not None
            for chunk in iter(lambda: process.stdout.read(4096), b""):
                log.write(chunk)
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
            exit_code = process.wait()
        manifest.update(status="exited", exit_code=exit_code)
    except BaseException as exc:
        manifest.update(status="wrapper_failed", exception_type=type(exc).__name__,
                        error=str(exc))
        raise
    finally:
        manifest.update(finished_at_utc=utc_now(), elapsed_seconds=time.monotonic() - started)
        if log_path.exists():
            manifest["console_sha256"] = hashlib.sha256(log_path.read_bytes()).hexdigest()
        path.write_text(json.dumps(manifest, indent=2) + "\n")
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-root", type=Path, default=ROOT / "log" / "runs")
    parser.add_argument("task", choices=ENTRY_MODULES)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    arguments = ([args.task] if args.task != "formal" else []) + args.arguments
    command = [sys.executable, "-m", ENTRY_MODULES[args.task], *arguments]
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    directory = args.log_root / args.task / run_id
    print(f"Execution evidence: {directory}", flush=True)
    return run_recorded(command, directory)


if __name__ == "__main__":
    raise SystemExit(main())
