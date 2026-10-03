"""Verify the exact released formal CSV and manifest files against their index."""
from pathlib import Path
import argparse
import hashlib
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("result/summary/formal_v1"))
    args = parser.parse_args()
    root = args.root.resolve()
    index = json.loads((root / "INDEX.json").read_text())
    count = 0
    for run in index:
        for relative, expected in run["files"].items():
            path = (root / run["run"] / relative).resolve()
            if not path.is_relative_to(root):
                raise ValueError("Result index contains an out-of-root path")
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f"Result hash mismatch: {run['run']}/{relative}")
            count += 1
    snapshots = root / "executed_sources" / "manifest.json"
    snapshot_count = 0
    if snapshots.exists():
        for row in json.loads(snapshots.read_text())["files"]:
            path = (root / row["file"]).resolve()
            if not path.is_relative_to(root):
                raise ValueError("Snapshot manifest contains an out-of-root path")
            if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError(f"Executed source hash mismatch: {row['file']}")
            snapshot_count += 1
    event_root = root.parent / "event_matching_v1"
    event_count = 0
    locked_sources = 0
    if (event_root / "export_manifest.json").exists():
        for row in json.loads((event_root / "export_manifest.json").read_text())["files"]:
            path = (event_root / row["file"]).resolve()
            if not path.is_relative_to(event_root.resolve()):
                raise ValueError("Event manifest contains an out-of-root path")
            if hashlib.sha256(path.read_bytes()).hexdigest() != row["export_sha256"]:
                raise ValueError(f"Event artifact hash mismatch: {row['file']}")
            event_count += 1
        lock_path = event_root / "LOCK.json"
        lock = json.loads(lock_path.read_text())
        for name, digest in lock["source_hashes"].items():
            path = (event_root / "source_snapshot" / name).resolve()
            if not path.is_relative_to(event_root.resolve()):
                raise ValueError("LOCK contains an out-of-root source path")
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError(f"Frozen event source does not match LOCK: {name}")
            locked_sources += 1
        lock_digest = hashlib.sha256(lock_path.read_bytes()).hexdigest()
        if json.loads((event_root / "TEST_STARTED.json").read_text())["lock_sha256"] != lock_digest:
            raise ValueError("Test-start marker does not match event LOCK")
        if hashlib.sha256((event_root / "protocol_locked.md").read_bytes()).hexdigest() != lock["protocol_sha256"]:
            raise ValueError("Saved event protocol does not match event LOCK")
    print(json.dumps({"status": "passed", "completed_runs": len(index), "verified_files": count,
                      "verified_executed_sources": snapshot_count,
                      "verified_event_artifacts": event_count, "event_locked_sources": locked_sources}))


if __name__ == "__main__":
    main()
