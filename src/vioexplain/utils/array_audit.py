"""Audit research-array reads while allowing a run to verify its own outputs.

This is a guard against accidental partition access, not a security sandbox.
The caller still validates hashes, row identities and fitting partitions.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys


class ArrayAccessAudit:
    """Allow declared inputs and arrays created by this run inside its output root.

    Merely being inside the output directory does not authorize an existing
    array. A new output must first be opened for writing while this guard is
    active. Its later read for hashing, serialization checks or inference is
    then recorded as a generated-output read, not a new research-data input.
    """

    EXTENSIONS = {".npy", ".npz", ".rdata", ".dat"}

    def __init__(self, inputs, output_root):
        self.inputs = {Path(path).resolve() for path in inputs}
        self.output_root = Path(output_root).resolve()
        self.read_inputs = set()
        self.created_outputs = set()
        self.read_outputs = set()
        self.active = False
        self.installed = False

    def __enter__(self):
        if self.installed:
            raise RuntimeError("Create a new array audit for each run")
        self.active = True
        self.installed = True
        sys.addaudithook(self._on_event)
        return self

    def __exit__(self, *exception):
        self.active = False

    def _on_event(self, event, arguments):
        if not self.active or event != "open" or not arguments:
            return
        name = arguments[0]
        if not isinstance(name, (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(name)).resolve()
        if path.suffix.lower() not in self.EXTENSIONS:
            return
        mode = arguments[1] if len(arguments) > 1 else None
        flags = arguments[2] if len(arguments) > 2 else 0
        writing = (isinstance(mode, str) and any(char in mode for char in "wax+")) or (
            isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
        )
        if writing:
            if not path.is_relative_to(self.output_root) or path in self.inputs:
                raise RuntimeError(f"Research-array write outside this run: {path}")
            self.created_outputs.add(path)
        elif path in self.inputs:
            self.read_inputs.add(path)
        elif path in self.created_outputs and path.is_relative_to(self.output_root):
            self.read_outputs.add(path)
        else:
            raise RuntimeError(f"Undeclared research-array read: {path}")

    def report(self):
        return {
            "declared_inputs": sorted(map(str, self.inputs)),
            "inputs_actually_read": sorted(map(str, self.read_inputs)),
            "arrays_created_in_run": sorted(map(str, self.created_outputs)),
            "generated_arrays_read_back": sorted(map(str, self.read_outputs)),
            "scope": "partition-access audit; hashes and split identities require separate verification",
        }
