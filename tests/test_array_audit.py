"""Holdout protection must not reject integrity reads of freshly saved artifacts."""
import hashlib
import os

import numpy as np
import pytest

from vioexplain.utils.array_audit import ArrayAccessAudit


def test_declared_input_and_created_output_are_distinct(tmp_path):
    source = tmp_path / "fit.npz"
    np.savez(source, x=np.arange(4))
    output = tmp_path / "run"
    output.mkdir()
    with ArrayAccessAudit([source], output) as audit:
        with np.load(source, allow_pickle=False) as values:
            x = values["x"].copy()
        generated = output / "features.npz"
        np.savez_compressed(generated, features=x * 2)
        digest = hashlib.sha256(generated.read_bytes()).hexdigest()
        with np.load(generated, allow_pickle=False) as values:
            np.testing.assert_array_equal(values["features"], x * 2)
    assert len(digest) == 64
    assert audit.read_inputs == {source.resolve()}
    assert audit.read_outputs == {generated.resolve()}


def test_existing_output_and_holdout_are_not_implicitly_authorized(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    holdout = tmp_path / "test.npy"
    existing = output / "earlier.npy"
    np.save(holdout, [1])
    np.save(existing, [2])
    with ArrayAccessAudit([], output):
        for path in [holdout, existing]:
            with pytest.raises(RuntimeError, match="Undeclared"):
                path.read_bytes()


def test_symlink_cannot_reclassify_external_file_as_generated(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    external = tmp_path / "heldout.npy"
    np.save(external, [1])
    (output / "alias.npy").symlink_to(external)
    with ArrayAccessAudit([], output):
        with pytest.raises(RuntimeError, match="outside"):
            (output / "alias.npy").write_bytes(b"changed")
    np.testing.assert_array_equal(np.load(external), [1])


def test_low_level_output_write_can_be_hashed(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    generated = output / "stage.dat"
    with ArrayAccessAudit([], output) as audit:
        descriptor = os.open(generated, os.O_WRONLY | os.O_CREAT, 0o600)
        os.write(descriptor, b"artifact")
        os.close(descriptor)
        assert generated.read_bytes() == b"artifact"
    assert generated.resolve() in audit.read_outputs
