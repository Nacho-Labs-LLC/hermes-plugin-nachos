"""Synthetic-only source acquisition boundary tests; no inventory wiring."""

import os
from pathlib import Path

import pytest

from nachos_core.reconciliation.boundary import AcquisitionError, SourceBoundary


def _ident(p):
    st = p.lstat()
    return (st.st_mode, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_nlink)


def test_file_snapshot_is_immutable_and_does_not_write(tmp_path):
    source = tmp_path / "MEMORY.md"
    source.write_bytes(b"fixture memory\n")
    before = _ident(source)
    snapshot = SourceBoundary(tmp_path, allowed_labels=("MEMORY.md",)).read_file("MEMORY.md")
    assert snapshot.data == b"fixture memory\n"
    assert snapshot.label == "MEMORY.md"
    assert _ident(source) == before
    assert list(tmp_path.iterdir()) == [source]
    with pytest.raises((AttributeError, TypeError)):
        snapshot.data = b"changed"


@pytest.mark.parametrize(
    "label",
    [
        "../MEMORY.md",
        "/MEMORY.md",
        "a/../MEMORY.md",
        "a\\b",
        "a:b",
        "a\0b",
        "./MEMORY.md",
        "a//b",
        "a/",
    ],
)
def test_rejects_noncanonical_labels(tmp_path, label):
    with pytest.raises(AcquisitionError, match="invalid_label"):
        SourceBoundary(tmp_path, allowed_labels=(label,))


def test_allowlist_is_exact_and_credentials_are_not_sources(tmp_path):
    (tmp_path / "other.md").write_text("fixture")
    boundary = SourceBoundary(tmp_path, allowed_labels=("MEMORY.md",))
    with pytest.raises(AcquisitionError, match="not_allowlisted"):
        boundary.read_file("other.md")
    with pytest.raises(AcquisitionError, match="credential_path"):
        SourceBoundary(tmp_path, allowed_labels=(".env",))


def test_rejects_symlinks_hardlinks_nonregular_and_missing(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    (real / "memory").write_bytes(b"fixture")
    (tmp_path / "alias").symlink_to(real, target_is_directory=True)
    (tmp_path / "link").symlink_to(real / "memory")
    os.link(real / "memory", tmp_path / "hard")
    labels = ("alias/memory", "link", "hard", "real", "missing")
    boundary = SourceBoundary(tmp_path, allowed_labels=labels)
    for label in labels:
        with pytest.raises(AcquisitionError):
            boundary.read_file(label)
    assert not (tmp_path / "missing").exists()


def test_file_limit(tmp_path):
    (tmp_path / "memory").write_bytes(b"12345")
    boundary = SourceBoundary(tmp_path, allowed_labels=("memory",))
    with pytest.raises(AcquisitionError, match="size_limit"):
        boundary.read_file("memory", max_bytes=4)


def test_trusted_root_must_be_absolute_and_not_symlink(tmp_path):
    with pytest.raises(AcquisitionError):
        SourceBoundary(Path("relative"), allowed_labels=("memory",))
    (tmp_path / "actual").mkdir()
    (tmp_path / "actual" / "memory").write_bytes(b"fixture")
    # The same file is readable through the real root, so only the root alias rejects it.
    assert SourceBoundary(tmp_path / "actual", allowed_labels=("memory",)).read_file("memory").data
    (tmp_path / "alias").symlink_to(tmp_path / "actual", target_is_directory=True)
    boundary = SourceBoundary(tmp_path / "alias", allowed_labels=("memory",))
    with pytest.raises(AcquisitionError):
        boundary.read_file("memory")
