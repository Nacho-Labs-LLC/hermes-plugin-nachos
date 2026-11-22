"""Bounded source bytes, not review approval or imported-locator authority.

Only trusted application adapters may construct roots/allowlists. No discovery,
recursive enumeration, configuration lookup, or writes occur here. Metadata/hash
checks detect ordinary drift, not malicious noncooperating concurrent writers.
"""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

MAX_BYTES = 64 * 1024 * 1024
_CHUNK = 128 * 1024


class AcquisitionError(ValueError):
    """Stable local failure reason; never includes source content."""


@dataclass(frozen=True)
class SourceBytes:
    """Detached immutable bytes; digest proves neither trust nor review."""

    label: str
    data: bytes
    sha256: str


def validate_label(label: str) -> tuple[str, ...]:
    """Accept only canonical relative POSIX labels, not arbitrary locators."""
    if not isinstance(label, str) or any(c in label for c in ("\0", "\\", ":")):
        raise AcquisitionError("invalid_label")
    parts = tuple(label.split("/"))
    if any(p in ("", ".", "..") for p in parts):
        raise AcquisitionError("invalid_label")
    for part in parts:
        name = part.casefold()
        if (
            name.startswith(".env")
            or name in {".ssh", ".aws", ".gnupg"}
            or any(word in name for word in ("credential", "secret", "token", "password"))
        ):
            raise AcquisitionError("credential_path")
    return parts


def bounded_limit(value: int) -> int:
    if type(value) is not int or not 0 < value <= MAX_BYTES:
        raise AcquisitionError("invalid_size_limit")
    return value


def identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


@dataclass
class OpenedSource:
    """Internal pinned descriptors. Never open SQLite's original with sqlite3."""

    fd: int
    parent_fd: int
    name: str
    links: list[tuple[int, str, int]]

    def verify_membership(self) -> None:
        for parent, name, child in self.links:
            current = os.stat(name, dir_fd=parent, follow_symlinks=False)
            if identity(current) != identity(os.fstat(child)):
                raise AcquisitionError("source_drift")

    def check_regular(self) -> os.stat_result:
        info = os.fstat(self.fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise AcquisitionError("nonregular_or_aliased_source")
        self.verify_membership()
        return info

    def read_once(self, max_bytes: int) -> bytes:
        os.lseek(self.fd, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(self.fd, min(_CHUNK, max_bytes + 1 - total))
            if not chunk:
                return b"".join(chunks)
            total += len(chunk)
            if total > max_bytes:
                raise AcquisitionError("size_limit")
            chunks.append(chunk)

    def stable_bytes(self, max_bytes: int) -> bytes:
        before = self.check_regular()
        if before.st_size > max_bytes:
            raise AcquisitionError("size_limit")
        first = self.read_once(max_bytes)
        if identity(before) != identity(self.check_regular()):
            raise AcquisitionError("source_drift")
        second = self.read_once(max_bytes)
        if (
            identity(before) != identity(self.check_regular())
            or first != second
            or len(first) != before.st_size
        ):
            raise AcquisitionError("source_drift")
        return first


@dataclass(frozen=True)
class SourceBoundary:
    """Explicit trusted root and exact adapter-owned allowlist; never from imports."""

    root: Path
    allowed_labels: tuple[str, ...]

    def __post_init__(self) -> None:
        root = Path(self.root)
        if not root.is_absolute() or ".." in root.parts:
            raise AcquisitionError("invalid_root")
        object.__setattr__(self, "root", root)
        object.__setattr__(self, "allowed_labels", tuple(self.allowed_labels))
        for label in self.allowed_labels:
            validate_label(label)

    def require_label(self, label: str) -> tuple[str, ...]:
        parts = validate_label(label)
        if label not in self.allowed_labels:
            raise AcquisitionError("not_allowlisted")
        return parts

    @contextmanager
    def open_source(self, label: str) -> Iterator[OpenedSource]:
        """Pin every directory from /; refuse symlinks and nonregular aliases."""
        parts = self.require_label(label)
        if os.name != "posix" or not hasattr(os, "O_NOFOLLOW"):
            raise AcquisitionError("unsupported_platform")
        descriptors: list[int] = []
        links: list[tuple[int, str, int]] = []
        flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
        try:
            parent = os.open("/", flags | os.O_DIRECTORY)
            descriptors.append(parent)
            for part in (*self.root.parts[1:], *parts[:-1]):
                child = os.open(part, flags | os.O_DIRECTORY, dir_fd=parent)
                descriptors.append(child)
                links.append((parent, part, child))
                parent = child
            source = os.open(parts[-1], flags, dir_fd=parent)
            descriptors.append(source)
            links.append((parent, parts[-1], source))
            opened = OpenedSource(source, parent, parts[-1], links)
            opened.check_regular()
            yield opened
        except OSError as exc:
            raise AcquisitionError("source_unavailable") from exc
        finally:
            for descriptor in reversed(descriptors):
                os.close(descriptor)

    def read_file(self, label: str, *, max_bytes: int = MAX_BYTES) -> SourceBytes:
        """Read at most 64 MiB twice, retaining only a stable immutable snapshot."""
        limit = bounded_limit(max_bytes)
        with self.open_source(label) as source:
            data = source.stable_bytes(limit)
        return SourceBytes(label, data, hashlib.sha256(data).hexdigest())
