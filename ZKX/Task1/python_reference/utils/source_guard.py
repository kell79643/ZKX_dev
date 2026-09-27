"""原版 cuSignal 23.08.00 源码完整性门禁。"""

from __future__ import annotations

import hashlib
from pathlib import Path


EXPECTED_FILE_COUNT = 152
EXPECTED_TREE_SHA256 = "ca9e760bdebb0a481e18171546881b2e75214a422482f7442a0814871476c163"
EXPECTED_RADARTOOLS_SHA256 = "438fdde707ea86e9249cc38cb5ad8addca3e65da7f38e7d06218809236f2f650"


def _source_files(source_root: Path) -> list[Path]:
    return sorted(
        path
        for path in source_root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix.lower() not in {
            ".pyc", ".pyo", ".png", ".jpg", ".jpeg", ".bmp", ".gif", ".svg"
        }
        and ".git" not in path.parts
    )


def cusignal_tree_sha256(source_root: Path) -> tuple[int, str]:
    root = source_root.resolve()
    files = _source_files(root)
    manifest = "".join(
        f"{path.relative_to(root).as_posix()}\t"
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}\n"
        for path in files
    ).encode("utf-8")
    return len(files), hashlib.sha256(manifest).hexdigest()


def verify_immutable_cusignal_source(cusignal_python: Path) -> dict[str, object]:
    source_root = cusignal_python.resolve().parent
    radartools = source_root / "python/cusignal/radartools/radartools.py"
    if not radartools.is_file():
        raise RuntimeError(f"cuSignal radartools source is missing: {radartools}")
    radartools_sha256 = hashlib.sha256(radartools.read_bytes()).hexdigest()
    file_count, tree_sha256 = cusignal_tree_sha256(source_root)
    if radartools_sha256 != EXPECTED_RADARTOOLS_SHA256:
        raise RuntimeError(
            "cusignal-23.08.00 is immutable: radartools.py SHA-256 changed; "
            f"expected={EXPECTED_RADARTOOLS_SHA256} actual={radartools_sha256}"
        )
    if file_count != EXPECTED_FILE_COUNT or tree_sha256 != EXPECTED_TREE_SHA256:
        raise RuntimeError(
            "cusignal-23.08.00 is immutable: source tree changed; "
            f"expected_count={EXPECTED_FILE_COUNT} actual_count={file_count} "
            f"expected_sha256={EXPECTED_TREE_SHA256} actual_sha256={tree_sha256}"
        )
    return {
        "source_root": str(source_root),
        "file_count": file_count,
        "tree_sha256": tree_sha256,
        "radartools_sha256": radartools_sha256,
    }
