"""Generate the release checksum manifest.

This maintainer command is intentionally separate from verification. Normal
users should run ``verify_manifest.py``; that command never rewrites evidence.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "RELEASE_SHA256.json"
TEXT_SUFFIXES = {".py", ".md", ".txt", ".csv", ".json", ".yaml", ".yml"}


def included_files() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and path != MANIFEST
        and ".git" not in path.parts
        and "__pycache__" not in path.parts
        and path.suffix.lower() != ".pyc"
    )


def sha256(path: Path) -> str:
    data = path.read_bytes()
    if path.suffix.lower() in TEXT_SUFFIXES or path.name in {".gitignore", ".gitattributes"}:
        # Repeated replacement also normalizes the occasional CRCRLF sequence
        # produced when a generated table already contains CRLF on Windows.
        while b"\r\n" in data:
            data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    manifest = {
        path.relative_to(ROOT).as_posix(): sha256(path)
        for path in included_files()
    }
    MANIFEST.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"WROTE RELEASE MANIFEST: {len(manifest)} files")


if __name__ == "__main__":
    main()
