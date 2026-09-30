"""Read-only verification of every file declared in RELEASE_SHA256.json."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "RELEASE_SHA256.json"
TEXT_SUFFIXES = {".py", ".md", ".txt", ".csv", ".json", ".yaml", ".yml"}


def included_files() -> dict[str, Path]:
    return {
        path.relative_to(ROOT).as_posix(): path
        for path in ROOT.rglob("*")
        if path.is_file()
        and path != MANIFEST
        and ".git" not in path.parts
        and "__pycache__" not in path.parts
        and path.suffix.lower() != ".pyc"
    }


def sha256(path: Path) -> str:
    data = path.read_bytes()
    if path.suffix.lower() in TEXT_SUFFIXES or path.name in {".gitignore", ".gitattributes"}:
        while b"\r\n" in data:
            data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    if not MANIFEST.is_file():
        raise SystemExit("MANIFEST VERIFICATION FAILED: RELEASE_SHA256.json is missing")
    declared = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if not isinstance(declared, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in declared.items()
    ):
        raise SystemExit("MANIFEST VERIFICATION FAILED: invalid JSON schema")

    actual = included_files()
    missing = sorted(set(declared) - set(actual))
    undeclared = sorted(set(actual) - set(declared))
    mismatched = sorted(
        name for name in set(declared) & set(actual)
        if sha256(actual[name]) != declared[name]
    )
    if missing or undeclared or mismatched:
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if undeclared:
            details.append("undeclared: " + ", ".join(undeclared))
        if mismatched:
            details.append("checksum mismatch: " + ", ".join(mismatched))
        raise SystemExit("MANIFEST VERIFICATION FAILED\n" + "\n".join(details))
    print(f"MANIFEST VERIFICATION PASSED: {len(actual)} files")


if __name__ == "__main__":
    main()
