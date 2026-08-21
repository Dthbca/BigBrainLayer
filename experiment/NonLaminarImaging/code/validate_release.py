"""Validate the synchronized NonLaminarImaging release."""

from __future__ import annotations

import argparse
import csv
import hashlib
from html.parser import HTMLParser
from pathlib import Path


class _ReferenceParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.references: list[str] = []

    def handle_starttag(self, tag, attrs) -> None:
        for key, value in attrs:
            if key in {"src", "href", "data"} and value:
                self.references.append(value)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(root: Path) -> None:
    manifest_path = root / "MANIFEST_SHA256.csv"
    with manifest_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    failures: list[str] = []
    for row in rows:
        path = root / row["path"]
        if not path.is_file():
            failures.append(f"missing manifest file: {row['path']}")
            continue
        if path.stat().st_size != int(row["bytes"]):
            failures.append(f"size mismatch: {row['path']}")
        if sha256(path) != row["sha256"]:
            failures.append(f"hash mismatch: {row['path']}")

    html_path = root / "report" / "nonlaminar_imaging_report.html"
    parser = _ReferenceParser()
    parser.feed(html_path.read_text(encoding="utf-8"))
    checked = 0
    for reference in parser.references:
        if reference.startswith(("http://", "https://", "#", "mailto:")):
            continue
        checked += 1
        target = (html_path.parent / reference).resolve()
        if not target.is_file():
            failures.append(f"missing HTML dependency: {reference}")

    if failures:
        raise SystemExit("\n".join(failures))
    print(f"PASS files={len(rows)} html_dependencies={checked}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", type=Path, default=Path(__file__).parents[1])
    validate(parser.parse_args().root.resolve())

