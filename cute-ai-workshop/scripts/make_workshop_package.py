#!/usr/bin/env python3
"""Create a zip package that can be uploaded to Google Drive or Colab."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


REQUIRED_FILES = [
    "data/workshop/metadata.csv",
    "data/workshop/embeddings.npy",
    "notebooks/02_participant_workshop.ipynb",
]


def make_package(args: argparse.Namespace) -> None:
    root_dir = Path(args.root_dir)
    output_path = root_dir / args.output

    missing = [path for path in REQUIRED_FILES if not (root_dir / path).exists()]
    images_dir = root_dir / "data" / "workshop" / "images"
    if not images_dir.exists() or not any(images_dir.glob("*.jpg")):
        missing.append("data/workshop/images/*.jpg")
    if missing:
        raise FileNotFoundError("Missing required workshop files:\n" + "\n".join(missing))

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Only distribute teaching materials, never local environments or raw data.
        candidates = [
            root_dir / name for name in
            ("README.md", "requirements.txt", "pyproject.toml", "uv.lock", ".python-version",
             "start_workshop.bat", "start_workshop.command")
        ]
        for folder in ("scripts", "notebooks", "docs", "data/workshop"):
            candidates.extend((root_dir / folder).rglob("*"))
        for path in candidates:
            relative = path.relative_to(root_dir)
            if any(part.startswith(".") or part == "__pycache__" for part in relative.parts):
                if relative.as_posix() != ".python-version":
                    continue
            if path.is_file() and path.suffix not in (".pyc", ".zip"):
                zf.write(path, relative)

    print(f"Created: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", default=".", help="Project root directory")
    parser.add_argument("--output", default="cute-ai-workshop-package.zip")
    return parser.parse_args()


if __name__ == "__main__":
    make_package(parse_args())
