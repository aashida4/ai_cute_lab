#!/usr/bin/env python3
"""Prepare a small Oxford-IIIT Pet subset for the workshop."""

from __future__ import annotations

import argparse
import random
import shutil
import tarfile
import urllib.request
from pathlib import Path

import pandas as pd
from tqdm import tqdm


IMAGES_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"

DEFAULT_BREEDS = {
    "Abyssinian": "cat",
    "Bengal": "cat",
    "Birman": "cat",
    "British_Shorthair": "cat",
    "Egyptian_Mau": "cat",
    "Persian": "cat",
    "beagle": "dog",
    "boxer": "dog",
    "chihuahua": "dog",
    "golden_retriever": "dog",
    "pomeranian": "dog",
    "samoyed": "dog",
}


def download(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        print(f"Already downloaded: {output_path}")
        return

    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as response:
        total = int(response.headers.get("Content-Length", 0))
        with output_path.open("wb") as f, tqdm(total=total, unit="B", unit_scale=True) as pbar:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                pbar.update(len(chunk))


def extract(tar_path: Path, output_dir: Path) -> None:
    marker = output_dir / ".extracted"
    if marker.exists():
        print(f"Already extracted: {output_dir}")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {tar_path}")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(output_dir)
    marker.write_text("ok\n", encoding="utf-8")


def breed_from_filename(path: Path) -> str:
    stem = path.stem
    parts = stem.split("_")
    return "_".join(parts[:-1])


def prepare(args: argparse.Namespace) -> None:
    root_dir = Path(args.root_dir)
    raw_dir = root_dir / "data" / "raw"
    workshop_dir = root_dir / "data" / "workshop"
    images_out = workshop_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    archive_path = raw_dir / "images.tar.gz"
    download(IMAGES_URL, archive_path)
    extract(archive_path, raw_dir)

    source_images = raw_dir / "images"
    if not source_images.exists():
        raise FileNotFoundError(f"Oxford images directory not found: {source_images}")

    rng = random.Random(args.seed)
    selected_breeds = DEFAULT_BREEDS
    base_per_breed = max(1, args.total_images // len(selected_breeds))
    remainder = max(0, args.total_images % len(selected_breeds))

    records = []
    for breed_index, (breed, species) in enumerate(selected_breeds.items()):
        candidates = sorted(p for p in source_images.glob("*.jpg") if breed_from_filename(p) == breed)
        if not candidates:
            print(f"Warning: no images found for breed {breed}")
            continue
        rng.shuffle(candidates)
        take = base_per_breed + (1 if breed_index < remainder else 0)
        for source_path in candidates[:take]:
            dest_name = f"{breed}__{source_path.name}"
            dest_path = images_out / dest_name
            shutil.copy2(source_path, dest_path)
            records.append(
                {
                    "image_id": dest_path.stem,
                    "filename": dest_name,
                    "image_path": f"images/{dest_name}",
                    "species": species,
                    "breed": breed,
                    "source_filename": source_path.name,
                }
            )

    rng.shuffle(records)
    metadata = pd.DataFrame(records)
    metadata.insert(0, "workshop_index", range(len(metadata)))
    metadata.to_csv(workshop_dir / "metadata.csv", index=False)

    print(f"Created {len(metadata)} workshop images")
    print(f"Images:   {images_out}")
    print(f"Metadata: {workshop_dir / 'metadata.csv'}")
    print(metadata.groupby(["species", "breed"]).size())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", default=".", help="Project root directory")
    parser.add_argument("--total-images", type=int, default=240, help="Target image count, usually 150-300")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
