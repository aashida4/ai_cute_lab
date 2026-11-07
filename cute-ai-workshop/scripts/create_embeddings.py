#!/usr/bin/env python3
"""Create image embeddings with a pretrained EfficientNet-B0 model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
from tqdm import tqdm


class WorkshopImageDataset(Dataset):
    def __init__(self, workshop_dir: Path, metadata: pd.DataFrame, transform):
        self.workshop_dir = workshop_dir
        self.metadata = metadata.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int):
        row = self.metadata.iloc[idx]
        image_path = self.workshop_dir / row["image_path"]
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), idx


def build_model(device: torch.device):
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)
    model.classifier = torch.nn.Identity()
    model.eval()
    model.to(device)
    return model, weights.transforms()


def create_embeddings(args: argparse.Namespace) -> None:
    root_dir = Path(args.root_dir)
    workshop_dir = root_dir / "data" / "workshop"
    metadata_path = workshop_dir / "metadata.csv"
    output_path = workshop_dir / "embeddings.npy"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"{metadata_path} が見つかりません。先に scripts/prepare_dataset.py を実行してください。"
        )

    metadata = pd.read_csv(metadata_path)
    if args.cpu:
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    model, transform = build_model(device)
    dataset = WorkshopImageDataset(workshop_dir, metadata, transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    embeddings = np.zeros((len(dataset), 1280), dtype=np.float32)
    with torch.inference_mode():
        for images, indices in tqdm(loader, desc="Embedding"):
            images = images.to(device)
            features = model(images).detach().cpu().numpy().astype(np.float32)
            embeddings[indices.numpy()] = features

    np.save(output_path, embeddings)
    print(f"Saved: {output_path}")
    print(f"Shape: {embeddings.shape}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", default=".", help="Project root directory")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even when CUDA or MPS is available")
    return parser.parse_args()


if __name__ == "__main__":
    create_embeddings(parse_args())
