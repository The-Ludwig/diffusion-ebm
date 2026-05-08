"""
Memorization analysis helpers for Project 2 (Tasks 4 and 5).

All distances here are computed directly in **pixel space** on flattened
(1, 32, 32) -> 1024-dim tensors. No classifier features are used.

Provided:
  - pixel_l2_nearest_neighbor: L2 nearest neighbor of a generated image in
    a set of training images (used for Task 4).
  - improved_pr_pixel: Improved Precision and Recall with k=5 in pixel
    space (used for Task 5).
  - load_mnist_tensors: convenience loader that returns MNIST images in the
    same (N, 1, 32, 32) layout and [-1, 1] range as the sampler output.
"""
from __future__ import annotations

import argparse
from tqdm import tqdm

from model import MicroDiT, NUM_CLASSES, PAD_SIZE
from sample import sample_images, plot_sweep_grid, load_model

import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path

DATA_DIR = Path(__file__).absolute().parents[2]/"data"


def load_mnist_tensors(train: bool = True, root: Path | str = DATA_DIR, limit: int | None = None):
    """Load MNIST as (N, 1, 32, 32) tensors in [-1, 1], matching the sampler.

    Args:
        train: if True, return the training split; else the test split.
        root: directory to download/load MNIST.
        limit: optional max number of images to return.

    Returns:
        images: (N, 1, 32, 32) float tensor in [-1, 1].
        labels: (N,) long tensor of class labels.
    """
    transform = transforms.Compose([
        transforms.Pad(2),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    ds = datasets.MNIST(root=root, train=train, download=True, transform=transform)
    loader = DataLoader(ds, batch_size=1024, shuffle=False)
    xs, ys = [], []
    collected = 0
    for x, y in loader:
        xs.append(x)
        ys.append(y)
        collected += x.shape[0]
        if limit is not None and collected >= limit:
            break
    images = torch.cat(xs, dim=0)
    labels = torch.cat(ys, dim=0)
    if limit is not None:
        images = images[:limit]
        labels = labels[:limit]
    return images, labels


def load_cifar_tensors(train: bool = True, root: Path | str = DATA_DIR, limit: int | None = None):
    """Load MNIST as (N, 1, 32, 32) tensors in [-1, 1], matching the sampler.

    Args:
        train: if True, return the training split; else the test split.
        root: directory to download/load MNIST.
        limit: optional max number of images to return.

    Returns:
        images: (N, 1, 32, 32) float tensor in [-1, 1].
        labels: (N,) long tensor of class labels.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    ds = datasets.CIFAR10(root=root, train=train, download=True, transform=transform)
    loader = DataLoader(ds, batch_size=1024, shuffle=False)
    xs, ys = [], []
    collected = 0
    for x, y in loader:
        xs.append(x)
        ys.append(y)
        collected += x.shape[0]
        if limit is not None and collected >= limit:
            break
    images = torch.cat(xs, dim=0)
    labels = torch.cat(ys, dim=0)
    if limit is not None:
        images = images[:limit]
        labels = labels[:limit]
    return images, labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="micro_dit_checkpoint.pt")
    parser.add_argument("--w", type=float, default=3.0, help="guidance scale")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="memorization.png")
    parser.add_argument("--num-per-class", type=int, default=1)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_id = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(device_id)
        print(f"Using CUDA device: {gpu_name}")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    torch.manual_seed(args.seed)

    N = args.num_per_class * NUM_CLASSES

    initial_noise = torch.randn(
        N, 1, PAD_SIZE, PAD_SIZE, device=device
    )

    model = load_model(args.checkpoint, device)
    labels = torch.arange(NUM_CLASSES, device=device).repeat_interleave(args.num_per_class)

    sample = sample_images(model, labels, guidance_scale=args.w, device=device, initial_noise=initial_noise)

    print("Load training data")
    training_data, training_labels = load_mnist_tensors(train=True)
    print("Done.")

    plot_closest_imgs(sample, training_data, args.out)
   
    print(f"Saved {args.out}")
