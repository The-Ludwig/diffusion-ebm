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

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


@torch.no_grad()
def pixel_l2_nearest_neighbor(generated: torch.Tensor, training_data: torch.Tensor):
    """L2 nearest neighbor of one generated image in a training set.

    Args:
        generated: (1, H, W) or (C, H, W) single generated image.
        training_data: (N, C, H, W) training images.

    Returns:
        best_idx: int, index of the nearest neighbor in ``training_data``.
        best_dist: float, L2 distance to the nearest neighbor.
    """
    gen_flat = generated.reshape(1, -1).float()
    train_flat = training_data.reshape(training_data.shape[0], -1).float()
    dists = torch.cdist(gen_flat, train_flat).squeeze(0)
    best_idx = int(dists.argmin().item())
    best_dist = float(dists[best_idx].item())
    return best_idx, best_dist


@torch.no_grad()
def improved_pr_pixel(generated: torch.Tensor, real: torch.Tensor, k: int = 5):
    """Improved Precision and Recall in pixel space (Kynkaanniemi et al., 2019).

    For each point the method defines a hypersphere whose radius equals the
    distance to its k-th nearest neighbor *within its own set*. A generated
    sample is counted as "precise" if it falls inside at least one real
    hypersphere; a real sample is "recalled" if it falls inside at least one
    generated hypersphere.

    Args:
        generated: (N_gen, C, H, W) generated images.
        real:      (N_real, C, H, W) real images.
        k: number of nearest neighbors used to define hypersphere radii.

    Returns:
        precision: float in [0, 1].
        recall:    float in [0, 1].
    """
    gen_flat = generated.reshape(generated.shape[0], -1).float()
    real_flat = real.reshape(real.shape[0], -1).float()

    real_dists = torch.cdist(real_flat, real_flat)
    real_radii = real_dists.topk(k + 1, dim=1, largest=False).values[:, -1]

    gen_dists = torch.cdist(gen_flat, gen_flat)
    gen_radii = gen_dists.topk(k + 1, dim=1, largest=False).values[:, -1]

    cross_dists = torch.cdist(gen_flat, real_flat)

    inside_real = (cross_dists < real_radii.unsqueeze(0)).any(dim=1)
    precision = float(inside_real.float().mean().item())

    inside_gen = (cross_dists.t() < gen_radii.unsqueeze(0)).any(dim=1)
    recall = float(inside_gen.float().mean().item())

    return precision, recall


def load_mnist_tensors(
    train: bool = True, root: str = "./data", limit: int | None = None
):
    """Load MNIST as (N, 1, 32, 32) tensors in [-1, 1], matching the sampler.

    Args:
        train: if True, return the training split; else the test split.
        root: directory to download/load MNIST.
        limit: optional max number of images to return.

    Returns:
        images: (N, 1, 32, 32) float tensor in [-1, 1].
        labels: (N,) long tensor of class labels.
    """
    transform = transforms.Compose(
        [
            transforms.Pad(2),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
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
