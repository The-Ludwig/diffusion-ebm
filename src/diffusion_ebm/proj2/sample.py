"""
DDPM sampling with Classifier-Free Guidance for Project 2 (Task 3).

Usage:
    python sample.py                     # one sample per class at w=3
    python sample.py --w 7.0             # change guidance scale
    python sample.py --num-per-class 10  # more samples per class

You must fill in the two TODOs inside ``sample_images`` for Task 3.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from diffusion_ebm.proj2.model import MicroDiT, NUM_CLASSES, PAD_SIZE

T = 1000
beta_start = 1e-4
beta_end = 0.02


def make_schedule(device):
    betas = torch.linspace(beta_start, beta_end, T, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_one_minus_alphas_cumprod": torch.sqrt(1.0 - alphas_cumprod),
        "sqrt_recip_alphas": torch.sqrt(1.0 / alphas),
        "posterior_variance": betas
        * (1.0 - alphas_cumprod_prev)
        / (1.0 - alphas_cumprod),
    }


@torch.no_grad()
def sample_images(model, labels, guidance_scale=3.0, initial_noise=None, device=None):
    """Generate samples with DDPM + Classifier-Free Guidance.

    Args:
        model: a MicroDiT in eval mode.
        labels: (N,) long tensor of class labels in [0, NUM_CLASSES).
        guidance_scale: the CFG scale ``w``.
        initial_noise: optional (N, 1, PAD_SIZE, PAD_SIZE) starting noise.
        device: torch device.

    Returns:
        x: (N, 1, PAD_SIZE, PAD_SIZE) generated images in [-1, 1] (approx.).
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    sch = make_schedule(device)
    N = labels.shape[0]
    x = (
        initial_noise
        if initial_noise is not None
        else torch.randn(N, 1, PAD_SIZE, PAD_SIZE, device=device)
    )
    null_labels = torch.full_like(labels, model.null_class_id)

    for t in tqdm(reversed(range(T)), total=T, desc="Sampling", leave=False):
        t_batch = torch.full((N,), t, device=device, dtype=torch.long)

        eps_uncond = model(x, t_batch, null_labels)
        eps_cond = model(x, t_batch, labels)

        eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

        sqrt_alpha_t = 1 / sch["sqrt_recip_alphas"][t]
        beta_t = sch["betas"][t]
        sqrt_1_minus_bar_alpha_t = sch["sqrt_one_minus_alphas_cumprod"][t]

        mean = (1 / sqrt_alpha_t) * (x - (beta_t / sqrt_1_minus_bar_alpha_t) * eps)

        if t > 0:
            z = torch.randn_like(x)
            x = mean + torch.sqrt(sch["posterior_variance"][t]) * z
        else:
            x = mean

    return x


def load_model(checkpoint_path, device):
    model = MicroDiT().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


def plot_row(samples, path, title=None):
    N = samples.shape[0]
    fig, axes = plt.subplots(1, N, figsize=(2 * N, 2))
    if N == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        img = samples[i].detach().cpu().squeeze().clamp(-1, 1) * 0.5 + 0.5
        ax.imshow(img, cmap="gray")
        ax.set_title(f"{i % NUM_CLASSES}")
        ax.axis("off")
    if title:
        plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()


def plot_sweep_grid(rows_by_w, path, row_labels=None):
    """Save a (num_rows x num_w) grid where each column is a guidance scale.

    Args:
        rows_by_w: dict mapping guidance scale ``w`` (float) to a tensor of
            shape ``(num_rows, 1, H, W)`` of samples generated at that w.
        path: output file path.
        row_labels: optional list of labels (one per row); defaults to
            ``range(num_rows)``.
    """
    w_values = sorted(rows_by_w.keys())
    any_tensor = next(iter(rows_by_w.values()))
    num_rows = any_tensor.shape[0]
    row_labels = (
        row_labels if row_labels is not None else [str(i) for i in range(num_rows)]
    )

    fig, axes = plt.subplots(
        num_rows,
        len(w_values),
        figsize=(2 * len(w_values), 2 * num_rows),
        squeeze=False,
    )
    for j, w in enumerate(w_values):
        samples = rows_by_w[w]
        for i in range(num_rows):
            ax = axes[i, j]
            img = samples[i].detach().cpu().squeeze().clamp(-1, 1) * 0.5 + 0.5
            ax.imshow(img, cmap="gray")
            ax.axis("off")
            if i == 0:
                ax.set_title(f"w = {w}")
            if j == 0:
                ax.text(-5, img.shape[0] / 2, row_labels[i], va="center", ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="micro_dit_checkpoint.pt")
    parser.add_argument("--w", type=float, default=3.0, help="guidance scale")
    parser.add_argument("--num-per-class", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="samples.png")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    torch.manual_seed(args.seed)

    model = load_model(args.checkpoint, device)
    labels = torch.arange(NUM_CLASSES, device=device).repeat_interleave(
        args.num_per_class
    )
    samples = sample_images(model, labels, guidance_scale=args.w, device=device)
    plot_row(samples, args.out, title=f"w = {args.w}")
    print(f"Saved {args.out}")
