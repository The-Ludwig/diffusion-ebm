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
import numpy as np
import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity, schedule
from tqdm.auto import tqdm

from diffusion_ebm.proj2.model import MicroDiT, NUM_CLASSES, PAD_SIZE
from diffusion_ebm.models import EBM
from diffusion_ebm.ema import EMA

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
        "posterior_variance": betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod),
    }


@torch.no_grad()
def sample_images(model, labels, guidance_scale=3.0, initial_noise=None, device=None, energy_model=None):
    """Generate samples with DDPM + Classifier-Free Guidance.

    Args:
        model: a MicroDiT in eval mode.
        labels: (N,) long tensor of class labels in [0, NUM_CLASSES).
        guidance_scale: the CFG scale ``w``.
        initial_noise: optional (N, 1, PAD_SIZE, PAD_SIZE) starting noise.
        device: torch device.

    Returns:
        x: ([N_guidance_strength,] N, 1, PAD_SIZE, PAD_SIZE) generated images in [-1, 1] (approx.).
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    # Let's see if this speeds up things 
    model = torch.compile(model, mode="reduce-overhead")
    sch = make_schedule(device)
    N = labels.shape[0]
    x = initial_noise.clone() if initial_noise is not None else torch.randn(
        N, 1, PAD_SIZE, PAD_SIZE, device=device
    )

    if isinstance(guidance_scale, float):
        guidance_scale = [guidance_scale]

    guidance_scale = torch.tensor(guidance_scale).to(device)
    n_w = guidance_scale.size().numel()
        
    x = x.unsqueeze(0).repeat(n_w, *([1] * x.dim()))
    labels = labels.unsqueeze(0).repeat(n_w, *([1] * labels.dim()))

    null_labels = torch.full_like(labels, model.null_class_id)
    # further speed-up tricks
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for t in tqdm(reversed(range(T)), total=T, desc="Sampling", leave=False):
        
            t_batch = torch.full((N*n_w,), t, device=device, dtype=torch.long)

            x = x.flatten(0, 1)
            null_labels = null_labels.flatten(0, 1)
            labels = labels.flatten(0, 1)

            if energy_model is not None:
                energy = energy_model(x)
                print(f"Energy at step {t}: {energy.mean().item():.4f}")
                
            eps_uncond = model(x, t_batch, null_labels).unflatten(0, (n_w, N)).clone()
            eps_cond = model(x, t_batch, labels).unflatten(0, (n_w, N))

            x = x.unflatten(0, (n_w, N))
            null_labels = null_labels.unflatten(0, (n_w, N))
            labels = labels.unflatten(0, (n_w, N))

            eps = eps_uncond + guidance_scale[:, None, None, None, None] * (eps_cond - eps_uncond)

            sqrt_recip_alpha = sch["sqrt_recip_alphas"][t]
            beta_fac = sch["betas"][t]/sch["sqrt_one_minus_alphas_cumprod"][t]

            mean = sqrt_recip_alpha * (x - beta_fac * eps)

            if t > 0:
                z = torch.randn_like(x)
                x = mean + torch.sqrt(sch["posterior_variance"][t]) * z
            else:
                x = mean

    return x.squeeze(0)


def load_model(checkpoint_path, device):
    model = MicroDiT().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


def plot_row(samples, path, title=None, num_per_class=None):
    N = samples.shape[0]

    if num_per_class is None:
        nrows = 1
        ncols = N
    else:
        nrows = num_per_class
        ncols = N // num_per_class

    fig, axes = plt.subplots(nrows, ncols, figsize=(2 * ncols, 2*nrows))
    if N == 1:
        axes = np.array([[axes]])
    for idx, ax in np.ndenumerate(axes):
        i = idx[1]*nrows+idx[0]

        img = samples[i].detach().cpu().squeeze().clamp(-1, 1) * 0.5 + 0.5
        ax.imshow(img, cmap="gray")
        ax.set_title(f"{idx[1]}")
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
    row_labels = row_labels if row_labels is not None else [str(i) for i in range(num_rows)]

    fig, axes = plt.subplots(
        num_rows, len(w_values),
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
    parser.add_argument("--energy-model", type=str, default=None, help="optional energy model checkpoint for monitoring energy during sampling")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    torch.manual_seed(args.seed)

    if args.energy_model is not None:
        energy_model = EBM()
        model_ema = EMA(energy_model, decay=0.999, burn_in=500)

        loaded_state = torch.load(args.energy_model)

        energy_model.load_state_dict(loaded_state['model_state_dict'])
        model_ema.shadow = loaded_state['ema_shadow']

        energy_model.to(device)
        energy_model.eval()
        model_ema.apply_shadow()

        print(f"Loaded energy model from {args.energy_model}")


    model = load_model(args.checkpoint, device)
    labels = torch.arange(NUM_CLASSES, device=device).repeat_interleave(args.num_per_class)
    if energy_model is None:
        samples = sample_images(model, labels, guidance_scale=args.w, device=device)
    else:
        samples = sample_images(model, labels, guidance_scale=args.w, device=device, energy_model=energy_model) 
    plot_row(samples, args.out, title=f"w = {args.w}", num_per_class=args.num_per_class)
    print(f"Saved {args.out}")
