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

from model import MicroDiT, NUM_CLASSES, PAD_SIZE


T = 1000
beta_start = 1e-4
beta_end = 0.02


def load_model(checkpoint_path, device):
    model = MicroDiT().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model



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
        print("Using CUDA")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    torch.manual_seed(args.seed)

    model = load_model(args.checkpoint, device)
    labels = torch.arange(NUM_CLASSES, device=device).repeat_interleave(args.num_per_class)
    samples = sample_images(model, labels, guidance_scale=args.w, device=device)
    plot_row(samples, args.out, title=f"w = {args.w}", num_per_class=args.num_per_class)
    print(f"Saved {args.out}")
