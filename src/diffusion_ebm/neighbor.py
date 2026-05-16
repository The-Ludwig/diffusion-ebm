import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")

with app.setup:
    import matplotlib.pyplot as plt
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    from pathlib import Path

    from diffusion_ebm.models import EBM
    from diffusion_ebm.proj2.sample import load_model, sample_images
    from diffusion_ebm.proj2.memorization import pixel_l2_nearest_neighbor
    from diffusion_ebm.train_ebm import show_replay_buffer_samples


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Imports
    """)
    return


@app.cell
def _():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Mac)")
    else:
        device = torch.device("cpu")
        print("Warning: Using CPU")

    root = Path(__file__).absolute().parent.parent.parent
    return device, root


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Load data
    """)
    return


@app.cell
def _(root):
    conditioning = False
    transform = transforms.Compose([
            transforms.Pad(2),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
    dataset = datasets.MNIST(root=root/"data", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    n_classes = 10 if conditioning else 1

    val_dataset = datasets.MNIST(root=root/"data", train=False, download=True, transform=transform)
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )
    return dataset, val_dataset


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Load models and buffer for EBM
    """)
    return


@app.cell
def _(device, root):
    dit_path = root/"micro_dit_checkpoint.pt"
    dit_model = load_model(dit_path, device=device)


    ebm_path = root/"checkpoints/nicer_lr_schedule/step_30000.pt"
    ebm_model = EBM()

    loaded_state = torch.load(f=ebm_path, map_location=device)
    ebm_model.load_state_dict(loaded_state['model_state_dict'])
    buffer = loaded_state['replay_buffer']
    return buffer, dit_model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Sample from DiT
    """)
    return


@app.cell
def _(device, dit_model):
    # sample 1000 images, 10 from each digit
    num_classes = 10
    num_per_class = 100
    dit_labels = torch.arange(num_classes, device=device).repeat_interleave(num_per_class)
    dit_samples = sample_images(dit_model, dit_labels, guidance_scale=3.0, device=device)

    # clamp result to enable plotting
    dit_samples = dit_samples.clamp(-1, 1)
    return (dit_samples,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Plot EBM samples
    """)
    return


@app.cell
def _(buffer, device):
    gen_images = buffer[:1000]
    gen_images.shape
    show_replay_buffer_samples(gen_images, n_samples=20, plt_title=None, device=device)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Plot DiT samples
    """)
    return


@app.cell
def _(device, dit_samples):
    show_replay_buffer_samples(dit_samples, n_samples=20, determinisitic=False, plt_title=None, device=device)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Nearest neighbors in training data
    """)
    return


@app.cell
def _(dataset, device, val_dataset):
    train_imgs = torch.stack([dataset[i][0] for i in range(len(dataset))]).to(device)
    val_imgs = torch.stack([val_dataset[i][0] for i in range(len(val_dataset))]).to(device)
    return train_imgs, val_imgs


@app.function
def plot_nearest_neighbors(samples, train_imgs, n_show=10, plt_title="Nearest neighbors"):
    fig, axes = plt.subplots(2, n_show, figsize=(2 * n_show, 4.5))
    for i in range(n_show):
        # find closest sample to gen in training data
        gen = samples[i].detach().cpu().clamp(-1, 1)
        idx, dist = pixel_l2_nearest_neighbor(samples[i], train_imgs)
        nn = train_imgs[idx].detach().cpu()

        # plot gen
        axes[0, i].imshow(gen.squeeze() * 0.5 + 0.5, cmap="gray")
        axes[0, i].axis("off")

        # plot nn
        axes[1, i].imshow(nn.squeeze() * 0.5 + 0.5, cmap="gray")
        axes[1, i].set_title(f"d={dist:.2f}")
        axes[1, i].axis("off")

    axes[0, 0].text(-5, 16, "gen", va="center", ha="right")
    axes[1, 0].text(-5, 16, "NN", va="center", ha="right")
    fig.suptitle(plt_title)
    plt.tight_layout(h_pad=2.2)
    return fig


@app.cell
def _(dit_samples, train_imgs):
    perm = torch.randperm(dit_samples.shape[0])
    plot_nearest_neighbors(dit_samples[perm], train_imgs, plt_title="DiT samples vs train NN")
    return


@app.cell
def _(buffer, train_imgs):
    plot_nearest_neighbors(buffer[:1000], train_imgs, plt_title="EBM buffer samples vs train NN")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Distance statistics
    """)
    return


@app.function
@torch.no_grad()
def compute_nn_distances(samples, train_imgs, batch_size=128):
    """Return a 1D tensor of L2 NN distances from each sample to the training set."""
    samples = samples.clamp(-1, 1).float()
    train_flat = train_imgs.reshape(train_imgs.shape[0], -1).float()
    out = []
    for i in range(0, samples.shape[0], batch_size):
        batch_flat = samples[i:i + batch_size].reshape(-1, train_flat.shape[1])
        d = torch.cdist(batch_flat, train_flat).min(dim=1).values
        out.append(d.cpu())
    return torch.cat(out)


@app.function
def summarize_distances(name, dists):
    return {
        "name": name,
        "n": int(dists.numel()),
        "mean": float(dists.mean()),
        "std": float(dists.std()),
        "min": float(dists.min()),
        "median": float(dists.median()),
        "max": float(dists.max()),
    }


@app.cell
def _(buffer, dit_samples, train_imgs, val_imgs):
    dit_dists = compute_nn_distances(dit_samples, train_imgs)
    ebm_dists = compute_nn_distances(buffer[:1000], train_imgs)
    real_dists = compute_nn_distances(val_imgs[:1000], train_imgs)

    # shared bin edges so the three histograms are directly comparable
    all_dists = torch.cat([dit_dists, ebm_dists, real_dists])
    bins = torch.linspace(all_dists.min(), all_dists.max(), 81).numpy()

    series = [
        ("DiT", dit_dists, "C0"),
        ("EBM buffer", ebm_dists, "C1"),
        ("real (val)", real_dists, "C2"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=True)
    for ax, (name, d, color) in zip(axes, series):
        ax.hist(d.numpy(), bins=bins, color=color, alpha=0.8)
        ax.set_title(name)
        ax.set_xlabel("L2 distance to nearest train neighbor")
    axes[0].set_ylabel("count")
    plt.tight_layout()

    stats = [
        summarize_distances("DiT", dit_dists),
        summarize_distances("EBM", ebm_dists),
        summarize_distances("real", real_dists),
    ]
    cols = ["name", "n", "mean", "std", "min", "median", "max"]
    widths = {"name": 6, "n": 6, "mean": 10, "std": 10, "min": 10, "median": 10, "max": 10}
    header = "".join(f"{c:>{widths[c]}}" for c in cols)
    print(header)
    print("-" * len(header))
    for s in stats:
        row = "".join(
            f"{s[c]:>{widths[c]}}" if c in ("name", "n")
            else f"{s[c]:>{widths[c]}.3f}"
            for c in cols
        )
        print(row)
    fig
    return


if __name__ == "__main__":
    app.run()
