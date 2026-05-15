import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")

with app.setup:
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    from pathlib import Path

    from diffusion_ebm.models import EBM
    from diffusion_ebm.proj2.sample import load_model, sample_images
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
    return


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
    return (dit_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Sample from DiT
    """)
    return


@app.cell
def _(device, dit_model):
    num_classes = 10
    num_per_class = 100
    dit_labels = torch.arange(num_classes, device=device).repeat_interleave(num_per_class)
    dit_samples = sample_images(dit_model, dit_labels, guidance_scale=3.0, device=device)
    return (dit_samples,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Plot samples
    """)
    return


@app.cell
def _(device, dit_samples):
    show_replay_buffer_samples(dit_samples.clamp(-1, 1), n_samples=16, determinisitic=False, device=device)

    #gen_images = buffer[:1000]
    #gen_images.shape
    #show_replay_buffer_samples(gen_images, n_samples=10, device=device)
    return


if __name__ == "__main__":
    app.run()
