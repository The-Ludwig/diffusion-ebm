import marimo

__generated_with = "0.23.6"
app = marimo.App()

with app.setup:
    # Initialization code that runs before all other cells
    import numpy as np
    import matplotlib.pyplot as plt
    import marimo
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    import torchvision

    from pathlib import Path
    import argparse

    from diffusion_ebm.models import EBM, load_energy_model
    from diffusion_ebm.ema import EMA

    from diffusion_ebm.proj1.helper import load_classifier

    if marimo.running_in_notebook():
        import tqdm.notebook as _tqdm

        # marimo's patched class lives at tqdm.notebook.tqdm at runtime
        _Patched = _tqdm.tqdm

        def _set_postfix(self, ordered_dict=None, refresh=True, **kwargs):
            items = {}
            if ordered_dict:
                items.update(ordered_dict)
            items.update(kwargs)
            subtitle = ", ".join(f"{k}={v}" for k, v in items.items())
            try:
                self.progress.update(increment=0, subtitle=subtitle)
            except Exception as e:
                print("Oh no: ", e)
                pass

        def _set_postfix_str(self, s, refresh=True):
            try:
                self.update(increment=0, subtitle=str(s))
                self.progress.update(increment=0, subtitle=str(s))
            except Exception:
                pass

        def _set_description(self, desc=None, refresh=True):
            try:
                self.progress.update(increment=0, title=str(desc) if desc else None)
            except Exception:
                pass

        def _noop(self, *a, **kw):
            pass

        # Only patch if missing — don't clobber a future real implementation
        for _name, _fn in [
            ("set_postfix", _set_postfix),
            ("set_postfix_str", _set_postfix_str),
            ("set_description", _set_description),
            ("set_description_str", _set_description),
            ("refresh", _noop),
            ("clear", _noop),
            ("reset", _noop),
            ("close", _noop),
        ]:
            if not hasattr(_Patched, _name):
                setattr(_Patched, _name, _fn)

        tqdm = _tqdm.tqdm
    else:
        from tqdm import tqdm


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
    return (device,)


@app.cell
def _():
    # Parse settings for EBM training
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, default="checkpoints/const_lr_schedule/step_180000.pt")

    if marimo.running_in_notebook():
        args, _ = parser.parse_known_args()

    else:
        args = parser.parse_args()

    # get root path for saving checkpoints and logs
    root = Path(__file__).absolute().parent.parent.parent

    checkpoint_arg = Path(args.checkpoint)
    if checkpoint_arg.is_absolute():
        folder = checkpoint_arg
    else:
        folder = root / checkpoint_arg
    return folder, root


@app.cell
def _(device, folder):
    model, model_ema, state= load_energy_model(folder, device=device)
    model.eval()
    return (model,)


@app.cell
def _():
    # thanks claude!
    # get hinton
    import io, urllib.request
    from PIL import Image
    from torchvision import transforms as T

    def get_image_from_url(url, crop = (0.25, 0.08, 0.95, 0.65)):
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

        image = Image.open(io.BytesIO(urllib.request.urlopen(req).read())).convert("L")  # convert to grayscale
        w, h = image.size
        image = image.crop((int(w*crop[0]), int(h*crop[1]), int(w*crop[2]), int(h*crop[3])))
        tfm = T.Compose([
            T.Resize(36),
            T.CenterCrop(32),
            T.ToTensor()
        ])

        return tfm(image)*2-1

    x_hinton = get_image_from_url("https://upload.wikimedia.org/wikipedia/commons/thumb/8/82/Geoffrey_Hinton_in_2026.jpg/250px-Geoffrey_Hinton_in_2026.jpg")

    print(x_hinton.min(), x_hinton.max(), x_hinton.shape)  # sanity check
    plt.imshow(x_hinton.squeeze(), cmap="gray")

    x_pica = get_image_from_url("https://archives.bulbagarden.net/media/upload/e/e7/Spr_1g_001.png", crop=(0, 0.3, 1, 1))
    plt.imshow(x_pica.squeeze(), cmap="gray")
    return x_hinton, x_pica


@app.cell
def _(device, model, root, x_hinton, x_pica):
    torch.manual_seed(1701)
    n_steps = 1000
    grad_clip = 1000
    step_size = 300
    noise_std = 0.01
    clamp = True
    plot_every = 50
    log_plot_step = True
    n_plots = n_steps//plot_every
    x_init_rand = torch.randn(3, 1, 32, 32, device=device)
    x_init_0 = torch.zeros(1, 1, 32, 32, device=device)-1  # start from a single blank image
    x_init_1 = torch.ones(1, 1, 32, 32, device=device)  # start from a single white image

    x_init = torch.cat([x_init_rand, x_init_0, x_init_1, x_hinton.unsqueeze(1).to(device), x_pica.unsqueeze(1).to(device)], dim=0)
    n_samples = x_init.shape[0]

    x_plot = [x_init.detach().cpu()]  # include initial state in the plot

    log_plot_steps = np.geomspace(1, n_steps-1, n_plots-1, dtype=int)
    log_plot_steps = np.unique(log_plot_steps)  # remove duplicates
    n_plots = len(log_plot_steps) + 1 # +1 for the initial state
    print(f"Log plot steps: {log_plot_steps}")

    energies = [model.energy(x_init).mean().item()]
    x = x_init.detach().clone().requires_grad_(True)
    for i in range(n_steps):
        energy = model.energy(x).mean()
        energies.append(energy.item())
        grad = torch.autograd.grad(energy, x)[0]
        if grad_clip is not None:
            grad = grad.clamp(-grad_clip, grad_clip)  # prevent blowups

        x = x - step_size * grad + noise_std * torch.randn_like(x)
        if clamp:
            x = x.clamp(-1, 1)

        if log_plot_step:
            if i in log_plot_steps:
                x_plot.append(x.detach().cpu())

        else:
            if i % plot_every == 0:
                x_plot.append(x.detach().cpu())

        x = x.detach().requires_grad_(True)


    fig, (ax, axE) = plt.subplots(2, 1, figsize=(n_plots/2, n_samples/2+3.5))

    axE.plot(energies)
    axE.set_ylabel("Mean Energy $E(x)$")
    axE.set_xlabel("Langevin Step")
    # axE.set_yscale("log")
    axE.set_xscale("log")
    axE.set_xlim(0, n_steps)
    for log_step in log_plot_steps:
        axE.axvline(log_step, color="gray", linestyle="--", alpha=0.5)
    axE.grid(False)

    x_plot = torch.stack(x_plot)  # (n_plots, n_samples, 1, 32, 32)
    print(f"After stack: {x_plot.shape}")
    # Add channel dimension if missing (in case it's 4D)
    if x_plot.ndim == 4:
        x_plot = x_plot.unsqueeze(2)
        print(f"Added channel dim: {x_plot.shape}")
    # Permute to (n_samples, n_plots, 1, 32, 32) so each row shows one sample's evolution
    x_plot = x_plot.permute(1, 0, 2, 3, 4)
    print(f"After permute: {x_plot.shape}")
    # Reshape to (n_samples*n_plots, 1, 32, 32) for make_grid
    x_plot = x_plot.reshape(n_samples * n_plots, 1, 32, 32)
    print(f"Before make_grid: {x_plot.shape}")
    # Create grid with n_plots columns so each row is one sample
    x_grid = torchvision.utils.make_grid((x_plot+1)/2, nrow=n_plots, padding=2, normalize=False)
    print(f"After make_grid: {x_grid.shape}")
    # Convert from (C, H, W) to (H, W, C) for matplotlib, then squeeze to remove channel dim
    x_grid = x_grid.permute(1, 2, 0).squeeze(-1)
    ax.imshow(x_grid.cpu().numpy(), cmap="gray")
    ax.axis("off")

    fig.savefig(root/"plots/langevin_sampling.png")

    fig
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
