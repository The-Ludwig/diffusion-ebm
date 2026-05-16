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

    from diffusion_ebm.proj2.model import MicroDiT, NUM_CLASSES, PAD_SIZE
    from diffusion_ebm.proj2.sample_energy import load_model, make_schedule
    from diffusion_ebm.models import EBM, load_energy_model
    from diffusion_ebm.ema import EMA

    T = 1000
    beta_start = 1e-4
    beta_end = 0.02

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

    parser.add_argument("--checkpoint", type=str, default="micro_dit_checkpoint.pt")
    parser.add_argument("--w", type=float, default=3.0, help="guidance scale")
    parser.add_argument("--num-per-class", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="samples.png")
    parser.add_argument("--energy-model", type=str, default="checkpoints/const_lr_schedule/step_180000.pt", help="optional energy model checkpoint for monitoring energy during sampling")

    if marimo.running_in_notebook():
        args, _ = parser.parse_known_args()

    else:
        args = parser.parse_args()

    dit_checkpoint = args.checkpoint
    guidance_scale = args.w
    num_per_class = args.num_per_class
    seed = args.seed
    out_path = args.out
    energy_model = args.energy_model

    # get root path for saving checkpoints and logs
    root = Path(__file__).absolute().parent.parent.parent
    return args, dit_checkpoint, energy_model, num_per_class, root


@app.cell
def _(device, dit_checkpoint, energy_model, root):
    # load models
    if energy_model is not None:
        ebm, ebm_ema, _ = load_energy_model(root/energy_model)

    dit = load_model(root/dit_checkpoint, device=device)
    return dit, ebm


@app.function
def sample_images(model, labels, guidance_scale=3.0, initial_noise=None, device=None, energy_model=None, energy_every=10):
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

    if energy_model is not None:
        energies = []
        samples = []
        cossims = []
        l2dists = []
        cossimf = torch.nn.CosineSimilarity(dim=1)

    # further speed-up tricks
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for t in tqdm(reversed(range(T)), total=T, desc="Sampling", leave=False):

            t_batch = torch.full((N*n_w,), t, device=device, dtype=torch.long)

            x = x.flatten(0, 1)
            null_labels = null_labels.flatten(0, 1)
            labels = labels.flatten(0, 1)


            eps_uncond = model(x, t_batch, null_labels).unflatten(0, (n_w, N)).clone()
            eps_cond = model(x, t_batch, labels).unflatten(0, (n_w, N))

            if energy_model is not None and t % energy_every == 0:
                with torch.enable_grad():
                    x_dif = x.detach().clone().requires_grad_(True)
                    energy = energy_model(x_dif)
                    energy_grad = -torch.autograd.grad(energy.sum(), x_dif)[0]
                score = eps_uncond / sch["sqrt_one_minus_alphas_cumprod"][t]

                l2_dist = torch.norm(score.squeeze().flatten(-2) - energy_grad.squeeze().flatten(-2), dim=1).cpu().to(torch.float64).numpy()
                cosine_sim = cossimf(score.squeeze().flatten(-2), energy_grad.squeeze().flatten(-2)).cpu().to(torch.float64).numpy()

                energies.append(energy.cpu().to(torch.float64).numpy())
                samples.append(x.cpu())
                cossims.append(cosine_sim)
                l2dists.append(l2_dist)

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

    if energy_model is None:
        return x.squeeze(0)

    return x.squeeze(0), energies, samples, cossims, l2dists


@app.cell
def _(args, device, dit, ebm, energy_model):
    labels = torch.arange(NUM_CLASSES, device=device).repeat_interleave(args.num_per_class)

    if energy_model is None:
        samples = sample_images(dit, labels, guidance_scale=args.w, device=device)
    else:
        samples, energies, samples_at_step, cossims, l2dists = sample_images(dit, labels, guidance_scale=args.w, device=device, energy_model=ebm) 
    return cossims, energies, l2dists, samples_at_step


@app.cell
def _(energies, num_per_class, samples_at_step):
    from matplotlib.colors import ListedColormap

    _fig, _ax = plt.subplots()

    _steps = np.arange(T-10, -10, -10)

    _energies = np.array(energies)

    _base = plt.get_cmap("viridis", lut=NUM_CLASSES)
    _cbar_cmap = ListedColormap([_base(i) for i in range(NUM_CLASSES)] + [(1, 0, 0, 1)])

    for _c in range(NUM_CLASSES):
        _ax.plot(_steps, _energies[:,_c], color=_base(_c))

    _ax.plot(_steps, _energies.mean(axis=1), color="red")


    _cbar = plt.colorbar(
        plt.cm.ScalarMappable(cmap=_cbar_cmap, norm=plt.Normalize(vmin=0, vmax=NUM_CLASSES+1)),
        ax=_ax,
        ticks=np.arange(NUM_CLASSES+1) + 0.5,
    )
    _cbar.set_ticklabels([f"{i}" for i in range(NUM_CLASSES)] + ["mean"])



    for _t in np.arange(0, 1050, 50):
        _idx = np.argmin(np.abs(_steps - _t))
        _ax.imshow(samples_at_step[_idx][:10*num_per_class:num_per_class,0].flatten(0,1), cmap="gray", vmin=-1, vmax=1, extent=(_t, _t-50, 0.1*_energies.max()+_energies.min(), 0.9*_energies.max()), aspect="auto")

    _ax.set_ylim(0, _energies.max())
    _ax.set_xlim(T, 0)

    _ax.set_xlabel("Denoising step $t$")
    _ax.set_ylabel("Energy / a.u.")
    return (ListedColormap,)


@app.cell
def _(ListedColormap, cossims):
    _fig, _ax = plt.subplots()

    _steps = np.arange(T-10, -10, -10)

    _cossims = np.array(cossims)

    _base = plt.get_cmap("viridis", lut=NUM_CLASSES)
    _cbar_cmap = ListedColormap([_base(i) for i in range(NUM_CLASSES)] + [(1, 0, 0, 1)])

    for _c in range(NUM_CLASSES):
        _ax.plot(_steps, _cossims[:,_c], color=_base(_c))
    print(_cossims.shape)

    _ax.plot(_steps, _cossims.mean(axis=1), color="red")


    _cbar = plt.colorbar(
        plt.cm.ScalarMappable(cmap=_cbar_cmap, norm=plt.Normalize(vmin=0, vmax=NUM_CLASSES+1)),
        ax=_ax,
        ticks=np.arange(NUM_CLASSES+1) + 0.5,
    )
    _cbar.set_ticklabels([f"{i}" for i in range(NUM_CLASSES)] + ["mean"])



    # for _t in np.arange(0, 1050, 50):
    #     _idx = np.argmin(np.abs(_steps - _t))
    #     _ax.imshow(samples_at_step[_idx][:10*num_per_class:num_per_class,0].flatten(0,1), cmap="gray", vmin=-1, vmax=1, extent=(_t, _t-50, -0.15, 0.3), aspect="auto")

    _ax.set_ylim(-0.25, 0.4)
    _ax.set_xlim(T, 0)

    _ax.text(500, 0.35, r"$\uparrow$ Similar $\uparrow$", ha="center")
    _ax.text(500, -0.22, "$\downarrow$ Dissimilar $\downarrow$", ha="center")

    _ax.set_xlabel("Denoising step $t$")
    _ax.set_ylabel(r"Cosine Similarity $\nabla_\mathbf{x} E_{\phi}/s_{\theta}(\mathbf{x})$")
    _fig
    return


@app.cell
def _(dit, ebm):
    print(f"Trainable parameters in EBM: {sum(p.numel() for p in ebm.parameters() if p.requires_grad)}")
    print(f"Trainable parameters in DiT: {sum(p.numel() for p in dit.parameters() if p.requires_grad)}")
    return


@app.cell
def _(ListedColormap, l2dists, num_per_class, samples_at_step):
    _fig, _ax = plt.subplots()

    _steps = np.arange(T-10, -10, -10)

    _l2dists = np.array(l2dists)

    _base = plt.get_cmap("viridis", lut=NUM_CLASSES)
    _cbar_cmap = ListedColormap([_base(i) for i in range(NUM_CLASSES)] + [(1, 0, 0, 1)])

    for _c in range(NUM_CLASSES):
        _ax.plot(_steps, _l2dists[:,_c], color=_base(_c))
    print(_l2dists.shape)

    _ax.plot(_steps, _l2dists.mean(axis=1), color="red")


    _cbar = plt.colorbar(
        plt.cm.ScalarMappable(cmap=_cbar_cmap, norm=plt.Normalize(vmin=0, vmax=NUM_CLASSES+1)),
        ax=_ax,
        ticks=np.arange(NUM_CLASSES+1) + 0.5,
    )
    _cbar.set_ticklabels([f"{i}" for i in range(NUM_CLASSES)] + ["mean"])



    for _t in np.arange(0, 1050, 50):
        _idx = np.argmin(np.abs(_steps - _t))
        _ax.imshow(samples_at_step[_idx][:10*num_per_class:num_per_class,0].flatten(0,1), cmap="gray", vmin=-1, vmax=1, extent=(_t, _t-50, 1, 12), aspect="auto")

    _ax.set_ylim(20, 10000)
    _ax.set_yscale("log")
    _ax.set_xlim(T, 0)

    _ax.set_xlabel("Denoising step $t$")
    _ax.set_ylabel(r"L2 Distance $E_{\phi}/s_{\theta}$")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
