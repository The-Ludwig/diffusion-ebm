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
    return


@app.cell
def _():
    # Parse settings for EBM training
    parser = argparse.ArgumentParser()

    parser.add_argument("--folder", type=str, default="checkpoints/const_lr_schedule")

    if marimo.running_in_notebook():
        args, _ = parser.parse_known_args()

    else:
        args = parser.parse_args()

    folder = args.folder

    # get root path for saving checkpoints and logs
    root = Path(__file__).absolute().parent.parent.parent
    return folder, root


@app.function
def inception_score(sample, classifier, splits=1):
    with torch.no_grad():
        p_pred = torch.softmax(classifier(sample), dim=-1)

    # Bootstrap-like estimate of variance
    split_len = len(sample) // splits
    scores = []
    for i in range(splits):
        part = p_pred[i*split_len:(i+1)*split_len]

        # this finds the mean in each class
        p_marginal = part.mean(dim=0) 

        # this sums over the different classes
        kl_div = (part*(part.log()-p_marginal.log())).sum(dim=1)

        scores.append(kl_div.mean().exp().item())

    return np.mean(scores), np.std(scores)


@app.function
def get_gaussian_params(data):
    mean = data.mean(dim=0)
    cov = torch.cov(data.T)
    return mean, cov


@app.function
def sqrtm(cov):
    # Eigen decomposition
    eigvals, eigvecs = torch.linalg.eigh(cov)

    # Handle small eigenvalues for numerical stability
    eigvals_clipped = torch.clamp(eigvals, min=1e-10)

    # Square root of eigenvalues
    sqrt_eigvals = torch.sqrt(eigvals_clipped)

    # Reconstruct the square root matrix
    sqrt_cov = eigvecs @ torch.diag(sqrt_eigvals) @ eigvecs.T
    return sqrt_cov


@app.function
def fid(sample, real_sample, classifier, remove_last_layer=True):
    if remove_last_layer:
        # extract one layer above
        p_gen = classifier(sample, return_features=True)
        p_real = classifier(real_sample, return_features=True)
    else:
        p_gen = classifier(sample)
        p_real = classifier(real_sample)


    mean_gen, cov_gen = get_gaussian_params(p_gen)
    mean_real, cov_real = get_gaussian_params(p_real)

    # Frechet distance
    diff = mean_gen - mean_real

    sqrt_cov_gen = sqrtm(cov_gen)
    inner = sqrt_cov_gen @ cov_real @ sqrt_cov_gen
    # Symmetrize to kill numerical asymmetry
    inner = 0.5 * (inner + inner.T)
    covmean = sqrtm(inner)

    return (diff @ diff + torch.trace(cov_gen + cov_real - 2 * covmean)).item()


@app.cell
def _(root):
    classifier = load_classifier(root/"checkpoints/classifier_mnist_resnet.pth", device="cuda").to("cuda")
    return (classifier,)


@app.cell
def _(root):
    transform = transforms.Compose([
            transforms.Pad(2),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])


    val_dataset = datasets.MNIST(root=root/"data", train=False, download=True, transform=transform)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1025*4,      # often larger than train batch — no backward pass, more memory headroom
        shuffle=True,       # order doesn't matter, and not shuffling makes results reproducible
        num_workers=4,
        pin_memory=True,
        drop_last=False,     # keep the last partial batch — you want every sample evaluated
    )
    real_sample, _ = next(iter(val_loader))
    real_sample = real_sample.to("cuda")
    return (real_sample,)


@app.cell
def _():
    from diffusion_ebm.proj2.memorization import improved_pr_pixel

    return (improved_pr_pixel,)


@app.cell
def _(classifier, folder, improved_pr_pixel, real_sample, root):
    checkpoints = list((root/folder).glob("*.pt"))

    def get_step(path):
        return int(path.stem.split("_")[-1] if path.stem.split("_")[-1].isdigit() else path.stem.split("_")[-2])

    checkpoints.sort(key=get_step)  # sort by step number, but if not possible (e.g. "checkpoint_final.pt"), put at the end

    cut_percentile = 10

    is_list = []
    fid_list = []
    p_list = []
    r_list = []
    with torch.no_grad():
        pbar = tqdm(checkpoints)
        for cp in pbar:
            step = get_step(cp)

            model, model_ema, state = load_energy_model(cp, device="cuda")
            model_ema.apply_shadow()


            imgs = state['replay_buffer'].to("cuda")
            energy = model(imgs)
            sort_idx = torch.argsort(energy)
            mask = sort_idx[:int(len(sort_idx)*(1-cut_percentile/100))]

            is_ = inception_score(imgs[mask], classifier, splits=10)
            is_list.append(is_)


            idxs = torch.randperm(mask.shape[0], device="cuda")[:real_sample.shape[0]]
            img_sampled = imgs[mask[idxs]]

            fid_ = fid(img_sampled, real_sample, classifier)
            fid_list.append(fid_)

            p, r = improved_pr_pixel(img_sampled, real_sample)
            p_list.append(p)
            r_list.append(r)

            pbar.set_postfix({"Inception Score": f"{is_[0]:.2f}±{is_[1]:.2f}", "FID": f"{fid_:.2f}", "Precision": f"{p:.2f}", "Recall": f"{r:.2f}"})
    return checkpoints, fid_list, get_step, is_list, p_list, r_list


@app.cell
def _(checkpoints, fid_list, folder, get_step, is_list, root):
    iscores = np.array(is_list)
    steps = [get_step(cpt) for cpt in checkpoints][:iscores.shape[0]]
    name = Path(folder).name

    _fig, _ax = plt.subplots()

    _ax.fill_between(steps, iscores[:,0]-iscores[:,1], iscores[:,0]+iscores[:,1], alpha=0.3)
    _ax.plot(steps, iscores[:,0])

    twinx = _ax.twinx()
    twinx.plot(steps, fid_list, color="C1")
    twinx.set_yscale("log")
    # twinx.set_ylim(0, 1)
    twinx.grid(False)
    twinx.set_ylabel(r"FID $\downarrow$", color="C1")

    _ax.set_ylabel(r"IS $\uparrow$", color="C0")
    _ax.set_xlabel("Step")

    _fig.savefig(root/"plots"/name/"is_fid.png")

    _fig
    return name, steps


@app.cell
def _(cov_gen, cov_real, name, p_list, r_list, root, sqrt_cov_gen, steps):
    _fig, _ax = plt.subplots()

    _ax.plot(steps, p_list, label="Precision")
    _ax.plot(steps, r_list, label="Recall")
    _ax.legend()

    _ax.set_ylabel("Precision/Recall")
    _ax.set_xlabel("Step")

    _fig.savefig(root/"plots"/name/"pr.png")

    _figsqrt_cov_gen = sqrtm(cov_gen)
    inner = sqrt_cov_gen @ cov_real @ sqrt_cov_gen
    # Symmetrize to kill numerical asymmetry
    inner = 0.5 * (inner + inner.T)
    covmean = sqrtm(inner)
    return


@app.cell
def _():
    return


@app.cell
def _(
    FrechetInceptionDistance,
    InceptionScore,
    classifier,
    get_dataloader,
    get_sample,
    load_model,
):
    def main(n_samples=100):
        model = load_model()

        with torch.no_grad():
            sample = get_sample(model, n_samples=n_samples).to("cpu")

        print(f"Inception score: {inception_score(sample, classifier, splits=10)}")
        print(f"Inception score: {inception_score(sample, classifier, splits=1)}")


        # Double check
        score = InceptionScore(feature=classifier, normalize=False, compute_with_cache=False)

        print(f"Inception score (torchmetrics): {score(sample)}")

        # FID 
        # Load the real samples from MNIST 
        dl = get_dataloader(batch_size=n_samples)
        real_sample, _ = next(iter(dl))
        real_sample = real_sample.to(sample.device)

        print(f"FID: {fid(sample, real_sample, classifier)}")
        print(f"FID (before softmax): {fid(sample, real_sample, classifier, remove_last_layer=False)}")

        fid_score = FrechetInceptionDistance(
            feature=classifier,
            input_img_size=real_sample.shape[-3:],
            normalize=True,
        ).to(sample.device)
        fid_score.update(sample, real=False)

        fid_score.update(real_sample, real=True)

        print(f"FID (torchmetrics): {fid_score.compute()}")

    return


if __name__ == "__main__":
    app.run()
