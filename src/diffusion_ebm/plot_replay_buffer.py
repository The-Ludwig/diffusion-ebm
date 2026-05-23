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

    from diffusion_ebm.train_ebm import show_replay_buffer_samples

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


@app.cell
def _(folder, root):
    checkpoints = list((root/folder).glob("*.pt"))

    def get_step(path):
        return int(path.stem.split("_")[-1] if path.stem.split("_")[-1].isdigit() else path.stem.split("_")[-2])

    checkpoints.sort(key=get_step)  # sort by step number, but if not possible (e.g. "checkpoint_final.pt"), put at the end

    cut_percentile = 0
    _name = Path(folder).name
    _plotfolder = root/"plots"/_name/"replay_buffer"
    _plotfolder.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        pbar = tqdm(checkpoints)
        for cp in pbar:
            step = get_step(cp)
            epoch = step // 938
            loaded = torch.load(cp)

            imgs = loaded['replay_buffer']

            fig, ax = show_replay_buffer_samples(imgs, n_samples=5**2, step=epoch, plt_title="Epoch")
            fig.savefig(_plotfolder/f"{step}.png")
            plt.close(fig)
    return


if __name__ == "__main__":
    app.run()
