import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")

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
    from torch.utils.tensorboard import SummaryWriter

    from pathlib import Path
    import argparse

    from diffusion_ebm.langevin import langevin_sample_train

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
    parser.add_argument("--name", type=str)
    parser.add_argument("--checkpoint", type=str, default="latest")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", type=str, default="resnet")
    parser.add_argument("--conditioning", action="store_true")

    # EBM settings
    parser.add_argument("--replay_size", type=int, default=1024*8)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--reinit_prob", type=float, default=0.05)
    parser.add_argument("--langevin_steps", type=int, default=60)
    parser.add_argument("--langevin_step_size", type=float, default=0.1)
    parser.add_argument("--langevin_noise_scale", type=float, default=0.005)

    # Training settings 
    parser.add_argument("--noise_level", type=float, default=0.005)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--val_every", type=int, default=200)
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--lr_schedule", type=str, default="cosine_old")

    if marimo.running_in_notebook():
        args, _ = parser.parse_known_args()

        name = "notebook_run"
        args.conditioning = False
    else:
        args = parser.parse_args()

        name = args.name

        if name is None:
            raise ValueError("Please provide a name for the run using --name")

    seed = args.seed
    model_type = args.model
    checkpoint = args.checkpoint
    conditioning = args.conditioning

    replay_size = args.replay_size
    alpha = args.alpha
    reinit_prob = args.reinit_prob
    langevin_steps = args.langevin_steps
    langevin_step_size = args.langevin_step_size
    langevin_noise_scale = args.langevin_noise_scale

    batch_size = args.batch_size
    num_epochs = args.num_epochs
    val_every = args.val_every
    save_every = args.save_every
    noise_level = args.noise_level
    lr_schedule = args.lr_schedule


    # get root path for saving checkpoints and logs
    root = Path(__file__).absolute().parent.parent.parent
    return (
        alpha,
        checkpoint,
        conditioning,
        lr_schedule,
        model_type,
        name,
        noise_level,
        num_epochs,
        reinit_prob,
        replay_size,
        root,
        save_every,
        val_every,
    )


@app.cell
def _(conditioning, root):
    ##############
    # Load data
    ##############
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
        batch_size=64,      # often larger than train batch — no backward pass, more memory headroom
        shuffle=True,       # order doesn't matter, and not shuffling makes results reproducible
        num_workers=4,
        pin_memory=True,
        drop_last=False,     # keep the last partial batch — you want every sample evaluated
    )
    return dataloader, n_classes, val_loader


@app.cell
def _(
    conditioning,
    dataloader,
    device,
    lr_schedule,
    model_type,
    n_classes,
    num_epochs,
):
    ###########################
    # Load model and optimizer
    ###########################

    from diffusion_ebm.models import EBM
    from diffusion_ebm.ema import EMA

    if model_type == "resnet":
        if conditioning:
            model = EBM(n_out=n_classes)
        else:
            model = EBM()
    else:
        raise NotImplementedError(f"Model type {model_type} not implemented")

    model_ema = EMA(model, decay=0.999, burn_in=500)

    model.train()
    model.to(device)
    # model = torch.compile(model, mode="reduce-overhead")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.0, 0.999))

    if lr_schedule == "cosine_old":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5000)
    elif lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs*len(dataloader), eta_min=1e-6)
    elif lr_schedule == "constant":
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    else:
        raise ValueError(f"Unknown lr_schedule: {lr_schedule}")

    print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    return model, model_ema, optimizer, scheduler


@app.cell
def _(dataloader, num_epochs):
    num_epochs*len(dataloader)
    return


@app.cell
def _(model, replay_size):
    ##########################################
    # Set up replay buffer and training state
    ##########################################
    replay_buffer = torch.rand(replay_size, model.in_channels, model.img_size, model.img_size, device="cuda") * 2 - 1  # init in [-1, 1]
    state = {'step': 0, 'loss_reg': [], 'val_energy': {},  'val_rand_energy': {}, 'e_pos': [], 'e_neg': [], 'e_rand': [], 'lr': []}
    return replay_buffer, state


@app.cell
def _(
    checkpoint,
    model,
    model_ema,
    name,
    optimizer,
    replay_buffer,
    root,
    scheduler,
    state,
):
    #######################
    # Load checkpoint
    #######################

    path = root / f"checkpoints/{name}/"
    plot_path = root / f"plots/{name}/"
    plot_path.mkdir(parents=True, exist_ok=True)

    if path.exists():
        if checkpoint == "new" or checkpoint == "none" or checkpoint is None:
            raise ValueError(f"Checkpoint {checkpoint} already exists. Choose a different name or delete it.")

        if checkpoint == "latest":
            checkpoints = list(path.glob("*.pt"))
            checkpoints.sort(key=lambda x: int(x.stem.split("_")[-1] if x.stem.split("_")[-1].isdigit() else x.stem.split("_")[-2]))  # sort by step number, but if not possible (e.g. "checkpoint_final.pt"), put at the end
            load_checkpoint = checkpoints[-1]
            print(f"Loading checkpoint: {load_checkpoint}")
        else:
            load_checkpoint = path / f"step_{checkpoint}.pt"
            if not load_checkpoint.exists():
                raise ValueError(f"Checkpoint {load_checkpoint} does not exist.")
            print(f"Loading checkpoint: {load_checkpoint}")
    
        # Load everything
        loaded_state = torch.load(load_checkpoint)
        model.load_state_dict(loaded_state['model_state_dict'])
        scheduler.load_state_dict(loaded_state['scheduler_state_dict'])
        optimizer.load_state_dict(loaded_state['optimizer_state_dict'])
        model_ema.shadow = loaded_state['ema_shadow']
        replay_buffer.copy_(loaded_state['replay_buffer'])
        for key in loaded_state['state']:
            state[key] = loaded_state['state'][key]
    else: 
        path.mkdir(parents=True, exist_ok=True)
    
    return path, plot_path


@app.function
@torch.no_grad()
def eval_ebm(model, val_loader, device="cuda", num_val_batches=10):
    model.eval()
    total = 0.0
    n = 0
    for i, (x, _) in enumerate(val_loader):
        if i >= num_val_batches:
            break
        x = x.to(device, non_blocking=True)
        e = model.energy(x).mean().item()
        total += e * x.shape[0]
        n += x.shape[0]

    x_unif = torch.rand(n, 1, 32, 32, device=device) * 2 - 1
    e_unif = model.energy(x_unif).mean().item()

    model.train()
    return total / n, e_unif


@app.function
def show_replay_buffer_samples(replay_buffer, n_samples=15**2, determinisitic=True, step=None):
    # visualize buffer images
    with torch.no_grad():
        n_samples =min(n_samples, replay_buffer.shape[0])
        if determinisitic:
            idxs = torch.arange(n_samples, device="cuda")
        else:
            idxs = torch.randperm(replay_buffer.shape[0], device="cuda")[:n_samples]
        samples = replay_buffer[idxs].cpu()
        samples = (samples + 1) / 2  # unnormalize from [-1, 1] to [0, 1]
        grid = torchvision.utils.make_grid(samples, nrow=np.sqrt(n_samples).astype(int))
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(grid.permute(1, 2, 0))
        ax.axis("off")
        if step is None:
            ax.set_title("Samples from Replay Buffer")
        else:
            ax.set_title(f"Samples from Replay Buffer Step {step}")

    return fig, ax


@app.cell
def _(conditioning):
    def train_ebm(model, dataloader, val_loader, replay_buffer, state, optimizer, scheduler, model_ema, path, alpha=0.1, reinit_prob=0.05, val_every=200, save_every=1000, num_epochs=50, noise_level=0.005):

        tb = SummaryWriter(log_dir=path/"logs")

        for epoch in tqdm(range(num_epochs)):
            try:
                pbar = tqdm(dataloader, desc=f"Epoch {epoch} (Total: {state['step']//len(dataloader)})", leave=False)
                for _x, _y in dataloader:
                    _x = _x.to("cuda")
                    _x = _x + noise_level * torch.randn_like(_x)
                    if conditioning:
                        _y = _y.to("cuda")

                    n_batch = _x.shape[0]

                    # --- Draw negatives from buffer, occasionally reinit from noise ---
                    idx = torch.randint(0, replay_buffer.shape[0], (n_batch,), device="cuda")
                    x_neg = replay_buffer[idx]
                    reinit_mask = torch.rand(n_batch, device="cuda") < reinit_prob
                    x_neg[reinit_mask] = torch.rand_like(x_neg[reinit_mask]) * 2 - 1

                    # --- Run Langevin to refine negatives ---
                    # x_neg = langevin_sample(model, x_neg)
                    x_neg_out = langevin_sample_train(model, x_neg)

                    # --- Write refined negatives back to buffer ---
                    replay_buffer[idx] = x_neg_out

                    # --- Compute loss ---
                    energies = model(torch.cat([_x, x_neg_out]))
                    e_pos, e_neg = energies.split(n_batch)

                    cd_loss = e_pos.mean() - e_neg.mean()
                    reg_loss = e_pos.pow(2).mean() + e_neg.pow(2).mean()
                    loss = cd_loss + alpha*reg_loss
                    state['loss_reg'].append(reg_loss.item())
                    state['e_pos'].append(e_pos.mean().item())
                    state['e_neg'].append(e_neg.mean().item())
                    x_rand = torch.rand_like(_x) * 2 - 1
                    state['e_rand'].append(model(x_rand).mean().item())
                    state['lr'].append(optimizer.param_groups[0]['lr'])

                    tb.add_scalar("Loss/reg", reg_loss, state['step'])
                    tb.add_scalar("Loss/cd", cd_loss, state['step'])
                    tb.add_scalar("Loss", loss, state['step'])
                    tb.add_scalar("Energy/e_pos-e_rand", state['e_pos'][-1] - state['e_rand'][-1], state['step'])
                    tb.add_scalar("Energy/e_neg-e_rand", state['e_neg'][-1] - state['e_rand'][-1], state['step'])
                    tb.add_scalar("Learning Rate", optimizer.param_groups[0]['lr'], state['step'])

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()

                    model_ema.update()

                    # --- Periodic validation ---
                    if state['step'] % val_every == 0:
                        val_energy, val_rand_energy = eval_ebm(model, val_loader, device="cuda")
                        state['val_energy'][state['step']] = val_energy
                        state['val_rand_energy'][state['step']] = val_rand_energy
                        tb.add_scalar("Val Energy/e_val-e_rand", val_energy - val_rand_energy, state['step'])
                        _fig, _ = show_replay_buffer_samples(replay_buffer, step=state['step'])
                        _fig.savefig(path/f"replay_buffer_step_{state['step']}.png")
                        plt.close(_fig)

                    if state['step'] % save_every == 0:
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'ema_shadow': model_ema.shadow,
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'replay_buffer': replay_buffer,
                            'state': state
                        }, path/f"step_{state['step']}.pt")

                    state['step'] = state['step'] + 1

                    pbar.set_postfix({
                        "loss_reg": f"{state['loss_reg'][-1]:.2f}",
                        "e_pos-e_rand": f"{state['e_pos'][-1] - state['e_rand'][-1]:.2f}",
                        "e_neg-e_rand": f"{state['e_neg'][-1] - state['e_rand'][-1]:.2f}",
                        "val_energy": f"{list(state['val_energy'].values())[-1]-list(state['val_rand_energy'].values())[-1]:.2f}" if len(state['val_energy']) > 0 else "N/A"
                    })

                    pbar.update()

            except KeyboardInterrupt:
                print("Training interrupted. Saving checkpoint...")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'ema_shadow': model_ema.shadow,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'replay_buffer': replay_buffer,
                    'state': state
                }, path/f"step_{state['step']}_interrupted.pt")
                _fig, _ = show_replay_buffer_samples(replay_buffer, step=state['step'])
                _fig.savefig(path/f"replay_buffer_step_{state['step']}_interrupted.png")
                plt.close(_fig)
                break

            pbar.close()
        tb.close()

    return (train_ebm,)


@app.cell
def _(
    alpha,
    dataloader,
    model,
    model_ema,
    noise_level,
    num_epochs,
    optimizer,
    path,
    reinit_prob,
    replay_buffer,
    save_every,
    scheduler,
    state,
    train_ebm,
    val_every,
    val_loader,
):
    # Train the EBM
    train_ebm(model=model, dataloader=dataloader, val_loader=val_loader, replay_buffer=replay_buffer, state=state, optimizer=optimizer, scheduler=scheduler, model_ema=model_ema, path=path, alpha=alpha, reinit_prob=reinit_prob, val_every=val_every, save_every=save_every, num_epochs=num_epochs, noise_level=noise_level)
    return


@app.cell
def _(dataloader, plot_path, state):
    _fig, _ax = plt.subplots()

    # moving average for smoother curves
    def moving_average(x, w=10):
        return np.convolve(x, np.ones(w), "valid") / w

    _e_pos_list = np.array(state['e_pos'][:len(state['e_rand'])])  # align lengths
    _e_neg_list = np.array(state['e_neg'][:len(state['e_rand'])])  # align lengths
    _e_rand_list = np.array(state['e_rand'])  # align lengths

    _ax.plot(moving_average(state['loss_reg']), label=r"$E(x^+)^2+E(x^-)^2$")
    _ax.plot(moving_average(_e_pos_list), label=r"$E(x^+)$")
    _ax.plot(moving_average(_e_neg_list), label=r"$E(x^-)$")
    # _ax.plot(moving_average(_e_rand_list), label=r"$E(x^{\mathrm{rand}})$")
    _ax.plot(moving_average(_e_pos_list-_e_rand_list), label=r"$E(x^+) - E(x^{\mathrm{rand}})$")

    n_val = list(state['val_energy'].keys())
    _ax.plot(n_val, np.array(list(state['val_energy'].values()))-np.array(list(state['val_rand_energy'].values())), label=r"$E(x^{\mathrm{val}}) - E(x^{\mathrm{rand}})$")

    # show epochs as vertical lines
    for _e in range(1, state['step']//len(dataloader) + 1):
        _ax.axvline(x=_e*len(dataloader), color="gray", linestyle="--", alpha=0.2)

    _ax.legend()

    _ax.set_xlabel("Training Step")
    _ax.set_ylabel("Value")
    _fig.savefig(plot_path/f"training_curves.png")
    _fig
    return


@app.cell
def _(replay_buffer):
    show_replay_buffer_samples(replay_buffer)
    return


@app.cell
def _(grads, model, model_ema):
    def show_samples(model, n=16, n_steps=5000, step_size=2.0):
        model.eval()
        model_ema.apply_shadow()
        x = torch.rand(n, 1, 32, 32, device="cuda") * 2 - 1
        x.requires_grad_(True)
        x = langevin_sample_train(model, x, n_steps=n_steps, step_size=step_size)

        model_ema.restore()
        model.train()

        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(4, 4, figsize=(6, 6))
        for ax, img in zip(axes.flat, x.detach().cpu()):
            ax.imshow(img.squeeze(), cmap="gray", vmin=-1, vmax=1)
            ax.axis("off")

        return fig

        fig, axes = plt.subplots()
        axes.plot(grads)
        axes.set_title("Gradient Magnitudes During Sampling")
        axes.set_xlabel("Sampling Step")
        axes.set_ylabel("Mean |grad| * epsilon/2")
        return fig

    show_samples(model, step_size=1, n_steps=2000)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
