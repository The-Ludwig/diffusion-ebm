import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    from tqdm.auto import tqdm
    import marimo
    import torchvision

    from pathlib import Path

    from diffusion_ebm.langevin import langevin_sample_train

    return (
        DataLoader,
        Path,
        datasets,
        langevin_sample_train,
        marimo,
        np,
        plt,
        torch,
        torchvision,
        transforms,
    )


@app.cell
def _(DataLoader, datasets, transforms):
    transform = transforms.Compose([
            transforms.Pad(2),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
    dataset = datasets.MNIST(root="../data", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    val_dataset = datasets.MNIST(root="../data", train=False, download=True, transform=transform)
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,      # often larger than train batch — no backward pass, more memory headroom
        shuffle=True,       # order doesn't matter, and not shuffling makes results reproducible
        num_workers=4,
        pin_memory=True,
        drop_last=False,     # keep the last partial batch — you want every sample evaluated
    )
    return dataloader, val_loader


@app.cell
def _(torch):
    from diffusion_ebm.models import EBM
    from diffusion_ebm.ema import EMA

    model = EBM()

    model_ema = EMA(model, decay=0.999, burn_in=500)

    model.train()
    model.to("cuda")
    # model = torch.compile(model, mode="reduce-overhead")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.0, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5000)

    sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model, model_ema, optimizer, scheduler


@app.cell
def _(torch):
    @torch.no_grad()
    def eval_ebm(model, val_loader, device="cuda", num_val_batches=10):
        model.eval()
        total = 0.0
        n = 0
        for i, (x, _) in enumerate(val_loader):
            if i >= num_val_batches:
                break
            x = x.to(device, non_blocking=True)
            e = model(x).mean().item()
            total += e * x.shape[0]
            n += x.shape[0]

        x_unif = torch.rand(n, 1, 32, 32, device=device) * 2 - 1
        e_unif = model(x_unif).mean().item()

        model.train()
        return total / n, e_unif

    return (eval_ebm,)


@app.cell
def _(model, torch):
    replay_buffer = torch.rand(1024*8, model.in_channels, model.img_size, model.img_size, device="cuda") * 2 - 1  # init in [-1, 1]
    state = {'step': 0, 'loss_reg': [], 'val_energy': {},  'val_rand_energy': {}, 'e_pos': [], 'e_neg': [], 'e_rand': []}
    return replay_buffer, state


@app.cell
def _(np, plt, torch, torchvision):
    def show_replay_buffer_samples(replay_buffer, n_samples=15**2, determinisitic=True):
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
            ax.set_title("Samples from Replay Buffer")

        return fig, ax

    return (show_replay_buffer_samples,)


@app.cell
def _(
    Path,
    model,
    model_ema,
    optimizer,
    replay_buffer,
    scheduler,
    state,
    torch,
):
    model_name = "model_test"
    path = Path(f"../checkpoints/{model_name}/")

    if path.exists():
        checkpoints = list(path.glob("*.pt"))
        checkpoints.sort(key=lambda x: int(x.stem.split("_")[-1]))
        latest_checkpoint = checkpoints[-1]
        print(f"Loading checkpoint: {latest_checkpoint}")

        loaded_state = torch.load(latest_checkpoint)
        model.load_state_dict(loaded_state['model_state_dict'])
        scheduler.load_state_dict(loaded_state['scheduler_state_dict'])
        optimizer.load_state_dict(loaded_state['optimizer_state_dict'])
        model_ema.shadow = loaded_state['ema_shadow']
        replay_buffer.copy_(loaded_state['replay_buffer'])
        for key in loaded_state['state']:
            state[key] = loaded_state['state'][key]
    return (path,)


@app.cell
def _(
    dataloader,
    eval_ebm,
    langevin_sample_train,
    marimo,
    model,
    model_ema,
    optimizer,
    path,
    replay_buffer,
    scheduler,
    show_replay_buffer_samples,
    state,
    torch,
    val_loader,
):
    alpha = .1
    reinit_prob = 0.05

    val_every = 200
    save_every = 1000
    num_epochs = 50
    noise_level = 0.005
    for epoch in range(num_epochs):
        with marimo.status.progress_bar(total=len(dataloader), title=f"Epoch {epoch}") as bar:
        # pbar = tqdm(dataloader, desc=f"Epoch {epoch}", dynamic_ncols=True)
            for _x, _ in dataloader:
                _x = _x.to("cuda")
                _x = _x + noise_level * torch.randn_like(_x)
                # y = y.to("cuda")

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

                if state['step'] % save_every == 0:
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'ema_shadow': model_ema.shadow,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'replay_buffer': replay_buffer,
                        'state': state
                    }, path/f"step_{state['step']}.pt")
                    _fig, _ = show_replay_buffer_samples(replay_buffer)
                    _fig.savefig(path/f"replay_buffer_step_{state['step']}.png")

                bar.update(
                    subtitle=f"loss: {state['loss_reg'][-1]:.2f}, e_pos-e_rand: {state['e_pos'][-1] - state['e_rand'][-1]:.2f}, val: {list(state['val_energy'].values())[-1]:.2f}" if len(state['val_energy']) > 0 else "N/A"
                )
                state['step'] = state['step'] + 1
    return


@app.cell
def _(np, plt, state):
    _fig, _ax = plt.subplots()

    # moving average for smoother curves
    def moving_average(x, w=10):
        return np.convolve(x, np.ones(w), "valid") / w

    _e_pos_list = np.array(state['e_pos'][:len(state['e_rand'])])  # align lengths
    _e_rand_list = np.array(state['e_rand'])  # align lengths

    _ax.plot(moving_average(state['loss_reg']), label="Total Loss")
    # ax.plot(moving_average(_e_pos_list), label="E_pos")
    # ax.plot(moving_average(_e_neg_list), label="E_neg")
    _ax.plot(moving_average(_e_pos_list-_e_rand_list), label="E_pos - E_rand")
    # ax.plot(list(state['val_energy'].keys()), list(state['val_energy'].values()), label="Val Energy")
    # ax.plot(list(state['val_rand_energy'].keys()), list(state['val_rand_energy'].values()), label="Val Rand Energy")

    n_val = [n if n < 10_000 else n-60_000+1_400 for n in state['val_energy'].keys()]
    _ax.plot(n_val, np.array(list(state['val_energy'].values()))-np.array(list(state['val_rand_energy'].values())), label="Val Energy - Rand Energy")

    _ax.legend()

    _ax.set_xlabel("Training Step")
    _ax.set_ylabel("Value")
    _fig.savefig("training_curves.png")
    _fig
    return


@app.cell
def _(state):
    list(state['val_energy'].keys())
    return


@app.cell
def _(replay_buffer, show_replay_buffer_samples):
    show_replay_buffer_samples(replay_buffer)
    return


@app.cell
def _(grads, langevin_sample_train, model, model_ema, torch):
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
        plt.tight_layout()
        plt.show()

        return None

        fig, axes = plt.subplots()
        axes.plot(grads)
        axes.set_title("Gradient Magnitudes During Sampling")
        axes.set_xlabel("Sampling Step")
        axes.set_ylabel("Mean |grad| * epsilon/2")
        plt.show()

    show_samples(model, step_size=2, n_steps=2000)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
