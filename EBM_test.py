import marimo

__generated_with = "0.23.3"
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
    from torch.nn.utils.parametrizations import spectral_norm

    return (
        DataLoader,
        datasets,
        marimo,
        nn,
        np,
        plt,
        spectral_norm,
        torch,
        torchvision,
        transforms,
    )


@app.cell
def _():
    from diffusion_ebm.model import MicroET

    return


@app.cell
def _(DataLoader, datasets, transforms):
    transform = transforms.Compose([
            transforms.Pad(2),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    val_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,      # often larger than train batch — no backward pass, more memory headroom
        shuffle=True,       # order doesn't matter, and not shuffling makes results reproducible
        num_workers=4,
        pin_memory=True,
        drop_last=False,     # keep the last partial batch — you want every sample evaluated
    )
    return dataloader, val_loader


@app.class_definition
class EMA:
    def __init__(self, model, decay=0.9999, burn_in=500):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        self.burn_in = burn_in
        self.step = 0

        # Initialize shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        self.step += 1

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow

                if self.step < self.burn_in:
                    # During burn-in, just copy the weights directly
                    self.shadow[name] = param.data.clone()
                    continue

                new_average = (
                    1.0 - self.decay
                ) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """Apply EMA weights to model (for sampling)"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """Restore original weights (after sampling)"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


@app.cell
def _(nn, spectral_norm, torch):
    # model = MicroET(in_channels=1, num_classes=10, depth=2)
    def sn_conv(in_c, out_c, k=3, s=1, p=1):
        return spectral_norm(nn.Conv2d(in_c, out_c, k, s, p))

    def sn_linear(in_f, out_f):
        return spectral_norm(nn.Linear(in_f, out_f))

    class ResBlock(nn.Module):
        def __init__(self, in_c, out_c, stride=1):
            super().__init__()
            self.conv1 = sn_conv(in_c, out_c, 3, stride, 1)
            self.conv2 = sn_conv(out_c, out_c, 3, 1, 1)
            self.act   = nn.SiLU()
            # 1x1 conv on the skip path when shape changes
            self.shortcut = (
                sn_conv(in_c, out_c, 1, stride, 0)
                if (in_c != out_c or stride != 1)
                else nn.Identity()
            )

        def forward(self, x):
            h = self.act(self.conv1(x))
            h = self.conv2(h)
            return self.act(h + self.shortcut(x))

    class EBM(nn.Module):
        """Small CNN energy: (B, in_channels, img_size, img_size) -> (B,) scalar energy."""
        def __init__(self, img_size=32, in_channels=1, ch=16, n_downsamples = 2):
            super().__init__()
            self.img_size = img_size
            self.in_channels = in_channels

            layers = [sn_conv(in_channels, ch, 3, 1, 1), nn.SiLU()]

            # Downsampling
            n_c = ch
            for i in range(n_downsamples):
                layers.append(ResBlock(n_c, n_c*2, stride=2))
                n_c *= 2

            # Linear output
            layers.append(nn.Flatten())

            final_size = img_size // (2 ** n_downsamples)

            layers.append(sn_linear(n_c * final_size * final_size, 1))

            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x).squeeze(-1)

    model = EBM()

    model_ema = EMA(model, decay=0.999, burn_in=500)

    model.train()
    model.to("cuda")
    # model = torch.compile(model, mode="reduce-overhead")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.0, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5000)
    return model, model_ema, optimizer, scheduler


@app.cell
def _(torch):
    def langevin_sample_train(
        model,
        x_init,
        n_steps=60,
        step_size=10.0,
        noise_std=0.005,
        grad_clip=0.01,
        clamp=True,
    ):
        model.eval()
        x = x_init.detach().clone().requires_grad_(True)
        for i in range(n_steps):
            # with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            energy = model(x).sum()
            grad = torch.autograd.grad(energy, x)[0]
            if grad_clip is not None:
                grad = grad.clamp(-grad_clip, grad_clip)  # prevent blowups

            x = x - step_size * grad + noise_std * torch.randn_like(x)
            if clamp:
                x = x.clamp(-1, 1)

            x = x.detach().requires_grad_(True)

        model.train()
        return x.detach()


    def langevin_sample_ebm(
        model,
        x_init,
        n_steps=500,
        epsilon=1e-3,
        temperatures=(1.0,),
        clamp_range=(-1, 1),              # e.g. (-1, 1); apply once at the END
        grad_clamp=(-1, 1),
    ):
        # go to a low-energy region at first 
        x = langevin_sample_train(model, x_init)

        x = x.detach().requires_grad_(True)

        grads = []

        for T in temperatures:
            for _ in range(n_steps):
                energy = model(x).sum()
                grad = torch.autograd.grad(energy, x)[0]

                if grad_clamp is not None:
                    grad = grad.clamp(*grad_clamp)

                x = x - (epsilon /  (2*T)) * grad + (epsilon)**0.5 * torch.randn_like(x)

                grads.append(grad.abs().mean().item()*epsilon/2)

                x = x.detach().requires_grad_(True)



        if clamp_range is not None:
            if clamp_range == True:
                x = x.clamp((-1, 1))
            else:
                x = x.clamp(*clamp_range)

        return x.detach(), grads

    return (langevin_sample_train,)


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
    def show_replay_buffer_samples(replay_buffer, n_samples=15**2):
        # visualize buffer images
        with torch.no_grad():
            n_samples =min(n_samples, replay_buffer.shape[0])
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
def _(model, model_ema, optimizer, scheduler, state, torch):
    if True:
        loaded_state = torch.load("checkpoints/step_59000.pt")
        model.load_state_dict(loaded_state['model_state_dict'])
        scheduler.load_state_dict(loaded_state['scheduler_state_dict'])
        optimizer.load_state_dict(loaded_state['optimizer_state_dict'])
        model_ema.shadow = loaded_state['ema_shadow']
        for key in loaded_state['state']:
            state[key] = loaded_state['state'][key]
    return


@app.cell
def _(state):
    state['step'] = 60_000
    return


@app.cell
def _(
    dataloader,
    eval_ebm,
    langevin_sample_train,
    marimo,
    model,
    model_ema,
    optimizer,
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
                        'state': state
                    }, f"checkpoints/step_{state['step']}.pt")
                    _fig, _ = show_replay_buffer_samples(replay_buffer)
                    _fig.savefig(f"checkpoints/replay_buffer_step_{state['step']}.png")

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
