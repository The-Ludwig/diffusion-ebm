import torch

def langevin_sample_train(
    model,
    x_init,
    n_steps=60,
    step_size=10.0,
    noise_std=0.005,
    grad_clip=0.01,
    clamp=True,
    conditioning=None
):
    model.eval()
    x = x_init.detach().clone().requires_grad_(True)
    for i in range(n_steps):
        # with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        if conditioning is None:
            energy = model.energy(x).sum()
        else:
            logits = model.classify(x)[:, conditioning]
            energy = -logits.sum()
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