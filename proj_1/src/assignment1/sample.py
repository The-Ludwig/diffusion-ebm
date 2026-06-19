import torch
from tqdm import tqdm


@torch.no_grad()
def sample(
    model, n_samples, image_channels, image_size, T, alpha, alpha_cumprod, beta, device
):
    model.eval()
    # Start from pure noise
    x = torch.randn((n_samples, image_channels, image_size, image_size), device=device)

    for i in tqdm(reversed(range(T)), total=T, desc="Sampling", leave=False):
        t = torch.full((n_samples,), i, device=device, dtype=torch.long)

        # Predict noise
        predicted_noise = model(x, t)

        # Get coefficients for the current timestep
        alpha_t = alpha[i]
        alpha_cumprod_t = alpha_cumprod[i]
        beta_t = beta[i]

        if i > 0:
            noise = torch.randn_like(x)
        else:
            noise = 0

        # DDPM Reverse Step formula
        # x_{t-1} = 1/sqrt(alpha_t) * (x_t - (1-alpha_t)/sqrt(1-alpha_cumprod_t) * eps_theta) + sigma_t * noise
        mean = (1 / torch.sqrt(alpha_t)) * (
            x - ((1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t)) * predicted_noise
        )
        std = torch.sqrt(beta_t)

        x = mean + std * noise

    model.train()
    # Rescale from [-1, 1] back to [0, 1] for visualization
    x = (x.clamp(-1, 1) + 1) / 2
    return x
