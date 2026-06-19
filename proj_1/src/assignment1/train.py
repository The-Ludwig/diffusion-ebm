import os
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm  # Import tqdm
from assignment1.data import get_dataloader
from assignment1.model import UNet
from assignment1.ema import EMA
from assignment1.sample import sample

# Hyperparameters
T = 300
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 1e-3
epochs = 50
batch_size = 128

# DDPM Schedule
beta = torch.linspace(1e-4, 0.02, T).to(device)
alpha = 1.0 - beta
alpha_cumprod = torch.cumprod(alpha, dim=0)


def forward_diffusion(x_0, t):
    noise = torch.randn_like(x_0)
    sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod[t]).view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - alpha_cumprod[t]).view(-1, 1, 1, 1)
    return sqrt_alpha_cumprod * x_0 + sqrt_one_minus_alpha_cumprod * noise, noise


def train():
    dataloader = get_dataloader(batch_size=batch_size)
    model = UNet().to(device)  # or ImprovedUNet
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Initialize EMA
    ema = EMA(model, decay=0.9999)

    train_writer = SummaryWriter("runs/ddpm_mnist/train")
    val_writer = SummaryWriter("runs/ddpm_mnist/val")
    os.makedirs("checkpoints", exist_ok=True)

    print(f"Training on {device}...")

    pbar = tqdm(range(epochs), desc="Epochs")
    for epoch in pbar:
        model.train()
        train_loss = 0

        batch_pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
        for images, _ in batch_pbar:
            images = images.to(device)
            t = torch.randint(0, T, (images.shape[0],), device=device).long()
            x_noisy, noise = forward_diffusion(images, t)

            predicted_noise = model(x_noisy, t)
            loss = F.mse_loss(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update EMA after each step
            ema.update()

            train_loss += loss.item()
            batch_pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = train_loss / len(dataloader)
        train_writer.add_scalar("Loss/epoch", avg_train_loss, epoch)

        # Validation (using regular model, not EMA)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for i, (images, _) in enumerate(dataloader):
                if i > 5:
                    break
                images = images.to(device)
                t = torch.randint(0, T, (images.shape[0],), device=device).long()
                x_noisy, noise = forward_diffusion(images, t)
                pred = model(x_noisy, t)
                val_loss += F.mse_loss(pred, noise).item()

        avg_val_loss = val_loss / 6
        val_writer.add_scalar("Loss/epoch", avg_val_loss, epoch)

        pbar.set_postfix(
            train_loss=f"{avg_train_loss:.4f}", val_loss=f"{avg_val_loss:.4f}"
        )

        # Checkpointing and Sampling with EMA weights
        if epoch % 5 == 0 or epoch == epochs - 1:
            # Save both regular and EMA checkpoints
            torch.save(
                {
                    "model": model.state_dict(),
                    "ema": ema.shadow,
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                },
                f"checkpoints/ddpm_epoch_{epoch}.pth",
            )

            tqdm.write(f"Generating samples at epoch {epoch} (using EMA)...")

            # Use EMA weights for sampling
            ema.apply_shadow()
            images = sample(model, 16, 1, 28, T, alpha, alpha_cumprod, beta, device)
            ema.restore()

            grid = torchvision.utils.make_grid(images, nrow=4)
            train_writer.add_image("Generated_Digits", grid, epoch)

    train_writer.close()
    val_writer.close()

    # Save final model with EMA
    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema.shadow,
        },
        "ddpm_mnist_final.pth",
    )


if __name__ == "__main__":
    train()
