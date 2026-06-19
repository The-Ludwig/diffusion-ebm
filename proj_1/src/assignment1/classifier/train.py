import torch
import torch.optim as optim
import torch.nn as nn
from assignment1.data import get_dataloader
from assignment1.classifier.model import ResNetMini


def train():
    # Hyperparameters
    epochs = 15
    lr = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize
    model = ResNetMini().to(device)
    train_loader = get_dataloader(batch_size=128)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print(f"Starting training on {device}...")

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}"
                )
                running_loss = 0.0

    save_path = "checkpoints/classifier_mnist_resnet.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved successfully to {save_path}")


if __name__ == "__main__":
    train()
