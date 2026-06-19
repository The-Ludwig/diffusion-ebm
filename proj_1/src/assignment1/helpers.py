import torch
from model import UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_for_inference(checkpoint_path):
    model = UNet().to(device)
    checkpoint = torch.load(checkpoint_path)

    # Load EMA weights directly into model
    model.load_state_dict(checkpoint["ema"])

    return model
