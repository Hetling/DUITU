import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import copy
import sys
import os
from tqdm import tqdm
import numpy as np
# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '../data')
duitu_root = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.append(duitu_root)

# Imports from DUITU
from models.Unet import UNet  # or UNetKernelSize
from scripts.dataloader import get_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()


def structured_filter_prune(model, amount=0.3):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            print(f"Structured pruning {name}: {int(amount * 100)}% filters")
            prune.ln_structured(module, name="weight", amount=amount, n=2, dim=0)
            prune.remove(module, "weight")


def evaluate(model, data_loader, max_batches=5):
    model.eval()
    total_loss = 0
    count = 0
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader):
            if i >= max_batches:
                break
            inputs = inputs.to(device)
            targets = torch.argmax(targets, dim=1).to(device)  # from one-hot to class indices
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            count += 1
    return total_loss / count if count > 0 else float('inf')


if __name__ == "__main__":
    print("📦 Loading model...")
    model = UNet(in_channels=3, num_classes=32)
    model.load_state_dict(torch.load("playground/unet_model.pth", map_location=device))
    model.to(device)

    print("📊 Loading data...")
    _, val_loader, _, _= get_dataloaders(batch_size=1, datadir=data_dir)

    best_loss = float('inf')
    best_amount = 0

    for amount in np.arange(0.05, 0.4, 0.01):
        print(f"\n🔧 Evaluating {int(amount * 100)}% pruning...")
        model_copy = copy.deepcopy(model)
        structured_filter_prune(model_copy, amount=amount)
        model_copy.to(device)
        loss = evaluate(model_copy, val_loader)
        print(f"Pruning {int(amount*100)}% → Validation Loss: {loss:.4f}")
        if loss < best_loss:
            best_loss = loss
            best_amount = amount

    print(f"\n✅ Best pruning level: {int(best_amount*100)}% → Loss: {best_loss:.4f}")
