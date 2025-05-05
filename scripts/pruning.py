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
from models.UNetKernelSize import UNetKernelSize    
from scripts.dataloader import get_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()


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
from models.UNetKernelSize import UNetKernelSize    
from scripts.dataloader import get_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()

def structured_filter_prune(model, amount=0.3):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.ln_structured(module, name="weight", amount=amount, n=2, dim=0)
            prune.remove(module, "weight")
            

def evaluate(model, sample_data):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in sample_data:
            inputs = inputs.to(device)
            
            targets = targets.to(device).long()
            targets = targets.squeeze(1)

            outputs = model(inputs)  
            loss = criterion(outputs, targets)  
            total_loss += loss.item()
    return total_loss / len(sample_data)

def find_best_pruning(model, val_samples, amounts):
    all_models = []
    all_amounts = []
    all_losses = []

    for amount in amounts:
        model_copy = copy.deepcopy(model)
        structured_filter_prune(model_copy, amount=amount)
        model_copy.to(device)

        loss = evaluate(model_copy, val_samples)

        all_models.append(model_copy)
        all_amounts.append(amount)
        all_losses.append(loss)
        print(f"Pruning amount: {amount:.2f}, Validation loss: {loss:.4f}")

    return all_amounts, all_losses, all_models

if __name__ == "__main__":
    from scripts.inferencetimer import load_model  

    model = UNetKernelSize(in_channels=3, num_classes=32, kernel_size=3).to(device)
    model = load_model(model, os.path.join(script_dir, '../playground/unet_model_reduced_classes.pth'))

    _, val_loader, _, class_dict = get_dataloaders(batch_size=1)

    val_samples = []
    for i, (x, y) in tqdm(enumerate(val_loader), total=100):
        if i == 100:
            break
        val_samples.append((x, y))

    # Pruning experiments
    all_amounts, all_losses, all_models = find_best_pruning(model, val_samples, amounts=[0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35])
    print("Best pruning amount:", all_amounts[np.argmin(all_losses)])
    print("Best validation loss:", min(all_losses))


