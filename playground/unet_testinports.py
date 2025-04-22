import torch
import torch.optim as optim
from pathlib import Path
from torch import nn
import os
from PIL import Image
import numpy as np
import pandas as pd
from diffusers import AutoPipelineForText2Image
import torch
from testimports import CustomImageDataset
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import sys
from tqdm import tqdm
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '../data')
duitu_root = os.path.abspath(os.path.join(script_dir, ".."))

# Add DUITU to the Python path
sys.path.append(duitu_root)
print('importing from ', script_dir)
print('data_dir is ', data_dir)
from models.Unet import UNet
from scripts.dataloader import get_dataloaders

transform = transforms.Compose([
        transforms.Resize((256, 256)),   # Resize to a standard size (optional)
        transforms.ToTensor(),           # Convert to tensor (scale to [0, 1])
    ])

# Create datasets for train, validation, and test
train_dataset = CustomImageDataset(
    images_dir=os.path.join(data_dir, 'train'),
    labels_dir=os.path.join(data_dir, 'train_labels'),
    class_dict_csv=os.path.join(data_dir, 'class_dict.csv'),
    transform=transform
)

val_dataset = CustomImageDataset(
    images_dir=os.path.join(data_dir, 'val'),
    labels_dir=os.path.join(data_dir, 'val_labels'),
    class_dict_csv=os.path.join(data_dir, 'class_dict.csv'),
    transform=transform
)

test_dataset = CustomImageDataset(
    images_dir=os.path.join(data_dir, 'test'),
    labels_dir=os.path.join(data_dir, 'test_labels'),
    class_dict_csv=os.path.join(data_dir, 'class_dict.csv'),
    transform=transform
)

train_loader, val_loader, test_loader, class_dict = get_dataloaders()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=3, num_classes=32)

criterion = nn.BCEWithLogitsLoss() # suitable for multi-class segmentation
optimizer = optim.Adam(model.parameters(), lr=1e-3)



# AI SLOP

def train(model, train_loader, val_loader, criterion, optimizer, device, epochs=10):
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        print(f"\n🔁 Epoch {epoch + 1}/{epochs}")

        # Training loop
        for images, masks in tqdm(train_loader, desc="Training"):
            images = images.to(device)
            masks = masks.to(device).float() 
            print(f"images shape: {images.shape}")
            print(f"masks shape: {masks.shape}")

            optimizer.zero_grad()
            outputs = model(images)  # [B, C, H, W]
            loss = criterion(outputs, masks)  # [B, C, H, W]
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        print(f"📊 Average Training Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc="Validating"):
                images = images.to(device)
                masks = masks.to(device).float()
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"🧪 Average Validation Loss: {avg_val_loss:.4f}")
        
# save model
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")
    
save_model_path = os.path.join(script_dir, 'unet_model.pth')
if os.path.exists(save_model_path):
    print("Loading existing trained model...")
    model.load_state_dict(torch.load(save_model_path, map_location=device))
else:
    print("Training model...")
    # Train the model
    train(model, train_loader, val_loader, criterion, optimizer, device, epochs=10)
save_model(model, save_model_path)

