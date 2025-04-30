import torch
import torch.optim as optim
from torch import nn
import os
from testimports import CustomImageDataset
from torchvision import transforms
import sys
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '../data')
duitu_root = os.path.abspath(os.path.join(script_dir, ".."))

# Add DUITU to the Python path
sys.path.append(duitu_root)
from models.Unet import UNet
from scripts.dataloader import get_dataloaders

# Enable cuDNN benchmark
torch.backends.cudnn.benchmark = True

# Enhanced transformations
transform = transforms.Compose([
        transforms.Resize((256, 256)),   # Resize to a standard size (optional)
        transforms.ToTensor(),           # Convert to tensor (scale to [0, 1])
    ])
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")
    
# Create datasets for train, validation, and test
train_dataset = CustomImageDataset(
    images_dir=os.path.join(data_dir, 'train'),
    labels_dir=os.path.join(data_dir, 'train_labels'),
    class_dict_csv=os.path.join(data_dir, 'class_dict.csv'),
    transform=transform
)

# Optimized DataLoader config
train_loader, val_loader, test_loader, class_dict = get_dataloaders(pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=3, num_classes=32).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scaler = torch.cuda.amp.GradScaler()  # For mixed precision training

def train(model, train_loader, val_loader, criterion, optimizer, device, epochs=10):
    model.to(device)

    epoch_loop = tqdm(range(epochs), desc="🔁 Epochs", bar_format="{l_bar}{bar} | Epoch {n_fmt}/{total_fmt} | Elapsed: {elapsed} | Remaining: {remaining}")
    for epoch in epoch_loop:
        try:
            model.train()
            running_loss = 0.0

            train_loop = tqdm(train_loader, desc=f"📊 Training. Current loss: {running_loss} ", leave=False, 
                              bar_format="{l_bar}{bar} | {n_fmt}/{total_fmt} batches | Elapsed: {elapsed} | Remaining: {remaining}")
            for images, masks in train_loop:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True).float()

                optimizer.zero_grad(set_to_none=True)

                # Mixed precision training
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, masks)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()
                train_loop.set_description(f"📊 Training. Current Avg. loss: {(running_loss / len(train_loader)):.4f}")

            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                val_loop = tqdm(val_loader, desc="Validating", leave=False,
                                bar_format="{l_bar}{bar} | {n_fmt}/{total_fmt} batches | Elapsed: {elapsed} | Remaining: {remaining}")
                for images, masks in val_loop:
                    images = images.to(device, non_blocking=True)
                    masks = masks.to(device, non_blocking=True).float()

                    with torch.cuda.amp.autocast():
                        outputs = model(images)
                        loss = criterion(outputs, masks)

                    val_loss += loss.item()
                    val_loop.set_description(f"Validating. Current Avg. loss: {(val_loss / len(val_loader)):.4f}")

            epoch_loop.write(f"Epoch {epoch + 1}/{epochs} - Training Loss: {running_loss / len(train_loader):.4f} - Validation Loss: {val_loss / len(val_loader):.4f}")
        except KeyboardInterrupt:
            print("Training interrupted. Saving model...")
            save_model(model, os.path.join(script_dir, 'unet_model.pth'))
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            break
        finally:
            save_model(model, os.path.join(script_dir, 'unet_model.pth'))
            print(f"Model saved after epoch {epoch + 1}")

# Save/load model
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

save_model_path = os.path.join(script_dir, 'unet_model.pth')
if os.path.exists(save_model_path):
    print("Loading existing trained model...")
    model.load_state_dict(torch.load(save_model_path))
    train(model, train_loader, val_loader, criterion, optimizer, device, epochs=10)
else:
    print("Training model...")
    train(model, train_loader, val_loader, criterion, optimizer, device, epochs=10)

save_model(model, save_model_path)
