import torch
import torch.optim as optim
from torch import nn
import os

import torch
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
from models.UNetKernelSize import UNetKernelSize
from scripts.dataloader import get_dataloaders

# Enable cuDNN benchmark
torch.backends.cudnn.benchmark = True

# Enhanced transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Create datasets
train_dataset = CustomImageDataset(
    images_dir=os.path.join(data_dir, 'train'),
    labels_dir=os.path.join(data_dir, 'train_labels'),
    class_dict_csv=os.path.join(data_dir, 'class_dict.csv'),
    transform=transform
)

# Save/load model
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

save_model_path = os.path.join(script_dir, 'unet7Kernelsize_model_attempt2.pth')

# Optimized DataLoader config
train_loader, val_loader, test_loader, class_dict = get_dataloaders(limit=10)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNetKernelSize(in_channels=3, num_classes=32, kernel_size = 7)

# Use smaller bottleneck in UNet (modify your UNet class)
# Original: self.bottle_neck = DoubleConv(512, 1024)
# Change to: self.bottle_neck = DoubleConv(512, 512)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)



def train(model, train_loader, val_loader, criterion, optimizer, device, epochs=10):
    model.to(device)

    epoch_loop = tqdm(range(epochs), desc="🔁 Epochs", bar_format="{l_bar}{bar} | Epoch {n_fmt}/{total_fmt} | Elapsed: {elapsed} | Remaining: {remaining}")
    for epoch in epoch_loop:
        model.train()
        running_loss = 0.0

        # Training loop
        early_stopping_counter = 0
        train_loop = tqdm(train_loader, desc=f"📊 Training.     Current loss: {running_loss} ", leave=False, 
                     bar_format="{l_bar}{bar} | {n_fmt}/{total_fmt} batches | Elapsed: {elapsed} | Remaining: {remaining}")
        for images, masks in train_loop:
                
            # Faster data transfer
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)  # More efficient
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            early_stopping_counter += 1
            train_loop.set_description(f"📊 Training. Current Avg. loss: {(running_loss / early_stopping_counter):.4f}")
            predicted_mask = torch.argmax(outputs, dim=1).squeeze(0).cpu().detach()
            # print nuber of unique classes in predicted mask
            unique_classes = torch.unique(predicted_mask)
            print("Unique classes in predicted mask:", unique_classes)
           

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            early_stopping_counter = 0
            val_loop = tqdm(val_loader, desc="Validating", leave=False,
                       bar_format="{l_bar}{bar} | {n_fmt}/{total_fmt} batches | Elapsed: {elapsed} | Remaining: {remaining}")
            for images, masks in val_loop:
                # if early_stopping_counter > 5:
                #     print("Early stopping triggered.")
                #     break
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True).float()
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                early_stopping_counter += 1

                val_loop.set_description(f"Validating. Current Avg. loss: {(val_loss / early_stopping_counter):.4f}")
        save_model(model, save_model_path)
        # Print epoch results
        epoch_loop.write(f"Epoch {epoch + 1}/{epochs} - Training Loss: {running_loss / len(train_loader):.4f} - Validation Loss: {val_loss / len(val_loader):.4f}")
        

    

if os.path.exists(save_model_path):
    print("Loading existing trained model...")
    model.load_state_dict(torch.load(save_model_path))

    #predict image and display besides original mask
    for i in range(len(test_loader.dataset)):
        print("Image", i)
        print("Image shape:", test_loader.dataset[i][0].shape)
        print("Mask shape:", test_loader.dataset[i][1].shape)
        model.to(device)  # Ensure the model is on the correct device
        y_pred = model(test_loader.dataset[i][0].unsqueeze(0).to(device, dtype=torch.float))

        # Set the most likely class for each pixel
        predicted_mask = torch.argmax(y_pred, dim=1).squeeze(0).cpu().detach().numpy()

        # Convert predicted mask to RGB
        rgb_predicted_mask = train_dataset.class_id_to_rgb(predicted_mask)

        # Get the ground truth mask
        ground_truth_mask = test_loader.dataset[i][1].cpu().detach().numpy()

        rgb_ground_truth_mask = train_dataset.class_id_to_rgb(ground_truth_mask)

        # Print shapes for debugging
        print("Predicted mask shape:", predicted_mask.shape)
        print("RGB Predicted mask shape:", rgb_predicted_mask.shape)
        print("Ground truth mask shape:", ground_truth_mask.shape)
        print("RGB Ground truth mask shape:", rgb_ground_truth_mask.shape)

        # Display the images
        import matplotlib.pyplot as plt


        plt.figure(figsize=(12, 6))
        plt.subplot(1, 3, 1)
        plt.imshow(test_loader.dataset[i][0].cpu().numpy().transpose(1, 2, 0))
        plt.title("Original Image")
        plt.axis("off")
        plt.subplot(1, 3, 2)
        plt.imshow(rgb_ground_truth_mask)
        plt.title("Ground Truth Mask")
        plt.axis("off")
        plt.subplot(1, 3, 3)
        plt.imshow(rgb_predicted_mask)
        plt.title("Predicted Mask")
        plt.axis("off")
        plt.show()
else:
    print("Training model...")
    train(model, train_loader, val_loader, criterion, optimizer, device, epochs=15)
save_model(model, save_model_path)