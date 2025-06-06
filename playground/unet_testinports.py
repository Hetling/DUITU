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
from models.UNetKernelSize import UNetKernelSize

# Enable cuDNN benchmark
torch.backends.cudnn.benchmark = True

# Enhanced transformations
transform = transforms.Compose([
        transforms.Resize((572, 572)),   # Resize to match UNet's input size requirements
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
num_classes = len(train_dataset.class_dict)
model = UNetKernelSize(in_channels=3, num_classes=num_classes, kernel_size = 3).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


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
                masks = masks.to(device, non_blocking=True).long()

                optimizer.zero_grad(set_to_none=True)

                # Mixed precision training
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

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
                    masks = masks.to(device, non_blocking=True).long()

                    with torch.amp.autocast('cuda'):
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

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")
save_model_path = os.path.join(script_dir, 'unet_model_reduced_classes.pth')

testing = True
if testing:
    print(device)
    model.load_state_dict(torch.load(save_model_path, map_location=device))

    #predict image and display besides original mask
    for i in range(len(test_loader.dataset)):
        print("Image", i)
        print("Image shape:", test_loader.dataset[i][0].shape)
        print("Mask shape:", test_loader.dataset[i][1].shape)
        model.to(device)  # Ensure the model is on the correct device
        y_pred = model(test_loader.dataset[i][0].unsqueeze(0).to(device, dtype=torch.float))

        # Set the most likely class for each pixel
        image_tensor = test_loader.dataset[i][0]
        label_tensor = test_loader.dataset[i][1]

        model_input = image_tensor.unsqueeze(0).to(device)  # [1, 3, H, W]
        y_pred = model(model_input)

        predicted_mask = torch.argmax(y_pred, dim=1).squeeze(0).cpu().numpy()  # [H, W]
        ground_truth_mask = label_tensor.cpu().numpy()  # [H, W]

        rgb_predicted_mask = train_dataset.class_id_to_rgb(predicted_mask)
        rgb_ground_truth_mask = train_dataset.class_id_to_rgb(ground_truth_mask)

        # Print shapes for debugging
        print("Predicted mask shape:", predicted_mask.shape)
        print("RGB Predicted mask shape:", rgb_predicted_mask.shape)
        print("Ground truth mask shape:", ground_truth_mask.shape)
        print("RGB Ground truth mask shape:", rgb_ground_truth_mask.shape)
        print(len(train_dataset.class_dict))

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

    quit()

if os.path.exists(save_model_path):
    print("Loading existing trained model...")
    model.load_state_dict(torch.load(save_model_path, map_location=device))
    train(model, train_loader, val_loader, criterion, optimizer, device, epochs=10)
else:
    print("Training model...")
    train(model, train_loader, val_loader, criterion, optimizer, device, epochs=10)

save_model(model, save_model_path)
