import segmentation_models_pytorch as smp
import torch
import os
import cv2  
from tqdm import tqdm
import numpy as np

MODEL_PATH = "unet_model.pth"   

path_data = f"data/"
train_path = f"{path_data}train/"
train_labels = f"{path_data}train_labels/"


def all_images(top_path, label: bool, size=(512, 512)):
    paths = os.listdir(top_path)
    images = []
    
    for path in tqdm(paths):
        full_path = os.path.join(top_path, path)
        image = cv2.imread(full_path)

        if image is None:
            print(f"Image {full_path} is empty or unreadable.")
            continue

        # Resize image/mask to fixed size
        if label:
            image = cv2.resize(image, size, interpolation=cv2.INTER_NEAREST)  # preserve label indices
            # Optional: convert RGB label mask to class indices
            # If your labels are already grayscale masks with class IDs per pixel, keep this:
            if image.ndim == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image_tensor = torch.from_numpy(image).long()  # [H, W]
        else:
            image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0  # [C, H, W]

        images.append(image_tensor)

    return images


images = all_images(train_path, label=False)
labels = all_images(train_labels, label=True)



def generate_colormap_from_labels(label_dir):
    unique_colors = set()

    for fname in tqdm(os.listdir(label_dir)):
        path = os.path.join(label_dir, fname)
        label = cv2.imread(path)
        if label is None:
            continue
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
        pixels = label.reshape(-1, 3)
        for color in np.unique(pixels, axis=0):
            unique_colors.add(tuple(color))

    sorted_colors = sorted(list(unique_colors))
    if len(sorted_colors) > 32:
        print(f"⚠️ Found more than 32 colors ({len(sorted_colors)}). You may need to review your masks.")
    print("Len of sorted colors: ", len(sorted_colors))
    color_to_class = {color: idx for idx, color in enumerate(sorted_colors)}
    return color_to_class

color_map = generate_colormap_from_labels(train_labels)
print(len(color_map))

model = smp.Unet(
    encoder_name="resnet34",        
    encoder_weights="imagenet",     
    in_channels=3,                  
    classes=32,        
    
)

def train(model, images, labels, epochs=10, batch_size=16):
    # set model to training mode
    model.train()
    
    # define loss function and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        for i in range(0, len(images), batch_size):
            # get batch of images and labels
            img_batch = images[i:i+batch_size]
            label_batch = labels[i:i+batch_size]
            
            # convert to tensors
            img_batch = torch.stack(img_batch).float()
            label_batch = torch.stack(label_batch).long()
            
            # forward pass
            outputs = model(img_batch)
            
            # calculate loss
            loss = loss_fn(outputs, label_batch)
            
            # backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")



if os.path.exists(MODEL_PATH):
    print("✅ Loading existing trained model...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
else:
    print("🚀 Training model...")
    train(model, images, labels, epochs=10, batch_size=16)
    torch.save(model.state_dict(), MODEL_PATH)
    print("💾 Model saved to disk.")

# Load validation images
val_images = all_images(f"{path_data}val/", label=False)
val_labels = all_images(f"{path_data}val_labels/", label=True)

# Show prediction

def decode_segmap(label, class_to_color):
    h, w = label.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for class_idx, color in class_to_color.items():
        rgb[label == class_idx] = color
    return rgb


def show_prediction(model, image, label, class_to_color):
    model.eval()
    
    # Ensure image is a normalized tensor [C, H, W]
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    image = image.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(image)

    pred_label = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    # Convert image back to [H, W, C] for cv2.imshow
    img_np = image.squeeze().permute(1, 2, 0).cpu().numpy()
    label_np = label.cpu().numpy() if torch.is_tensor(label) else label

    # Normalize masks for display (0–255 range)
    pred_vis = decode_segmap(pred_label, class_to_color)
    label_vis = decode_segmap(label_np, class_to_color)


    cv2.imshow("Input Image", (img_np * 255).astype(np.uint8))
    cv2.imshow("Predicted Mask", pred_vis)
    cv2.imshow("True Mask", label_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# View a sample prediction
show_prediction(model, val_images[0], val_labels[0], color_map)