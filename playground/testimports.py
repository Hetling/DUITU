import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np


script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '../data')

print('importing from ', script_dir)
print('data_dir is ', data_dir)
class CustomImageDataset(Dataset):
    def __init__(self, images_dir, labels_dir, class_dict_csv, transform=None):
        """
        Args:
            images_dir (str): Directory with all the color images (dashcam footage).
            labels_dir (str): Directory with all the label images (segmentation masks).
            class_dict_csv (str): Path to the CSV file containing class mappings (RGB -> class_id).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform

        # Load class mappings from CSV
        self.class_dict = pd.read_csv(class_dict_csv)
        self.class_mapping = {tuple(map(int, row[["r", "g", "b"]])): idx for idx, row in self.class_dict.iterrows()}
        
        # Get all image filenames (assuming they are png files)
        self.image_filenames = [f for f in os.listdir(images_dir) if f.endswith('.png')]
    
    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Get the image filename and label filename
        image_filename = self.image_filenames[idx]
        image_path = os.path.join(self.images_dir, image_filename)
        label_path = os.path.join(self.labels_dir, image_filename.replace('.png', '_L.png'))

        # Open the image and label image
        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path).convert('RGB')  # RGB label image

        # Convert the label image to class IDs
        label = np.array(label)  # Convert label to numpy array
        label = self._rgb_to_class_id(label)

        # Convert the label back to a PIL image (needed for transforms)
        label = Image.fromarray(label)

        # Apply the transform (resize, ToTensor, etc.) if specified
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        return image, label

    def _rgb_to_class_id(self, rgb_array):
        """
        Convert RGB values in the label image to class IDs.
        Assumes RGB is mapped to class IDs in the class_dict.
        """
        class_id_map = np.zeros(rgb_array.shape[:2], dtype=np.uint8)  # Create an empty array for class IDs

        # Iterate through each RGB value and assign the corresponding class ID
        for rgb, class_id in self.class_mapping.items():
            mask = np.all(rgb_array == rgb, axis=-1)  # Find pixels matching the RGB value
            class_id_map[mask] = class_id  # Assign the corresponding class ID

        return class_id_map


# Define the transformations
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

# Create DataLoader objects
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)



#show the first image and label
import matplotlib.pyplot as plt
def show_image_and_label(image, label):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image.permute(1, 2, 0))  # Convert from CxHxW to HxWxC
    plt.title('Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(label.permute(1, 2, 0))  # Convert from CxHxW to HxWxC for RGB
    plt.title('Label')
    plt.axis('off')

    plt.show()



show_image_and_label(*next(iter(train_loader)))