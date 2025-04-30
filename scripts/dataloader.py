import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '../data')

import numpy as np

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
            
        label = label.squeeze(0).long()  # shape: [H, W]
        label = F.one_hot(label, num_classes=len(self.class_mapping.keys()))  # shape: [H, W, num_classes]
        label = label.permute(2, 0, 1).float()  # shape: [num_classes, H, W]
        
        
        
        return image, label

    def _rgb_to_class_id(self, rgb_array):
        """
        Convert RGB v alues in the label image to class IDs.
        Assumes RGB is mapped to class IDs in the class_dict.
        """
        class_id_map = np.zeros(rgb_array.shape[:2], dtype=np.int32)  # Use int32 for whole integers

        # Iterate through each RGB value and assign the corresponding class ID
        for rgb, class_id in self.class_mapping.items():
            mask = np.all(rgb_array == rgb, axis=-1)  # Find pixels matching the RGB value
            class_id_map[mask] = int(class_id)  # Ensure class_id is a whole integer

        return class_id_map
    
    def _class_id_to_rgb(self, class_id_array):
        """
        Convert class IDs back to RGB values.
        This is optional and can be used for visualization.
        """
        rgb_array = np.zeros((*class_id_array.shape, 3), dtype=np.uint8)  # Create an empty array for RGB values
        for rgb, class_id in self.class_mapping.items():
            mask = class_id_array == class_id  # Find pixels matching the class ID
            rgb_array[mask] = rgb  # Assign the corresponding RGB value

        return rgb_array

def get_dataloaders(batch_size=4, limit=None):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
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
    if limit:
        train_dataset.image_filenames = train_dataset.image_filenames[:limit]
        val_dataset.image_filenames = val_dataset.image_filenames[:limit]
        test_dataset.image_filenames = test_dataset.image_filenames[:limit]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, train_dataset.class_dict


if __name__ == "__main__":
    from visualisations import show_image_and_label 
    print('Loading datasets...')
    train_loader, val_loader, test_loader, class_dict = get_dataloaders()
    print('Datasets loaded.')
    print('Number of training samples:', len(train_loader.dataset))
    print('Number of validation samples:', len(val_loader.dataset))
    print('Number of test samples:', len(test_loader.dataset))
    print('Class dictionary:')
    print(class_dict)

    # Show an example image and its label
    print('Showing an example image and its label...')
    x,y = next(iter(train_loader))
    show_image_and_label(x[0],y[0], 17)
    #id of left most corner pixel
    print('left most corner pixel id:', y[0][0][0][0])
    #print the name of the class
    class_id = torch.argmax(y[0, :, 0, 0]).item()

    # Look up the class name
    class_name = class_dict.iloc[class_id]['name']

    print('class name:', class_name)
    print('Label image shape:', y.shape)  # Should print: torch.Size([32, 32, 256, 256]) (batch_size, num_classes, height, width)
    print('Unique class IDs in the label:', torch.unique(y[0]))  # Unique class IDs in the first imageuv
    print(y[0])  # One-hot encoded tensor

    