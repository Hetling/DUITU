import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '../data')

import numpy as npz

class CustomImageDataset(Dataset):
    def __init__(self, images_dir, labels_dir, class_dict_csv, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform

        # Load class mappings
        self.class_dict = pd.read_csv(class_dict_csv)
        self.class_names = sorted(self.class_dict["name"].unique())
        self.class_to_id = {name: idx for idx, name in enumerate(self.class_names)}

        # Build RGB to class ID map
        self.class_mapping = {}
        for _, row in self.class_dict.iterrows():
            rgb = tuple(map(int, (row["r"], row["g"], row["b"])))
            self.class_mapping[rgb] = self.class_to_id[row["name"]]

        self.num_classes = len(self.class_names)
        self.image_filenames = [f for f in os.listdir(images_dir) if f.endswith(".png")]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_filename = self.image_filenames[idx]
        image_path = os.path.join(self.images_dir, image_filename)
        label_path = os.path.join(self.labels_dir, image_filename.replace(".png", "_L.png"))

        image = Image.open(image_path).convert("RGB")
        label = Image.open(label_path).convert("RGB")
        label = np.array(label)
        label = self._rgb_to_class_id(label)
        label = Image.fromarray(label)

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        label = label.squeeze(0).long()


        return image, label

    def _rgb_to_class_id(self, rgb_array):
        class_id_map = np.zeros(rgb_array.shape[:2], dtype=np.int32)
        for rgb, class_id in self.class_mapping.items():
            mask = np.all(rgb_array == rgb, axis=-1)
            class_id_map[mask] = int(class_id)
        return class_id_map

    def _class_id_to_rgb(self, class_id_array):
        rgb_array = np.zeros((*class_id_array.shape, 3), dtype=np.uint8)
        for rgb, class_id in self.class_mapping.items():
            mask = class_id_array == class_id
            rgb_array[mask] = rgb
        return rgb_array

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

def get_dataloaders(batch_size=4, pin_memory=True):

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    train_dataset = CustomImageDataset(
        images_dir=os.path.join(data_dir, 'train'),
        labels_dir=os.path.join(data_dir, 'train_labels'),
        class_dict_csv=os.path.join(data_dir, 'reduced_dict.csv'),
        transform=transform
    )

    val_dataset = CustomImageDataset(
        images_dir=os.path.join(data_dir, 'val'),
        labels_dir=os.path.join(data_dir, 'val_labels'),
        class_dict_csv=os.path.join(data_dir, 'reduced_dict.csv'),
        transform=transform
    )

    test_dataset = CustomImageDataset(
        images_dir=os.path.join(data_dir, 'test'),
        labels_dir=os.path.join(data_dir, 'test_labels'),
        class_dict_csv=os.path.join(data_dir, 'reduced_dict.csv'),
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
    
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
    show_image_and_label(x[0],y[0], 3)
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

    