from dataloader import get_dataloaders
from dataloader import CustomImageDataset
import os
import torch
import torch.nn as nn


train_loader, val_loader, test_loader, class_dict = get_dataloaders()

#print stats about the dataset
print('number of classes:', len(class_dict))
print('number of training images:', len(train_loader.dataset))
print('number of validation images:', len(val_loader.dataset))
print('number of test images:', len(test_loader.dataset))