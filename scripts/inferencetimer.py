import torch
import time 
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '../data')
duitu_root = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.append(duitu_root)
from models.Unet import UNet
from scripts.dataloader import get_dataloaders
from tqdm import tqdm

#import cross_entropy_loss

from torch import nn
criterion = nn.CrossEntropyLoss()

def load_model(model, path):

    weight_dict = torch.load(path, map_location='cpu')
    model.load_state_dict(weight_dict)
    return model

def infer(model, input, device):
    time_start = time.time()
    y_pred = model(input.unsqueeze(0).to(device))

    time_end = time.time()
    elapsed_time = time_end - time_start
    return y_pred, elapsed_time

def experiment(model, test_images, quantized=False):

    model.eval()
    total_time = 0
    total_loss = 0
    outputs = []
    
    with torch.no_grad():

        for i, (input, output) in tqdm(enumerate(test_images), total=100):
            input = input.to(device)
            output = output.to(device)
            #dim is [1, 1, 3, 256, 256] i want just [3, 256, 256]
            input = input.squeeze(0)
            #dim is [1, 32, 256, 256] i want just [32, 256, 256]
            output = output.squeeze(0)
            # Ensure input has the correct dimensions for the model
            if quantized:
                model = model.to(device).half()
                input = input.to(device).half()
                output = output.to(device).half()
            print("input shape", input.shape)
            y_pred, elapsed_time = infer(model, input, device)
            total_time += elapsed_time           
            outputs.append(y_pred)
            # Ensure y_pred and output have the same dimensions
            loss = criterion(y_pred.squeeze(0), output)
            total_loss += loss.item()
    avg_time_per_sample = total_time / len(test_images)
    avg_time_per_sample = total_time / len(images_mask)

    print(f"Average time per sample: {avg_time_per_sample:.4f} seconds")
    print(f"Average loss: {total_loss / len(test_images):.4f}")
        
    avg_time_per_sample = total_time / len(images_mask)  
    





if __name__ == "__main__":
    _ , _, test_loader, class_dict = get_dataloaders(data_dir, batch_size=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, num_classes=32).to(device)
    model = load_model(model, os.path.join(script_dir, '../playground/unet_model.pth'))

    #for testing batch size 1 Get out 100 images
    images_mask = []
    for i, (input, output) in tqdm(enumerate(test_loader), total=100):
        if i == 1:
            break
        images_mask.append((input, output))
    print("Number of images:", len(images_mask))
    experiment(model, images_mask)
    print("running quantized model")
    experiment(model, images_mask, quantized=True)



