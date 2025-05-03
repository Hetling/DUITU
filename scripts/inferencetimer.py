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



def load_model(model, path):
    weight_dict = torch.load(path, map_location='cpu')
    model.load_state_dict(weight_dict)
    return model

def infer(model, input, device):
    time_start = time.time()
    y_pred = model(input.unsqueeze(0).to(device, dtype=torch.float))
    time_end = time.time()
    elapsed_time = time_end - time_start
    return y_pred, elapsed_time

def experiment(model, images_mask):
    model.eval()
    total_time = 0
    
    outputs, labels = [], []
    
    with torch.no_grad():
        for i, (input, output) in enumerate(images_mask):
        
            input = input.to(device, dtype=torch.float)
            output = output.to(device, dtype=torch.float)
            y_pred, elapsed_time = infer(model, input, device)
            total_time += elapsed_time           
            outputs.append(y_pred)
            labels.append(output)
        
    avg_time_per_sample = total_time / len(images_mask)
    





if __name__ == "__main__":
    _ , _, test_loader, class_dict = get_dataloaders()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, num_classes=32).to(device)
    model = load_model(model, os.path.join(script_dir, 'model.pth'))
    images_mask = next(iter(test_loader))
    experiment(model, images_mask)


