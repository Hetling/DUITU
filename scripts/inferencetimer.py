import torch
import time 
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '../data')
duitu_root = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.append(duitu_root)
from models.Unet import UNet
from models.UNetKernelSize import UNetKernelSize
from scripts.dataloader import get_dataloaders
from tqdm import tqdm
from scipy import stats
from scripts.pruning import find_best_pruning
import numpy as np

#import cross_entropy_loss

from torch import nn
criterion = nn.CrossEntropyLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    
    timers = []
    with torch.no_grad():

        for i, (input, output) in tqdm(enumerate(test_images), total=100):
            input = input.to(device)
            output = output.to(device)
            #dim is [1, 1, 3, 256, 256] i want just [3, 256, 256]
            input = input.squeeze(0)
            #dim is [1, 32, 256, 256] i want just [32, 256, 256]
            output = output.long()
            if output.ndim == 4 and output.shape[1] == 1:
                output = output.squeeze(1) 
            # Ensure input has the correct dimensions for the model
            if quantized:
                model = model.to(device).half()
                input = input.to(device).half()
                output = output.to(device).half()
            y_pred, elapsed_time = infer(model, input, device)
            total_time += elapsed_time           
            outputs.append(y_pred)
            timers.append(elapsed_time)
            # Ensure y_pred and output have the same dimensions
            loss = criterion(y_pred, output)
            total_loss += loss.item()
    avg_time_per_sample = total_time / len(test_images)
    avg_time_per_sample = total_time / len(test_images)

    print(f"Average time per sample: {avg_time_per_sample:.4f} seconds")
    print(f"Average loss: {total_loss / len(test_images):.4f}")
        
    avg_time_per_sample = total_time / len(test_images) 
 
    return avg_time_per_sample, total_loss / len(test_images), timers


def t_test(timer1, timer2):
    # Perform a t-test to compare the two sets of timers

    t_stat, p_value = stats.ttest_ind(timer1, timer2)
    print(f"T-statistic: {t_stat}, P-value: {p_value}")
    if p_value < 0.05:
        print("The difference is statistically significant.")
    else:
        print("The difference is not statistically significant.")


if __name__ == "__main__":
    _, _, test_loader, class_dict = get_dataloaders(batch_size=1)
    print("Number of classes:", len(class_dict))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetKernelSize(in_channels=3, num_classes=32, kernel_size=3).to(device)  
    model = load_model(model, os.path.join(script_dir, '../playground/unet_model_reduced_classes.pth'))

    images_mask = []
    for i, (input, output) in tqdm(enumerate(test_loader), total=100):
        if i == 100:
            break
        images_mask.append((input, output))

    print("Number of images:", len(images_mask))
    experiment(model, images_mask)
    pruning_amounts = [0.01,0.025,0.05,0.07,0.1,0.125,0.15,0.175,0.2,0.225,0.25,0.275,0.3,0.325,0.35,0.375,0.4,0.425,0.45,0.475,0.5]
    all_amounts, all_losses, all_models = find_best_pruning(model, images_mask, pruning_amounts)

    best_index = np.argmin(all_losses)
    best_model = all_models[best_index]
    best_amount = all_amounts[best_index]
    best_loss = all_losses[best_index]
    
    # save all_models all_amounts and all_losses in a pickle file
    import pickle
    with open(os.path.join(script_dir, 'pruning_results.pkl'), 'wb') as f:
        pickle.dump((all_models, all_amounts, all_losses), f)
        

    print(f"\n✅ Best pruning amount: {best_amount:.2f}")
    print(f"📉 Best validation loss: {best_loss:.4f}")

    # Run experiment on best pruned model
    print("\n🔍 Running experiment on best pruned model...")
    experiment(best_model, images_mask)