from train import CellDataset

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import json
import pandas as pd
import os
from glob import glob
from skimage import io
import numpy as np


def classes_percent(base_dir, split="train"):
    masks = glob(os.path.join(base_dir, split, "masks", "*.tif"))
    bg = 0.0
    cell = 0.0
    for mask in masks:
        msk = io.imread(mask)
        cell += (msk > 0).sum()
        bg += (msk == 0).sum()
        
    cell_percent = cell / (len(masks) * np.prod(msk.shape))
    bg_percent = bg / (len(masks) * np.prod(msk.shape))
    
    print(f"{split=}")
    print(f"{cell_percent=}")
    print(f"{bg_percent=}")
    
def cal_mean_std():
    # Define the transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Create the custom dataset
    base_dir = 'data/patches'
    data_df = pd.read_csv(os.path.join(base_dir, 'data.csv'))
    train_dataset = CellDataset(data_df, 'train')

    # Create a DataLoader with tqdm
    dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    # Calculate mean and std for each channel
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_samples = len(train_dataset)

    # Wrap the DataLoader with tqdm
    for data in tqdm(dataloader, total=len(dataloader)):
        img = data[0]
        mean += torch.mean(img, dim=(0, 2, 3))
        std += torch.std(img, dim=(0, 2, 3))

    mean /= total_samples
    std /= total_samples

    # Convert mean and std to Python lists for JSON serialization
    mean = mean.tolist()
    std = std.tolist()

    # Save mean and std to JSON file
    stats = {'mean': mean, 'std': std}
    with open('mean_std.json', 'w') as json_file:
        json.dump(stats, json_file)

if __name__ == "__main__":
    base_dir = os.path.join('data', 'Fluo-N3DH-SIM+')
    base_dir = os.path.join('content', 'My Drive', '3D Segmentation', 'Fluo-N3DH-SIM+')
    for s in ['train', 'test', 'val']:
        classes_percent(base_dir, s)