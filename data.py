import os
import torch
from torch.utils.data import Dataset

import numpy as np
from skimage import io


# Dataset class
class CellDataset(Dataset):
    def __init__(self, df, split, transform=None):
        self.df = df[df.split == split]
        self.df = self.df.reset_index(drop=True)
        self.images_paths = self.df.path.to_list()
        self.filenames = [os.path.basename(path) for path in self.df.path.to_list()]
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Read image
        img_path = self.images_paths[index]
        img = io.imread(img_path)
        img = img.astype(np.float32)
        # Normalize
        # img /= img.max()
        # img = torch.tensor(img)
        # img = img.unsqueeze(0)
        # img = img.repeat(3, 1, 1, 1)
        # Read mask
        msk_path = img_path.replace("image", "mask")
        msk = io.imread(msk_path)
        msk = msk.astype(np.float32)
        # msk = torch.tensor(msk)
        # msk = msk.unsqueeze(0)

        if self.transform:
            data = {"image": img, "mask": msk}
            data = self.transform(**data)
            img, msk = data["image"], data["mask"]

        data = dict()
        data["image"] = img
        data["mask"] = msk
        return data
