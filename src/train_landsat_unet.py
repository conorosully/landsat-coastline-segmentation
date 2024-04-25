# Trianing unet for coastline detection
# Conor O'Sullivan
# 07 Feb 2023

# Imports
import numpy as np
import pandas as pd
import sys
import random
import glob

import cv2 as cv
from PIL import Image
from osgeo import gdal

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

from network import U_Net, R2U_Net, AttU_Net, R2AttU_Net

# Variables
train_path = "../data/training/"  # UPDATE
save_path = "../models/{}.pth"  # UPDATE
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("mps")  # UPDATE
batch_size = 32
early_stopping = 10

# System arguments
try:
    model_name = sys.argv[1]
    sample = sys.argv[2] == "True"
    incl_bands = np.array(list(sys.argv[3])).astype(int) - 1
    model_type = sys.argv[4]

    print("Training model: {}".format(model_name))
    print("Sample: {}".format(sample))
    print("Include bands: {}".format(incl_bands))
    print("Model type: {}".format(model_type))
    print("Using device: {}\n".format(device))
except:
    model_name = "DEFAULT"
    sample = False
    incl_bands = np.array([0, 1, 2, 3, 4, 5, 6])
    model_type = "U_Net"


# Classes
class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, paths, target=-1):
        self.paths = paths
        self.target = target

    def __getitem__(self, idx):
        """Get image and binary mask for a given index"""

        path = self.paths[idx]
        instance = np.load(path)

        # Get spectral bands
        bands = instance[:, :, :-1]
        bands = bands[:, :, incl_bands]  # Only include specified bands
        bands = bands.astype(np.float32)

        # Normalise bands
        bands = np.clip(bands * 0.0000275 - 0.2, 0, 1)

        # Convert to tensor
        bands = bands.transpose(2, 0, 1)
        bands = torch.tensor(bands)

        # Get target
        mask_1 = instance[:, :, self.target].astype(np.int8)  # Water = 1, Land = 0
        mask_0 = 1 - mask_1

        target = np.array([mask_0, mask_1])
        target = torch.Tensor(target).squeeze()

        return bands, target

    def __len__(self):
        return len(self.paths)


# Functions
def load_data():
    """Load data from disk"""

    paths = glob.glob(train_path + "*")
    print("Training images: {}".format(len(paths)))

    if sample:
        paths = paths[:1000]

    # Shuffle the paths
    random.seed(42)
    random.shuffle(paths)

    # Create a datasets for training and validation
    split = int(0.9 * len(paths))
    train_data = TrainDataset(paths[:split])
    valid_data = TrainDataset(paths[split:])

    # Prepare data for Pytorch model
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size)

    print("Training images: {}".format(train_data.__len__()))
    print("Validation images: {}".format(valid_data.__len__()))

    #
    bands, target = train_data.__getitem__(0)

    print("Bands shape: {}".format(bands.shape))
    print("Min: {} Max: {} Avg: {}".format(bands.min(), bands.max(), bands.mean()))
    print("Target shape: {}".format(target.shape))
    print("Target unique: {}".format(torch.unique(target)))

    return train_loader, valid_loader


def train_model(train_loader, valid_loader, ephocs=50):
    # define the model
    if model_type == "U_Net":
        model = U_Net(len(incl_bands), 2)
    elif model_type == "R2U_Net":
        model = R2U_Net(len(incl_bands), 2)
    elif model_type == "AttU_Net":
        model = AttU_Net(len(incl_bands), 2)
    elif model_type == "R2AttU_Net":
        model = R2AttU_Net(len(incl_bands), 2)

    model.to(device)

    # specify loss function (binary cross-entropy)
    criterion = nn.CrossEntropyLoss()
    #sm = nn.Softmax(dim=1)

    # specify optimizer
    optimizer = torch.optim.Adam(model.parameters())

    # Train the model
    min_loss = np.inf
    epochs_no_improve = 0  # Counter for epochs with no improvement in validation loss


    for epoch in range(ephocs):
        print("Epoch {} |".format(epoch + 1), end=" ")

        model = model.train()

        for images, target in iter(train_loader):
            images = images.to(device)
            target = target.to(device)

            # Zero gradients of parameters
            optimizer.zero_grad()

            # Execute model to get outputs
            output = model(images)
            #output = sm(output)

            # Calculate loss
            loss = criterion(output, target)

            # Run backpropogation to accumulate gradients
            loss.backward()

            # Update model parameters
            optimizer.step()

        # Calculate validation loss
        model = model.eval()

        valid_loss = 0
        for images, target in iter(valid_loader):
            images = images.to(device)
            target = target.to(device)

            output = model(images)

            loss = criterion(output, target) 

            valid_loss += loss.item()

        valid_loss /= len(valid_loader)
        print("| Validation Loss: {}".format(round(valid_loss, 5)))

        if valid_loss < min_loss:
            print("Saving model")
            torch.save(model, save_path.format(model_name))

            min_loss = valid_loss
            epochs_no_improve = 0  # Reset counter
        else:
            epochs_no_improve += 1
        
        if early_stopping != -1 and epochs_no_improve >= early_stopping:
            print("Early stopping triggered after {} epochs with no improvement.".format(epochs_no_improve))
            break  # Break out of the loop


if __name__ == "__main__":
    # Load data
    train_loader, valid_loader = load_data()

    # Train the model
    train_model(train_loader, valid_loader)
