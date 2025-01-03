# Trianing unet for coastline detection
# Conor O'Sullivan
# 07 Feb 2023

# Imports
import numpy as np

import os
import random
import glob
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from network import U_Net, R2U_Net, AttU_Net, R2AttU_Net

def main():
    # Define the argument parser
    parser = argparse.ArgumentParser(description="Train a model on specified dataset")

    # Adding arguments
    parser.add_argument("--model_name", type=str, help="Name of the model to train")
    parser.add_argument("--sample", type=bool, default=False, help="Whether to use a sample dataset")
    parser.add_argument("--incl_bands",type=str,default="[0,1,2,3,4,5,6]",help="Bands to include, specified as a string of digits")
    parser.add_argument("--target_pos",type=int,default=7,help="Position of the target band in the dataset (0-indexed)")
    parser.add_argument("--model_type", type=str, default="U_Net", help="Type of model to train")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train for")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for training")
    parser.add_argument("--split", type=float, default=0.9, help="Train/Validation split")
    parser.add_argument("--early_stopping", type=int, default=-1, help="Number of epochs to wait before stopping training. -1 to disable.")
    parser.add_argument("--train_path",type=str,default="../../data/LICS/train/",help="Path to the training data",)
    parser.add_argument("--save_path",type=str,default="../models/",help="Path template for saving the model",)
    parser.add_argument("--device",type=str,default="cuda",choices=["cuda", "cpu", "mps"],help="Device to use for training",)
    parser.add_argument("--seed",type=int,default=42,help="Random seed for shuffling the dataset",)

    # Parse the arguments
    args = parser.parse_args()

    # Process incl_bands to be a numpy array of integers
    args.incl_bands = np.array(eval(args.incl_bands)) 

    # Set device based on argument
    args.device = torch.device(args.device)

    # Set random seed
    set_seed(args.seed)

    # Load data
    train_loader, valid_loader = load_data(args)

    # Train the model
    train_model(train_loader, valid_loader, args)

def set_seed(seed_value):
    """Set seed for reproducibility."""
    random.seed(seed_value)       # Python random module
    np.random.seed(seed_value)    # Numpy module
    torch.manual_seed(seed_value) # PyTorch
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True  # to ensure determinism
    torch.backends.cudnn.benchmark = False     # if benchmark=True, deterministic might not be guaranteed

def sense_check(args, train_data, valid_data):

    """Prints out the arguments for the model training 
    and sense checks the data before training."""

    # Print out the arguments for experiment tracking
    print("\nTraining model with the following arguments:")
    print("Training model: {}".format(args.model_name))
    train_len = len(glob.glob(args.train_path + "*"))
    print("Training data: {} images".format(train_len))
    print("Sample: {}".format(args.sample))
    print("Include bands: {}".format(args.incl_bands))
    print("Target band position: {}".format(args.target_pos))
    print("Model type: {}".format(args.model_type))
    print("Batch size: {}".format(args.batch_size))
    print("Epochs: {}".format(args.epochs))
    print("Learning rate: {}".format(args.lr))
    print("Train/Validation split: {}".format(args.split))
    print("Early stopping: {}".format(args.early_stopping))
    print("Using device: {}".format(args.device))
    print("Random seed: {}".format(args.seed))
    print()

    # Sense check the data
    paths = glob.glob(args.train_path + "*.npy")

    print("Total images: {}".format(len(paths)))
    print("Training images: {}".format(train_data.__len__()))
    print("Validation images: {}".format(valid_data.__len__()))

    # Ensure we have correct target
    train_sample = np.load(args.train_path + "LC08_L2SP_205023_20170505_20200904_02_T1_0.npy")
    print("Image shape: {}".format(train_sample.shape)) # (256, 256, 8)
    target = train_sample[:, :, args.target_pos]
    print("Avg target: {}".format(target.mean()))  #0.564727783203125

    # Check the bands and target have been loaded correctly
    bands, target = train_data.__getitem__(0)
    print("Bands shape: {}".format(bands.shape)) # (n, 256, 256)
    print("Min: {} Max: {} Avg: {}".format(bands.min(), bands.max(), bands.mean()))
    print("Target shape: {}".format(target.shape))
    print("Target unique: {}".format(torch.unique(target)))

    

# Classes
class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, paths, args):
        self.paths = paths
        self.target = args.target_pos
        self.incl_bands = args.incl_bands

    def __getitem__(self, idx):
        """Get image and binary mask for a given index"""

        path = self.paths[idx]
        instance = np.load(path)

        # Get spectral bands
        #bands = instance[:, :, :-1]
        bands = instance[:, :, self.incl_bands]  # Only include specified bands
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
def load_data(args):
    """Load data from disk"""

    paths = glob.glob(args.train_path + "*.npy")

    if args.sample:
        paths = paths[:1000]

    # Shuffle the paths
    random.shuffle(paths)

    # Create a datasets for training and validation
    split = int(0.9 * len(paths))
    train_data = TrainDataset(paths[:split],args)
    valid_data = TrainDataset(paths[split:],args)

    # Prepare data for Pytorch model
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size)

    # Data sense check
    sense_check(args, train_data, valid_data)

    return train_loader, valid_loader


def train_model(train_loader, valid_loader, args):
    # define the model
    n = len(args.incl_bands)
    if args.model_type == "U_Net":
        model = U_Net(n, 2)
    elif args.model_type == "R2U_Net":
        model = R2U_Net(n, 2)
    elif args.model_type == "AttU_Net":
        model = AttU_Net(n, 2)
    elif args.model_type == "R2AttU_Net":
        model = R2AttU_Net(n, 2)

    model.to(args.device)

    # specify loss function (binary cross-entropy)
    criterion = nn.CrossEntropyLoss()
    #sm = nn.Softmax(dim=1)

    # specify optimizer
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)

    # Train the model
    min_loss = np.inf
    epochs_no_improve = 0  # Counter for epochs with no improvement in validation loss

    for epoch in range(args.epochs):
        print("Epoch {} |".format(epoch + 1), end=" ")

        model = model.train()

        for images, target in iter(train_loader):
            images = images.to(args.device)
            target = target.to(args.device)

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
            images = images.to(args.device)
            target = target.to(args.device)

            output = model(images)

            loss = criterion(output, target) 

            valid_loss += loss.item()

        valid_loss /= len(valid_loader)
        print("| Validation Loss: {}".format(round(valid_loss, 5)))

        if valid_loss < min_loss:
            print("Saving model...")
            min_loss = valid_loss
            epochs_no_improve = 0  # Reset counter

            # Save the model
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)

            save_path = os.path.join(args.save_path, args.model_name + ".pth")

            torch.save(model.state_dict(), save_path)
        else:
            epochs_no_improve += 1
        
        # Check if early stopping is to be performed
        if args.early_stopping != -1 and epochs_no_improve >= args.early_stopping:
            print("Early stopping triggered after {} epochs with no improvement.".format(epochs_no_improve))
            break  # Break out of the loop

if __name__ == "__main__":
    # Load data
    main()
