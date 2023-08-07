# Trianing unet for coastline detection
# Conor O'Sullivan 
# 07 Feb 2023

#Imports
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

# Variables
train_path = "../data/training/" #UPDATE
save_path = "../models/{}.pth" #UPDATE
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device('mps')  #UPDATE
batch_size = 32

# System arguments
try:
    model_name = sys.argv[1]
    sample = bool(sys.argv[2])
    incl_bands = np.array(list(sys.argv[3])).astype(int) - 1

    print("Training model: {}".format(model_name))
    print("Sample: {}".format(sample))
    print("Include bands: {}".format(incl_bands))
    print("Using device: {}\n".format(device))
except:
    model_name = "DEFAULT"
    sample = False
    incl_bands = np.array([0,1,2,3,4,5,6])



# Classes
class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, paths):
        self.paths = paths


    def __getitem__(self, idx):
        """Get image and binary mask for a given index"""

        path = self.paths[idx]
        instance = np.load(path)

        # Get spectral bands
        bands = instance[:,:,:-1]
        bands = bands[:,:,incl_bands] # Only include specified bands
        bands = bands.astype(np.float32)

        # Normalise bands
        bands = np.clip(bands*0.0000275-0.2, 0, 1)

        # Convert to tensor
        bands = bands.transpose(2,0,1)
        bands = torch.tensor(bands)

        # Get target
        mask_1 = instance[:,:,-1].astype(np.int8) # Water = 1, Land = 0
        mask_0 = 1-mask_1 

        target = np.array([mask_0,mask_1])
        target = torch.Tensor(target).squeeze()
    
        return bands, target

    def __len__(self):
        return len(self.paths)

class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.elu = nn.ELU()
    

    def forward(self, inputs):
        #Layer 1
        x = self.conv1(inputs) #convolution
        x = self.elu(x) #activation
        x = self.bn1(x) #normalisation
        
        #Layer 2
        x = self.conv2(x)
        x = self.elu(x)
        x = self.bn2(x)
        
        return x

class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = conv_block(in_c, out_c) 
        self.pool = nn.MaxPool2d((2, 2)) 
    def forward(self, inputs):
        x = self.conv(inputs) #convolutional block
        p = self.pool(x) #max pooling
        return x, p

class decoder_block(nn.Module):

    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0) 
        self.conv = conv_block(out_c+out_c, out_c) 

    def forward(self, inputs, skip):
        x = self.up(inputs) #upsampling
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x) #convolutional block
        return x

class build_unet(nn.Module):
    def __init__(self):
        super().__init__()
        """ Encoder """
        n_bands = len(incl_bands)
        self.e1 = encoder_block(n_bands, 32)
        self.e2 = encoder_block(32, 64)
        self.e3 = encoder_block(64, 128)
        self.e4 = encoder_block(128, 256)
        """ Bottleneck """
        self.b = conv_block(256, 512)
        """ Decoder """
        self.d1 = decoder_block(512, 256)
        self.d2 = decoder_block(256, 128)
        self.d3 = decoder_block(128, 64)
        self.d4 = decoder_block(64, 32)
        """ Classifier """
        self.outputs = nn.Conv2d(32, 2, kernel_size=1, padding=0)
        self.sm = nn.Softmax(dim=1)

    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        """ Bottleneck """
        b = self.b(p4)
        """ Decoder """
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)
        """ Classifier """
        outputs = self.outputs(d4)
        outputs = self.sm(outputs) 

        return outputs

# Functions
def load_data():
    """Load data from disk"""

    paths = glob.glob(train_path + "*")
    print("Training images: {}".format(len(paths)))
    
    if sample: 
        paths = paths[:1000]

    # Shuffle the paths
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

def train_model(train_loader, valid_loader,ephocs=50):
    
    # define the model
    model = build_unet()
    model.to(device)
    
    # specify loss function (binary cross-entropy)
    criterion = nn.CrossEntropyLoss()

    # specify optimizer
    optimizer = torch.optim.Adam(model.parameters())
    
    # Train the model
    min_loss = np.inf

    for epoch in range(ephocs):
        print("Epoch {} |".format(epoch+1),end=" ")
        
        model = model.train()

        for images, target in iter(train_loader):

            images = images.to(device)
            target = target.to(device)

            # Zero gradients of parameters
            optimizer.zero_grad()  

            # Execute model to get outputs
            output = model(images)
         
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
        print("| Validation Loss: {}".format(round(valid_loss,5)))
        
        if valid_loss < min_loss:
            print("Saving model")
            torch.save(model, save_path.format(model_name))

            min_loss = valid_loss

if __name__ == "__main__":

    # Load data
    train_loader, valid_loader = load_data()
   
    # Train the model
    train_model(train_loader, valid_loader)


    
