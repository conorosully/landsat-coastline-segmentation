# landsat-coastline-segmentation
Segmenting satellite images of Ireland's coastline into land and ocean pixels.

This repository contains the code required to reproduce the results in the conference paper:

> To update

This code is only for academic and research purposes. Please cite the above paper if you intend to use whole/part of the code. 

## Data Files

We have used the following dataset in our analysis: 

1. Sentinel-2 Water Edges Dataset (SWED) found [here](https://zenodo.org/records/8414665).

 The data is available under the Creative Commons Attribution 4.0 International license.

## Code Files
You can find the following files in the src folder:

- `1_selecting_scenes.ipynb` Obtain the metadata for all potential Landsat scenes, select 100 scenes for model development and download the Landsat Collection 2 Level-2 Science Products.
- `2_processing_data.ipynb` This file is used: (1) Create RGB images from Landsat scenes used to produce rough annotations with Label Studio and (2) Create npy file for each scene that includes the necessary spectral bands and rough annotation.
- `3_model_data.ipynb` Crop 30,000 training images and 100 test images for the modelling dataset. This file is used in combination with 3_label_studio.ipynb. 
- `3_label_studio.ipynb` Used to help create the precise test annotations using Label Studio.
- `4_model_results.ipynb` Get predictions on the test set from various segmentation approaches --- NDWI, XGBoost and U-NET.
- `5_model_evaluation.ipynb`Produce metrics and visualisations for the performance of all segmentation approaches.
- `utils.py` Helper functions used to perform the analysis. 
- `evaluation.py` Help functions used to evaluate the segmentation approaches.
- `network.py` Deep learning model code.
- `train_landsat_unet.py` Used to train deep learning models.
