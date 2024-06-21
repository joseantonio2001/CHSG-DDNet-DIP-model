# Introduction
This repository is an extension of the original repository [CHSG-DDNET](https://github.com/vipgugr/CHSG-DDNet) where methods for restoring and enhancing images using DIP (Deep Image Prior) are included.

# Requirements
* Python >= 3.8
* Numpy
* Scikit-image
* Pytorch == 1.7.1
* pytorch_msssim
* Scipy


# Installation
```console
conda create -n ddnet python=3.8 scipy scikit-image
conda activate ddnet
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
pip install pytorch_msssim
```

# Usage
First, activate conda environment: 
```console 
conda activate ddnet
```

## NO_DIP model

### Data preparation training
Download dataset from this [link](https://drive.google.com/drive/folders/109VwKx-GI_MbqIdAfGp6WJ0ekpA1N9_T?usp=sharing) and extract it. Modify `TRAIN_FILE_PATH` and `EVAL_FILE_PATH` in config.py to the paths of the train data folder and the validation data folder.

### Train
Our systems uses two models:
1. Luminance model or "y model". Script: train_y.py
2. Color model or "cbcr model".  Script: train_cbcr.py

Before training, modify both `W_PATH_SAVE` and `W_COLOR_PATH_SAVE` to the paths of the folders where you want to save the weights of both models.

```console
python train_y.py
python train_cbcr.py
```

### Val/Test
To process a folder and generate the restores images of its contents, use predict.py.
You will need to download our weights from [here](https://drive.google.com/drive/folders/109VwKx-GI_MbqIdAfGp6WJ0ekpA1N9_T?usp=sharing) or provide your own using the training scripts.

```console
python predict.py <image_path> <psf_path> <output_path> <model_y_weights_path> <model_cbcr_weights_path>
#python predict.py data/degraded_noisy/test2017reduced data/restored_noisy_denoise/test2017reduced/psfs/ out model_y_weights.pt model_cbcr_weights.pt
```
* <image_path>:  Path to folder containing the blur images. They have to be in png or jpg formats.
* <psf_path>:    Path to folder containing the estimated PSFs. They must be saved as a matrix in a npy file. Each one must have the same name as its corresponding blur image with the subfix `_psf.npy` added.
* <output_path>: Path where the restored images would be saved.
* <model_y_weights_path>: Path to the weights of model y.
* <model_cbcr_weights_path>: Path to the weights of model cbcr.

## DIP model

### Data preparation training
Modify the paths of the `config_deep_image_prior.py` file with the paths of the directories where the working data resides (`.mat`, `.npy`, `.png`).

**Description of the Paths**
- `ORIGINAL_PNGs_PATH`: Directory containing the original images in PNG format.
- `TRAIN_PNGs_PATH`: Directory with the degraded images used for training.
- `TRAIN_NPYs_PATH`: Directory with the numpy (.npy) files for training.
- `TRAIN_NPYs_PSFs_PATH`: Directory with the PSF (.npy) files for training.
- `PREDICTS_PNGs_PATH`: Directory with the degraded images to be restored.
- `PREDICTS_PSFs_MATs_PATH`: Directory with the PSF (.mat) files for restoration.
- `PREDICTS_PSFS_PNGs_PATH`: Directory with the PSF (.png) files for restoration.
- `W_Y_PATH`: Path to the weights of the y model.
- `W_COLOR_PATH`: Path to the weights of the cbcr model.

### Train and test
To use the model, just run the `deep_image_prior.py` script. Once the execution is finished, a directory `DEEP_IMAGE_PRIOR` will be created, where you will be able to see:
1. The restored images with the base model in the `DEEP_IMAGE_PRIOR/RESTORE_DATA/NO_DIP/` directory.
2. The images restored with the Deep Image Prior model in the directory `DEEP_IMAGE_PRIOR/RESTORE_DATA/DIP/`.
3. The calculated metrics in the file `DEEP_IMAGE_PRIOR/METRICS.csv`.
4. The updated weights used to restore images using Deep Image Prior in the `DEEP_IMAGE_PRIOR/UPDATE_WEIGHTS` directory.
5. All data used to train and restore organised in the `DEEP_IMAGE_PRIOR/TEST_DATA` directory.

