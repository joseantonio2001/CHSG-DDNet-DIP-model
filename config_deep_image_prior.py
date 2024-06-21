'''
    This file contains the configuration to test Deep Image Prior Model
'''

#Paths
ORIGINAL_PNGs_PATH     = '/data/datasets/Restoration/Dataset_CORREGIDO/groundtruth_noisy_denoised/val2017/' # Directory containing the original images in PNG format
TRAIN_PNGs_PATH     = '/data/datasets/Restoration/Dataset_CORREGIDO/degraded_noisy/val2017/'  # Directory with the degraded images used for training
TRAIN_NPYs_PATH     = '/data/datasets/Restoration/Dataset_CORREGIDO/numpy/val/'   # Directory with the numpy (.npy) files for training
TRAIN_NPYs_PSFs_PATH     = '/data/datasets/Restoration/Dataset_CORREGIDO/numpy/val/'  # Directory with the PSF (.npy) files for training
PREDICTS_PNGs_PATH     = '/data/datasets/Restoration/Dataset_CORREGIDO/degraded_noisy/val2017/'   # Directory with the degraded images to be restored   
PREDICTS_PSFs_MATs_PATH     = '/data/datasets/Restoration/Dataset_CORREGIDO/restored_noisy_denoise/val2017/psfs/' # Directory with the PSF (.mat) files for restoration
PREDICTS_PSFS_PNGs_PATH     = '/data/datasets/Restoration/Dataset_CORREGIDO/restored_noisy_denoise/val2017/psfs/' # Directory with the PSF (.png) files for restoration
W_Y_PATH         = 'model_y_weights.pt'  # Path of the weights of the y model
W_COLOR_PATH   = 'model_cbcr_weights.pt'    # Path of the weights of the cbcr model    

#Data Parameters
NUM_IMAGES = 14 # Number of images from the directory ORIGINAL_PNGs_PATH to be used in the script

#Training Parameters  
DIP_EPOCHS = 110                 # Number of epochs to train y model
DIP_ITERS_PER_EPOCH = 3000      # Number of iterations per epoch