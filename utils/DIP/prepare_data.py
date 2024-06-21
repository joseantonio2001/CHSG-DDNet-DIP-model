import os
import shutil
import sys

# Obtiene el directorio padre de la carpeta S
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

# Agrega el directorio base al sys.path
sys.path.insert(0, base_dir)


from config_deep_image_prior import (
    ORIGINAL_PNGs_PATH, TRAIN_PNGs_PATH, TRAIN_NPYs_PATH, TRAIN_NPYs_PSFs_PATH,
    PREDICTS_PNGs_PATH, PREDICTS_PSFs_MATs_PATH, PREDICTS_PSFS_PNGs_PATH, NUM_IMAGES
)

def main():
    
    # Delete data
    if os.path.exists('DEEP_IMAGE_PRIOR'):
        os.system(f"rm -r DEEP_IMAGE_PRIOR")
    
    # Step 1: Get image names
    image_names = sorted(os.listdir(ORIGINAL_PNGs_PATH.rstrip('/')))[:NUM_IMAGES]

    # Step 2: Extract names
    image_names = [name.split('_')[0] + '_' + name.split('_')[1] for name in image_names]

    # Step 3: Iterate through each image
    for name in image_names:
        # Check if all required files exist
        degraded_path = f"{TRAIN_PNGs_PATH}{name}.png"
        psf_mat_path = f"{PREDICTS_PSFs_MATs_PATH}{name}_psf.mat"
        psf_png_path = f"{PREDICTS_PSFS_PNGs_PATH}{name}_psf.png"
        numpy_path = f"{TRAIN_NPYs_PATH}{name}.npy"
        numpy_h_path = f"{TRAIN_NPYs_PSFs_PATH}{name}_H.npy"
        
        if not (os.path.exists(degraded_path) and os.path.exists(psf_mat_path) and os.path.exists(psf_png_path)
                and os.path.exists(numpy_path) and os.path.exists(numpy_h_path)):
            continue  # Skip to next image if any required file is missing

        # Create directories if they don't exist
        dirs = [
            "DEEP_IMAGE_PRIOR/TEST_DATA/ORIGINAL_IMGs/",
            f"DEEP_IMAGE_PRIOR/TEST_DATA/TRAIN_PNGs/{name}",
            f"DEEP_IMAGE_PRIOR/TEST_DATA/PREDICT_PNGs/{name}",
            f"DEEP_IMAGE_PRIOR/TEST_DATA/PREDICT_PSFs/{name}",
            f"DEEP_IMAGE_PRIOR/TEST_DATA/TRAIN_NPYs/{name}"
        ]
        
        for dir_name in dirs:
            os.makedirs(dir_name, exist_ok=True)

        # Copy files to respective directories
        shutil.copy(f"{ORIGINAL_PNGs_PATH}{name}_firls.png", f"DEEP_IMAGE_PRIOR/TEST_DATA/ORIGINAL_IMGs/{name}_firls.png")
        shutil.copy(degraded_path, f"DEEP_IMAGE_PRIOR/TEST_DATA/TRAIN_PNGs/{name}/{name}.png")
        shutil.copy(degraded_path, f"DEEP_IMAGE_PRIOR/TEST_DATA/PREDICT_PNGs/{name}/{name}.png")
        shutil.copy(psf_mat_path, f"DEEP_IMAGE_PRIOR/TEST_DATA/PREDICT_PSFs/{name}/{name}_psf.mat")
        shutil.copy(psf_png_path, f"DEEP_IMAGE_PRIOR/TEST_DATA/PREDICT_PSFs/{name}/{name}_psf.png")
        shutil.copy(numpy_path, f"DEEP_IMAGE_PRIOR/TEST_DATA/TRAIN_NPYs/{name}/{name}.npy")
        shutil.copy(numpy_h_path, f"DEEP_IMAGE_PRIOR/TEST_DATA/TRAIN_NPYs/{name}/{name}_H.npy")
        
    os.makedirs('DEEP_IMAGE_PRIOR/RESTORE_DATA/DIP/', exist_ok=True)    
    os.makedirs('DEEP_IMAGE_PRIOR/RESTORE_DATA/DIP+ES_VAR/', exist_ok=True)
    os.makedirs('DEEP_IMAGE_PRIOR/RESTORE_DATA/NO_DIP/', exist_ok=True)
    os.makedirs('DEEP_IMAGE_PRIOR/UPDATE_WEIGHTS', exist_ok=True)

    print("Done!")

if __name__ == "__main__":
    main()
