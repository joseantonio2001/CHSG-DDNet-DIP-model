import os
from config_deep_image_prior import W_Y_PATH, W_COLOR_PATH

# Preparar directorios que almacenan los datos de entrenamiento y donde se guardaran los resultados 
#prepare_dirs = 'python prepare_data.py'
prepare_dirs = 'python utils/DIP/prepare_data.py'
os.system(prepare_dirs)

# Ruta de la carpeta que contiene las imágenes
dir_ORIGINAL_IMGs = "DEEP_IMAGE_PRIOR/TEST_DATA/ORIGINAL_IMGs/"

# Obtener la lista de archivos en la carpeta
archivos = os.listdir(dir_ORIGINAL_IMGs)

# Extraer los nombres de las imágenes
image_names = [name.split('_')[0] + '_' + name.split('_')[1] for name in archivos if name.endswith(".png")]

# Eliminar el archivo 'METRIC.csv' si existe y lo genera de nuevo para borrar datos anteriores
metric_file = os.path.join('DEEP_IMAGE_PRIOR/', 'METRICS.csv')
if os.path.exists(metric_file):
    os.remove(metric_file)
#command_gen_metrics = f"python gen_metrics_doc.py"
command_gen_metrics = f"python utils/DIP/gen_metrics_doc.py"
os.system(command_gen_metrics)

# Iterar sobre cada nombre de imagen
for nombre in image_names:
    print("")
    print("################# IMAGE " + nombre + ".png #####################")
    print("** RE-TRAIN LUMINANCE MODEL")
    
    # Construir las rutas de los archivos necesarios para ejecutar el script
    npy_file = f"DEEP_IMAGE_PRIOR/TEST_DATA/TRAIN_NPYs/{nombre}/"
    weights_file = f"DEEP_IMAGE_PRIOR/UPDATE_WEIGHTS/{nombre}"
    weights_file_var = f"DEEP_IMAGE_PRIOR/UPDATE_WEIGHTS/{nombre}+ES_VAR"
    
    # Ejecutar el script deep_image_prior_train_y.py con los argumentos requeridos
    command_train = f"python deep_image_prior_train_y.py {npy_file} {weights_file}"
    os.system(command_train)
    # Ejecutar el script deep_image_prior_train_y_ES_VAR.py con los argumentos requeridos
    command_train_ES = f"python deep_image_prior_train_y_ES_VAR.py {npy_file} {weights_file_var}"
    os.system(command_train_ES)
    
    
    print("** PREDICTING WITH RESTORED DIP MODEL")
    # Ejecutar el programa predict.py con los argumentos requeridos para la restauración DIP
    command_dip = f"python predict.py DEEP_IMAGE_PRIOR/TEST_DATA/PREDICT_PNGs/{nombre}/ DEEP_IMAGE_PRIOR/TEST_DATA/PREDICT_PSFs/{nombre}/ DEEP_IMAGE_PRIOR/RESTORE_DATA/DIP/ DEEP_IMAGE_PRIOR/UPDATE_WEIGHTS/{nombre}_updated_model_y_weights.pt {W_COLOR_PATH}"
    os.system(command_dip)
    
    print("** PREDICTING WITH RESTORED DIP EARLY STOP VAR MODEL")
    # Ejecutar el programa predict.py con los argumentos requeridos para la restauración DIP Early Stop VAR
    command_dip = f"python predict.py DEEP_IMAGE_PRIOR/TEST_DATA/PREDICT_PNGs/{nombre}/ DEEP_IMAGE_PRIOR/TEST_DATA/PREDICT_PSFs/{nombre}/ DEEP_IMAGE_PRIOR/RESTORE_DATA/DIP+ES_VAR/ DEEP_IMAGE_PRIOR/UPDATE_WEIGHTS/{nombre}+ES_VAR_updated_model_y_weights.pt {W_COLOR_PATH}"
    os.system(command_dip)
    
    print("** PREDICTING WITH RESTORED NO DIP MODEL")
    # Ejecutar el programa predict.py con los argumentos requeridos para la restauración sin DIP
    command_no_dip = f"python predict.py DEEP_IMAGE_PRIOR/TEST_DATA/PREDICT_PNGs/{nombre}/ DEEP_IMAGE_PRIOR/TEST_DATA/PREDICT_PSFs/{nombre}/ DEEP_IMAGE_PRIOR/RESTORE_DATA/NO_DIP/ {W_Y_PATH} {W_COLOR_PATH}"
    os.system(command_no_dip)

    print("** COMPARE RESULTS")
    # Ejecutar el programa MSE_SSIM_PSNR.py con los argumentos requeridos para comparar resultados DIP, DIP+ES_VAR, NO_DIP
    #command_compare = f"python MSE_SSIM_PSNR.py DEEP_IMAGE_PRIOR/RESTORE_DATA/DIP/{nombre}_ddnet.png DEEP_IMAGE_PRIOR/RESTORE_DATA/DIP+ES_VAR/{nombre}_ddnet.png DEEP_IMAGE_PRIOR/RESTORE_DATA/NO_DIP/{nombre}_ddnet.png DEEP_IMAGE_PRIOR/TEST_DATA/ORIGINAL_IMGs/{nombre}_firls.png"
    command_compare = f"python utils/DIP/MSE_SSIM_PSNR.py DEEP_IMAGE_PRIOR/RESTORE_DATA/DIP/{nombre}_ddnet.png DEEP_IMAGE_PRIOR/RESTORE_DATA/DIP+ES_VAR/{nombre}_ddnet.png DEEP_IMAGE_PRIOR/RESTORE_DATA/NO_DIP/{nombre}_ddnet.png DEEP_IMAGE_PRIOR/TEST_DATA/ORIGINAL_IMGs/{nombre}_firls.png"
    os.system(command_compare)
    
    print("################################################################")

print("")
print("* FINAL METRICS")
#os.system("python get_mean_standard_deviation.py")
os.system("python utils/DIP/get_mean_standard_deviation.py")
print("")
print('~ All metrics can be found in ‘DEEP_IMAGE_PRIOR/METRICS.csv’')