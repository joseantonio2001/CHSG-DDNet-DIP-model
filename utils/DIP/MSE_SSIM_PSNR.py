import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import sys
import os.path
import csv

# Obtiene el directorio padre de la carpeta S
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

# Agrega el directorio base al sys.path
sys.path.insert(0, base_dir)

def calcular_diferencia(imagen1, imagen2):
    
    imagen1_gris = cv2.imread(imagen1, cv2.IMREAD_GRAYSCALE)
    imagen2_gris = cv2.imread(imagen2, cv2.IMREAD_GRAYSCALE)
    
    # Calcular el Error Cuadrático Medio (MSE)
    mse = np.mean((imagen1_gris - imagen2_gris) ** 2)
    
    # Calcular el Índice de Similitud Estructural (SSIM)
    ssim_score, _ = ssim(imagen1_gris, imagen2_gris, full=True)
    
    # Calcular el PSNR
    if mse == 0:
        psnr = float('inf')
    else:
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    
    return mse, ssim_score, psnr

if __name__ == "__main__":
    
    if len(sys.argv) != 5:
        print("ERROR - Usage: python script_name.py imagen_DIP imagen_NO_DIP imagen_ORIGINAL")
        sys.exit(1)
    
    mse_DIP, ssim_score_DIP, psnr_DIP = calcular_diferencia(sys.argv[1], sys.argv[4]) 
    print(" * DIP results")
    print("     Error Cuadrático Medio (MSE):", mse_DIP)
    print("     Índice de Similitud Estructural (SSIM):", ssim_score_DIP)
    print("     Relación Señal-Ruido de Pico (PSNR):", psnr_DIP)

    mse_DIP_ES_VAR, ssim_score_DIP_ES_VAR, psnr_DIP_ES_VAR = calcular_diferencia(sys.argv[2], sys.argv[4])
    print(" * DIP+ES_VAR results") 
    print("     Error Cuadrático Medio (MSE):", mse_DIP_ES_VAR)
    print("     Índice de Similitud Estructural (SSIM):", ssim_score_DIP_ES_VAR)
    print("     Relación Señal-Ruido de Pico (PSNR):", psnr_DIP_ES_VAR)   
   
    mse_NO_DIP, ssim_score_NO_DIP, psnr_NO_DIP = calcular_diferencia(sys.argv[3], sys.argv[4])
    print(" * NO_DIP results") 
    print("     Error Cuadrático Medio (MSE):", mse_NO_DIP)
    print("     Índice de Similitud Estructural (SSIM):", ssim_score_NO_DIP)
    print("     Relación Señal-Ruido de Pico (PSNR):", psnr_NO_DIP)        
    
    # Obtener la parte final entre '/' y '_firls.png' de la segunda imagen
    img_filename = os.path.basename(sys.argv[3])
    img_name = img_filename.split('/')[-1].split('_firls.png')[0]
    with open('DEEP_IMAGE_PRIOR/METRICS.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([img_name,mse_DIP,mse_DIP_ES_VAR,mse_NO_DIP,ssim_score_DIP,ssim_score_DIP_ES_VAR,ssim_score_NO_DIP,psnr_DIP,psnr_DIP_ES_VAR,psnr_NO_DIP])
            
            
            
            