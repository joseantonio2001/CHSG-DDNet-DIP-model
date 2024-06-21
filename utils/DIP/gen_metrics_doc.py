import os
import csv
import sys

# Obtiene el directorio padre de la carpeta S
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

# Agrega el directorio base al sys.path
sys.path.insert(0, base_dir)

def crear_o_actualizar_csv():
    # Comprobar si existe el archivo METRICS.csv
    if not os.path.isfile('METRICS.csv'):
        # Si no existe, crear el archivo y escribir las columnas
        with open('DEEP_IMAGE_PRIOR/METRICS.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['IMG', 'MSE_DIP', 'MSE_DIP+ES_VAR', 'MSE_NO_DIP', 'SSIM_DIP', 'SSIM_DIP+ES_VAR', 'SSIM_NO_DIP', 'PSNR_DIP', 'PSNR_DIP+ES_VAR', 'PSNR_NO_DIP'])
    else:
        # Si existe, escribir en la siguiente fila disponible
        with open('DEEP_IMAGE_PRIOR/METRICS.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([])  # Escribir una fila vacía para ir a la siguiente fila disponible

# Llamar a la función para crear o actualizar el archivo CSV
crear_o_actualizar_csv()
