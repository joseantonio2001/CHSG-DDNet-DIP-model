import csv
import numpy as np
import sys
import os.path

# Obtiene el directorio padre de la carpeta S
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

# Agrega el directorio base al sys.path
sys.path.insert(0, base_dir)

# Nombre del archivo CSV
archivo_csv = "DEEP_IMAGE_PRIOR/METRICS.csv"

# Separador para el print
separador = '#######################################################################'

# Lista para almacenar los valores
MSE_DIP = []
MSE_DIP_ES = []
MSE_NO_DIP = []
SSIM_DIP = []
SSIM_DIP_ES = []
SSIM_NO_DIP = []
PSNR_DIP = []
PSNR_DIP_ES = []
PSNR_NO_DIP = []

# Abre el archivo CSV y lee los valores de las columnas de cada fila
with open(archivo_csv, newline='') as csvfile:
    # Crea un objeto lector CSV
    lector_csv = csv.reader(csvfile)
    
    # Salta la primera fila
    next(lector_csv)
    
    # Itera sobre las filas restantes
    for fila in lector_csv:        
        MSE_DIP.append(float(fila[1]))
        MSE_DIP_ES.append(float(fila[2]))
        MSE_NO_DIP.append(float(fila[3]))
        SSIM_DIP.append(float(fila[4]))
        SSIM_DIP_ES.append(float(fila[5]))
        SSIM_NO_DIP.append(float(fila[6]))
        PSNR_DIP.append(float(fila[7]))
        PSNR_DIP_ES.append(float(fila[8]))
        PSNR_NO_DIP.append(float(fila[9]))

# Calcular las medias       
media_MSE_DIP = np.mean(MSE_DIP)
media_MSE_DIP_ES = np.mean(MSE_DIP_ES)
media_MSE_NO_DIP = np.mean(MSE_NO_DIP)
media_SSIM_DIP = np.mean(SSIM_DIP)
media_SSIM_DIP_ES = np.mean(SSIM_DIP_ES)
media_SSIM_NO_DIP = np.mean(SSIM_NO_DIP)
media_PSNR_DIP = np.mean(PSNR_DIP)
media_PSNR_DIP_ES = np.mean(PSNR_DIP_ES)
media_PSNR_NO_DIP = np.mean(PSNR_NO_DIP)

# Calcular las desviaciones estándar
std_MSE_DIP = np.std(MSE_DIP)
std_MSE_DIP_ES = np.std(MSE_DIP_ES)
std_MSE_NO_DIP = np.std(MSE_NO_DIP)
std_SSIM_DIP = np.std(SSIM_DIP)
std_SSIM_DIP_ES = np.std(SSIM_DIP_ES)
std_SSIM_NO_DIP = np.std(SSIM_NO_DIP)
std_PSNR_DIP = np.std(PSNR_DIP)
std_PSNR_DIP_ES = np.std(PSNR_DIP_ES)
std_PSNR_NO_DIP = np.std(PSNR_NO_DIP)

# Imprimir los resultados en el formato deseado
print(" ** MSE")
print(" Mean                 DIP: {:.2f}   DIP+ES_VAR: {:.2f}   NO_DIP: {:.2f}".format(media_MSE_DIP, media_MSE_DIP_ES, media_MSE_NO_DIP))
print(" Standard Deviation   DIP: {:.2f}   DIP+ES_VAR: {:.2f}   NO_DIP: {:.2f}".format(std_MSE_DIP, std_MSE_DIP_ES, std_MSE_NO_DIP))

print(" ** SSIM")
print(" Mean                 DIP: {:.2f}   DIP+ES_VAR: {:.2f}   NO_DIP: {:.2f}".format(media_SSIM_DIP, media_SSIM_DIP_ES, media_SSIM_NO_DIP))
print(" Standard Deviation   DIP: {:.2f}   DIP+ES_VAR: {:.2f}   NO_DIP: {:.2f}".format(std_SSIM_DIP, std_SSIM_DIP_ES, std_SSIM_NO_DIP))

print(" ** PSNR")
print(" Mean                 DIP: {:.2f}   DIP+ES_VAR: {:.2f}   NO_DIP: {:.2f}".format(media_PSNR_DIP, media_PSNR_DIP_ES, media_PSNR_NO_DIP))
print(" Standard Deviation   DIP: {:.2f}   DIP+ES_VAR: {:.2f}   NO_DIP: {:.2f}".format(std_PSNR_DIP, std_PSNR_DIP_ES ,std_PSNR_NO_DIP))

#Escribir los resultados en METRICS.csv
with open('DEEP_IMAGE_PRIOR/METRICS.csv', mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([])
    writer.writerow([separador])
    writer.writerow([' ** MSE'])
    writer.writerow([" Mean                 DIP: {:.2f}   DIP+ES_VAR: {:.2f}   NO_DIP: {:.2f}".format(media_MSE_DIP, media_MSE_DIP_ES, media_MSE_NO_DIP)])
    writer.writerow([" Standard Deviation   DIP: {:.2f}   DIP+ES_VAR: {:.2f}   NO_DIP: {:.2f}".format(std_MSE_DIP, std_MSE_DIP_ES, std_MSE_NO_DIP)])
    writer.writerow([' ** SSIM'])
    writer.writerow([" Mean                 DIP: {:.2f}   DIP+ES_VAR: {:.2f}   NO_DIP: {:.2f}".format(media_SSIM_DIP, media_SSIM_DIP_ES, media_SSIM_NO_DIP)])
    writer.writerow([" Standard Deviation   DIP: {:.2f}   DIP+ES_VAR: {:.2f}   NO_DIP: {:.2f}".format(std_SSIM_DIP, std_SSIM_DIP_ES, std_SSIM_NO_DIP)])
    writer.writerow([' ** PSNR'])
    writer.writerow([" Mean                 DIP: {:.2f}   DIP+ES_VAR: {:.2f}   NO_DIP: {:.2f}".format(media_PSNR_DIP, media_PSNR_DIP_ES, media_PSNR_NO_DIP)])
    writer.writerow([" Standard Deviation   DIP: {:.2f}   DIP+ES_VAR: {:.2f}   NO_DIP: {:.2f}".format(std_PSNR_DIP, std_PSNR_DIP_ES ,std_PSNR_NO_DIP)])
    writer.writerow([separador])
    
    
    
    

        

