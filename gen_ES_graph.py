import os
import numpy as np
from utils.DIP.train_y_ES_VAR_graph import train_method
import matplotlib.pyplot as plt


train_data = 'DEEP_IMAGE_PRIOR/TEST_DATA/TRAIN_NPYs/'

# Parámetros: numero de imágnes a considerar y durante cuantas épocas se va a entrenar
num_imgs = 65
num_epochs = 1200

# Initialize empty 2D arrays for PSNR and VAR
PSNR = {}
VAR = {}
for i in range(num_epochs):
    PSNR[i] = []
    VAR[i] = []

EPOCH = []
for i in range(num_imgs):
    EPOCH.append(num_epochs)

#print(PSNR)
#print(VAR)


count = 1
for archivo in os.listdir(train_data):
    if(count > num_imgs):
        break
    #print(archivo)
    PSNR, VAR, EPOCH = train_method(train_data+archivo+'/',PSNR,VAR,EPOCH, num_epochs, count)
    
    count += 1
    
#print(PSNR)
#print()
#print(VAR)
#print()
#print(EPOCH)

PSNR_mean = []
VAR_mean = []
# Calculate and print the mean of each array within each key in PSNR and VAR
for key, value_list in PSNR.items():
    if value_list:  # Check if the list is not empty to avoid division by zero
        PSNR_mean.append(np.mean(value_list))

for key, value_list in VAR.items():
    if value_list:  # Check for empty lists
        VAR_mean.append(np.mean(value_list))
        
EPOCH_mean = np.mean(EPOCH)
        
#print(len(PSNR_mean))
#print(VAR_mean)
#print(EPOCH_mean)


Eje_X = np.arange(0, num_epochs, 1)

# Create the plot
fig, ax1 = plt.subplots()

# PSNR (red)
color = 'tab:red'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('PSNR', color=color)
ax1.plot(Eje_X[6:], PSNR_mean[6:], color=color, label='PSNR')

# Variance (blue)
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Variance', color=color)
ax2.plot(Eje_X[6:], VAR_mean[6:], color=color, label='Variance')

# Add vertical line at EPOCH_mean (yellow)
ax1.axvline(x=EPOCH_mean, color='yellow', linestyle='dashed', linewidth=2, label='Epoch Mean')

# Crear cadena con el número medio de épocas
epoch_mean_legend = f'Epoch Mean: {EPOCH_mean:.2f}'

# Add legend with color codes
legend_elements = [
    plt.Line2D([0], [0], color='red', lw=2, label='PSNR'),
    plt.Line2D([0], [0], color='blue', lw=2, label='Variance'),
    plt.Line2D([0], [0], color='yellow', linestyle='dashed', lw=2, label=epoch_mean_legend)
]
ax1.legend(handles=legend_elements, loc='upper right')

# Adjust layout and title
fig.tight_layout()
plt.title('ES Training History')

# Display the plot
plt.savefig("ES_Training_History")
