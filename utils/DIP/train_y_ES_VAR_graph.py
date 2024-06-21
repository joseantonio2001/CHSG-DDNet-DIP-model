import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models.aff import AffWrapper, DDNet, AAplusModel
from models.fade2black_matrix import FadeToBlackMatrix, FadeToBlackMatrixNoPad
from utils.read import RestorationsDataset
import sys
import config
from config_deep_image_prior import W_Y_PATH, DIP_EPOCHS, DIP_ITERS_PER_EPOCH
import numpy as np
import os

# Obtiene el directorio padre de la carpeta S
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

# Agrega el directorio base al sys.path
sys.path.insert(0, base_dir)

def calculate_psnr(pred, target):
  """
  Calculates PSNR between two tensors (predicted and target).
  Args:
      pred: Predicted image tensor.
      target: Target image tensor.
  Returns:
      PSNR value in dB.
  """
  mse = torch.mean((pred - target)**2)  # Mean squared error
  max_val = torch.max(pred).item()  # Maximum value of the image
  psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
  return psnr.item()  # Convert to scalar


class Preprocess_deblur():
    def __init__(self, epsilon, fd_black1, fd_black2):
        self.epsilon = epsilon
        self.fd_black1 = fd_black1
        self.fd_black2 = fd_black2

    def __call__(self, X, gpu):
        with torch.no_grad():
            x, y, H = X
            H = H.unsqueeze(dim=1)

            if gpu:
                x = Variable(x.float().cuda() / 255.0)
                y = Variable(y.float().cuda() / 255.0)
                H = Variable(H.float().cuda())
            else:
                x = Variable(x.float() / 255.0)
                y = Variable(y.float() / 255.0)
                H = Variable(H.float())

            amortised_model = AAplusModel(H, fd_black=self.fd_black2, epsilon=self.epsilon)
            x_r = amortised_model.Aplus_only(y)
            amortised_model = AAplusModel(H, fd_black=self.fd_black1, epsilon=self.epsilon)
            y = y[:, :, config.MAX_PSF_SIZE:-config.MAX_PSF_SIZE, config.MAX_PSF_SIZE:-config.MAX_PSF_SIZE].contiguous()
            x = x[:, :, config.MAX_PSF_SIZE:-config.MAX_PSF_SIZE, config.MAX_PSF_SIZE:-config.MAX_PSF_SIZE].contiguous()
            data = [amortised_model, x_r, y]

            return data, x, x.size(0)
 
class EarlyStopVariance:
    def __init__(self, window_size, patience):
        # Inicializa la clase con el tamaño de la ventana y la paciencia
        self.window_size = window_size  # Tamaño de la ventana para calcular la varianza
        self.patience = patience  # Paciencia para detener el entrenamiento
        self.wait_count = 0  # Contador para controlar la paciencia
        self.best_variance = float('inf')  # Mejor varianza observada
        self.best_epoch = 0  # Época en la que se observó la mejor varianza
        self.img_collection = []  # Colección de imágenes para calcular la varianza
        self.stop = False  # Indicador para detener el entrenamiento

    def calculate_variance(self, images):
        # Calcula la varianza entre las imágenes en la colección
        mean_image = np.mean(images, axis=0)  # Calcula la imagen media
        variance = np.mean([(img - mean_image) ** 2 for img in images])  # Calcula la varianza
        return variance

    def update_img_collection(self, cur_img):
        # Actualiza la colección de imágenes con una nueva imagen
        self.img_collection.append(cur_img)  # Agrega la nueva imagen
        if len(self.img_collection) > self.window_size:
            self.img_collection.pop(0)  # Elimina la imagen más antigua si se excede el tamaño de la ventana

    def check_stop(self, cur_img, cur_epoch):
        # Verifica si se debe detener el entrenamiento basado en la varianza calculada
        self.update_img_collection(cur_img)  # Actualiza la colección de imágenes
        #print("CALCULANDO VARIANZA")
        if len(self.img_collection) == self.window_size:
            # Si la colección alcanza el tamaño de la ventana
            current_variance = self.calculate_variance(self.img_collection)  # Calcula la varianza actual
            #print("current_variance: ", current_variance)
            
            if current_variance < self.best_variance:
                # Si la varianza actual es mejor que la mejor varianza anterior
                self.best_variance = current_variance  # Actualiza la mejor varianza
                self.best_epoch = cur_epoch  # Actualiza la época de la mejor varianza
                self.wait_count = 0  # Reinicia el contador de paciencia
            else:
                self.wait_count += 1  # Incrementa el contador de paciencia
            if self.wait_count >= self.patience:
                # Si se excede la paciencia
                self.stop = True  # Indica que se debe detener el entrenamiento
        return self.stop, self.calculate_variance(self.img_collection)  # Retorna el indicador de detención del entrenamiento
        
class CharbonnierLoss():
    def __init__(self, eps=1e-3):
        self.eps= eps

    def __call__(self, pred, target):
        loss = torch.sqrt((pred - target)**2 + self.eps).mean()

        return loss

def train_method(train_data, PSNR, VAR, EPOCH, epochs_ES, count):
    
    print(train_data)
    
    # Load pre-trained weights
    #print("Loading pre-trained weights for model _Y...")
    G = AffWrapper(
            DDNet(
                channels_c=1,
                n_features=config.N_FEATURES,
                nb=config.N_DENSE_BLOCKS,
                df_size=config.DYNAMIC_FILTER_SIZE
            ),
        df_size=config.DYNAMIC_FILTER_SIZE
    ).cuda()
    G.load(W_Y_PATH)
    
    # Define loss function and optimizer
    loss_fn = CharbonnierLoss(eps=config.EPS)
    optimizer = optim.Adam(G.parameters(), lr=config.LR_1, weight_decay=config.W_DECAY)
    
    # Load the image for updating weights
    #print("Loading the image for updating weights...")
    train_dataset = RestorationsDataset(train_data, ['.npy'], config.IM_SIZE, config.SIGMA_NOISE)
    train_data_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)
    
    # Preprocessing function
    fd_black1 = FadeToBlackMatrix([config.IM_SIZE, config.IM_SIZE], [config.MAX_PSF_SIZE, config.MAX_PSF_SIZE]).cuda()
    fd_black2 = FadeToBlackMatrixNoPad([config.IM_SIZE + 2 * config.MAX_PSF_SIZE, config.IM_SIZE + 2 * config.MAX_PSF_SIZE], [config.MAX_PSF_SIZE, config.MAX_PSF_SIZE]).cuda()
    preprocess_func = Preprocess_deblur(epsilon=config.EPS_WIENER, fd_black1=fd_black1, fd_black2=fd_black2)
    
    # Initialize EarlyStopVariance
    window_size = 30
    patience = 6
    earlystop = EarlyStopVariance(window_size=window_size, patience=patience)
    LAST_EPOCH_mod = False
    for epoch in range(epochs_ES):
        for i, batch in enumerate(train_data_loader):
            optimizer.zero_grad()
            data, x, _ = preprocess_func(batch, gpu=True)
            p, _ = G(data)
            loss = loss_fn(p, x)
            loss.backward()
            optimizer.step()
            
            # Check early stopping criteria
            if i % DIP_ITERS_PER_EPOCH == 0:
                print('Epoch [{}/{}]'.format(epoch + 1, epochs_ES))
                p_np = p.detach().cpu().numpy()
                x_np = x.detach().cpu().numpy()
                
                # Calculate PSNR and VAR
                psnr = calculate_psnr(torch.from_numpy(p_np), torch.from_numpy(x_np))
                stop, variance = earlystop.check_stop(p_np, epoch)
                PSNR[epoch].append(psnr)
                VAR[epoch].append(variance)
                
                if stop:
                    print(f"Stopping early at epoch {epoch}")
                    break
        
        if earlystop.stop and not LAST_EPOCH_mod:
            EPOCH[count-1] = (epoch + 1)
            LAST_EPOCH_mod = True
            #break
        
    return PSNR, VAR, EPOCH