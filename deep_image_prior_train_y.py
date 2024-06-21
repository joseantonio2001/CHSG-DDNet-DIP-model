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
        
class CharbonnierLoss():
    def __init__(self, eps=1e-3):
        self.eps= eps

    def __call__(self, pred, target):
        loss = torch.sqrt((pred - target)**2 + self.eps).mean()

        return loss

if __name__ == '__main__':
    
    train_data = sys.argv[1]
    out_dir = sys.argv[2]
    
    # Load pre-trained weights
    print("Loading pre-trained weights for model _Y...")
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
    print("Loading the image for updating weights...")
    train_dataset = RestorationsDataset(train_data, ['.npy'], config.IM_SIZE, config.SIGMA_NOISE)
    train_data_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)
    
    # Preprocessing function
    fd_black1 = FadeToBlackMatrix([config.IM_SIZE, config.IM_SIZE], [config.MAX_PSF_SIZE, config.MAX_PSF_SIZE]).cuda()
    fd_black2 = FadeToBlackMatrixNoPad([config.IM_SIZE + 2 * config.MAX_PSF_SIZE, config.IM_SIZE + 2 * config.MAX_PSF_SIZE], [config.MAX_PSF_SIZE, config.MAX_PSF_SIZE]).cuda()
    preprocess_func = Preprocess_deblur(epsilon=config.EPS_WIENER, fd_black1=fd_black1, fd_black2=fd_black2)
    
    # Training loop
    print("Updating weights for model _Y...")
    for epoch in range(DIP_EPOCHS):
        for i, batch in enumerate(train_data_loader):
            optimizer.zero_grad()
            data, x, _ = preprocess_func(batch, gpu=True)
            p, _ = G(data)
            loss = loss_fn(p, x)
            loss.backward()
            optimizer.step()

            #if i % 1000 == 0:
            if i % DIP_ITERS_PER_EPOCH == 0:
                #print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, config.EPOCHS, i + 1, len(train_data_loader), loss.item()))
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, DIP_EPOCHS, i + 1, len(train_data_loader), loss.item()))

    # Save updated weights
    print("Saving updated weights for model _Y...")
    G.save(out_dir+ '_updated_model_y_weights.pt')
    
    print("Training _Y model completed.")
