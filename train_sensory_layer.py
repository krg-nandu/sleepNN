import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D, proj3d
from matplotlib import offsetbox

from torchvision import datasets, transforms
from models.autoencoder import BasicAutoEncoder, BasicAutoEncoderMLP
import os, glob, cv2
import numpy as np
import tqdm 
import time
from torch.autograd import Variable

from sklearn.decomposition import PCA

def proj(X, ax1, ax2):
    """ From a 3D point in axes ax1, 
    calculate position in 2D in ax2 """
    x,y,z = X
    x2, y2, _ = proj3d.proj_transform(x,y,z, ax1.get_proj())
    return ax2.transData.inverted().transform(ax1.transData.transform((x2, y2)))

def image(ax,arr,xy):
    """ Place an image (arr) as annotation at position xy """
    im = offsetbox.OffsetImage(arr, zoom=0.32)
    im.image.axes = ax
    ab = offsetbox.AnnotationBbox(im, xy, xybox=(-30., 30.),
                         xycoords='data', boxcoords="offset points",
                       pad=0.3, arrowprops=dict(arrowstyle="->"))
    ax.add_artist(ab)

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if alpha is not None:
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        #import ipdb; ipdb.set_trace()
        logpt = F.log_softmax(input, dim=1)

        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        #logpt = logpt.gather(0, target.squeeze())

        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, cache=False, transform=None):
        self.transform = transform
        images = glob.glob(os.path.join(data_path, '*.png'))
        if cache:
            print('Loading dataset...')
            t1 = time.time()
            self.ims = np.load('SM_v2.npy')
            self.ims = (self.ims > 240).astype(np.float32)
            t2 = time.time()
            print('Done. Time elapsed: {}'.format(t2-t1))
        else:
            self.ims = []
            for im in tqdm.tqdm(images):
                x = cv2.cvtColor(cv2.imread(im), cv2.COLOR_BGR2GRAY)
                self.ims.append(x)
            self.ims = np.dstack(self.ims)
            self.ims = np.expand_dims(self.ims, axis=-2).astype(np.float32)
            np.save('SM_v2',self.ims)

    def __len__(self):
        return self.ims.shape[-1]

    def __getitem__(self, index):
        # replace this with the actual image label
        
        sample = self.ims[...,index]
        if self.transform:
            sample = self.transform(sample)
        return sample, 0

def train(model, params, num_epochs=5, device='cpu', batch_size=64, learning_rate=1e-3):
    torch.manual_seed(42)

    #criterion = nn.MSELoss() # mean square error loss
    criterion = FocalLoss(gamma=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate, 
                                 weight_decay=1e-5)
    transform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize((0.), (255.))
            ]) 

    train_set = Dataset(
            data_path='/media/data_cifs/projects/prj_working-mem/SM_v2',
            transform = transform,
            cache = True)
    train_loader = torch.utils.data.DataLoader(train_set, **params)

    outputs = []

    fig, (ax1, ax2) = plt.subplots(1,2)

    for epoch in range(num_epochs):
        for data in train_loader:
            #import ipdb; ipdb.set_trace()
            img, _ = data
            img = img.to(device)
            recon = model(img)
            mask = (img != 1)
            loss = criterion(recon, img.long()) #+ 1e-5*torch.sum(mask*recon)
            #loss = criterion(recon, img)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print('Epoch:{}, Loss:{:.4f}'.format(epoch+1, float(loss)))
        #outputs.append((epoch, img, recon),)
        n_im = img.shape[0]
        idx = np.random.randint(n_im)
        X = torch.argmax(recon, axis=1)
        ax1.imshow(img[idx,0,...].cpu().detach().numpy())
        ax2.imshow(X[idx].cpu().detach().numpy())
        plt.pause(0.5)
    return model

def model_train():

    h, w, c = 64, 64, 1
    params = {'batch_size': 256,
              'shuffle': True,
              'num_workers': 1}
    dims = [params['batch_size'],c,h,w]

    device = torch.device('cuda:0')
    model = BasicAutoEncoder()

    #model = BasicAutoEncoderMLP(dims=dims)
    model = model.to(device)

    max_epochs = 100
    model = train(model, params, num_epochs=max_epochs, device=device)

    torch.save(model.state_dict(), 'ckpts/cnn_ae.pth')

def draw_trajectory(model, pca, transform, device, ax, ax2, a, b, c, color, draw_im=False):
    folder = '/media/data_cifs/projects/prj_working-mem/SM_v2'
    file_names = [os.path.join(folder, '{}_{}_{}_{}.png'.format(a,b,c,x)) for x in range(0,360, 10)]
    test_ims = []
    for im in tqdm.tqdm(file_names):
        x = cv2.cvtColor(cv2.imread(im), cv2.COLOR_BGR2GRAY)
        test_ims.append(x)
    test_ims = (np.dstack(test_ims) > 240).astype(np.float32)
    test_ims = transform(test_ims).unsqueeze(1).to(device)

    test_emb = model.get_embedding(test_ims)
    test_recon = torch.argmax(model(test_ims), axis=1)

    em = test_emb.flatten(start_dim=1)
    y = pca.transform(em.cpu().detach().numpy())

    if not draw_im:
        ax.plot(y[:,0], y[:,1], y[:,2], color)
        ax.scatter(y[:, 0], y[:, 1], y[:, 2], c=y[:,2], cmap='Greens')
    else:
        for k in [np.random.randint(test_recon.shape[0])]: #range(0, test_recon.shape[0], 19):
            xx, yy = proj((y[k,0], y[k,1], y[k,2]), ax, ax2)
            #im = test_recon[k,...].cpu().detach().numpy().astype(np.uint8)
            im = test_ims[k,0,...].cpu().detach().numpy().astype(np.uint8)
            image(ax2, im,[xx, yy])

def model_eval():
    h, w, c = 64, 64, 1
    params = {'batch_size': 4096,
              'shuffle': True,
              'num_workers': 1}
    dims = [params['batch_size'],c,h,w]

    device = torch.device('cuda:0')
    model = BasicAutoEncoder()
    model = model.to(device)

    model.load_state_dict(torch.load('ckpts/cnn_ae.pth'))
    model = model.eval()

    transform = transforms.Compose([
            transforms.ToTensor(),
            ]) 

    image_set = Dataset(
            data_path='/media/data_cifs/projects/prj_working-mem/SM_v2',
            transform = transform,
            cache = True)
    im_loader = torch.utils.data.DataLoader(image_set, **params)

    ims, recons, embs = [], [], []
    with torch.no_grad():
        for batch in im_loader:
            img, _ = batch
            img = img.to(device)

            recon = model(img)
            recon = torch.argmax(recon, axis=1)
            emb = model.get_embedding(img)

            ims.append(img.cpu().detach().numpy())
            embs.append(emb.cpu().detach().numpy())
            recons.append(recon.cpu().detach().numpy())

    Xim = np.vstack(ims)
    Xrecon = np.vstack(recons)
    Xemb = np.vstack(embs)
    Xemb = Xemb.reshape(Xemb.shape[0], -1)

    pca = PCA(n_components=3)
    pca.fit(Xemb)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.grid('off')
    ax.set_xlabel('PC1 expl var {}%'.format(int(pca.explained_variance_ratio_[0]*1000)/10.))
    ax.set_ylabel('PC2 expl var {}%'.format(int(pca.explained_variance_ratio_[1]*1000)/10.))
    ax.set_zlabel('PC3 expl var {}%'.format(int(pca.explained_variance_ratio_[2]*1000)/10.)) 
 
    ax2 = fig.add_subplot(111, frame_on=False)
    ax2.axis('off')

    draw_trajectory(model, pca, transform, device, ax, ax2, 4, 4, 4, 'gray')
    draw_trajectory(model, pca, transform, device, ax, ax2, 3, 1, 2, 'blue')
    draw_trajectory(model, pca, transform, device, ax, ax2, 4, 0, 0, 'black')
    draw_trajectory(model, pca, transform, device, ax, ax2, 4, 4, 0, 'red')

    draw_trajectory(model, pca, transform, device, ax, ax2, 4, 4, 4, 'gray', draw_im=True)
    draw_trajectory(model, pca, transform, device, ax, ax2, 3, 1, 2, 'blue', draw_im=True)
    draw_trajectory(model, pca, transform, device, ax, ax2, 4, 0, 0, 'black', draw_im=True)
    draw_trajectory(model, pca, transform, device, ax, ax2, 4, 4, 0, 'red', draw_im=True)

    plt.grid('off')
    plt.savefig('cnn_ae_latent.png', bbox_inches='tight') 
    plt.show()


if __name__ == '__main__':
    #model_train()
    model_eval()
