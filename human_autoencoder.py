import torch
from torch import nn

import torch.nn.functional as F
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import glob, tqdm, os
from pathlib import Path 
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.io as S
import argparse, copy, pickle
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt
from config import Config

from utils import extract_sleep_bouts, butter_lowpass_filter, make_pcs
from matplotlib.animation import FuncAnimation 

import h5py as H

class BasicAutoEncoderMLP(nn.Module):
    def __init__(self, in_feat):
        super(BasicAutoEncoderMLP, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=in_feat, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=32, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=in_feat)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        #x = torch.reshape(x, [-1, 2, *self.dims[2:]])
        return x

class Dataset(torch.utils.data.Dataset):
    def __init__(self, cfg, transform):
        self.cfg = cfg
        self.dataset_size = cfg['dataset_size']
        self.n_timesteps = cfg['n_timesteps']
        self.base_path = cfg['base_path']
        self.arrs = []
        self.transform = transform
        N = 100000

        arr = H.File(os.path.join(self.base_path, cfg['dataset']))
        self.train_data = arr['h5eeg']['eeg'][:N, np.array(cfg['train_chan'])]
        self.test_data = arr['h5eeg']['eeg'][:N, np.array(cfg['test_chan'])]

        # normalize
        self.mu = self.train_data.mean(axis=0, keepdims=True)
        self.sig = self.train_data.std(axis=0, keepdims=True)
        self.train_data = (self.train_data - self.mu)/self.sig

        self.s_len = len(self.train_data)
        print('Preparing train-test splits...')
        self.rnd_idx = np.random.randint(self.s_len - self.n_timesteps - 1 , size=(self.dataset_size,)) 
 
    def __len__(self):
        'Denotes the total number of samples'
        return self.rnd_idx.shape[0]

    def __getitem__(self, index):
        'Generates one sample of data'
        idx = self.rnd_idx[index]
        sample = np.expand_dims(self.train_data[idx: idx + self.n_timesteps], 0)
        sample = self.transform(sample)
        return sample

def model_eval(device):
    np.random.seed(0)
    cfg = Config()

    transform = transforms.Compose([
            transforms.ToTensor(),
            ]) 

    dataset = Dataset(cfg, cfg.data_path, cfg.experiments, transform)
    generator = torch.utils.data.DataLoader(dataset, **cfg.train_params)
    model = BasicAutoEncoderMLP(1024)
    model.load_state_dict(torch.load('ckpts/rsc_autoencode.pth'))
    model = model.eval()

    fig = plt.figure()
    for k in range(16):
        ax = fig.add_subplot(4, 4, k+1)
        val_sample = dataset.__getitem__(np.random.randint(100))
        val_recon = model(val_sample)
        val_recon = val_recon.detach().squeeze()

        ax.plot(val_sample.squeeze(), c='tab:pink', label='original')
        ax.plot(val_recon, c='tab:gray', label='recon')
        
        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

    plt.legend()
    plt.show()
 
def train_fn(device):
    np.random.seed(0)
    transform = transforms.Compose([
            transforms.ToTensor(),
            ]) 

    cfg = {
        'base_path': '/media/data_cifs/projects/prj_working-mem/human_data/h5_notch20',
        'dataset': 'm00083.h5',
        'dataset_size': 100000,
        'n_timesteps': 1024,
        'train_params': {
                'batch_size': 1024,
                'shuffle': False,
                'num_workers': 1
                },
        'train_chan': [0],
        'test_chan': [0]
    }

    training_set = Dataset(cfg, transform)
    training_generator = torch.utils.data.DataLoader(training_set, **cfg['train_params'])
    model = BasicAutoEncoderMLP(1024)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=1e-4, 
                                 weight_decay=1e-5)

    for epoch in range(100):
        model = model.train()
        for data in training_generator:
            batch = data
            recon = model(batch)
            loss = criterion(recon, batch)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print('Epoch: {}, Loss: {:.4f}'.format(epoch+1, float(loss)))

        model = model.eval()

        fig = plt.figure()
        for k in range(25):
            ax = fig.add_subplot(5, 5, k+1)
            ax.clear()
            val_sample = training_set.__getitem__(np.random.randint(100))
            val_recon = model(val_sample)
            val_recon = val_recon.detach().squeeze()
            ax.plot(val_sample.squeeze(), label='original')
            ax.plot(val_recon, label='recon')
            ax.axis('off')
        plt.legend()
        plt.savefig('figures/precuneus_train/img_%03d.png'%epoch, bbox_inches='tight')
        plt.close()

    torch.save(model.state_dict(), 'ckpts/precuneus_autoencode.pth')

if __name__ == '__main__':
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', action='store_true')

    args = parser.parse_args()

    if args.train:
        train_fn(device)
    elif args.eval:
        model_eval(device)
    else:
        raise NotImplementedError
