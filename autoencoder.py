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
    def __init__(self, cfg, base_path, exp_name, transform):
        self.cfg = cfg
        self.dataset_size = cfg.dataset_size
        self.n_timesteps = cfg.n_timesteps
        self.base_path = base_path
        self.arrs = []
        self.transform = transform

        for dataset in exp_name:
            arr = S.loadmat(os.path.join(base_path, dataset))
            if cfg.exp == 'PFC':
                arr = arr[dataset.strip('.mat').replace('LFP', 'lfp')][0][0][1]
            elif cfg.exp == 'RSC':
                arr = arr['lfp'][0][0][1]

            filter_arr, bouts = extract_sleep_bouts(arr.squeeze(), cfg)
            filter_arr = filter_arr.astype(np.float32)
            for bout in bouts:
                self.arrs.append(filter_arr[bout[0] : bout[1]])

        if cfg.train_zscore:
            self.arrs = np.vstack(self.arrs).astype(np.float32)

            self.mean = np.mean(self.arrs)
            self.std = np.std(self.arrs)

            self.arrs = (self.arrs - np.mean(self.arrs))/np.std(self.arrs)

        self.n_datapoints = len(self.arrs)

        print('Preparing train-test splits...')
        self.rnd_idx = np.random.randint(self.n_datapoints, size=(self.dataset_size,)) 
        #, np.random.randint(self.n_duration-self.n_timesteps-1, size=(self.dataset_size,))
 
    def __len__(self):
        'Denotes the total number of samples'
        return self.rnd_idx.shape[0]

    def __getitem__(self, index):
        'Generates one sample of data'
        idx = self.rnd_idx[index]
        # sample start index
        arr_len = self.arrs[idx].shape[0]

        start_idx = np.random.randint(arr_len - self.cfg.n_timesteps - 1)    
        sample = np.expand_dims(self.arrs[idx][start_idx:start_idx + self.cfg.n_timesteps], 0)
        sample = self.transform(sample)

        return sample

def model_eval(device):
    np.random.seed(0)
    cfg = Config()

    transform = transforms.Compose([
            transforms.ToTensor(),
            ]) 

    import ipdb; ipdb.set_trace()
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
    cfg = Config()

    transform = transforms.Compose([
            transforms.ToTensor(),
            ]) 

    training_set = Dataset(cfg, cfg.data_path, cfg.experiments, transform)
    training_generator = torch.utils.data.DataLoader(training_set, **cfg.train_params)
    model = BasicAutoEncoderMLP(1024)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=1e-4, 
                                 weight_decay=1e-5)

    training_set.__getitem__(0)
    for epoch in range(100):
        for data in training_generator:
            batch = data
            recon = model(batch)
            loss = criterion(recon, batch)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print('Epoch: {}, Loss: {:.4f}'.format(epoch+1, float(loss)))

    torch.save(model.state_dict(), 'ckpts/rsc_autoencode.pth')
    model = model.eval()

    fig = plt.figure()
    for k in range(25):
        ax = fig.add_subplot(5, 5, k+1)
        val_sample = training_set.__getitem__(np.random.randint(100))
        val_recon = model(val_sample)
        val_recon = val_recon.detach().squeeze()
        ax.plot(val_sample.squeeze(), label='original')
        ax.plot(val_recon, label='recon')
        ax.axis('off')
    plt.legend()
    plt.show()
 
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
