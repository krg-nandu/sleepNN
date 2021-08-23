import torch
import torch.nn.functional as F
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import glob, tqdm, os
from pathlib import Path 
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.io as S
import argparse
import copy

import numpy as np
import matplotlib.pyplot as plt
from config import Config

from utils import extract_sleep_bouts, butter_lowpass_filter

class Dataset(torch.utils.data.Dataset):
    def __init__(self, cfg, base_path, exp_name):
        self.cfg = cfg
        self.dataset_size = cfg.dataset_size
        self.n_timesteps = cfg.n_timesteps

        self.base_path = base_path
        self.arrs = []

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

        return self.arrs[idx][start_idx:start_idx + self.cfg.n_timesteps], self.arrs[idx][start_idx+self.cfg.n_timesteps]

class ARLSTM(torch.nn.Module):

   def __init__(self, embedding_dim, hidden_dim, output_dim):
      super(ARLSTM, self).__init__()
      self.hidden_dim = hidden_dim
      self.output_dim = output_dim

      # The LSTM takes action labels as inputs, and outputs hidden states
      # with dimensionality hidden_dim.
      self.lstm = torch.nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=3)
      self.output_fn = torch.nn.Linear(hidden_dim, output_dim)
      self.act = F.softplus

   def forward(self, x):
      lstm_out, _ = self.lstm(x.transpose(-1,0).unsqueeze(-1))
      pred = self.output_fn(lstm_out[-1])
      # is there a good activation function for this? not sure..
      pred = torch.cat([pred[:, :int(self.output_dim/2)], self.act(pred[:, int(self.output_dim/2):])], axis=-1)

      return pred


def _log_beta(x, y):
    return torch.lgamma(x) + torch.lgamma(y) - torch.lgamma(x + y)

def compute_logprob(k, a, b):
    n = torch.zeros_like(a) + 9. # defining the support
    log_factorial_n = torch.lgamma(n + 1)
    log_factorial_k = torch.lgamma(k + 1)
    log_factorial_nmk = torch.lgamma(n - k + 1)

    return -1*(log_factorial_n - log_factorial_k - log_factorial_nmk +
                _log_beta(k + a, n - k + b) -
                _log_beta(b, a))

'''Isotropic heteroskedastic loss
'''
def iso_het_loss(x, mu, var):
    return torch.sum( ((x - mu) ** 2)/var + torch.log(var) )

'''Cheat to try replicate a confidence interval
'''
def lsqr(x, mu, var):
    return torch.sum( (x - (mu+var))**2 + (x - (mu-var))**2)

def train_fn(device):
    cfg = Config()

    training_set = Dataset(cfg, cfg.data_path, cfg.experiments)
    training_generator = torch.utils.data.DataLoader(training_set, **cfg.train_params)

    # set up the model
    model = ARLSTM(embedding_dim=1, hidden_dim=128, output_dim=2)
    
    #loss_fn = iso_het_loss
    loss_fn = lsqr

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    if False:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model.to(device)
    model.train()

    per_epoch_loss = []
    for epoch in range(cfg.max_epochs):
       losses = []
       # training loop
       for cnt, (batch, lab) in enumerate(training_generator):

          batch, lab = batch.to(device), lab.to(device)
          model.zero_grad()
          y = model(batch)

          # y has the "parameters" of the distribution that we want to model
          mu, var = y[:, 0], y[:, 1]
          loss = loss_fn(lab, mu, var)

          loss.backward()
          optimizer.step()
          losses.append(loss.detach().item())

       #scheduler.step()
       per_epoch_loss.append(np.mean(losses))
       print('Epoch:{} | Batch: {}/{} | loss: {}'.format(epoch, cnt, training_generator.__len__(), per_epoch_loss[-1]))

    # saving model
    torch.save(model.state_dict(), os.path.join(cfg.save_path, '{}_{}.pth'.format(cfg.model_name, cfg.model_type)))


def gen_sequence(model, seq, len_sample_traj):

    seq2 = seq.clone()
    pred_seq = list(seq.cpu().numpy().flatten())

    mu, sd = [], []
    with torch.no_grad():
        for k in tqdm.tqdm(range(len_sample_traj)):
            pred = model(seq)
            
            # sample based on this distribution
            next_token = torch.normal(pred[:,0], pred[:,1])
            sd.append(pred[:, 1])
            mu.append(pred[:, 0])
            pred_seq.append(next_token)
            
            # update seq
            seq2[:, :-1] = seq[:, 1:].data
            seq2[:, -1] = next_token
            seq = seq2.clone()

    return pred_seq


def generate(device):
    np.random.seed(0)
    cfg = Config()

    model = ARLSTM(embedding_dim=1, hidden_dim=128, output_dim=2)
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(cfg.save_path, '{}_{}'.format(cfg.model_name, cfg.model_type))))
    model.eval()

    arr = S.loadmat(os.path.join(cfg.data_path, cfg.test_experiment))
    if cfg.exp == 'PFC':
        arr = arr[dataset.strip('.mat').replace('LFP', 'lfp')][0][0][1]
    elif cfg.exp == 'RSC':
        arr = arr['lfp'][0][0][1]
    arr = butter_lowpass_filter(arr.squeeze(), cfg.cutoff, cfg.fs, cfg.order)
    arr = arr.astype(np.float32)

    # pick a random sample
    start_idx = 100000
    time_window = cfg.n_timesteps 
    synthetic_trajectory_length = 5 * 600
    
    init_seq = torch.tensor(arr[start_idx : start_idx+time_window]).unsqueeze(0).to(device)
    real_trajectory = arr[start_idx: start_idx+time_window+synthetic_trajectory_length]
    rtrajectory = copy.deepcopy(real_trajectory)

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(rtrajectory, c = 'b', alpha=0.75, label='original')

    for k in range(1):
        syn_trajectory = gen_sequence(model, init_seq, synthetic_trajectory_length)
        ax.plot(syn_trajectory, c = 'r', alpha=0.75)

    plt.legend()
    plt.show()


if __name__ == '__main__':
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--generate', action='store_true')

    args = parser.parse_args()

    if args.train:
        train_fn(device)
    elif args.generate:
        generate(device)
    else:
        raise NotImplementedError
