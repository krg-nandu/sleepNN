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
import copy, pickle

import numpy as np
import matplotlib.pyplot as plt
from config import Config

from utils import extract_sleep_bouts, butter_lowpass_filter, make_pcs
from matplotlib.animation import FuncAnimation 

class DatasetPCA(torch.utils.data.Dataset):
    def __init__(self, cfg, base_path, exp_name):
        self.cfg = cfg
        self.dataset_size = cfg.dataset_size
        self.n_timesteps = cfg.n_timesteps
        self.base_path = base_path
        self.arrs = []

        if not os.path.exists('{}_components.p'.format(cfg.exp)):
            self.pca, self.arrs, self.bouts = make_pcs(cfg)
        else:
            self.pca = pickle.load(open('{}_components.p'.format(cfg.exp), 'rb'))
            self.arrs = pickle.load(open('{}_transformed.p'.format(cfg.exp), 'rb'))
            self.bouts = pickle.load(open('{}_bouts.p'.format(cfg.exp), 'rb'))

        self.n_datapoints = len(cfg.experiments) #len(self.arrs)

        print('Preparing train-test splits...')
        self.rnd_idx = np.random.randint(self.n_datapoints, size=(self.dataset_size,)) 
 
    def __len__(self):
        'Denotes the total number of samples'
        return self.rnd_idx.shape[0]

    def __getitem__(self, index):
        'Generates one sample of data'
        idx = self.rnd_idx[index]

        # sample start index 
        arr_len = len(self.bouts[self.cfg.model_type][idx])
        start_idx = np.random.randint(arr_len)
        start_idx = int(self.bouts[self.cfg.model_type][idx][start_idx]/2)

        # overall training
        #arr_len = self.arrs[idx].shape[0]
        #start_idx = np.random.randint(arr_len - self.cfg.n_timesteps - 1)    

        return self.arrs[idx][start_idx:start_idx + self.cfg.n_timesteps, :], self.arrs[idx][start_idx+self.cfg.n_timesteps, :]


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
      self.n_layers = 3

      self.lstm = torch.nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=self.n_layers)
      self.output_fn = torch.nn.Linear(hidden_dim, output_dim)
      self.act = F.softplus

   def init_hidden(self, batch_size):
      hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to('cuda:0')
      cell = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to('cuda:0')
      self.hidden_state = (hidden, cell)

   def forward(self, x, only_last=False, reset_hidden=False):
      if len(x.shape) == 3:
        inp = x.transpose(1, 0)
      else:
        inp = x.transpose(-1,0).unsqueeze(-1)

      if reset_hidden:
        self.init_hidden(inp.shape[1])
        lstm_out, self.hidden_state = self.lstm(inp, self.hidden_state)
      else:
        lstm_out, _ = self.lstm(inp)

      nparams = int(self.output_dim/2)
      if only_last:
        pred = self.output_fn(lstm_out[-1])
      else:      
        pred = self.output_fn(lstm_out)

      pred = torch.cat([pred[..., :nparams], self.act(pred[..., nparams:])], axis=-1)

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
    #return torch.sum( (x - (mu+var))**2 + (x - (mu-var))**2)
    return torch.sum( (x - mu)**2)

def train_fn(device):
    np.random.seed(0)
    cfg = Config()

    if cfg.style == 'raw':
        training_set = Dataset(cfg, cfg.data_path, cfg.experiments)
    elif cfg.style == 'pca':
        training_set = DatasetPCA(cfg, cfg.data_path, cfg.experiments)

    training_generator = torch.utils.data.DataLoader(training_set, **cfg.train_params)

    # set up the model
    model = ARLSTM(embedding_dim=cfg.embedding_dim, hidden_dim=cfg.hidden_dim, output_dim=2*cfg.output_dim)
    
    loss_fn = iso_het_loss
    #loss_fn = lsqr

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    if False:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # just for the large model
    #model.load_state_dict(torch.load(os.path.join(cfg.save_path, '{}_{}_T{}_S{}.pth'.format(cfg.model_name, cfg.model_type, cfg.n_timesteps, cfg.style))))
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

          if cfg.per_timestep_supervision:          
            xs = batch.clone().transpose(1,0)
            xs[:-1, :, :] = xs[1:, :, :]
            xs[-1, :, :] = lab
            mu, var = y[..., :cfg.output_dim], y[..., cfg.output_dim:]
            loss = loss_fn(xs, mu, var)
          else:
            mu, var = y[-1, :, :cfg.output_dim], y[-1, :, cfg.output_dim:]
            loss = loss_fn(lab, mu, var)

          loss.backward()
          optimizer.step()
          losses.append(loss.detach().item())

       #scheduler.step()
       per_epoch_loss.append(np.mean(losses))
       print('Epoch:{} | Batch: {}/{} | loss: {}'.format(epoch, cnt, training_generator.__len__(), per_epoch_loss[-1]))

    # saving model
    torch.save(model.state_dict(), os.path.join(cfg.save_path, 'rodent{}_{}_T{}_S{}.pth'.format(cfg.exp, cfg.model_type, cfg.n_timesteps, cfg.style)))


def gen_sequence(model, seq, len_sample_traj):

    seq2 = seq.clone()
    pred_seq = list(seq.cpu().numpy().flatten())

    mu, sd = [], []
    with torch.no_grad():
        for k in tqdm.tqdm(range(len_sample_traj)):
            pred = model(seq)
            
            # sample based on this distribution
            next_token = torch.normal(pred[:,0], pred[:,1])
            #next_token = pred[:,0]

            sd.append(pred[:, 1])
            mu.append(pred[:, 0])
            pred_seq.append(next_token)
            
            # update seq
            seq2[:, :-1] = seq[:, 1:].data
            seq2[:, -1] = next_token
            seq = seq2.clone()

    return pred_seq

def gen_sequence_pca(model, seq, len_sample_traj, nparam):

    seq2 = seq.clone()
    pred_seq = list(seq.cpu().numpy())

    with torch.no_grad():
        for k in tqdm.tqdm(range(len_sample_traj)):
            pred = model(seq, only_last=True, reset_hidden=True)
            
            # sample based on this distribution
            next_token = torch.normal(pred[0, :nparam], pred[0, nparam:])
            pred_seq.append(next_token.detach().cpu().numpy())
            
            # update seq
            seq2[:, :-1] = seq[:, 1:].data
            seq2[:, -1] = next_token
            seq = seq2.clone()

    return np.vstack(pred_seq)

def inverse_transform(idx, pca, rtrajectory, syn_trajectory):
    rval = pca.inverse_transform(rtrajectory[idx])
    sval = pca.inverse_transform(syn_trajectory[idx])
    plt.figure()
    plt.plot(rval, c='r', label='recorded')
    plt.plot(sval, c='k', label='simulated')
    plt.legend()
    plt.grid()
    plt.show()

def generate(device):
    np.random.seed(0)
    cfg = Config()
    
    if cfg.style == 'raw':
        validation_set = Dataset(cfg, cfg.data_path, [cfg.test_experiment])
    elif cfg.style == 'pca':
        validation_set = DatasetPCA(cfg, cfg.data_path, cfg.experiments)

    model = ARLSTM(embedding_dim=cfg.embedding_dim, hidden_dim=cfg.hidden_dim, output_dim=2*cfg.output_dim)

    model.to(device)
    model.load_state_dict(torch.load(os.path.join(cfg.save_path, 'rodent{}_{}_T{}_S{}.pth'.format(cfg.exp, cfg.model_type, cfg.n_timesteps, cfg.style))))
    model.eval()

    # pick a random sample
    time_window = cfg.n_timesteps 
    synthetic_trajectory_length = 5 * 600

    if cfg.style == 'raw':
        '''
        arr = S.loadmat(os.path.join(cfg.data_path, cfg.test_experiment))
        if cfg.exp == 'PFC':
            arr = arr[dataset.strip('.mat').replace('LFP', 'lfp')][0][0][1]
        elif cfg.exp == 'RSC':
            arr = arr['lfp'][0][0][1]
        arr = butter_lowpass_filter(arr.squeeze(), cfg.cutoff, cfg.fs, cfg.order)
        arr = arr.astype(np.float32)

        start_idx = 100000
        init_seq = torch.tensor(arr[start_idx : start_idx+time_window]).unsqueeze(0).to(device)
        real_trajectory = arr[start_idx: start_idx+time_window+synthetic_trajectory_length]
        rtrajectory = copy.deepcopy(real_trajectory)
        '''

        start_idx = validation_set.rnd_idx[0]
        init_seq = torch.tensor(validation_set.arrs[start_idx][:time_window]).unsqueeze(0).to(device)
        real_trajectory = validation_set.arrs[start_idx][:time_window+synthetic_trajectory_length]
        rtrajectory = copy.deepcopy(real_trajectory)
        
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(rtrajectory, c = 'b', alpha=0.75, label='original')

        for k in range(1):
            syn_trajectory = gen_sequence(model, init_seq, synthetic_trajectory_length)
            ax.plot(syn_trajectory, c = 'r', alpha=0.75)

        plt.legend()
        plt.show()

    elif cfg.style == 'pca':
        '''        
        idx = 0
        arr_len = len(validation_set.bouts[validation_set.cfg.model_type][idx])
        start_idx = 5000 #np.random.randint(arr_len)
        start_idx = int(validation_set.bouts[cfg.model_type][idx][start_idx]/2)
        init_seq = torch.tensor(validation_set.arrs[idx][start_idx: start_idx+time_window, :]).unsqueeze(0).to(device)
        real_trajectory = validation_set.arrs[idx][start_idx:start_idx+time_window+synthetic_trajectory_length, :]
        
        '''
        start_idx = validation_set.rnd_idx[0]
        init_seq = torch.tensor(validation_set.arrs[start_idx][:time_window, :]).unsqueeze(0).to(device)
        real_trajectory = validation_set.arrs[start_idx][:time_window+synthetic_trajectory_length, :]
        
        rtrajectory = copy.deepcopy(real_trajectory)
        syn_trajectory = gen_sequence_pca(model, init_seq, synthetic_trajectory_length, cfg.output_dim)
       
        lfp = validation_set.pca.inverse_transform(rtrajectory)
        syn = validation_set.pca.inverse_transform(syn_trajectory)
        #pickle.dump({'lfp': lfp, 'syn':syn}, open('data/syn_sleep/{}_T{}_I{}.p'.format(cfg.exp, cfg.n_timesteps, start_idx), 'wb'))
        plt.plot(lfp[:, -1]); plt.plot(syn[:, -1]); plt.show()
        
        #pickle.dump({'lfp': rtrajectory, 'syn':syn_trajectory}, open('data/syn_sleep/awake_trajectory.p', 'wb'))

        import ipdb; ipdb.set_trace()
        
        fig = plt.figure()
        ax1 = fig.add_subplot(121,projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')

        for i in range(10, synthetic_trajectory_length):
            ax1.clear()
            ax2.clear()

            ax1.plot(rtrajectory[i-10:i+10,0], rtrajectory[i-10:i+10,1], rtrajectory[i-10:i+10,2])
            ax1.set_xlim([-10, 10])
            ax1.set_ylim([-10, 10])
            ax1.set_zlim([-10, 10])
            ax1.grid()

            ax2.plot(syn_trajectory[i-10:i+10,0], syn_trajectory[i-10:i+10,1], syn_trajectory[i-10:i+10,2])
            ax2.set_xlim([-10, 10])
            ax2.set_ylim([-10, 10])
            ax2.set_zlim([-10, 10])
            ax2.grid()
            plt.pause(0.01)

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
