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
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    a = 1
    y = lfilter(b, a, data)
    return y

def func(arr):
    # Filter requirements.
    order = 6
    fs = 600.0       # sample rate, Hz
    cutoff = 6.  # desired cutoff frequency of the filter, Hz

    # Get the filter coefficients so we can check its frequency response.
    b, a = butter_lowpass(cutoff, fs, order)

    # Plot the frequency response.
    w, h = freqz(b, a, worN=8000)
    plt.subplot(2, 1, 1)
    plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
    plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
    plt.axvline(cutoff, color='k')
    plt.xlim(0, 0.5*fs)
    plt.title("Lowpass Filter Frequency Response")
    plt.xlabel('Frequency [Hz]')
    plt.grid()

    '''
    # Demonstrate the use of the filter.
    # First make some data to be filtered.
    T = 5.0         # seconds
    n = int(T * fs) # total number of samples
    t = np.linspace(0, T, n, endpoint=False)
    # "Noisy" data.  We want to recover the 1.2 Hz signal from this.
    data = np.sin(1.2*2*np.pi*t) + 1.5*np.cos(9*2*np.pi*t) + 0.5*np.sin(12.0*2*np.pi*t)
    '''

    # Filter the data, and plot both the original and filtered signals.
    y = butter_lowpass_filter(arr, cutoff, fs, order)

    plt.subplot(2, 1, 2)
    #plt.plot(arr, 'b-', label='data')
    #plt.plot(y, 'g-', linewidth=2, label='filtered data')
    #plt.xlabel('Time [sec]')
    power = np.convolve(y**2, np.ones(600,dtype=int),'valid')
    plt.hist(power, bins=1000, range=(np.quantile(power, 0.01), np.quantile(power, 0.99)))
    plt.grid()
    plt.legend()

    plt.subplots_adjust(hspace=0.35)
    plt.show()


class Dataset(torch.utils.data.Dataset):
    def __init__(self, base_path, exp_name):
        self.dataset_size = 300000
        self.n_timesteps = 32

        self.base_path = base_path
        self.arrs = []
        for dataset in exp_name:
            arr = S.loadmat(os.path.join(base_path, dataset))
            #arr = arr[dataset.strip('.mat').replace('LFP', 'lfp')][0][0][1]
            arr = arr['lfp'][0][0][1]
            self.arrs.append(arr)

            #import ipdb; ipdb.set_trace()
            func(arr.squeeze())

        self.arrs = np.vstack(self.arrs).astype(np.float32)

        # z score?
        self.mean = np.mean(self.arrs)
        self.std = np.std(self.arrs)

        self.arrs = (self.arrs - np.mean(self.arrs))/np.std(self.arrs)
        self.n_datapoints, self.n_duration = self.arrs.shape[0], self.arrs.shape[1]

        print('Preparing train-test splits...')
        self.rnd_idx, self.rnd_start = np.random.randint(self.n_datapoints, size=(self.dataset_size,)), np.random.randint(self.n_duration-self.n_timesteps-1, size=(self.dataset_size,))
 
    def __len__(self):
        'Denotes the total number of samples'
        return self.rnd_idx.shape[0]

    def __getitem__(self, index):
        'Generates one sample of data'
        return self.arrs[self.rnd_idx[index], self.rnd_start[index]:self.rnd_start[index]+self.n_timesteps], self.arrs[self.rnd_idx[index], self.rnd_start[index]+self.n_timesteps]

class ARLSTM(torch.nn.Module):

   def __init__(self, embedding_dim, hidden_dim, output_dim):
      super(ARLSTM, self).__init__()
      self.hidden_dim = hidden_dim

      # The LSTM takes action labels as inputs, and outputs hidden states
      # with dimensionality hidden_dim.
      self.lstm = torch.nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=3)
      self.output_fn = torch.nn.Linear(hidden_dim, output_dim)
      self.act = F.log_softmax

   def forward(self, x):
      lstm_out, _ = self.lstm(x.transpose(-1,0).unsqueeze(-1))
      pred = self.output_fn(lstm_out[-1])
      # is there a good activation function for this? not sure..

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
    return torch.mean(((x - mu) ** 2)/(var ** 2) + 0.5 * torch.log((var ** 2)))

def train_fn(device):
    params = {'batch_size': 1024,
              'shuffle': False,
              'num_workers': 1}

    max_epochs = 100

    #training_set = Dataset('data/', ['PFC_LFP_rat1.mat'])
    training_set = Dataset('data/', ['RSC_LFP_rat3_600Hz.mat'])

    training_generator = torch.utils.data.DataLoader(training_set, **params)

    # set up the model
    model = ARLSTM(embedding_dim=1, hidden_dim=128, output_dim=2)
    
    loss_fn = iso_het_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model.to(device)
    model.train()

    per_epoch_loss = []
    for epoch in range(max_epochs):
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
    torch.save(model.state_dict(), 'ckpts/rodentV0.pth')


def gen_sequence(model, seq, len_sample_traj):

    seq2 = seq.clone()
    pred_seq = list(seq.cpu().numpy().flatten())

    with torch.no_grad():
        for k in tqdm.tqdm(range(len_sample_traj)):
            pred = model(seq)
            
            # sample based on this distribution
            next_token = torch.normal(pred[:,0], pred[:,1])
            pred_seq.append(next_token)
            
            # update seq
            seq2[:, :-1] = seq[:, 1:].data
            seq2[:, -1] = next_token
            seq = seq2.clone()

    return pred_seq

'''
Sample a bunch of synthetic action trajectories from the autoreg LSTM
'''
def simulate_seq(model, val_set, filename, r_idx=0, len_sample_traj=500, n_repeats=100):
    seq = val_set.preds[val_set.rnd_idx[r_idx], val_set.rnd_start[r_idx]:val_set.rnd_start[r_idx]+16]
    #seq = torch.tensor([2, 2, 2, 5, 4, 4, 4, 4, 5, 3, 5, 5, 5, 2, 5, 2])
    #seq = seq.unsqueeze(0).float().to(device)
    seq = torch.tensor(seq).unsqueeze(0).float().to(device)

    fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    ax = fig.add_subplot(111)

    sequences = np.zeros((n_repeats+1, len_sample_traj+16))    
    sequences[0,:] = val_set.preds[val_set.rnd_idx[r_idx], val_set.rnd_start[r_idx]:val_set.rnd_start[r_idx]+16+len_sample_traj]


    for k in range(n_repeats):
        pred_seq = gen_sequence(model, seq, len_sample_traj)
        pred_seq = torch.tensor(pred_seq).unsqueeze(-1).unsqueeze(-1).to(device)
        zs, _ = model.lstm(pred_seq)
    
        sequences[k+1, :] = pred_seq.cpu().numpy().squeeze()

        #traj = [x.cpu().data.numpy().squeeze() for x in zs]
        #traj = np.vstack(traj)
        #ax.plot(traj[:,0], traj[:,1], traj[:,2])

    cmap = ax.imshow(sequences, interpolation='none', cmap='Set1')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(cmap, cax=cax)

    cbar.set_ticks(np.arange(9))
    cbar.set_ticklabels(BEH_LABELS)
    plt.setp(cbar.ax.get_yticklabels(), fontsize=8)

    ax.set_xlabel('time')
    ax.set_xticks([0, len_sample_traj])
    ax.set_xticklabels(['0s', '%0.2fs'%(len_sample_traj/30)])
    ax.set_yticks([0])
    ax.set_yticklabels(['original'])
    #ax.set_ylabel('simulated', fontsize=8)
    #plt.axis('off')

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    plt.savefig(os.path.join('simulated_seqs', filename))
    plt.close()


def generate(device, save_path):
    np.random.seed(0)
    params = {'batch_size': 32,
              'shuffle': False,
              'num_workers': 1}

    # this is a bad way, but doing this just to keep track of the params of the distribution
    training_set = Dataset('data/', ['PFC_LFP_rat1.mat'])
    #import ipdb; ipdb.set_trace()
 
    model = ARLSTM(embedding_dim=1, hidden_dim=128, output_dim=2)
    model.to(device)
    model.load_state_dict(torch.load(save_path))
    model.eval()

    arr = S.loadmat('data/PFC_LFP_rat1.mat')
    arr = (arr['PFC_lfp_rat1'][0][0][1][0].astype(np.float32) - training_set.mean)/training_set.std

    # pick a random sample
    start_idx = 1000
    time_window = 32 

    init_seq = torch.tensor(arr[start_idx : start_idx+time_window]).unsqueeze(0).to(device)
    synthetic_trajectory_length = 2048
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
        generate(device, 'ckpts/rodentV0.pth')
    else:
        raise NotImplementedError
