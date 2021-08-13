'''
Author: Lakshmi
Code skeleton adopted from Masse et. al. 2019
'''

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np

import pickle
import time
from parameters import par
import os, sys

class dRNN(nn.Module):

    def __init__(self, device):
        super(dRNN, self).__init__()

        self.device = device
        # network activities
        self.h = []
        # list of matrices. each matrix describes synapse-specific parameters
        self.syn_x = []
        self.syn_u = []
        # network readouts
        self.y = []
        self.rectify = nn.ReLU()

        self._U = torch.tensor(par['U']).to(device)
        self.alpha_std = torch.tensor(par['alpha_std']).to(device)
        self.alpha_stf = torch.tensor(par['alpha_stf']).to(device)

        # network trainable parameters
        self.W_in = nn.Parameter(torch.empty(par['n_input'], par['n_hidden']))
        self.W_rec = nn.Parameter(torch.empty(par['n_hidden'], par['n_hidden']))
        self.W_bias = nn.Parameter(torch.empty(par['n_hidden']))
        self.W_out = nn.Parameter(torch.empty(par['n_hidden'], par['n_output']))
        self.W_out_b = nn.Parameter(torch.empty(par['n_output']))

        # initialize parameter values
        init.orthogonal_(self.W_in)
        init.orthogonal_(self.W_rec)
        init.orthogonal_(self.W_out)

        init.constant_(self.W_bias, 0.)
        init.constant_(self.W_out_b, 0.)

    '''forward pass through the RNN
    x: TxN_in temporal input
    function returns the latent activities, synaptic states
    '''
    def forward(self, x):
        self.h.clear()
        self.syn_x.clear()
        self.syn_u.clear()
        self.y.clear()

        # initialize these values
        self.h.append(torch.zeros((par['batch_size'], par['n_hidden'])).to(self.device))
        self.syn_x.append(torch.ones(par['batch_size'], par['n_hidden'], par['n_hidden']).to(self.device))
        self.syn_u.append((par['U_std'] + torch.ones(par['batch_size'], par['n_hidden'], par['n_hidden'])).to(self.device))
        for step in range(x.shape[0]):

            h, syn_x, syn_u = self.rnn_step(x[step].float())

            self.h.append(h)
            self.syn_x.append(syn_x)
            self.syn_u.append(syn_u)

            y = self.readout(h)
            self.y.append(y)

        return self.h, self.y, self.syn_x, self.syn_u

    def readout(self, h):
        z = torch.matmul(h, self.W_out) + self.W_out_b
        prob = torch.sigmoid(z)
        return prob

    def rnn_step(self, inp):
        syn_x = self.syn_x[-1]
        syn_u = self.syn_u[-1]
        h = self.h[-1]
        
        syn_x = syn_x + torch.mul(self.alpha_std, 1-syn_x) - (par['dt']/1000.)*torch.mul(torch.mul(syn_u, syn_x), h.unsqueeze(-1).repeat(1,1,par['n_hidden']))
        syn_u = syn_u + torch.mul(self.alpha_stf, self._U - syn_u) + (par['dt']/1000.)*torch.matmul(torch.mul(self._U, 1-syn_u), h.unsqueeze(-1).repeat(1,1,par['n_hidden']))

        #TODO  mask out the non-dynamic synapse factors

        # cap the variables
        syn_x = torch.clamp(syn_x, 0., 1.)
        syn_u = torch.clamp(syn_u, 0., 1.)

        # only the exc input weights are used
        # ignoring noise for now
        dt = par['dt']/par['membrane_time_constant']
        h = self.rectify((1-dt)*h + \
                    dt*(torch.matmul(h.unsqueeze(1), torch.mul(torch.mul(syn_x, syn_u), self.W_rec)).squeeze() + self.W_bias) + \
                    torch.matmul(inp, self.W_in) )
        return h, syn_x, syn_u
