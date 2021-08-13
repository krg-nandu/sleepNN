import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BasicAutoEncoder(nn.Module):
    def __init__(self):
        super(BasicAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 4, 3, stride=2, padding=1),
            nn.Conv2d(4, 4, 3, stride=2, padding=1),
            nn.Conv2d(4, 4, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 8, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 5)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 8, 5),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, 3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(4, 4, 3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(4, 4, 3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(4, 2, 3, stride=2, padding=1, output_padding=1),
            #nn.Sigmoid()
        )

    '''
    def __init__(self):
        super(BasicAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 4, 3, stride=2, padding=1),
            nn.Conv2d(4, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8,16,3, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 64, 7)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 16, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(16,8,3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, 3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(4, 2, 3, stride=2, padding=1, output_padding=1),
            #nn.Sigmoid()
        )
    '''

    def forward(self, x):
        #import ipdb; ipdb.set_trace()
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def get_embedding(self, x):
        x = self.encoder(x)
        return x


class BasicAutoEncoderMLP(nn.Module):
    def __init__(self, dims):
        super(BasicAutoEncoderMLP, self).__init__()
        self.dims = dims
        self.encoder = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=np.prod(dims[1:]), out_features=2048),
            nn.ReLU(),
            nn.Linear(in_features=2048, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64)
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=64, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=2048),
            nn.ReLU(),
            nn.Linear(in_features=2048, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=2*np.prod(dims[1:])),
            #nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = torch.reshape(x, [-1, 2, *self.dims[2:]])
        return x
