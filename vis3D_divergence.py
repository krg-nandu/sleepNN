import torch
import torch.nn.functional as F
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import glob, tqdm, os
from pathlib import Path 
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.io as S
import copy, pickle
import numpy as np
import matplotlib.pyplot as plt
from config import Config

from utils import extract_sleep_bouts, butter_lowpass_filter, make_pcs
from matplotlib.animation import FuncAnimation 
from mpl_toolkits.axisartist.axislines import SubplotZero

sleep = pickle.load(open('data/syn_sleep/asleep_trajectory.p','rb'))
awake = pickle.load(open('data/syn_sleep/awake_trajectory.p','rb'))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

def update(i): 
    ax.clear()

    ax.plot(sleep['lfp'][i-10:i+10,0], sleep['lfp'][i-10:i+10,1], sleep['lfp'][i-10:i+10,2], c='r', label='LFP')
    ax.plot(sleep['syn'][i-10:i+10,0], sleep['syn'][i-10:i+10,1], sleep['syn'][i-10:i+10,2], c='k', label='model-sleep')
    ax.plot(awake['syn'][i-10:i+10,0], awake['syn'][i-10:i+10,1], awake['syn'][i-10:i+10,2], c='tab:orange', label='model-wake')

    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_zlim([-5, 5])
    ax.set_xlabel('$PC\ 1$')
    ax.set_ylabel('$PC\ 2$')
    ax.set_zlabel('$PC\ 3$')

    ax.legend()

anim = FuncAnimation(fig, update, frames=np.arange(10, 1000), interval=10)
anim.save('gifs/sleep_wake_diverge.gif', dpi=80, writer='imagemagick')
#plt.show()
