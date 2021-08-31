import numpy as np
from sklearn.decomposition import PCA
import scipy.io as S
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from utils import butter_lowpass_filter
from config import Config

cfg = Config()

arr = S.loadmat('data/PFC_LFP_rat1.mat')
arr = arr['PFC_lfp_rat1'][0][0][1][0] * 1e-3
arr = butter_lowpass_filter(arr, cfg.cutoff, cfg.fs, cfg.order)

time_window = 100
half_window = int(time_window/2)
n_dp = arr.shape[0]

start_idx = 250000

pts = np.vstack([arr[j - half_window: j + half_window] for j in range(half_window, n_dp - half_window)])

pca = PCA(n_components=3)
t_pts = pca.fit_transform(pts)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

'''
ax.plot(t_pts[:10,0], t_pts[:10,1], t_pts[:10,2])
ax.set_xlim([-400, 400])
ax.set_ylim([-400, 400])
ax.set_zlim([-400, 400])
ax.grid(False)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_xticks([-400, 400], ['-400', '400'])
'''

#def update(i):
for i in range(start_idx, start_idx+1200):
    ax.clear()
    ax.plot(t_pts[start_idx+500: start_idx+600, 0], t_pts[start_idx+500: start_idx+600, 1], t_pts[start_idx+500: start_idx+600, 2], c='r')
    ax.plot(t_pts[i-10:i+10,0], t_pts[i-10:i+10, 1], t_pts[i-10:i+10, 2])
    ax.set_xlim([-4, 4])
    ax.set_ylim([-4, 4])
    ax.set_zlim([-4, 4])

    ax.grid(False)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')

    plt.savefig('figures/intropca/%06d.png'%(i-start_idx), bbox_inches='tight')
    #return ax

#anim = FuncAnimation(fig, update, frames=np.arange(start_idx, start_idx+1500), interval=5)
#anim.save('gifs/pfc_trajectory.gif', dpi=70, writer='imagemagick')
#plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(arr[250000+500 : 250000+800], c='r')
ax.set_ylabel('LFP (in mv)')
ax.set_xlabel('Time (in ms)')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.show()

