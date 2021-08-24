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
arr = arr['PFC_lfp_rat1'][0][0][1][0]
arr = butter_lowpass_filter(arr, cfg.cutoff, cfg.fs, cfg.order)

time_window = 100
half_window = int(time_window/2)
n_dp = arr.shape[0]

pts = np.vstack([arr[j - half_window: j + half_window] for j in range(half_window, n_dp - half_window)])

pca = PCA(n_components=3)
t_pts = pca.fit_transform(pts)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.plot(t_pts[:10,0], t_pts[:10,1], t_pts[:10,2])
ax.set_xlim([-400, 400])
ax.set_ylim([-400, 400])
ax.set_zlim([-400, 400])
ax.grid(False)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_xticks(
              [-400, 400], 
              ['-400', '400'])


def update(i):
    ax.clear()
    ax.plot(t_pts[i-10:i+10,0], t_pts[i-10:i+10, 1], t_pts[i-10:i+10, 2])
    ax.set_xlim([-400, 400])
    ax.set_ylim([-400, 400])
    ax.set_zlim([-400, 400])

    ax.grid(False)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')

    ax.set_xticks(
              [-400, 400], 
              ['-400', '400'])

    return ax

anim = FuncAnimation(fig, update, frames=np.arange(10, 2000), interval=5)
anim.save('gifs/pfc_trajectory.gif', dpi=80, writer='imagemagick')

plt.show()
