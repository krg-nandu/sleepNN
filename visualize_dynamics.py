import numpy as np
from sklearn.decomposition import PCA
import scipy.io as S
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

arr = S.loadmat('data/PFC_LFP_rat1.mat')
arr = arr['PFC_lfp_rat1'][0][0][1][0]

time_window = 16
half_window = int(time_window/2)
n_dp = arr.shape[0]

pts = np.vstack([arr[j - half_window: j + half_window] for j in range(half_window, n_dp - half_window)])

pca = PCA(n_components=3)
t_pts = pca.fit_transform(pts)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.plot(t_pts[:10,0], t_pts[:10,1], t_pts[:10,2])
ax.set_xlim([-200, 200])
ax.set_ylim([-200, 200])
ax.set_zlim([-200, 200])
ax.grid(False)

def update(i):
    ax.clear()
    ax.plot(t_pts[i-10:i+10,0], t_pts[i-10:i+10, 1], t_pts[i-10:i+10, 2])
    ax.set_xlim([-200, 200])
    ax.set_ylim([-200, 200])
    ax.set_zlim([-200, 200])
    ax.grid(False)
    return ax

anim = FuncAnimation(fig, update, frames=np.arange(10, 100), interval=5)
anim.save('gifs/random_clip1.gif', dpi=80, writer='imagemagick')

plt.show()
