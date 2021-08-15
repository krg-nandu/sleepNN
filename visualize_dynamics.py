import numpy as np
from sklearn.decomposition import PCA
import scipy.io as S
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

arr = S.loadmat('data/PFC_LFP_rat1.mat')
arr = arr['PFC_lfp_rat1'][0][0][1][0]

time_window = 16
half_window = int(time_window/2)
n_dp = arr.shape[0]

pts = np.vstack([arr[j - half_window: j + half_window] for j in range(half_window, n_dp - half_window)])

#import ipdb; ipdb.set_trace()
pca = PCA(n_components=3)
t_pts = pca.fit_transform(pts)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.plot(t_pts[:1000,0], t_pts[:1000, 1], t_pts[:1000, 2])
plt.show()
