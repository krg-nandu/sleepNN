import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz
from sklearn.mixture import GaussianMixture
import scipy.io as S
from sklearn.decomposition import PCA
import os, pickle

def identify_threshold(power):
    lq = np.quantile(power, 0.01)
    uq = np.quantile(power, 0.99)
    vals = power[np.logical_and(power > lq, power < uq)]

    # generally power values are too low
    vals = np.expand_dims(vals, -1) * 1e+10

    gm = GaussianMixture(n_components=2, random_state=0)
    labs = gm.fit_predict(vals) 

    thr = np.mean(gm.means_) * 1e-10
    return thr

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    #a = 1
    y = lfilter(b, a, data)
    return y

def extract_sleep_bouts(arr, cfg, make_plot=True, force_mode=None):
    # Filter the data, and plot both the original and filtered signals.
    y = butter_lowpass_filter(arr, cfg.cutoff, cfg.fs, cfg.order)
    power = np.convolve(y**2, np.ones(int(cfg.window), dtype=int),'valid')
    thr = identify_threshold(power)

    if make_plot:
        # Get the filter coefficients so we can check its frequency response.
        b, a = butter_lowpass(cfg.cutoff, cfg.fs, cfg.order)

        # Plot the frequency response.
        w, h = freqz(b, a, worN=8000)
        plt.subplot(2, 1, 1)
        plt.plot(0.5 * cfg.fs * w/np.pi, np.abs(h), 'b')
        plt.plot(cfg.cutoff, 0.5 * np.sqrt(2), 'ko')
        plt.axvline(cfg.cutoff, color='k')
        plt.xlim(0, 0.5 * cfg.fs)
        plt.title("Lowpass Filter Frequency Response")
        plt.xlabel('Frequency [Hz]')
        plt.grid()

        plt.subplot(2, 1, 2)
        plt.hist(power[power <= thr], bins=500, range=(np.quantile(power, 0.01), np.quantile(power, 0.99)), color='b', label='wake')
        plt.hist(power[power > thr], bins=500, range=(np.quantile(power, 0.01), np.quantile(power, 0.99)), color='r', label='sleep')

        plt.xlabel('Average Power')
        plt.ylabel('Frequency')

        plt.grid()
        plt.legend()
        plt.subplots_adjust(hspace=0.35)
        plt.show()

    if not(force_mode == None):
        mtype = force_mode
    else:
        mtype = cfg.model_type

    if mtype == 'asleep':
        idx = np.where(power >= thr)[0]
    elif mtype == 'awake':
        idx = np.where(power <= thr)[0]
    else:
        raise NotImplementedError

    changepts = np.where(idx[1:] - idx[:-1] > 1)[0]
    bouts = []
    bouts.append((idx[0], idx[changepts[0]]))
    for k in range(changepts.shape[0]-1):
        bouts.append((idx[changepts[k]+1], idx[changepts[k+1]]))

    join_bouts = []
    cur_bout = bouts[0]

    # consider joining bouts
    for k in range(1, len(bouts)):
        # merge
        if bouts[k][0] - cur_bout[1] <= cfg.merge_thresh:
            cur_bout = (cur_bout[0], bouts[k][1])
        else:
            join_bouts.append(cur_bout)
            cur_bout = bouts[k]

    # prune bouts
    bouts = []
    for k in range(len(join_bouts)):
        if join_bouts[k][1] - join_bouts[k][0] > cfg.min_bout_duration:
            bouts.append(join_bouts[k])

    if make_plot:           
        plt.plot(y)
        for k in range(len(bouts)):
            plt.plot(np.arange(bouts[k][0], bouts[k][1]), y[bouts[k][0]:bouts[k][1]], c='r')
        plt.grid()
        plt.xlabel('Time (in ms)')  
        plt.ylabel('LFP (in mV)')

        plt.show()

    return y, bouts

def make_pcs(cfg):

    half_window = int(cfg.pca_window / 2)
    all_pts = []
    pca = PCA(n_components=3)
    all_bouts = {'asleep': [], 'awake': []}

    for dataset in cfg.experiments:
        print(dataset)
        arr = S.loadmat(os.path.join(cfg.data_path, dataset))
        if cfg.exp == 'PFC':
            arr = arr[dataset.strip('.mat').replace('LFP', 'lfp')][0][0][1] * 1e-3
        elif cfg.exp == 'RSC':
            arr = arr['lfp'][0][0][1]

        # low pass filter the data
        filter_arr = butter_lowpass_filter(arr.squeeze(), cfg.cutoff, cfg.fs, cfg.order)
        filter_arr = filter_arr.astype(np.float32)
        n_dp = filter_arr.shape[0]
        
        pts = np.vstack([filter_arr[j - half_window: j + half_window] for j in range(half_window, n_dp - half_window, 2)])

        all_pts.append(pts)

        _, sleep_bouts = extract_sleep_bouts(arr.squeeze(), cfg, force_mode='asleep')
        sbouts = [y - half_window for x in sleep_bouts for y in range(x[0], x[1]) if y - half_window >= 0]
        all_bouts['asleep'].append(sbouts)

        _, wake_bouts = extract_sleep_bouts(arr.squeeze(), cfg, force_mode='awake')
        wbouts = [y - half_window for x in wake_bouts for y in range(x[0], x[1]) if y - half_window >= 0]
        all_bouts['awake'].append(wbouts)

    my_pts = np.vstack(all_pts)
    
    # build the pca model
    pca = pca.fit(my_pts)
    print(pca.explained_variance_ratio_)
    pickle.dump(pca, open('{}_components.p'.format(cfg.exp), 'wb'))

    pickle.dump(all_bouts, open('{}_bouts.p'.format(cfg.exp),'wb'))

    arr = []
    for k in range(len(all_pts)):
        myp = pca.fit_transform(all_pts[k])
        arr.append(myp)

    pickle.dump(arr, open('{}_transformed.p'.format(cfg.exp), 'wb'))

    return pca, arr, all_bouts
