import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz
from sklearn.mixture import GaussianMixture

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
    a = 1
    y = lfilter(b, a, data)
    return y

def extract_sleep_bouts(arr, cfg, make_plot=True):
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
        plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
        plt.plot(cfg.cutoff, 0.5*np.sqrt(2), 'ko')
        plt.axvline(cfg.cutoff, color='k')
        plt.xlim(0, 0.5*fs)
        plt.title("Lowpass Filter Frequency Response")
        plt.xlabel('Frequency [Hz]')
        plt.grid()

        plt.subplot(2, 1, 2)
        plt.hist(power[power <= thr], bins=500, range=(np.quantile(power, 0.01), np.quantile(power, 0.99)), c='b', label='wake')
        plt.hist(power[power > thr], bins=500, range=(np.quantile(power, 0.01), np.quantile(power, 0.99)), c='r', label='sleep')

        plt.grid()
        plt.legend()
        plt.subplots_adjust(hspace=0.35)
        plt.show()

    if cfg.model_type == 'asleep':
        idx = np.where(power >= thr)[0]
    elif cfg.model_type == 'awake':
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
        plt.show()

    return y, bouts
