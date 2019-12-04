import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import saturation as sat
from scipy.signal import welch
import sounddevice as sd
import scipy.io.wavfile as sio
from scipy.signal import decimate, resample
from customwidgets import *

def psd(arr, fs):
    N = 1024
    f, Pxx = welch(arr, fs=fs, window='hamming', nperseg=N, noverlap=N/2, nfft=N, return_onesided=True, scaling='spectrum')
    return Pxx, f

# -- init
f0 = 1000.0
fs = 16000.0
amp = 1.0
nsamples = int(0.2 * fs)
sig = amp * np.sin(2.0 * 3.1415 * f0 / fs * np.arange(0, nsamples))

s = sat.Saturator()
s.threshold = 0.1
s.ratio = 1000.0
s.symmetry = 1.0
s.update_params()

res = s.process(sig)

# -- plot
fig, ax = plt.subplots(figsize=(6,8))
p1 = plt.subplot(311)
plt.title("Original signal")
nrj, w = psd(res, fs)
l1, = plt.semilogy(w, nrj, lw=1)
l1bis = plt.fill_between(w, nrj, -10000.0 * np.ones(len(nrj)), color=(0.0, 0.0, 1.0, 0.15))
plt.ylabel("Magnitude (dB)")
plt.xlabel("Frequencies (Hz)")
plt.grid(linestyle='--', color='gray')
ax.axis('tight')
plt.ylim([1e-15, 1.0])

p2 = plt.subplot(312)
plt.title("Saturated signal with/without AA")
nrj, w = psd(res, fs)
l2, = plt.semilogy(w, nrj, lw=1)
l3 = plt.fill_between(w, nrj, -10000.0 * np.ones(len(nrj)), color=(0.0, 0.0, 1.0, 0.15))

plt.grid(linestyle='--', color='gray')
plt.ylabel("Magnitude (dB)")
plt.xlabel("Frequencies (Hz)")
plt.ylim([1e-15, 1.0])
fig.tight_layout()

# -- commands
axcolor = 'white'
ax_freq = plt.axes([0.2, 0.05, 0.65, 0.03], facecolor=axcolor)
ax_osf = plt.axes([0.2, 0.10, 0.65, 0.03], facecolor=axcolor)
ax_listen = plt.axes([0.4, 0.2, 0.3, 0.03], facecolor=axcolor)

s_freq = Slider(ax_freq, 'Frequency', 0.0, fs/2.0, valinit=100.0, valstep=1)
s_osf = CustomRadioButtons(ax_osf, ('No AA', 'AA x2', 'AA x4', 'AA x16'), active=0, activecolor='crimson', orientation="horizontal")
b_listen = Button(ax_listen, 'Export', color=axcolor, hovercolor='0.8')

def update(val):
    hzdict = {'No AA': 1, 'AA x2': 2, 'AA x4': 4, 'AA x16': 16}
    ds_factor = int(hzdict[s_osf.value_selected])

    f0 = s_freq.val
    sig = amp * np.sin(2.0 * 3.1415 * f0 / fs * np.arange(0, nsamples))

    # -- upsampling
    sig2 = resample(sig, int(len(sig) * ds_factor))

    # -- saturation
    res = s.process(sig2)

    # -- downsampling
    res = decimate(res, ds_factor, ftype='iir', zero_phase=True)

    nrj, w = psd(sig, fs)
    l1.set_ydata(nrj)
    p1.collections.clear()
    p1.fill_between(w, nrj, -10000.0 * np.ones(len(nrj)), color=(0.0, 0.0, 1.0, 0.15))

    nrj, w = psd(res, fs)
    l2.set_ydata(nrj)
    p2.collections.clear()
    p2.fill_between(w, nrj, -10000.0 * np.ones(len(nrj)), color=(0.0, 0.0, 1.0, 0.15))

    fig.canvas.draw_idle()

def export(val):
    sio.write("out.wav", int(fs), res)
    return

# -- tie widgets to actions
s_freq.on_changed(update)
s_osf.on_clicked(update)
b_listen.on_clicked(export)

plt.show()