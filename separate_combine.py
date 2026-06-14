from enum import auto
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import stft
from scipy.fft import fft, fftfreq, fftshift, ifft


sample_rate, data = wavfile.read('staying_alive.wav')
left_data = data[:, 0]
right_data = data[:, 1] 

reconstructed = np.stack((left_data, right_data), axis=1)
wavfile.write('unedited_reconstructed.wav', sample_rate, reconstructed)
