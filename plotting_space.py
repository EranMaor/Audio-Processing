from enum import auto
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import stft
from scipy.fft import fft, fftfreq, fftshift, ifft

y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 100, 10])
x = range(10)
plt.plot(x, y)
plt.show()