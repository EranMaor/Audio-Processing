from enum import auto
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import stft
from scipy.fft import fft, fftfreq, fftshift, irfft


def transform(file_name):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    right_limit = 23000


    sample_rate, data = wavfile.read(file_name)
    left_data = data[:, 0]
    right_data = data[:, 1]
    samples = range(100000,101000)
    f_transform = fft(left_data)
    f = fftfreq(len(left_data), 1/sample_rate)
    amplitude = np.abs(f_transform) / (len(left_data) / 2)
    print(data.size)

    axs[0].set_title(file_name + ' - Amplitude vs Frequencies (pre-shift)')
    axs[0].set_ylabel('Amplitude')
    axs[0].set_xlabel('Frequency [Hz]')
    axs[0].set_xlim(left=0, right=right_limit)
    axs[0].plot(f, amplitude)

    f_transform[0] = 0
    amplitude = np.abs(f_transform) / (len(left_data) / 2)

    axs[1].set_title(file_name + ' - Amplitude vs Frequencies')
    axs[1].set_ylabel('Amplitude')
    axs[1].set_xlabel('Frequency [Hz]')
    axs[1].set_xlim(left=0, right=right_limit)
    axs[1].plot(f,amplitude)
    plt.tight_layout()  # Prevents overlapping text
    plt.show()

    return f_transform, amplitude

f_transform, amplitude = transform('a_milli.wav')



