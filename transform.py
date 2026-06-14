from enum import auto
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft
from scipy.fft import fft, fftfreq, fftshift, irfft

def transform(file_name):
    sample_rate, data = wavfile.read(file_name)
    left_data = data[:, 0]
    right_data = data[:, 1]
    
    f_transform_left = fft(left_data)
    f_left = fftfreq(len(left_data), 1/sample_rate)
    amplitude_left = np.abs(f_transform_left) / (len(left_data) / 2)

    f_transform_right = fft(right_data)
    f_right = fftfreq(len(right_data), 1/sample_rate)
    amplitude_right = np.abs(f_transform_right) / (len(right_data) / 2)

    return left_data, f_transform_left, f_left, amplitude_left, right_data, f_transform_right, f_right, amplitude_right

def remove_dc_offset(f_transform, data):
    f_transform[0] = 0
    amplitude = np.abs(f_transform) / (len(data) / 2)