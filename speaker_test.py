from enum import auto
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import stft, iirfilter
from scipy.fft import fft, fftfreq, fftshift, ifft, rfft, rfftfreq, irfft
import math

def transform(file_name):
    sample_rate, data = wavfile.read(file_name)
    length=len(data)/2
    f = rfftfreq(len(data), 1/sample_rate)

    if len(data.shape) > 1:
        # It's stereo: average the left (channel 0) and right (channel 1) to make it mono
        mono_data = (data[:, 0].astype(float) + data[:, 1].astype(float)) / 2
    else:
        # It's already mono
        mono_data = data.astype(float)

    f_transform = rfft(mono_data)
    amplitude = np.abs(f_transform) / length


    return data, f_transform, amplitude, f, sample_rate

def plot(x, y, title, xlabel, ylabel):
    plt.figure(figsize=(10, 4))
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.plot(x, y)
    plt.tight_layout()
    plt.show()

def plot_freqs(x, y, title, xlabel, ylabel):
    plt.figure(figsize=(10, 4))
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.xlim(0, 250)
    #plt.semilogx(x, y)
    plt.plot(x, y)
    plt.tight_layout()
    plt.show()

room_data, room_f_transform, room_amplitude, room_f, room_sample_rate = transform('room.wav')
plot_freqs(room_f, room_amplitude,  'Room Noise - Magnitude vs Frequencies', 'Frequency [Hz]', 'Magnitude')

phone_data, phone_f_transform, phone_amplitude, phone_f, phone_sample_rate = transform('phone_recording.wav')
plot_freqs(phone_f, phone_amplitude,  'Phone - Magnitude vs Frequencies', 'Frequency [Hz]', 'Magnitude')