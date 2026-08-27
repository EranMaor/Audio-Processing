import matplotlib.pyplot as plt
import numpy as np

from file_operations import read_mono


# Helper function for graphing
def plot(x, y, title, xlabel, ylabel, xmax=None):
    plt.figure(figsize=(10, 4))
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.plot(x, y)
    
    if xmax:
        plt.xlim(0, xmax)

    plt.tight_layout()
    plt.show()

# Convert song to mono and place it in both channels
# Returns: mono 1D float (for plots/FFT), stereo int16 (for writing WAV), freqs, sample_rate
def convert_to_mono(input_file_name):
    data, f, sample_rate = read_mono(input_file_name)

    combined = np.stack((data.copy(), data.copy()), axis=1)
    audio_data = np.int16(np.clip(combined, -32768, 32767.0))

    return data, audio_data, f, sample_rate

# Convert song to mono and place it in the left channel
def convert_to_single_channel(input_file_name):
    data, f, sample_rate = read_mono(input_file_name)
    zero_data = np.zeros_like(data.copy())

    combined = np.stack((data.copy(), zero_data), axis=1)
    audio_data = np.int16(np.clip(combined, -32768, 32767.0))

    return data, audio_data, f, sample_rate
