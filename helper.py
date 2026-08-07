import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np

from manipulation import adjust_frequencies_in_time
from file_operations import read_mono


# Helper function for graphing
def plot(x, y, title, xlabel, ylabel, xmax=25000):
    plt.figure(figsize=(10, 4))
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.plot(x, y)
    
    if xmax != 25000:
        plt.xlim(0, xmax)

    plt.tight_layout()
    plt.show()

# Convert song to mono and place it in both channels
def convert_to_mono(input_file_name, output_file_name):
    data, f, sample_rate = read_mono(input_file_name)

    combined = np.stack((data.copy(), data.copy()), axis=1)
    audio_data = np.int16(np.clip(combined, -32768, 32767.0))

    return audio_data, f, sample_rate

# Convert song to mono and place it in the left channel
def convert_to_single_channel(input_file_name, output_file_name):
    data, f, sample_rate = read_mono(input_file_name)

    zero_data = np.zeros_like(data.copy())
    combined = np.stack((data.copy(), zero_data), axis=1)
    audio_data = np.int16(np.clip(combined, -32768, 32767.0))

    return audio_data, f, sample_rate

# Split song into left and right channels based on frequency band
def split(input_file_name, output_file_name):
    data, f, sample_rate = read_mono(input_file_name)

    # Apply HPF to isolate highs in the left channel 
    left = adjust_frequencies_in_time(sample_rate=sample_rate, time_signal=data.copy(), min_freq=1000, filter='hpf', filter_order=4)

    # Apply HPF to isolate < 1khz in the right channel
    right = adjust_frequencies_in_time(sample_rate=sample_rate, time_signal=data.copy(), max_freq=1000, filter='lpf', filter_order=4)

    combined = np.stack((left, right), axis=1)
    audio_data = np.int16(np.clip(combined, -32768, 32767.0))

    return audio_data, f, sample_rate