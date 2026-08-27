from helper import plot, convert_to_single_channel
from file_operations import read_mono
from manipulation import time_to_frequency
from scipy.io import wavfile
import numpy as np

# # Get single channel mono to test speaker
data, audio_data, f, sample_rate = convert_to_single_channel('original_songs/hotel_california.wav')
plot(f, time_to_frequency(data), "Frequency domain", "Frequency (Hz)", "Magnitude")
# wavfile.write('single_channel_songs/hotel_california_single_channel.wav', sample_rate, audio_data)
# Compare frequency spectrum of single channel and recording
original_data, f, sample_rate = read_mono('calibration/hotel_california_snip.wav')
recording_data, recording_f, sample_rate = read_mono('calibration/hotel_california_recording.wav')
plot(f, time_to_frequency(original_data), "Frequency domain (10s)", "Frequency (Hz)", "Magnitude")
plot(recording_f, time_to_frequency(recording_data), "Frequency domain (10s)", "Frequency (Hz)", "Magnitude")


# Test frequency spectrum
# Should show: 20, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700
data, f, sample_rate = read_mono('calibration/bass_frequencies.wav')
amplitude = time_to_frequency(data)
# amplitude = np.where(amplitude > 30, amplitude, 0)
plot(f, amplitude, "Frequency domain - Bass Frequencies", "Frequency (Hz)", "Magnitude", 1000)
