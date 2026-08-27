from helper import plot
from manipulation import time_to_frequency
from file_operations import read_mono

noise_data, noise_f, noise_sample_rate = read_mono('speaker_noise.wav')
plot(noise_f, time_to_frequency(noise_data), "Speaker Noise - Magnitude vs Frequency", "Frequency (Hz)", "Magnitude")