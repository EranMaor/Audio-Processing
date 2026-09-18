from scipy.io import wavfile
import numpy as np

from manipulation import nld, time_to_frequency, split
from helper import plot, convert_to_mono, convert_to_single_channel

# CONSTANTS
SONG_NAME = 'Best Part (10 sec)'
FILE_NAME = 'best_part_93-103'
SPEAKER_THRESHOLD_FREQ = 500
A = 0
B = 1.5
C = 0.4

input_file_name = f'segmented_songs/{FILE_NAME}.wav'
single_channel_file_name = f'single_channel_songs/{FILE_NAME}_single_channel.wav'
mono_file_name = f'mono_songs/{FILE_NAME}_mono.wav'
split_file_name = f'split_songs/{FILE_NAME}_split.wav'
output_file_name = f'enhanced_songs/{FILE_NAME}_harmonics.wav'

# 1) Get single channel mono for analysis + stereo-mono WAV for listening
#    data      = 1D float array  -> use for plots / FFT
#    audio_data = (N, 2) int16   -> same mono in L and R, for file writing
data, single_channel_audio_data, f, sample_rate = convert_to_single_channel(input_file_name)
wavfile.write(single_channel_file_name, sample_rate, single_channel_audio_data)
_, mono_audio_data, _, sample_rate = convert_to_mono(input_file_name)
wavfile.write(mono_file_name, sample_rate, mono_audio_data)

# 2) Split mono into left and right channels
# Left channel is mids/highs, right channel is lows
split_left, split_right, split_audio_data, _, sample_rate = split(input_file_name, SPEAKER_THRESHOLD_FREQ*2)
wavfile.write(split_file_name, sample_rate, split_audio_data)

# 3) Apply harmonics enhancement and write enhanced file
left, right, _, sample_rate = nld(input_file_name, speaker_threshold_freq=100, lower_threshold_freq=40, bin_band=10, a=0, b=1.5, c=0.4, B=[1,1,1,1,1,1], w=[1,1,1,1,1,1], device_type='f_p')
combined = np.stack((left, right), axis=1)
enhanced_data = np.int16(np.clip(combined, -32768, 32767.0))
wavfile.write(output_file_name, sample_rate, enhanced_data)

# Calculations for graph display
duration = len(data) / sample_rate
time = np.linspace(0.0, duration, num=len(data))

# 4) Graph original mono vs enhanced channels (time domain)
plot(time, data, f'ORIGINAL: {SONG_NAME} MONO AMPLITUDE vs TIME', 'time(s)', 'Amplitude')
plot(time, enhanced_data[:, 0].astype(float), 'ENHANCED MIDS/HIGHS AMPLITUDE vs TIME', 'time(s)', 'Amplitude')
plot(time, enhanced_data[:, 1].astype(float), 'ENHANCED LOWS AMPLITUDE vs TIME', 'time(s)', 'Amplitude')

# 5) Compare frequencies for original and enhanced song
#    Graph original mono vs enhanced channels (frequency domain)
original_amplitude = time_to_frequency(data)
enhanced_left_amplitude = time_to_frequency(enhanced_data[:, 0].astype(float))
enhanced_right_amplitude = time_to_frequency(enhanced_data[:, 1].astype(float))
plot(f, original_amplitude, f'Original: {SONG_NAME} - Magnitude vs Frequency', 'Frequency (Hz)', 'Magnitude')
plot(f, enhanced_left_amplitude, f'MIDS/HIGHS: {SONG_NAME} - Magnitude vs Frequency', 'Frequency (Hz)', 'Magnitude')
plot(f, enhanced_right_amplitude, f'LOWS: {SONG_NAME} - Magnitude vs Frequency', 'Frequency (Hz)', 'Magnitude')
