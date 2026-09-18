import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.fft import rfftfreq

from file_operations import read_mono, write_file
from manipulation import time_to_frequency, nld, filter as freq_filter
from helper import plot

mono_data, _, sample_rate = read_mono("original_songs/best_part.wav")
write_file(mono_data, mono_data, sample_rate, "mono_songs/best_part_mono.wav")

start_sec = 93
end_sec = 103
start_sample = int(start_sec * sample_rate)
end_sample = int(end_sec * sample_rate)
segment = mono_data[start_sample:end_sample]

output_file_name = f"segmented_songs/best_part_{start_sec}-{end_sec}.wav"
write_file(segment, segment, sample_rate, output_file_name)

single_channel_file_name = f"single_channel_songs/best_part_{start_sec}-{end_sec}_single_channel.wav"
write_file(segment, np.zeros_like(segment), sample_rate, single_channel_file_name)


highs = freq_filter(sample_rate=sample_rate, time_signal=segment.copy(), min_freq=100, filter='hpf', filter_order=4)
hpf_file_name = f"hpf_songs/best_part_{start_sec}-{end_sec}_hpf.wav"
write_file(highs, np.zeros_like(highs), sample_rate, hpf_file_name)

left, right, _, sample_rate = nld(output_file_name, speaker_threshold_freq=100, lower_threshold_freq=40, bin_band=10, a=0, b=1.5, c=0.4, B=[1,1,1,1,1,1], w=[1,1,1,1,1,1], device_type='f_p')
enhanced_file_name = f"enhanced_songs/best_part_{start_sec}-{end_sec}_harmonics.wav"
write_file(left, right, sample_rate, enhanced_file_name)
