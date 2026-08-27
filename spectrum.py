import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.fft import rfftfreq

from file_operations import read_mono, write_file
from manipulation import time_to_frequency
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

# Play the segment through the speakers while recording the microphone
playback = np.stack((segment, segment), axis=1).astype(np.float32) / 32768.0
input("Press Enter to play the segment and record the microphone...")
print(f"Playing and recording for {end_sec - start_sec} seconds...")
recording = sd.playrec(playback, samplerate=sample_rate, channels=1)
sd.wait()
print("Recording finished.")

recording = recording.flatten().astype(np.float64) * 32767.0
recording_file_name = f"calibration/best_part_{start_sec}-{end_sec}_recording.wav"
write_file(recording, recording, sample_rate, recording_file_name)

f = rfftfreq(len(segment), 1 / sample_rate)
frequency_data = time_to_frequency(segment)
plot(f, frequency_data, f"Frequency Spectrum ({start_sec}-{end_sec} s)", "Frequency (Hz)", "Amplitude")

recording_f = rfftfreq(len(recording), 1 / sample_rate)
recording_frequency_data = time_to_frequency(recording)
plot(recording_f, recording_frequency_data, f"Recorded Spectrum ({start_sec}-{end_sec} s)", "Frequency (Hz)", "Amplitude")
plt.show()