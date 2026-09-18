import math

from scipy import signal
from scipy.fft import rfft
import numpy as np

from helper import plot
from file_operations import read_mono


# Filtering function
# Uses butterworth IIR filter
# SOS (second-order sections) method breaks down filtration into small pieces to reduce noise
def filter(sample_rate, time_signal, min_freq=0, max_freq=0, filter='hpf', filter_order=1):
    # High pass filter attenuates frequencies below cutoff
    if filter == 'hpf':
        sos = signal.butter(N=filter_order, Wn=min_freq, btype='highpass', fs=sample_rate, output='sos')
        filtered_signal = signal.sosfiltfilt(sos, time_signal)
    # Low pass filter attenuates frequencies above cutoff
    elif filter == 'lpf':
        sos = signal.butter(N=filter_order, Wn=max_freq, btype='lowpass', fs=sample_rate, output='sos')
        filtered_signal = signal.sosfiltfilt(sos, time_signal)
    # Band pass filter attenuates frequencies outside of frequency band defined
    elif filter == 'bpf':
        sos = signal.butter(N=filter_order, Wn=[min_freq, max_freq], btype='bandpass', fs=sample_rate, output='sos')
        filtered_signal = signal.sosfiltfilt(sos, time_signal)
        
    return filtered_signal

# Split song into left and right channels based on frequency band
def split(input_file_name, threshold_freq=1000):
    data, f, sample_rate = read_mono(input_file_name)

    # Apply HPF to isolate highs in the left channel 
    # Apply LPF to isolate < threshold_freq in the right channel
    left = filter(sample_rate=sample_rate, time_signal=data.copy(), min_freq=threshold_freq, filter='hpf', filter_order=4)
    right = filter(sample_rate=sample_rate, time_signal=data.copy(), max_freq=threshold_freq, filter='lpf', filter_order=4)

    combined = np.stack((left, right), axis=1)
    audio_data = np.int16(np.clip(combined, -32768, 32767.0))

    return left, right, audio_data, f, sample_rate

#Helper function to graph FFT amplitudes
def time_to_frequency(data):
    f_transform = rfft(data)
    amplitude = np.abs(f_transform) / len(data)

    return amplitude

# a: DC offset
# b: Produces odd harmonics
# c: Produces even harmonics
def nld(input_file_name, speaker_threshold_freq=100, lower_threshold_freq=40, bin_band=10, a=0, b=1.5, c=0.4, B=[1,1,1,1,1,1], w=[1,1,1,1,1,1], device_type='tanh'):
    data, f, sample_rate = read_mono(input_file_name)

    # Apply BPF to isolate mids/highs in the left channel 
    left = filter(sample_rate=sample_rate, time_signal=data.copy(), min_freq=speaker_threshold_freq, max_freq=20000, filter='bpf', filter_order=4)
    right = np.zeros_like(left)

    for freq_bin in range((speaker_threshold_freq-lower_threshold_freq)//bin_band):
        # Step 1 of bass frequency processing:
        # Apply BPF to each bucket to isolate original lows (within audible range) in right channel
        bpf_freq = filter(sample_rate=sample_rate, time_signal=data.copy(), min_freq=lower_threshold_freq+freq_bin*bin_band, max_freq=(lower_threshold_freq+freq_bin*bin_band+bin_band), filter='bpf', filter_order=4)

        if device_type=='tanh':
            # Step 2 of bass frequency processing:
            # Normalize to range +- k (divide by max amplitude and multiply by k) to optimize harmonic saturation
            max_amplitude = np.max(np.abs(bpf_freq)) or 1
            k = 2.5
            norm_output = bpf_freq / max_amplitude * k
            nld_output = np.empty_like(norm_output)

            # Step 3 of bass frequency processing:
            # Apply NLD to the right channel (non-linear function) to create harmonics
            # Inner term (a + bx + cx^2) produces both even and odd harmonics
            np.tanh((a + b*norm_output + c*np.square(norm_output)), out=norm_output)
        if device_type=='f_p':
            # Step 2 of bass frequency processing:
            # Normalize to range +- Beta_p to optimize harmonic saturation
            B_p = B[freq_bin]
            max_amplitude = np.max(np.abs(bpf_freq)) or 1
            norm_output = B_p / max_amplitude * bpf_freq
            nld_output = np.empty_like(norm_output)

            # Step 3 of bass frequency processing:
            # Apply NLD to the right channel (non-linear function) to create harmonics
            for i, x in enumerate(norm_output):
                nld_output[i] = 2.5*math.atan(0.9*x)+2.5*math.sqrt(1-(0.9*x)**2)-2.5 if x >= 0 else math.tanh(2.25*x)


        # Step 4 of bass frequency processing:
        # Apply another BPF to limit number of harmonics
        # Add weighted output to right channel
        filtered_output = filter(sample_rate=sample_rate, time_signal=nld_output.copy(), min_freq=speaker_threshold_freq, max_freq=3*speaker_threshold_freq, filter='bpf', filter_order=8)
        right += (w[freq_bin]*filtered_output)

    # Step 5 of bass frequency processing:
    # Normalize right channel to match left channel peak amplitude
    left_peak = np.max(np.abs(left)) or 1.0
    right_peak = np.max(np.abs(right)) or 1.0
    right = right * (left_peak / right_peak) #* 0.5

    return left, right, f, sample_rate
