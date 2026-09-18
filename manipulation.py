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
def harmonics(input_file_name, speaker_threshold_freq=100, lower_threshold_freq=40, upper_threshold_freq=80, a=0, b=1.5, c=0.4):
    data, f, sample_rate = read_mono(input_file_name)

    # Apply HPF to isolate mids/highs in the left channel 
    left = filter(sample_rate=sample_rate, time_signal=data.copy(), min_freq=speaker_threshold_freq, filter='hpf', filter_order=4)

    # Step 1 of bass frequency processing:
    # Apply BPF to isolate original lows (within audible range) in right channel
    # May change parameter but using BPF from 100hz to speaker_threshold_freq
    #
    # Normalize amplitudes to range +- k (divide by max amplitude and multiply by k) to optimize harmonic saturation
    bpf_low = filter(sample_rate=sample_rate, time_signal=data.copy(), min_freq=lower_threshold_freq, max_freq=speaker_threshold_freq, filter='bpf', filter_order=4)
    max_amplitude = np.max(np.abs(bpf_low)) or 1
    k = 2.5
    right1 = bpf_low / max_amplitude * k

    # Step 2 of bass frequency processing:
    # To stay within radius of convergence for tanh Taylor Series, the amplitudes where scaled to +-pi/2 in previous step
    # Apply tanh to the right channel (non-linear function) to create harmonics
    # Inner term (a + bx + cx^2) produces both even and odd harmonics
    np.tanh((a + b*right1 + c*np.square(right1)), out=right1)

    # Step 3 of bass frequency processing:
    # Apply another BPF on right channel to limit number of harmonics
    right = filter(sample_rate=sample_rate, time_signal=right1.copy(), min_freq=upper_threshold_freq, max_freq=3*speaker_threshold_freq, filter='bpf', filter_order=8)

    # Step 4 of bass frequency processing:
    # Normalize right channel to match left channel peak amplitude
    left_peak = np.max(np.abs(left)) or 1.0
    right_peak = np.max(np.abs(right)) or 1.0
    right = right * (left_peak / right_peak) * 0.5

    return left, right, f, sample_rate
