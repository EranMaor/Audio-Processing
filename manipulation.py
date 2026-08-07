from scipy import signal
from scipy.fft import rfft
import numpy as np

from file_operations import read_mono

# Filtering function
# Uses butterworth IIR filter
# SOS (second-order sections) method breaks down filtration into small pieces to reduce noise
def adjust_frequencies_in_time(sample_rate, time_signal, min_freq=0, max_freq=0, filter='hpf', filter_order=1):
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

#Helper function to graph FFT amplitudes
def transform(data):
    n = len(data)

    f_transform = rfft(data)
    amplitude = np.abs(f_transform) / n

    return amplitude

# a: DC offset
# b: Produces odd harmonics
# c: Produces even harmonics
def harmonics(input_file_name, output_file_name, a=0, b=1.5, c=0.4):
    data, f, sample_rate = read_mono(input_file_name)

    # Apply HPF to isolate mids/highs in the left channel 
    left = adjust_frequencies_in_time(sample_rate=sample_rate, time_signal=data.copy(), min_freq=100, filter='hpf', filter_order=4)

    # Step 1 of bass frequency processing:
    # Apply BPF to isolate original lows (within audible range) in right channel
    # (could use LPF, but <20hz frequencies are not audible anyways)
    #
    # Normalize amplitudes to range +-1 (divide by max amplitude)
    # Then scale by pi/2 for next step to be within radius of convergence
    bpf_low = adjust_frequencies_in_time(sample_rate=sample_rate, time_signal=data.copy(), min_freq=20, max_freq=100, filter='bpf', filter_order=4)
    max_amplitude = np.max(np.abs(bpf_low))
    if max_amplitude == 0:
        max_amplitude = 1

    normalized_base = bpf_low / max_amplitude
    right1 = normalized_base * (np.pi / 2)

    # Step 2 of bass frequency processing:
    # To stay within radius of convergence for tanh Taylor Series, the amplitudes where scaled to +-pi/2 in previous step
    # Apply tanh to the right channel (non-linear function) to create harmonics
    # Inner term (a + bx + cx^2) creates both even and odd harmonics
    np.tanh((a + b*right1 + c*np.square(right1)), out=right1)

    # Step 3 of bass frequency processing:
    # Apply another BPF on right channel to limit number of harmonics
    right = adjust_frequencies_in_time(sample_rate=sample_rate, time_signal=right1.copy(), min_freq=100, max_freq=500, filter='bpf', filter_order=8)

    return left, right, f, sample_rate
