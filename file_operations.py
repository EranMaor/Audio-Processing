from scipy.io import wavfile
import numpy as np
from scipy.fft import rfftfreq

# Read WAV file and return data and properties
def read_mono(file_name):
    sample_rate, data = wavfile.read(file_name)
    
    # Check if the file is stereo or mono
    if data.ndim > 1:
        # It's stereo: keep only the left channel
        mono_data = data[:, 0]
    else:
        # It's already mono
        mono_data = data

    mono_data = mono_data.astype(np.float64)
    f = rfftfreq(len(mono_data), (1/sample_rate))

    return mono_data, f, sample_rate

# Write finalized output file
def write_file(reconstructed_left, reconstructed_right, sample_rate, output_filename):
    # reconstructed_left represents the mids/highs that are outputted from the left channel device
    # reconstructed_right represents the lows that are outputted from right channel device
    reconstructed_real = np.stack((reconstructed_left, reconstructed_right), axis=1)
    audio_data_int16 = np.int16(np.clip(reconstructed_real, -32768, 32767.0))

    wavfile.write(output_filename, sample_rate, audio_data_int16)

    return audio_data_int16