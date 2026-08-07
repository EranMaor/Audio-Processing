from scipy.io import wavfile
import numpy as np
from scipy.fft import rfftfreq, rfft

# Read WAV file and return data and properties
def read_mono(file_name):
    sample_rate, data = wavfile.read(file_name)
    
    # Check if the file is stereo or mono
    if data.ndim > 1:
        # It's stereo: average the left (channel 0) and right (channel 1) to make it mono
        mono_data = mono_data = np.mean(data, axis=1)
    else:
        # It's already mono
        mono_data = data

    mono_data = mono_data.astype(np.float64)
    
    f = rfftfreq(len(mono_data), (1/sample_rate))
    return mono_data, f, sample_rate

# Write finalized output file
# Normalize and scale channels before combining and writing to output WAV file
def write_file(reconstructed_left, reconstructed_right, sample_rate, output_filename):
    # Find the max amplitude per channel
    max_left = np.max(np.abs(reconstructed_left))
    max_right = np.max(np.abs(reconstructed_right))
    if max_left == 0:
        max_left = 1
    if max_right == 0:
       max_right = 1

    # Normalize both channels to +-1
    normalized_left = reconstructed_left / max_left
    normalized_right = reconstructed_right / max_right

    # Scale both channels to 16 bit (word)
    scaled_left = normalized_left * 32767.0
    # Scale right channel (bass) to not dominate over mids/highs in left channel
    scaled_right = (normalized_right * 32767.0) / 2

    # scaled_left represents the mids/highs that are outputted from the left channel device
    # scaled_right represents the lows that are outputted from right channel device
    reconstructed_real = np.stack((scaled_left, scaled_right), axis=1)
    audio_data_int16 = np.int16(np.clip(reconstructed_real, -32768, 32767.0))

    wavfile.write(output_filename, sample_rate, audio_data_int16)

    return audio_data_int16