from plot import plot
from transform import transform
from transform import remove_dc_offset

right_limit = 23000

left_data, f_transform_left, f_left, amplitude_left, right_data, f_transform_right, f_right, amplitude_right = transform('a_milli.wav')
plot(f_left, amplitude_left,  'Left A Milli - Amplitude vs Frequencies (pre-shift)', 'Frequency [Hz]', 'Amplitude', [0, right_limit])
plot(f_right, amplitude_right,  'Right A Milli - Amplitude vs Frequencies (pre-shift)', 'Frequency [Hz]', 'Amplitude', [0, right_limit])