import numpy as np
import pandas as pd
import sys

import scipy.io.wavfile as wavfile

def analyze_wav_to_csv(wav_file_path, csv_output_path):
    """
    Analyze WAV file and output frequency data to CSV using Hanning window.
    """
    # Read WAV file
    sample_rate, audio_data = wavfile.read(wav_file_path)
    
    # Convert to mono if stereo
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    
    # Apply Hanning window
    window = np.hanning(len(audio_data))
    windowed_data = audio_data * window
    
    # Perform FFT
    fft_data = np.fft.fft(windowed_data)
    frequencies = np.fft.fftfreq(len(fft_data), 1/sample_rate)
    
    # Get magnitude spectrum (only positive frequencies)
    magnitude = np.abs(fft_data)
    positive_freq_indices = frequencies >= 0
    
    frequencies = frequencies[positive_freq_indices]
    magnitude = magnitude[positive_freq_indices]
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame({
        'Frequency_Hz': frequencies,
        'Magnitude': magnitude
    })
    
    df.to_csv(csv_output_path, index=False)
    print(f"Frequency analysis saved to {csv_output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input.wav> <output.csv>")
        sys.exit(1)
    
    wav_file = sys.argv[1]
    csv_file = sys.argv[2]
    
    try:
        analyze_wav_to_csv(wav_file, csv_file)
    except Exception as e:
        print(f"Error: {e}")