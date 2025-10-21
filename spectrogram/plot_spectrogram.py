import soundfile as sf
from pathlib import Path
import librosa
import librosa.display
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def generate_spectrogram(signal, fft_size=2048, hop_size=None, window_size=None):
    """
    Returns a db scaled spectrogram from the input signal

    :param signal: input array of audio
    :param fft_size: length of the windowed signal after padding with zeros
    :param hop_size: number of audio samples between adjacent STFT columns
    :param window_size: Smaller values improve temporal resolution of STFT but decrease window resolution
    :return: 2d array of stft decibel readings
    """ 

    if not window_size:
        window_size = fft_size
        
    if not hop_size:
        hop_size = window_size // 4

    stft = librosa.stft(signal, n_fft=fft_size, hop_length=hop_size, win_length=window_size)
    spectrogram = np.abs(stft)
    spectrogram_db = librosa.amplitude_to_db(spectrogram)
    return spectrogram_db
    


def main():
    script_dir = Path(__file__).parent.parent
    signal, sample_rate = sf.read(script_dir / 'audio' / "cello_suite.wav")
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    print(f'Sample rate: {sample_rate}')

    spectrogram_db = generate_spectrogram(signal)
    plt.figure(figsize=(10,4))
    img = librosa.display.specshow(spectrogram_db, y_axis='log', x_axis='time', sr=sample_rate, cmap='inferno')
    plt.show()

if __name__ == '__main__': 
    main()