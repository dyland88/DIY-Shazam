"""
Demo script showing the complete Shazam-style fingerprinting pipeline
"""

import soundfile as sf
from pathlib import Path
import numpy as np
import sys
sys.path.append(str(Path(__file__).parent.parent))

from spectrogram.plot_spectrogram import generate_spectrogram
from fingerprinting.audio_fingerprint import (
    fingerprint_spectrogram,
    visualize_peaks,
    visualize_constellation,
    store_fingerprints,
    match_fingerprints
)
import matplotlib.pyplot as plt

def add_gaussian_noise(signal, noise_level=0.1):
    """
    Adds Gaussian noise to the signal.
    """
    signal = np.asarray(signal)
    noise = np.random.normal(0, noise_level, size=signal.shape)
    signal = signal + noise
    signal = np.clip(signal, -1, 1)
    return signal

def main():
    # Load audio file
    script_dir = Path(__file__).parent.parent
    audio_path = script_dir / 'audio' / 'cello_suite.wav'
    
    print(f"Loading audio from: {audio_path}")
    signal, sample_rate = sf.read(audio_path)
    
    # Convert stereo to mono if needed
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Duration: {len(signal) / sample_rate:.2f} seconds")
    print(f"Samples: {len(signal)}")
    print("-" * 60)
    
    # Optional: Only test a portion of the audio
    start_time = 10.0
    duration = 10.0
    start = int(start_time * sample_rate)
    end = start + int(duration * sample_rate)
    #signal = signal[start:end]

    # Optional: Add noise to the signal
    #signal = add_gaussian_noise(signal, noise_level=0.025)
    #sf.write(script_dir / 'audio' / 'cello_suite_noisy.wav', signal, sample_rate)

    # Generate spectrogram
    print("\nGenerating spectrogram...")
    fft_size = 2048
    hop_size = 512
    spectrogram_db = generate_spectrogram(signal, 
                                         fft_size=fft_size, 
                                         hop_size=hop_size)
    
    print(f"Spectrogram shape: {spectrogram_db.shape}")
    print("-" * 60)
    
    # Fingerprint the spectrogram
    print("\nPerforming fingerprint analysis...")
    
    peaks, constellation, fingerprints = fingerprint_spectrogram(
        spectrogram_db,
        neighborhood_size=20,      # Size of local region for peak detection
        amplitude_threshold=15.0,  # Minimum dB for a peak
        fan_value=5,               # Number of pairs per anchor peak
        max_time_delta=200         # Maximum time gap between paired peaks (frames)
    )
    
    print("-" * 60)
    print("\nFingerprint Statistics:")
    print(f"  Total peaks detected: {len(peaks)}")
    print(f"  Constellation points: {len(constellation)}")
    print(f"  Unique fingerprints: {len(fingerprints)}")
    print(f"  Fingerprints per second: {len(fingerprints) / (len(signal) / sample_rate):.1f}")
    
    # Show some example fingerprints
    print("\nSample Fingerprints (first 5):")
    for i, (hash_val, time_offset) in enumerate(fingerprints[:5]):
        print(f"  {i+1}. Hash: {hash_val[:16]}... at time frame {time_offset}")

    #store_fingerprints(fingerprints, ("Cello Suite", "Johann Sebastian Bach", len(signal), audio_path.as_posix()))
    matches = match_fingerprints(fingerprints, k=3)
    print(matches)
if __name__ == '__main__':
    main()

