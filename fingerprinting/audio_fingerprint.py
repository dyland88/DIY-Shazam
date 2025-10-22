"""
Audio Fingerprinting Module - Shazam-style Implementation

This module implements the core fingerprinting algorithm used by Shazam:
1. Peak detection in spectrograms
2. Constellation map creation
3. Hash generation from peak pairs
"""

import numpy as np
from scipy.ndimage import maximum_filter
from typing import List, Tuple
import hashlib


def find_peaks(spectrogram: np.ndarray, 
               neighborhood_size: int = 20,
               amplitude_threshold: float = 10.0) -> List[Tuple[int, int]]:
    """
    Finds local maxima (peaks) in the spectrogram using a maximum filter approach.
    
    :param spectrogram: 2D array of spectrogram values (frequency bins x time frames)
    :param neighborhood_size: Size of the region to check for local maxima
    :param amplitude_threshold: Minimum amplitude (in dB) for a peak to be considered
    :return: List of (time_idx, freq_idx) tuples representing peak locations
    """
    # Apply maximum filter to find local maxima
    local_max = maximum_filter(spectrogram, size=neighborhood_size) == spectrogram
    
    # Apply amplitude threshold
    above_threshold = spectrogram > amplitude_threshold
    
    # Combine conditions: must be local maximum AND above threshold
    peaks = local_max & above_threshold
    
    # Get indices of peaks
    peak_indices = np.argwhere(peaks)
    
    # Return as list of (time, frequency) tuples
    # Note: argwhere returns (row, col) which is (freq, time) in spectrogram
    # We swap to (time, freq) for intuitive ordering
    peaks_list = [(int(time), int(freq)) for freq, time in peak_indices]
    
    return peaks_list


def generate_constellation_map(peaks: List[Tuple[int, int]],
                                fan_value: int = 5,
                                max_time_delta: int = 200) -> List[Tuple[int, int, int]]:
    """
    Creates a constellation map by pairing peaks according to Shazam's algorithm.
    
    Each peak is paired with up to `fan_value` peaks that occur after it within
    a time window of `max_time_delta` frames.
    
    :param peaks: List of (time_idx, freq_idx) tuples
    :param fan_value: Number of peaks to pair with each anchor peak
    :param max_time_delta: Maximum time difference (in frames) between paired peaks
    :return: List of (freq1, freq2, time_delta) tuples representing the constellation
    """
    # Sort peaks by time
    peaks_sorted = sorted(peaks, key=lambda x: x[0])
    
    constellation = []
    
    for i, (t1, f1) in enumerate(peaks_sorted):
        # Look only at future peaks within the time window
        fan_count = 0
        for j in range(i + 1, len(peaks_sorted)):
            t2, f2 = peaks_sorted[j]
            time_delta = t2 - t1
            
            if time_delta > max_time_delta:
                break
            
            # Add this peak pair to constellation
            # Format: (anchor_freq, target_freq, time_delta)
            constellation.append((f1, f2, time_delta))
            fan_count += 1
            
            if fan_count >= fan_value:
                break
    
    return constellation


def generate_fingerprints(constellation: List[Tuple[int, int, int]],
                          anchor_time: List[int] = None) -> List[Tuple[str, int]]:
    """
    Generates hash fingerprints from the constellation map.
    
    Each constellation point is hashed into a fingerprint, along with its
    absolute time offset in the song.
    
    :param constellation: List of (freq1, freq2, time_delta) tuples
    :param anchor_time: List of anchor peak times corresponding to constellation points
    :return: List of (hash_string, time_offset) tuples
    """
    fingerprints = []
    
    for i, (f1, f2, dt) in enumerate(constellation):
        # Create a hash from the frequency pair and time delta
        # Format: "f1|f2|dt" -> hash
        hash_input = f"{f1}|{f2}|{dt}".encode('utf-8')
        hash_value = hashlib.sha1(hash_input).hexdigest()
        
        # If anchor times are provided, use them; otherwise use index
        time_offset = anchor_time[i] if anchor_time else i
        
        fingerprints.append((hash_value, time_offset))
    
    return fingerprints


def fingerprint_spectrogram(spectrogram: np.ndarray,
                            neighborhood_size: int = 20,
                            amplitude_threshold: float = 10.0,
                            fan_value: int = 5,
                            max_time_delta: int = 200) -> Tuple[List[Tuple[int, int]], 
                                                                  List[Tuple[int, int, int]], 
                                                                  List[Tuple[str, int]]]:
    """
    Complete fingerprinting pipeline for a spectrogram.
    
    This function combines all steps:
    1. Peak detection
    2. Constellation map creation
    3. Fingerprint generation
    
    :param spectrogram: 2D array of spectrogram values (frequency bins x time frames)
    :param neighborhood_size: Size of region for peak detection
    :param amplitude_threshold: Minimum amplitude (dB) for peaks
    :param fan_value: Number of target peaks per anchor
    :param max_time_delta: Maximum time window for peak pairing (frames)
    :return: Tuple of (peaks, constellation_map, fingerprints)
    """
    # Step 1: Find peaks
    peaks = find_peaks(spectrogram, neighborhood_size, amplitude_threshold)
    
    #print(f"Found {len(peaks)} peaks in spectrogram")
    
    # Step 2: Generate constellation map
    constellation = generate_constellation_map(peaks, fan_value, max_time_delta)
    
    #print(f"Generated {len(constellation)} constellation points")
    
    # Step 3: Generate fingerprints
    # Extract anchor times from peaks for fingerprint time offsets
    constellation_with_time = []
    anchor_times = []
    
    peaks_sorted = sorted(peaks, key=lambda x: x[0])
    
    # Build constellation with anchor times
    for i, (t1, f1) in enumerate(peaks_sorted):
        for j in range(i + 1, len(peaks_sorted)):
            t2, f2 = peaks_sorted[j]
            time_delta = t2 - t1
            
            if time_delta > max_time_delta:
                break
            
            constellation_with_time.append((f1, f2, time_delta))
            anchor_times.append(t1)
            
            if len([c for c, t in zip(constellation_with_time, anchor_times) 
                   if c[0] == f1 and t == t1]) >= fan_value:
                break
    
    fingerprints = generate_fingerprints(constellation_with_time, anchor_times)
    
    #print(f"Generated {len(fingerprints)} unique fingerprints")
    
    return peaks, constellation, fingerprints


def visualize_peaks(spectrogram: np.ndarray, 
                   peaks: List[Tuple[int, int]], 
                   sample_rate: int = 44100,
                   hop_length: int = 512):
    """
    Visualizes the spectrogram with detected peaks overlaid.
    
    :param spectrogram: 2D array of spectrogram values
    :param peaks: List of (time_idx, freq_idx) tuples
    :param sample_rate: Audio sample rate
    :param hop_length: STFT hop length
    :return: matplotlib figure
    """
    import matplotlib.pyplot as plt
    import librosa.display
    
    plt.figure(figsize=(14, 6))
    
    # Plot spectrogram
    librosa.display.specshow(spectrogram, 
                            y_axis='log', 
                            x_axis='time',
                            sr=sample_rate,
                            hop_length=hop_length,
                            cmap='inferno')
    plt.colorbar(format='%+2.0f dB')
    
    # Overlay peaks
    if peaks:
        times, freqs = zip(*peaks)
        # Convert indices to actual values
        times_sec = np.array(times) * hop_length / sample_rate
        freq_bins = np.array(freqs)
        
        plt.scatter(times_sec, freq_bins, c='cyan', s=5, alpha=0.8, marker='o', 
                   label=f'Peaks ({len(peaks)})')
    
    plt.title('Spectrogram with Detected Peaks')
    plt.legend()
    plt.tight_layout()
    
    return plt.gcf()


def visualize_constellation(peaks: List[Tuple[int, int]],
                           constellation: List[Tuple[int, int, int]],
                           max_points: int = 500):
    """
    Visualizes the constellation map showing peak pairs.
    
    :param peaks: List of (time_idx, freq_idx) tuples
    :param constellation: List of (freq1, freq2, time_delta) tuples
    :param max_points: Maximum number of constellation points to display
    :return: matplotlib figure
    """
    import matplotlib.pyplot as plt
    
    # Sort peaks by time for indexing
    peaks_sorted = sorted(peaks, key=lambda x: x[0])
    
    # Build a mapping from (time, freq) to index
    peak_dict = {(t, f): idx for idx, (t, f) in enumerate(peaks_sorted)}
    
    plt.figure(figsize=(14, 6))
    
    # Plot all peaks
    if peaks:
        times, freqs = zip(*peaks_sorted)
        plt.scatter(times, freqs, c='blue', s=20, alpha=0.5, label='Peaks')
    
    # Plot constellation connections
    # Reconstruct the connections from constellation map
    count = 0
    for i, (t1, f1) in enumerate(peaks_sorted):
        if count >= max_points:
            break
            
        for j in range(i + 1, len(peaks_sorted)):
            t2, f2 = peaks_sorted[j]
            time_delta = t2 - t1
            
            # Check if this pair is in our constellation
            if (f1, f2, time_delta) in constellation:
                plt.plot([t1, t2], [f1, f2], 'r-', alpha=0.3, linewidth=0.5)
                count += 1
                
            if count >= max_points:
                break
    
    plt.xlabel('Time (frames)')
    plt.ylabel('Frequency (bins)')
    plt.title(f'Constellation Map (showing up to {max_points} connections)')
    plt.legend()
    plt.tight_layout()
    
    return plt.gcf()

