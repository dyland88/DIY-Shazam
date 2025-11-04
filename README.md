# DIY-Shazam

A University of Florida Signal Processing Society project recreating the Shazam audio recognition algorithm from scratch.

## How It Works

1. **Generate Spectrogram** - Convert audio to time-frequency representation using STFT
2. **Find Peaks** - Detect prominent features in the spectrogram
3. **Create Constellation Map** - Pair peaks to form unique audio patterns
4. **Generate Fingerprints** - Hash the peak pairs into compact identifiers
5. **Match Database** - Find the closest match using vector similarity search

## Installation

```bash
conda env create -f environment.yml
conda activate diy-shazam
```

## Usage Example

```python
import soundfile as sf
from spectrogram.plot_spectrogram import generate_spectrogram
from fingerprinting.audio_fingerprint import fingerprint_spectrogram

# Load and process audio
signal, sr = sf.read('audio/song.wav')
spec = generate_spectrogram(signal)
peaks, constellation, fingerprints = fingerprint_spectrogram(spec)
```

## Key Functions

#### `generate_spectrogram(signal, fft_size=2048, hop_size=None, window_size=None)`

Converts audio signal to decibel-scaled spectrogram → Returns 2D array (frequency × time)

#### `find_peaks(spectrogram, neighborhood_size=20, amplitude_threshold=10.0)`

Finds local maxima in the spectrogram → Returns list of `(time_idx, freq_idx)` tuples

#### `generate_constellation_map(peaks, fan_value=5, max_time_delta=200)`

Pairs peaks to create unique audio patterns → Returns list of `(freq1, freq2, time_delta)` tuples

#### `generate_fingerprints(constellation, anchor_time=None)`

Hashes constellation points into fingerprints → Returns list of `(hash_string, time_offset)` tuples

#### `fingerprint_spectrogram(spectrogram, ...)`

Complete pipeline: peak detection → constellation → fingerprints → Returns `(peaks, constellation_map, fingerprints)`

#### `visualize_peaks(spectrogram, peaks, sample_rate=44100, hop_length=512)`

Plots spectrogram with detected peaks overlaid

#### `visualize_constellation(peaks, constellation, max_points=500)`

Plots constellation map showing peak connections

## Tech Stack

- **NumPy/SciPy** - Signal processing
- **Librosa** - Audio analysis
- **Matplotlib** - Visualization
- **FAISS** - Vector similarity search
