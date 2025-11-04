# Audio Fingerprinting Module

This module implements **Shazam-style audio fingerprinting** for music identification, including database storage and retrieval.

## 🎯 Overview

The fingerprinting algorithm converts audio into unique "fingerprints" that can be stored in a database and matched against unknown audio clips. It works in three main steps:

1. **Peak Detection** - Find prominent frequency points in spectrograms
2. **Constellation Mapping** - Pair peaks to create robust patterns
3. **Hash Generation** - Create unique fingerprints from peak pairs
4. **Database Storage** - Store fingerprints for matching

## 📦 Module Contents

### Core Functions

#### `find_peaks(spectrogram, neighborhood_size=20, amplitude_threshold=10.0)`
Detects local maxima (peaks) in the spectrogram.

**Parameters:**
- `spectrogram`: 2D array of spectrogram values (freq bins × time frames)
- `neighborhood_size`: Size of region for local maximum detection
- `amplitude_threshold`: Minimum peak amplitude in dB

**Returns:** List of `(time_idx, freq_idx)` tuples

```python
peaks = find_peaks(spectrogram_db, neighborhood_size=20, amplitude_threshold=15.0)
```

---

#### `generate_constellation_map(peaks, fan_value=5, max_time_delta=200)`
Creates a constellation map by pairing peaks (Shazam's core algorithm).

**Parameters:**
- `peaks`: List of `(time_idx, freq_idx)` tuples
- `fan_value`: Number of target peaks to pair with each anchor
- `max_time_delta`: Maximum time gap between paired peaks (frames)

**Returns:** List of `(freq1, freq2, time_delta)` tuples

```python
constellation = generate_constellation_map(peaks, fan_value=5, max_time_delta=200)
```

---

#### `generate_fingerprints(constellation, anchor_times)`
Generates SHA1 hash fingerprints from constellation points.

**Parameters:**
- `constellation`: List of `(freq1, freq2, time_delta)` tuples
- `anchor_times`: List of anchor peak times

**Returns:** List of `(hash_string, time_offset)` tuples

```python
fingerprints = generate_fingerprints(constellation, anchor_times)
```

---

#### `fingerprint_spectrogram(spectrogram, ...)`
**All-in-one function** that performs the complete fingerprinting pipeline.

**Parameters:**
- `spectrogram`: 2D spectrogram array
- `neighborhood_size`: Peak detection region size (default: 20)
- `amplitude_threshold`: Minimum peak amplitude in dB (default: 10.0)
- `fan_value`: Peaks to pair per anchor (default: 5)
- `max_time_delta`: Max time window for pairing (default: 200)

**Returns:** Tuple of `(peaks, constellation, fingerprints)`

```python
peaks, constellation, fingerprints = fingerprint_spectrogram(
    spectrogram_db,
    neighborhood_size=20,
    amplitude_threshold=15.0,
    fan_value=5,
    max_time_delta=200
)
```

---

#### `store_fingerprints(fingerprints, song_details)`
Stores fingerprints in the PostgreSQL database.

**Parameters:**
- `fingerprints`: List of `(hash_string, time_offset)` tuples
- `song_details`: Tuple of `(song_name, artist, duration_ms, link)`

**Returns:** `song_id` of the inserted song (or `None` on error)

```python
song_details = ("Cello Suite No. 1", "Bach", 195370, "audio/cello_suite.wav")
song_id = store_fingerprints(fingerprints, song_details)
```

---

### Visualization Functions

#### `visualize_peaks(spectrogram, peaks, sample_rate, hop_length)`
Displays spectrogram with detected peaks overlaid.

```python
import matplotlib.pyplot as plt
visualize_peaks(spectrogram_db, peaks, sample_rate=44100, hop_length=512)
plt.show()
```

#### `visualize_constellation(peaks, constellation, max_points=500)`
Shows constellation map with peak connections.

```python
visualize_constellation(peaks, constellation, max_points=1000)
plt.show()
```

---

## 🗄️ Database Setup

### 1. Create `.env` File

Create a `.env` file in the project root with your database credentials:

### 2. Create Database and Tables

Run the database creation script:

```bash
python fingerprinting/create_db.py
```

This creates:
- **`songs`** table - Stores song metadata
- **`fingerprints`** table - Stores hash fingerprints
- Indexes on `hash` for fast lookups

### Database Schema

**songs table:**
```sql
song_id      INT PRIMARY KEY
name         VARCHAR(255) NOT NULL
artist       VARCHAR(255) NOT NULL
duration_ms  INT NOT NULL
link         VARCHAR(255)
```

**fingerprints table:**
```sql
hash            VARCHAR(255) NOT NULL
offset_time_ms  INT NOT NULL
song_id         INT NOT NULL (FOREIGN KEY)
PRIMARY KEY (hash, song_id, offset_time_ms)
```

---

## 🚀 Complete Workflow

### Step 1: Import Required Modules

```python
import soundfile as sf
from spectrogram import generate_spectrogram
from fingerprinting import fingerprint_spectrogram, store_fingerprints
```

### Step 2: Load Audio File

```python
signal, sample_rate = sf.read('audio/song.wav')

# Convert stereo to mono if needed
if signal.ndim > 1:
    signal = signal.mean(axis=1)
```

### Step 3: Generate Spectrogram

```python
spectrogram_db = generate_spectrogram(
    signal,
    fft_size=2048,
    hop_size=512
)
```

### Step 4: Extract Fingerprints

```python
peaks, constellation, fingerprints = fingerprint_spectrogram(
    spectrogram_db,
    neighborhood_size=20,
    amplitude_threshold=15.0,
    fan_value=5,
    max_time_delta=200
)

print(f"Generated {len(fingerprints)} fingerprints from {len(peaks)} peaks")
```

### Step 5: Store in Database

```python
# Calculate duration in milliseconds
duration_ms = int((len(signal) / sample_rate) * 1000)

# Prepare song details
song_details = (
    "My Song Name",      # song_name
    "Artist Name",       # artist
    duration_ms,         # duration_ms
    "audio/song.wav"     # link/path to file
)

# Store in database
song_id = store_fingerprints(fingerprints, song_details)
print(f"Song stored with ID: {song_id}")
```

---

## 📊 Example Output

For a 195-second audio file:

```
Generated 8600 fingerprints from 1723 peaks
Inserted song 'Cello Suite No. 1' by Bach with ID 1
Inserted 8600 fingerprints for song ID 1
Song stored with ID: 1
```

**Database State:**
- 1 row in `songs` table
- 8,600 rows in `fingerprints` table
- Ready for matching!

---

## 🎛️ Parameter Tuning Guide

### For More Fingerprints (Better Matching)
```python
peaks, constellation, fingerprints = fingerprint_spectrogram(
    spectrogram_db,
    neighborhood_size=15,      # Smaller = more peaks
    amplitude_threshold=10.0,  # Lower = more peaks
    fan_value=10,              # More pairs per anchor
    max_time_delta=300         # Wider time window
)
```

### For Fewer, Stronger Fingerprints (Faster, Less Storage)
```python
peaks, constellation, fingerprints = fingerprint_spectrogram(
    spectrogram_db,
    neighborhood_size=25,      # Larger = fewer peaks
    amplitude_threshold=20.0,  # Higher = only strong peaks
    fan_value=3,               # Fewer pairs per anchor
    max_time_delta=100         # Narrower time window
)
```

| Parameter | Default | Effect |
|-----------|---------|--------|
| `neighborhood_size` | 20 | ↑ = fewer peaks |
| `amplitude_threshold` | 10.0 dB | ↑ = fewer, stronger peaks |
| `fan_value` | 5 | ↑ = more fingerprints |
| `max_time_delta` | 200 frames | ↑ = wider temporal pairing |

---

## 📁 Files in This Module

```
fingerprinting/
├── audio_fingerprint.py      # Core algorithms and database functions
├── create_db.py              # Database and table creation
├── __init__.py               # Module exports
└── README.md                 # This file
```

---

## 🐛 Troubleshooting

### "Error: connection to database failed"
- Check PostgreSQL is running: `pg_ctl status`
- Verify `.env` file has correct credentials
- Ensure database exists: run `create_db.py`

### "Table does not exist"
- Run `python fingerprinting/create_db.py` to create tables

### Not enough fingerprints generated
- Lower `amplitude_threshold` (e.g., 10.0 → 8.0)
- Decrease `neighborhood_size` (e.g., 20 → 15)
- Increase `fan_value` (e.g., 5 → 10)

---

## 📝 Dependencies

- `numpy` - Array operations
- `scipy` - Signal processing, filters
- `librosa` - Audio visualization
- `matplotlib` - Plotting
- `psycopg2` - PostgreSQL connector
- `python-dotenv` - Environment variable management
- `soundfile` - Audio file I/O

Install all dependencies:
```bash
pip install numpy scipy librosa matplotlib psycopg2-binary python-dotenv soundfile
```
