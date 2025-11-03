# Database Module

Audio fingerprinting pipeline that processes songs and uploads them to Pinecone vector database.

## Files

- **`download_dataset.py`** - Downloads GTZAN dataset from Kaggle
- **`pipeline.py`** - Main processing pipeline (audio → spectrogram → fingerprint → Pinecone)
- **`search.py`** - Search for songs in the database
- **`test_accuracy.py`** - Test search accuracy on sample of songs
- **`vector.py`** - Old test file (deprecated)

## Quick Start

### 1. Download Dataset
```bash
python -m database.download_dataset
```
Downloads 1000 songs (10 genres, 100 songs each) and saves path to `dataset_path.txt`.
Can switch to different dataset to expand, just starting small for now. 

### 2. Run Pipeline
```bash
python -m database.pipeline
```
Processes all songs and uploads to Pinecone. Auto-detects CPU cores for parallel processing.

### 3. Search Database
```bash
python -m database.search /path/to/audio.wav [index_name]
```
Search for a song and see top 10 matches with scores.

### 4. Test Accuracy
```bash
python -m database.test_accuracy [index_name] [songs_per_genre]
```
Test accuracy on 30 songs (3 per genre by default).

## How It Works

```
Audio File (.wav)
    ↓
Load with soundfile
    ↓
Generate Spectrogram (STFT) [from spectrogram module]
    ↓
Extract Fingerprints [from fingerprinting module]
  - Peaks: (time, freq) locations
  - Constellation: (freq1, freq2, time_delta) pairs
  - Fingerprints: SHA1 hash strings
    ↓
Convert to 128-D Vector (multi-hash approach)
  - 5 hash strategies on constellation data
  - Peak position hashing
  - Pre-computed SHA1 fingerprints
  - Statistical features
    ↓
Upload to Pinecone (cosine similarity)
```

## Command-Line Options

```bash
# Basic usage
python -m database.pipeline

# Custom options
python -m database.pipeline \
  --data_dir /path/to/audio \
  --index_name my-index \
  --dimension 128 \
  --batch_size 100 \
  --num_workers 32
```

### Options:
- `--data_dir`: Audio directory (defaults to `dataset_path.txt`)
- `--index_name`: Pinecone index name (default: `audio-fingerprints`)
- `--dimension`: Vector dimension (default: `128`)
- `--batch_size`: Upload batch size (default: `100`)
- `--num_workers`: CPU cores to use (default: auto-detect)

## Performance

### Parallel Processing (CPU)
- **Auto-detects:** 2 workers on local (≤8 CPUs), all cores on HiperGator (>8 CPUs)
- Can override with `--num_workers N`

### Speed:
**Sequential:** ~1.6 sec/song  
**With 32 cores:** ~0.05 sec/song ⚡

For 1000 songs:
- Sequential: ~27 minutes
- 32 cores: ~50 seconds
- 64 cores: ~25 seconds

## HiperGator Usage

### SLURM Job Script
```bash
# Edit run_pipeline.sh (set your account)
nano run_pipeline.sh

# Submit job
sbatch run_pipeline.sh

# Check status
squeue -u $USER
```

The pipeline automatically uses all allocated CPU cores.

## Requirements

Set `PINECONE_API_KEY` in `.env` file:
```
PINECONE_API_KEY=your_key_here
```

## What Gets Stored

Each song is stored in Pinecone as:
- **ID**: Filename (without extension)
- **Vector**: 128-D fingerprint
- **Metadata**:
  - `filename`: Original filename
  - `genre`: Parent directory name
  - `path`: Full file path
  - `sample_rate`: Audio sample rate
  - `num_peaks`: Number of detected peaks
  - `num_fingerprints`: Number of hash fingerprints

## Searching & Testing

### Search for a Song
```bash
# Basic search (uses default index)
python -m database.search audio/song.wav

# Specify index
python -m database.search audio/song.wav audio-fingerprints-v4
```

Shows top 10 matches with scores and analyzes score distribution.

### Test Accuracy
```bash
# Test with 30 songs (3 per genre)
python -m database.test_accuracy audio-fingerprints-v4

# Test with 50 songs (5 per genre)
python -m database.test_accuracy audio-fingerprints-v4 5
```

Shows:
- Overall accuracy percentage
- Accuracy by genre
- Failed matches with details

## Vector Generation Details

The 128-D vector uses:
1. **Constellation hashing (5 strategies)**:
   - Frequency pair hash
   - Time-frequency hash
   - Combined component hash
   - Frequency difference hash
   - Position-weighted hash
2. **SHA1 fingerprints**: Pre-computed Shazam-style hashes
3. **Peak positions**: Spatial distribution
4. **Statistical features**: Means, medians, stds

This multi-hash approach creates unique vectors per song while maintaining similarity for identical songs.

## Notes

- Only processes `.wav` files
- Stereo files are converted to mono
- Uses cosine similarity for matching
- Batch uploads for efficiency
- Default dimension: 128 (can use 64 or 256)
