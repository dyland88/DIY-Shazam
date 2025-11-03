"""
Pipeline: Audio files → Spectrogram → Fingerprinting → Pinecone
Usage: python -m database.pipeline --data_dir /path/to/audio/files
   or: python -m database.pipeline  (reads from dataset_path.txt)
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import numpy as np
from tqdm import tqdm
import argparse
import soundfile as sf
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import multiprocessing

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import spectrogram
import fingerprinting

load_dotenv()


def process_audio_file_standalone(audio_path):
    """
    Standalone function to process a single audio file (for multiprocessing)
    
    This function is defined at module level so it can be pickled for parallel processing.
    
    :param audio_path: Path to audio file
    :return: (song_id, fingerprint_vector, metadata) or (None, None, None) on error
    """
    try:
        audio_path = Path(audio_path)
        
        # Step 1: Load audio file
        signal, sample_rate = sf.read(str(audio_path))
        
        # Convert to mono if stereo
        if signal.ndim > 1:
            signal = signal.mean(axis=1)
        
        # Step 2: Generate Spectrogram
        spec = spectrogram.generate_spectrogram(signal)
        
        # Step 3: Extract Fingerprint (peaks, constellation, hashes)
        peaks, constellation, fingerprints = fingerprinting.fingerprint_spectrogram(spec)
        
        # Step 4: Convert constellation + peaks + fingerprints to numeric vector for Pinecone
        fingerprint_vector = constellation_to_vector(constellation, peaks, fingerprints, target_dim=128)
        
        # Create metadata
        song_id = audio_path.stem  # filename without extension
        metadata = {
            'filename': audio_path.name,
            'genre': audio_path.parent.name,
            'path': str(audio_path),
            'sample_rate': int(sample_rate),
            'num_peaks': len(peaks),
            'num_fingerprints': len(fingerprints)
        }
        
        return song_id, fingerprint_vector, metadata
        
    except Exception as e:
        print(f"❌ Error in {audio_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def constellation_to_vector(constellation, peaks, fingerprints, target_dim=128):
    """
    Convert constellation map + fingerprints to fixed-size numeric vector for Pinecone
    
    Uses ALL data from the fingerprinting process:
    - Constellation patterns (freq pairs + time deltas)
    - Peak positions
    - Pre-computed hash fingerprints
    
    :param constellation: List of (freq1, freq2, time_delta) tuples
    :param peaks: List of (time_idx, freq_idx) tuples
    :param fingerprints: List of (hash_string, time_offset) tuples
    :param target_dim: Target dimension for the output vector
    :return: numpy array of shape (target_dim,)
    """
    if len(constellation) == 0 or len(peaks) == 0:
        return np.zeros(target_dim)
    
    vector = np.zeros(target_dim, dtype=np.float32)
    
    # Use multiple hash functions to distribute constellation patterns across all dimensions
    # This creates a much more unique fingerprint per song
    
    # Hash each constellation point into the vector using multiple strategies
    for i, (f1, f2, dt) in enumerate(constellation):
        # Strategy 1: Hash based on frequency pair (primes for better distribution)
        idx1 = (f1 * 73 + f2 * 181) % target_dim
        vector[idx1] += 1.0
        
        # Strategy 2: Hash based on time delta and first frequency
        idx2 = (dt * 97 + f1 * 23) % target_dim
        vector[idx2] += 0.8
        
        # Strategy 3: Hash based on all three components
        idx3 = (f1 * 31 + f2 * 17 + dt * 7) % target_dim
        vector[idx3] += 1.2
        
        # Strategy 4: Hash based on frequency difference
        freq_diff = abs(f2 - f1)
        idx4 = (freq_diff * 53 + dt * 11) % target_dim
        vector[idx4] += 0.6
        
        # Strategy 5: Position-weighted hash (early vs late in song matters)
        position_weight = i / len(constellation)  # 0 to 1
        idx5 = int((f1 * 41 + f2 * 13) * position_weight) % target_dim
        vector[idx5] += 0.5
    
    # Use the pre-computed hash fingerprints (actual Shazam hashes!)
    if len(fingerprints) > 0:
        for hash_str, time_offset in fingerprints[:target_dim * 2]:  # Use more fingerprints
            # Convert hash string to integer and map to vector dimension
            hash_int = int(hash_str[:8], 16)  # Use first 8 hex chars of SHA1 hash
            idx = hash_int % target_dim
            vector[idx] += 1.0
            
            # Also incorporate time offset information
            time_idx = (hash_int + time_offset) % target_dim
            vector[time_idx] += 0.7
    
    # Add peak distribution information to remaining dimensions
    peak_times = np.array([t for t, f in peaks])
    peak_freqs = np.array([f for t, f in peaks])
    
    # Distribute peak information across vector
    if len(peaks) > 0:
        for i, (t, f) in enumerate(peaks[:target_dim//4]):  # Use subset of peaks
            idx = (t * 67 + f * 37) % target_dim
            vector[idx] += 0.3
    
    # Add weighted statistical features in specific positions
    freqs_1 = np.array([f1 for f1, f2, dt in constellation])
    freqs_2 = np.array([f2 for f1, f2, dt in constellation])
    time_deltas = np.array([dt for f1, f2, dt in constellation])
    
    # Add stats at specific deterministic positions based on song characteristics
    stat_positions = [
        (int(np.mean(freqs_1)) % target_dim, np.std(freqs_1)),
        (int(np.mean(freqs_2)) % target_dim, np.std(freqs_2)),
        (int(np.mean(time_deltas)) % target_dim, np.std(time_deltas)),
        (int(np.median(freqs_1)) % target_dim, len(peaks) / 100.0),
        (int(np.median(freqs_2)) % target_dim, len(constellation) / 100.0),
    ]
    
    for idx, value in stat_positions:
        vector[idx] += value
    
    # Normalize to unit length for cosine similarity
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    
    return vector


class AudioPipeline:
    """Process audio files through spectrogram → fingerprinting → Pinecone"""
    
    def __init__(self, index_name="audio-fingerprints"):
        """Initialize Pinecone connection"""
        print("🔧 Initializing Pinecone...")
        
        api_key = os.environ.get("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("❌ PINECONE_API_KEY not found in .env file!")
        
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.index = None
        
    def create_index(self, dimension=128):
        """Create Pinecone index if it doesn't exist"""
        print(f"\n🗄️  Setting up Pinecone index: {self.index_name}")
        
        if self.index_name in self.pc.list_indexes().names():
            print(f"✓ Index '{self.index_name}' already exists")
        else:
            print(f"Creating new index '{self.index_name}'...")
            self.pc.create_index(
                name=self.index_name,
                dimension=dimension,
                metric="cosine",  # Cosine similarity for audio fingerprinting
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            print("✓ Index created")
        
        self.index = self.pc.Index(self.index_name)
        
        # Show current stats
        stats = self.index.describe_index_stats()
        print(f"✓ Current vectors in database: {stats.get('total_vector_count', 0)}")
        
    def process_single_audio(self, audio_path):
        """
        Process single audio file through the full pipeline
        
        audio_path → load audio → spectrogram.py → fingerprinting.py → vector conversion
        
        Returns: (song_id, fingerprint_vector, metadata)
        """
        try:
            audio_path = Path(audio_path)
            
            # Step 1: Load audio file
            signal, sample_rate = sf.read(str(audio_path))
            
            # Convert to mono if stereo
            if signal.ndim > 1:
                signal = signal.mean(axis=1)
            
            # Step 2: Generate Spectrogram
            spec = spectrogram.generate_spectrogram(signal)
            
            # Step 3: Extract Fingerprint (peaks, constellation, hashes)
            peaks, constellation, fingerprints = fingerprinting.fingerprint_spectrogram(spec)
            
            # Step 4: Convert constellation to numeric vector for Pinecone
            fingerprint_vector = constellation_to_vector(constellation, peaks, target_dim=128)
            
            # Create metadata
            song_id = audio_path.stem  # filename without extension
            metadata = {
                'filename': audio_path.name,
                'genre': audio_path.parent.name,
                'path': str(audio_path),
                'sample_rate': int(sample_rate),
                'num_peaks': len(peaks),
                'num_fingerprints': len(fingerprints)
            }
            
            return song_id, fingerprint_vector, metadata
            
        except Exception as e:
            print(f"❌ Error processing {audio_path.name}: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None
    
    def upload_to_pinecone(self, vectors_batch):
        """Upload batch of vectors to Pinecone"""
        try:
            self.index.upsert(vectors=vectors_batch)
            return True
        except Exception as e:
            print(f"❌ Upload error: {e}")
            return False
    
    def process_directory(self, data_dir, batch_size=100, num_workers=None):
        """
        Process all audio files in directory with parallel processing
        Efficient batch processing for large datasets using multiprocessing
        
        :param data_dir: Directory containing audio files
        :param batch_size: Number of vectors to upload at once
        :param num_workers: Number of parallel workers (default: CPU count)
        """
        data_dir = Path(data_dir)
        
        print(f"\n🎵 Scanning for audio files in: {data_dir}")
        
        # Find all .wav files
        audio_files = list(data_dir.rglob("*.wav"))
        
        if not audio_files:
            print(f"❌ No .wav files found in {data_dir}")
            return
        
        print(f"✓ Found {len(audio_files)} audio files")
        
        # Determine number of workers
        if num_workers is None:
            # Auto-detect: use 2 workers on local machine, all cores on HPC
            cpu_count = multiprocessing.cpu_count()
            if cpu_count <= 8:
                # Local machine - use 2 workers for easier debugging
                num_workers = 2
            else:
                # HPC/HiperGator - use all cores
                num_workers = min(cpu_count, len(audio_files))
        
        print(f"⚡ Using {num_workers} parallel workers (out of {multiprocessing.cpu_count()} CPUs)")
        
        # Process files in parallel
        vectors_batch = []
        successful = 0
        failed = 0
        
        print(f"\n🚀 Starting parallel pipeline processing...")
        print(f"   Audio → Spectrogram → Fingerprinting → Pinecone")
        print(f"   Processing {num_workers} songs at a time")
        
        # Use ProcessPoolExecutor for parallel processing
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(process_audio_file_standalone, audio_file): audio_file 
                for audio_file in audio_files
            }
            
            # Process completed tasks with progress bar
            for future in tqdm(as_completed(future_to_file), 
                             total=len(audio_files), 
                             desc="Processing", 
                             unit="song"):
                
                song_id, fingerprint, metadata = future.result()
                
                if song_id and fingerprint is not None:
                    # Prepare vector for upload
                    vectors_batch.append({
                        'id': song_id,
                        'values': fingerprint.tolist() if hasattr(fingerprint, 'tolist') else fingerprint,
                        'metadata': metadata
                    })
                    
                    # Upload in batches
                    if len(vectors_batch) >= batch_size:
                        if self.upload_to_pinecone(vectors_batch):
                            successful += len(vectors_batch)
                        else:
                            failed += len(vectors_batch)
                        vectors_batch = []
                else:
                    failed += 1
        
        # Upload remaining vectors
        if vectors_batch:
            if self.upload_to_pinecone(vectors_batch):
                successful += len(vectors_batch)
            else:
                failed += len(vectors_batch)
        
        # Final summary
        print(f"\n{'='*60}")
        print(f"🎉 Pipeline Complete!")
        print(f"{'='*60}")
        print(f"✓ Successfully processed: {successful} songs")
        print(f"❌ Failed: {failed} songs")
        print(f"📊 Success rate: {(successful/(successful+failed)*100):.1f}%")
        
        # Show final database stats
        stats = self.index.describe_index_stats()
        print(f"\n🗄️  Final database stats:")
        print(f"   Total vectors: {stats.get('total_vector_count', 0)}")
        print(f"   Dimension: {stats.get('dimension', 'N/A')}")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description='Process audio files and upload to Pinecone',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m database.pipeline --data_dir /path/to/audio/files
  python -m database.pipeline  (reads from dataset_path.txt)
  python -m database.pipeline --batch_size 50 --dimension 256
  python -m database.pipeline --num_workers 32  (use 32 CPU cores)
        """
    )
    
    parser.add_argument(
        '--data_dir', 
        type=str, 
        help='Directory containing audio files (reads from dataset_path.txt if not provided)'
    )
    parser.add_argument(
        '--index_name', 
        type=str, 
        default='audio-fingerprints',
        help='Pinecone index name (default: audio-fingerprints)'
    )
    parser.add_argument(
        '--dimension', 
        type=int, 
        default=128,
        help='Vector dimension (default: 128)'
    )
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=100,
        help='Batch size for uploads (default: 100)'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=None,
        help='Number of parallel workers (default: auto-detect CPU count)'
    )
    
    args = parser.parse_args()
    
    # Get data directory
    data_dir = args.data_dir
    
    # If not provided, try to read from dataset_path.txt
    if not data_dir:
        path_file = Path("dataset_path.txt")
        if path_file.exists():
            with open(path_file, "r") as f:
                data_dir = f.read().strip()
            print(f"📂 Using audio directory from {path_file}")
        else:
            print("❌ Error: No --data_dir provided and dataset_path.txt not found")
            print("Run: python download_dataset.py first")
            return
    
    # Verify directory exists
    if not Path(data_dir).exists():
        print(f"❌ Error: Directory not found: {data_dir}")
        return
    
    print(f"📂 Audio directory: {data_dir}")
    
    # Run pipeline
    pipeline = AudioPipeline(index_name=args.index_name)
    pipeline.create_index(dimension=args.dimension)
    pipeline.process_directory(data_dir, batch_size=args.batch_size, num_workers=args.num_workers)


if __name__ == "__main__":
    main()