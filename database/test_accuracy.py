"""
Test search accuracy: Query 30 songs (3 per genre) and check if top result is correct
Usage: python -m database.test_accuracy [index_name]
"""

import sys
import os
import random
from pathlib import Path
from dotenv import load_dotenv
from pinecone import Pinecone
import soundfile as sf
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import spectrogram
import fingerprinting
from database.pipeline import constellation_to_vector

load_dotenv()


def get_test_files(data_dir, songs_per_genre=3):
    """Get random sample of songs from each genre"""
    data_dir = Path(data_dir)
    test_files = []
    
    # Find all genre directories
    genre_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    
    for genre_dir in genre_dirs:
        # Get all wav files in this genre
        wav_files = list(genre_dir.glob("*.wav"))
        
        if len(wav_files) >= songs_per_genre:
            # Randomly sample
            selected = random.sample(wav_files, songs_per_genre)
            test_files.extend(selected)
        else:
            # Use all available if less than requested
            test_files.extend(wav_files)
    
    return test_files


def search_song(audio_path, index, return_top_match=True):
    """Search for a song and return results"""
    try:
        # Load audio and generate vector
        signal, sample_rate = sf.read(audio_path)
        if signal.ndim > 1:
            signal = signal.mean(axis=1)
        
        spec = spectrogram.generate_spectrogram(signal)
        peaks, constellation, fingerprints = fingerprinting.fingerprint_spectrogram(spec)
        query_vector = constellation_to_vector(constellation, peaks, fingerprints, target_dim=128)
        
        # Query
        results = index.query(
            vector=query_vector.tolist(),
            top_k=5,
            include_metadata=True
        )
        
        if return_top_match:
            return results['matches'][0]
        return results
        
    except Exception as e:
        print(f"❌ Error processing {audio_path.name}: {e}")
        return None


def run_accuracy_test(index_name="audio-fingerprints-v4", songs_per_genre=3):
    """Run accuracy test on sample of songs"""
    
    print("🎯 AUDIO FINGERPRINT ACCURACY TEST")
    print("=" * 70)
    
    # Get dataset path
    path_file = Path("dataset_path.txt")
    if path_file.exists():
        with open(path_file, "r") as f:
            data_dir = f.read().strip()
    else:
        print("❌ Error: dataset_path.txt not found")
        return
    
    print(f"📂 Dataset: {data_dir}")
    
    # Get test files
    print(f"🎲 Selecting {songs_per_genre} random songs from each genre...")
    test_files = get_test_files(data_dir, songs_per_genre)
    random.shuffle(test_files)  # Randomize order
    
    print(f"✓ Selected {len(test_files)} test songs\n")
    
    # Connect to Pinecone
    print(f"🔗 Connecting to Pinecone index: {index_name}")
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    index = pc.Index(index_name)
    print()
    
    # Test each file
    correct = 0
    total = 0
    results_detail = []
    
    print(f"🔍 Testing search accuracy...")
    print("=" * 70)
    
    for audio_path in tqdm(test_files, desc="Testing", unit="song"):
        total += 1
        
        # Expected ID (filename without extension)
        expected_id = audio_path.stem
        
        # Search
        top_match = search_song(audio_path, index)
        
        if top_match:
            found_id = top_match['id']
            score = top_match['score']
            is_correct = (found_id == expected_id)
            
            if is_correct:
                correct += 1
            
            results_detail.append({
                'file': audio_path.name,
                'genre': audio_path.parent.name,
                'expected': expected_id,
                'found': found_id,
                'score': score,
                'correct': is_correct
            })
    
    # Calculate accuracy
    accuracy = (correct / total * 100) if total > 0 else 0
    
    # Print results
    print("\n" + "=" * 70)
    print("📊 RESULTS")
    print("=" * 70)
    print(f"Total tests:      {total}")
    print(f"Correct matches:  {correct}")
    print(f"Incorrect:        {total - correct}")
    print(f"Accuracy:         {accuracy:.1f}%")
    print("=" * 70)
    
    # Show failures
    failures = [r for r in results_detail if not r['correct']]
    if failures:
        print(f"\n❌ FAILURES ({len(failures)}):")
        print("-" * 70)
        for f in failures:
            print(f"File: {f['file']}")
            print(f"  Expected: {f['expected']}")
            print(f"  Found:    {f['found']} (score: {f['score']:.4f})")
            print()
    
    # Genre breakdown
    genre_stats = {}
    for r in results_detail:
        genre = r['genre']
        if genre not in genre_stats:
            genre_stats[genre] = {'correct': 0, 'total': 0}
        genre_stats[genre]['total'] += 1
        if r['correct']:
            genre_stats[genre]['correct'] += 1
    
    print("\n📈 ACCURACY BY GENRE:")
    print("-" * 70)
    for genre in sorted(genre_stats.keys()):
        stats = genre_stats[genre]
        genre_acc = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
        print(f"{genre:12} {stats['correct']}/{stats['total']} ({genre_acc:.1f}%)")
    
    print("\n" + "=" * 70)
    
    if accuracy == 100:
        print("🎉 PERFECT! All songs correctly identified!")
    elif accuracy >= 90:
        print("✅ Excellent accuracy!")
    elif accuracy >= 75:
        print("👍 Good accuracy, but room for improvement")
    else:
        print("⚠️  Accuracy needs improvement")
    
    return accuracy, results_detail


if __name__ == "__main__":
    index_name = sys.argv[1] if len(sys.argv) > 1 else "audio-fingerprints-v4"
    songs_per_genre = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    
    print(f"\nTesting with index: {index_name}")
    print(f"Songs per genre: {songs_per_genre}\n")
    
    run_accuracy_test(index_name, songs_per_genre)

