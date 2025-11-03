"""
Search for a song in the Pinecone database
Usage: python -m database.search /path/to/audio.wav
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv
from pinecone import Pinecone
import soundfile as sf

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import spectrogram
import fingerprinting
from database.pipeline import constellation_to_vector

load_dotenv()


def search_song(audio_path, index_name="audio-fingerprints-v5", top_k=10):
    """
    Search for a song in the Pinecone database
    
    :param audio_path: Path to audio file
    :param index_name: Name of Pinecone index
    :param top_k: Number of results to return
    """
    print(f"🔍 Searching for: {audio_path}")
    
    # Load audio and generate vector
    print("📊 Processing audio...")
    signal, sample_rate = sf.read(audio_path)
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    
    spec = spectrogram.generate_spectrogram(signal)
    peaks, constellation, fingerprints = fingerprinting.fingerprint_spectrogram(spec)
    query_vector = constellation_to_vector(constellation, peaks, fingerprints, target_dim=128)
    
    print(f"✓ Generated vector with {len(peaks)} peaks, {len(constellation)} constellation points")
    print(f"✓ Vector norm: {query_vector.sum():.4f}")
    
    # Connect to Pinecone
    print(f"\n🔗 Connecting to Pinecone index: {index_name}")
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    index = pc.Index(index_name)
    
    # Query
    print(f"🔎 Searching for top {top_k} matches...\n")
    results = index.query(
        vector=query_vector.tolist(),
        top_k=top_k,
        include_metadata=True
    )
    
    # Display results
    print(f"{'='*70}")
    print(f"🎵 SEARCH RESULTS")
    print(f"{'='*70}\n")
    
    for i, match in enumerate(results['matches'], 1):
        print(f"{i}. Score: {match['score']:.4f}")
        print(f"   ID: {match['id']}")
        print(f"   Genre: {match['metadata'].get('genre', 'N/A')}")
        print(f"   Filename: {match['metadata'].get('filename', 'N/A')}")
        print()
    
    # Analysis
    scores = [m['score'] for m in results['matches']]
    print(f"{'='*70}")
    print(f"📊 SCORE ANALYSIS")
    print(f"{'='*70}")
    print(f"Top score:    {scores[0]:.4f}")
    print(f"10th score:   {scores[-1]:.4f}")
    print(f"Score range:  {scores[0] - scores[-1]:.4f}")
    
    if scores[0] - scores[-1] < 0.1:
        print("\n⚠️  Warning: Scores are very close together!")
        print("    This suggests vectors may not be discriminative enough.")
    else:
        print("\n✅ Score spread looks good!")
    
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m database.search /path/to/audio.wav [index_name]")
        print("\nExample:")
        print("  python -m database.search audio/cello_suite.wav")
        print("  python -m database.search audio/test.wav audio-fingerprints-v5")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    index_name = sys.argv[2] if len(sys.argv) > 2 else "audio-fingerprints-v5"
    
    if not Path(audio_path).exists():
        print(f"❌ Error: File not found: {audio_path}")
        sys.exit(1)
    
    search_song(audio_path, index_name)

