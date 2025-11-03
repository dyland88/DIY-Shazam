import kagglehub
from pathlib import Path

"""
Download GTZAN dataset from Kaggle
Usage: python download_dataset.py
"""

import kagglehub
from pathlib import Path

def download_gtzan():
    """Download GTZAN dataset and return path to audio files"""
    
    print("Downloading GTZAN dataset from Kaggle...")
    print("(This may take a few minutes on first download)")
    
    # Download dataset
    dataset_path = kagglehub.dataset_download(
        "andradaolteanu/gtzan-dataset-music-genre-classification"
    )
    
    print(f"\n✓ Dataset downloaded to: {dataset_path}")
    
    # Find the audio files directory
    dataset_path = Path(dataset_path)
    
    # Try different possible structures
    possible_paths = [
        dataset_path / "Data" / "genres_original",
        dataset_path / "genres_original",
        dataset_path / "GTZAN" / "genres_original",
    ]
    
    audio_dir = None
    for path in possible_paths:
        if path.exists():
            audio_dir = path
            break
    
    # If still not found, search recursively
    if not audio_dir:
        search_results = list(dataset_path.rglob("genres_original"))
        if search_results:
            audio_dir = search_results[0]
    
    if not audio_dir:
        print("❌ Could not find genres_original folder!")
        print(f"Dataset structure:")
        for item in dataset_path.rglob("*"):
            if item.is_dir():
                print(f"  📁 {item.relative_to(dataset_path)}")
        return None
    
    # Count audio files
    audio_files = list(audio_dir.rglob("*.wav"))
    
    print(f"\n✓ Found audio directory: {audio_dir}")
    print(f"✓ Total audio files: {len(audio_files)}")
    
    # Show genre breakdown
    print("\n📊 Genre breakdown:")
    genres = {}
    for audio_file in audio_files:
        genre = audio_file.parent.name
        genres[genre] = genres.get(genre, 0) + 1
    
    for genre, count in sorted(genres.items()):
        print(f"  {genre}: {count} files")
    
    # Save path to file for pipeline to use
    path_file = Path("dataset_path.txt")
    with open(path_file, "w") as f:
        f.write(str(audio_dir))
    
    print(f"\n✓ Audio path saved to {path_file}")
    print(f"\n🎵 Ready to run pipeline with: {audio_dir}")
    
    return audio_dir


if __name__ == "__main__":
    download_gtzan()