from pinecone import Pinecone

pc = Pinecone(api_key="pcsk_5jbA3t_Rmw7FnvLAcYYuporoMJdQfp7MSXtySoZRFZHxgNZFmYCynncVr5QVDrjMtaYfbc")
index_name = "shazam-3d"


#check if index exists, if not create one
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension = 3,
        metric = "cosine", # Use cosine similarity (good for audio, ignores volume differences)
        spec={"serverless": {"cloud": "aws", "region": "us-east-1"}}
    )


index = pc.Index(index_name)

#creates metadata about mock songs
songs = {
    0: "Pasadena by Elijah Fox",
    1: "Party Rock",
    2: "Old McDonald",
    3: "Right my Wrongs by Bryson Tiller",
    4: "My Favorite Things by John Coltrane",
    5: "In a Sentimental Mood by Duke Ellington and John Coltrane",
}


# Mock 3D vectors extracted from spectrograms (these would come from real audio later)
# Each vector represents the "fingerprint" of a song
mock_vectors = [
    [0.1, 0.5, 0.9],
    [0.2, 0.6, 0.8],
    [0.15, 0.55, 0.85],
    [0.8, 0.2, 0.1],
    [0.9, 0.1, 0.2],
    [0.85, 0.15, 0.15],
]

# Create a list of tuples to store in Pinecone
# This loops through each vector and creates a tuple with:
#   - str(i): unique ID as a string
#   - vector: the 3D vector itself
#   - {"song": songs[i]}: metadata dict with the song name
vectors_to_store = [
    (str(i), vector, {"song": songs[i]})
    for i, vector in enumerate(mock_vectors)
]

#uploads them
index.upsert(vectors=vectors_to_store)

#simulates vector we get from spectogram from random song
query_vector = [.12, .52, .88]
results = index.query(vector=query_vector, top_k = 1, include_metadata=True)

#it gives us a list of closest matches, we want the best one
match = results['matches'][0]
print(f"Found: {match['metadata']['song']}")
print(f"Similarity score: {match['score']}")
