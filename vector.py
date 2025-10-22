import numpy as np
import faiss

mock_database = [1.5523, 4.323231, 5.4213213, 9.231321, 10.2312414, 2.33334]

songs = {
    0: "Pasadena by Elijah Fox",
    1: "Party Rock",
    2: "Old McDonald",
    3: "Right my Wrongs by Bryson Tiller",
    4: "My Favorite Things by John Coltrane",
    5: "In a Sentimental Mood by Duke Ellington and John Coltrane",
}

d = 1                           # dimension

query_vector = (float(input("Enter query vector here: ")))

reshaped_query = np.array(query_vector).reshape(1,d)

mock_database = np.array(mock_database)
index_data = mock_database.astype("float32").reshape(len(mock_database),d)

index = faiss.IndexFlatL2(d)   # build the index

index.add(index_data)

D, I = index.search(reshaped_query, 1) 

print(songs[I.item()])
print(f"The similarity of the search to the database is {D}")