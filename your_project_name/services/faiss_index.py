import faiss
import numpy as np

dimension = 768  # Dimension of embeddings
index = faiss.IndexFlatL2(dimension)

def add_embeddings_to_index(post_id, embeddings):
    index.add(np.array([embeddings]))
    print(f"Embeddings for post {post_id} added to the index")
