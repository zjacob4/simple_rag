from extraction import get_embeddings
import numpy as np
import faiss
import pickle

def conduct_rag_retrieval(query):
    embedded_query = get_embeddings(query)

    knowledge_index = faiss.read_index("simple_rag.index")
    with open("text_lookup.pkl","rb") as f:
        text_lookup = pickle.load(f)

    # euclidean brute force search on embedded_query against knowledge (list of embeddings)
    distances, indices = knowledge_index.search(np.array(embedded_query, dtype=np.float32).reshape(1,-1), k=2)

    # get retrieved chunks by index using text lookup dictionary
    chunks = [text_lookup[i] for i in indices[0]]

    return chunks

if __name__ == "__main__":
    retrieved_chunks = conduct_rag_retrieval()
    print("Retrieved Chunks: ", retrieved_chunks)