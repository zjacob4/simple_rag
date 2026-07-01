from new_knowledge import add_knowledge
from extraction import get_embeddings
import numpy as np

def conduct_rag_retrieval(query):
    embedded_query = get_embeddings(query)
    
    # get embedded knowledge
    knowledge_index, text_lookup = add_knowledge()

    # cosine similarity search on embedded_query against knowledge (list of embeddings)
    distances, indices = knowledge_index.search(np.array(embedded_query, dtype=np.float32).reshape(1,-1), k=2)
    
    # convert returned indices to vectors
    #vectors = knowledge_index.reconstruct_batch(indices[0].tolist())

    # get retrieved chunks by index using text lookup dictionary
    chunks = [text_lookup[i] for i in indices[0]]

    return chunks

if __name__ == "__main__":
    retrieved_chunks = conduct_rag_retrieval()
    print("Retrieved Chunks: ", retrieved_chunks)