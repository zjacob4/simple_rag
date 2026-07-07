from extraction import get_embeddings
import numpy as np
import faiss
import pickle
from pinecone import Pinecone
from dotenv import load_dotenv
import os

# load pinecone api key from .env file
load_dotenv()
pinecone_key = os.getenv("PINECONE_API_KEY")

pc_client = Pinecone(api_key=pinecone_key)

def faiss_retrieval(query):
    embedded_query = get_embeddings(query)

    knowledge_index = faiss.read_index("simple_rag.index")
    with open("text_lookup.pkl","rb") as f:
        text_lookup = pickle.load(f)

    # euclidean brute force search on embedded_query against knowledge (list of embeddings)
    distances, indices = knowledge_index.search(np.array(embedded_query, dtype=np.float32).reshape(1,-1), k=2)

    # get retrieved chunks by index using text lookup dictionary
    chunks = [text_lookup[i] for i in indices[0]]

    return chunks

def pinecone_retrieval(query):

    embedded_query = get_embeddings(query)

    # get pinecone index
    index = pc_client.index("simple-rag")

    # execute similarity search
    response = index.query(
        vector = embedded_query,
        top_k = 5,
        include_values = False,
        include_metadata = True,
        namespace = "__default__"
    )
    
    # get retrieved chunks by index using text lookup dictionary
    chunks = [match["metadata"]["text"]+" with a score of "+str(match["score"]) for match in response["matches"]]

    return chunks

if __name__ == "__main__":
    retrieved_chunks = faiss_retrieval()
    print("Retrieved Chunks: ", retrieved_chunks)